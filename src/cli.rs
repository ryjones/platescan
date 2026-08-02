use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

/// Scan dashcam video for license plates and write a timestamped Markdown report.
#[derive(Parser, Debug)]
#[command(name = "platescan", version, about, long_about = None)]
pub struct Cli {
    /// Video files to scan (e.g. 2026_0801_131218_000289F.MP4), directories
    /// of them (scanned recursively, and grouped into trips), or a previous
    /// run's --json report to rebuild a KMZ without rescanning
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Markdown report path (default: <stem>-plates.md for one input)
    #[arg(short, long)]
    pub out: Option<PathBuf>,

    /// Also write raw findings as JSON
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// Also write a Google Earth KMZ: one placemark per sighting, positioned
    /// from the GPS track, with the verification crop embedded. The path is
    /// optional (--kmz=path to set it) and defaults to the report's name.
    /// To build one from a previous run without rescanning, pass that run's
    /// --json report as the input instead of a video.
    #[arg(long, num_args = 0..=1, default_missing_value = "auto", require_equals = true)]
    pub kmz: Option<PathBuf>,

    /// Group the inputs into trips — runs of clips whose recordings are
    /// contiguous — and write one consolidated report per trip. Front and
    /// rear clips recorded together land in the same trip. --out names a
    /// directory in this mode, and raw JSON findings are always written so
    /// a KMZ can be rebuilt later without rescanning.
    #[arg(long)]
    pub by_trip: bool,

    /// Seconds the camera may be off between clips that still count as the
    /// same trip (--by-trip)
    #[arg(long, default_value_t = 90.0)]
    pub trip_gap: f64,

    /// Recognition backend
    #[arg(long, value_enum, default_value_t = EngineKind::Alpr)]
    pub engine: EngineKind,

    /// Directory holding the ONNX detector and recognizer
    #[arg(long, default_value = "models")]
    pub models: PathBuf,

    /// Path to libonnxruntime (default: $ORT_DYLIB_PATH, then the usual
    /// Homebrew and system locations)
    #[arg(long)]
    pub ort_dylib: Option<PathBuf>,

    /// Minimum plate-detector objectness, 0.0-1.0 (alpr engine)
    #[arg(long, default_value_t = 0.35)]
    pub det_conf: f32,

    /// Require readings to match the --region plate format. Off by default for
    /// the alpr engine, whose detector already establishes that it is a plate.
    #[arg(long)]
    pub strict_format: bool,

    /// Frames sampled per second of video
    #[arg(long, default_value_t = 2.0)]
    pub fps: f64,

    /// Start offset in seconds
    #[arg(long, default_value_t = 0.0)]
    pub start: f64,

    /// End offset in seconds (default: end of clip)
    #[arg(long)]
    pub end: Option<f64>,

    /// Region of interest as fractions of the frame: x,y,w,h. The default is
    /// the road band of a forward dashcam: below the horizon, above the bonnet,
    /// and clear of the camera's own on-screen display.
    #[arg(long, default_value = "0.05,0.44,0.90,0.25", value_parser = parse_roi)]
    pub roi: Roi,

    /// OCR tile size in source pixels, WxH
    #[arg(long, default_value = "960x540", value_parser = parse_dims)]
    pub tile: (u32, u32),

    /// Fractional overlap between neighbouring tiles
    #[arg(long, default_value_t = 0.25)]
    pub overlap: f64,

    /// Upscale factor applied to each tile before OCR
    #[arg(long, default_value_t = 3.0)]
    pub upscale: f64,

    /// Minimum tesseract word confidence (0-100)
    #[arg(long, default_value_t = 60.0)]
    pub min_conf: f32,

    /// Minimum plate text height in source pixels
    #[arg(long, default_value_t = 9.0)]
    pub min_height: f32,

    /// Distinct sampled frames a plate must appear in to be reported
    #[arg(long, default_value_t = 2)]
    pub min_hits: usize,

    /// Seconds without a re-detection before a sighting is closed. Detection of
    /// an approaching vehicle is intermittent, so this needs to be generous or
    /// one car is reported several times.
    #[arg(long, default_value_t = 10.0)]
    pub gap: f64,

    /// Plate format preset
    #[arg(long, value_enum, default_value_t = Region::Generic)]
    pub region: Region,

    /// Extra plate regex (uppercase, no separators). Repeatable; replaces the
    /// preset patterns when given.
    #[arg(long = "pattern")]
    pub patterns: Vec<String>,

    /// Tesseract page segmentation mode
    #[arg(long, default_value_t = 11)]
    pub psm: u8,

    /// Tesseract language
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Let tesseract emit punctuation and lowercase instead of restricting it
    /// to A-Z0-9
    #[arg(long)]
    pub no_whitelist: bool,

    /// tessdata directory (default: tesseract's built-in search path)
    #[arg(long)]
    pub tessdata: Option<PathBuf>,

    /// Parallel OCR workers (default: CPU cores - 1)
    #[arg(short = 'j', long)]
    pub jobs: Option<usize>,

    /// Do not save a cropped still for each sighting
    #[arg(long)]
    pub no_crops: bool,

    /// Directory for sighting crops (default: <out>-crops next to the report)
    #[arg(long)]
    pub crop_dir: Option<PathBuf>,

    /// Overwrite an existing report instead of writing alongside it. Without
    /// this, a rerun never destroys the previous run's output.
    #[arg(long)]
    pub force: bool,

    /// Suppress progress output
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// ONNX plate detector plus plate-specific recognizer. Reads plates at
    /// road distances; needs the models and onnxruntime.
    Alpr,
    /// Tiled tesseract OCR. No models needed, but only reads plates that are
    /// large in frame.
    Tesseract,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Region {
    /// Any 5-8 character mix of letters and digits
    Generic,
    /// Common North American formats
    Us,
    /// Common UK / EU formats
    Eu,
}

#[derive(Copy, Clone, Debug)]
pub struct Roi {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Roi {
    /// Resolve to even-aligned pixel coordinates inside a `width` x `height` frame.
    pub fn resolve(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let even = |v: f64, max: u32| ((v.round() as i64).clamp(0, max as i64) as u32) & !1;
        let x = even(self.x * width as f64, width);
        let y = even(self.y * height as f64, height);
        let w = even(self.w * width as f64, width - x).max(2);
        let h = even(self.h * height as f64, height - y).max(2);
        (x, y, w.min(width - x), h.min(height - y))
    }
}

fn parse_roi(s: &str) -> Result<Roi> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("bad --roi value {s:?}: {e}"))?;
    let [x, y, w, h] = parts[..] else {
        bail!("--roi needs four comma-separated fractions: x,y,w,h");
    };
    for v in [x, y, w, h] {
        if !(0.0..=1.0).contains(&v) {
            bail!("--roi fractions must be within 0.0..=1.0, got {v}");
        }
    }
    if x + w > 1.0 || y + h > 1.0 {
        bail!("--roi extends past the frame edge");
    }
    if w <= 0.0 || h <= 0.0 {
        bail!("--roi width and height must be positive");
    }
    Ok(Roi { x, y, w, h })
}

fn parse_dims(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow::anyhow!("expected WxH, got {s:?}"))?;
    let w: u32 = w.trim().parse()?;
    let h: u32 = h.trim().parse()?;
    if w < 64 || h < 64 {
        bail!("tile size must be at least 64x64");
    }
    Ok((w, h))
}

impl Cli {
    pub fn validate(&self) -> Result<()> {
        if self.fps <= 0.0 {
            bail!("--fps must be positive");
        }
        if !(0.0..0.9).contains(&self.overlap) {
            bail!("--overlap must be within 0.0..0.9");
        }
        if !(1.0..=8.0).contains(&self.upscale) {
            bail!("--upscale must be within 1.0..=8.0");
        }
        if let Some(end) = self.end {
            if end <= self.start {
                bail!("--end must be greater than --start");
            }
        }
        if self.psm > 13 {
            bail!("--psm must be 0..=13");
        }
        if self.trip_gap <= 0.0 {
            bail!("--trip-gap must be positive");
        }
        Ok(())
    }

    pub fn workers(&self) -> usize {
        self.jobs.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(1).max(1))
                .unwrap_or(1)
        })
    }
}
