use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use leptess::{LepTess, Variable};

use crate::frames::Frame;
use crate::scan::{axis_offsets, Candidate, Scanner};

const CHAR_WHITELIST: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
/// Longest run of adjacent words on one line that we try to join. Plates split
/// into at most three fragments in practice ("8ABC 123", "CA 8ABC 123").
const MAX_JOIN: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct OcrSettings {
    pub tile: (u32, u32),
    pub overlap: f64,
    pub upscale: f64,
    pub psm: u8,
    pub min_conf: f32,
    pub min_height: f32,
    pub whitelist: bool,
}

pub struct Engine {
    tess: LepTess,
    settings: OcrSettings,
    png: Vec<u8>,
}

/// Leptonica writes clipping complaints straight to stderr for every tile it
/// dislikes, which buries the progress line. Silence it once per process.
fn silence_engines() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        leptonica_sys::setMsgSeverity(leptonica_sys::L_SEVERITY_NONE as i32);
    });
}

impl Engine {
    pub fn new(tessdata: Option<&Path>, lang: &str, settings: OcrSettings) -> Result<Engine> {
        silence_engines();
        let data_path = tessdata.map(|p| p.to_string_lossy().into_owned());
        let mut tess = LepTess::new(data_path.as_deref(), lang).with_context(|| {
            format!(
                "could not initialise tesseract for language {lang:?}{}. \
                 Install language data with `brew install tesseract-lang` or pass --tessdata",
                data_path
                    .as_deref()
                    .map(|p| format!(" using tessdata at {p}"))
                    .unwrap_or_default()
            )
        })?;
        // Tesseract narrates every tile ("Detected N diacritics") unless its
        // debug stream is redirected.
        let _ = tess.set_variable(Variable::DebugFile, "/dev/null");
        tess.set_variable(Variable::TesseditPagesegMode, &settings.psm.to_string())
            .map_err(|e| anyhow::anyhow!("failed to set page segmentation mode: {e}"))?;
        if settings.whitelist {
            tess.set_variable(Variable::TesseditCharWhitelist, CHAR_WHITELIST)
                .map_err(|e| anyhow::anyhow!("failed to set character whitelist: {e}"))?;
        }
        Ok(Engine {
            tess,
            settings,
            png: Vec::new(),
        })
    }

    /// OCR one frame tile by tile, returning plate-shaped text candidates.
    fn scan_frame(&mut self, frame: &Frame) -> Result<Vec<Candidate>> {
        let (tw, th) = (
            self.settings.tile.0.min(frame.width),
            self.settings.tile.1.min(frame.height),
        );
        let mut best: HashMap<String, Candidate> = HashMap::new();
        for oy in axis_offsets(frame.height, th, self.settings.overlap) {
            for ox in axis_offsets(frame.width, tw, self.settings.overlap) {
                for cand in self.scan_tile(frame, ox, oy, tw, th)? {
                    best.entry(cand.text.clone())
                        .and_modify(|c| {
                            if cand.conf > c.conf {
                                *c = cand.clone();
                            }
                        })
                        .or_insert(cand);
                }
            }
        }
        Ok(best.into_values().collect())
    }

    fn scan_tile(
        &mut self,
        frame: &Frame,
        ox: u32,
        oy: u32,
        tw: u32,
        th: u32,
    ) -> Result<Vec<Candidate>> {
        let tile = frame.crop_gray(ox, oy, tw, th);
        let scale = self.settings.upscale;
        let scaled = if scale > 1.0 {
            image::imageops::resize(
                &tile,
                (tw as f64 * scale).round() as u32,
                (th as f64 * scale).round() as u32,
                image::imageops::FilterType::CatmullRom,
            )
        } else {
            tile
        };

        self.png.clear();
        PngEncoder::new_with_quality(&mut self.png, CompressionType::Fast, PngFilter::NoFilter)
            .write_image(
                scaled.as_raw(),
                scaled.width(),
                scaled.height(),
                ExtendedColorType::L8,
            )
            .context("failed to encode tile for OCR")?;

        self.tess
            .set_image_from_mem(&self.png)
            .map_err(|e| anyhow::anyhow!("tesseract rejected the tile image: {e}"))?;
        self.tess.set_source_resolution(300);
        let tsv = self
            .tess
            .get_tsv_text(0)
            .context("tesseract returned invalid UTF-8")?;

        // Map upscaled tile coordinates back to the full source frame.
        let to_source = |v: f32, origin: u32, roi: u32| v / scale as f32 + origin as f32 + roi as f32;
        let mut out = Vec::new();
        for c in candidates(&tsv) {
            let h_src = c.h / scale as f32;
            let w_src = c.w / scale as f32;
            if c.conf < self.settings.min_conf || h_src < self.settings.min_height {
                continue;
            }
            // Plate text sits in a wide, shallow box; anything squarer or
            // taller than wide is signage or bodywork.
            let aspect = w_src / h_src.max(1.0);
            if !(1.2..=9.0).contains(&aspect) {
                continue;
            }
            out.push(Candidate {
                text: c.text,
                conf: c.conf,
                rect: (
                    to_source(c.x, ox, frame.origin.0).max(0.0) as u32,
                    to_source(c.y, oy, frame.origin.1).max(0.0) as u32,
                    w_src.max(1.0) as u32,
                    h_src.max(1.0) as u32,
                ),
                detector_score: None,
                region: None,
            });
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct TsvWord {
    line: (u32, u32, u32),
    text: String,
    conf: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// Words, plus every run of up to `MAX_JOIN` adjacent words on the same line,
/// so plates split across OCR words are still recovered.
fn candidates(tsv: &str) -> Vec<TsvWord> {
    let words = parse_tsv(tsv);
    let mut out: Vec<TsvWord> = Vec::new();
    for (idx, w) in words.iter().enumerate() {
        out.push(w.clone());
        let mut joined = w.clone();
        for next in words.iter().skip(idx + 1).take(MAX_JOIN - 1) {
            if next.line != w.line {
                break;
            }
            let right = (joined.x + joined.w).max(next.x + next.w);
            let bottom = (joined.y + joined.h).max(next.y + next.h);
            joined = TsvWord {
                line: joined.line,
                text: format!("{}{}", joined.text, next.text),
                conf: joined.conf.min(next.conf),
                x: joined.x.min(next.x),
                y: joined.y.min(next.y),
                w: right - joined.x.min(next.x),
                h: bottom - joined.y.min(next.y),
            };
            out.push(joined.clone());
        }
    }
    out
}

fn parse_tsv(tsv: &str) -> Vec<TsvWord> {
    let mut out = Vec::new();
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        // level page block par line word left top width height conf text
        if f.len() < 12 || f[0] != "5" {
            continue;
        }
        let num = |i: usize| f[i].parse::<f32>().ok();
        let idx = |i: usize| f[i].parse::<u32>().ok();
        let (Some(block), Some(par), Some(ln)) = (idx(2), idx(3), idx(4)) else {
            continue;
        };
        let (Some(x), Some(y), Some(w), Some(h), Some(conf)) =
            (num(6), num(7), num(8), num(9), num(10))
        else {
            continue;
        };
        let text = f[11].trim();
        if text.is_empty() || conf < 0.0 {
            continue;
        }
        out.push(TsvWord {
            line: (block, par, ln),
            text: text.to_string(),
            conf,
            x,
            y,
            w,
            h,
        });
    }
    out
}

impl Scanner for Engine {
    fn scan(&mut self, frame: &Frame) -> Result<Vec<Candidate>> {
        self.scan_frame(frame)
    }

    fn describe(&self) -> String {
        let s = &self.settings;
        format!(
            "{}; tiles {}x{}, {:.0}% overlap, {:.1}x upscale, psm {}",
            tesseract_version(),
            s.tile.0,
            s.tile.1,
            s.overlap * 100.0,
            s.upscale,
            s.psm
        )
    }
}

fn tesseract_version() -> String {
    std::process::Command::new("tesseract")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_else(|| "tesseract".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_word_rows_only() {
        let tsv = "level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext\n\
                   4\t1\t1\t1\t1\t0\t0\t0\t100\t20\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t10\t20\t60\t18\t91.5\t8ABC\n\
                   5\t1\t1\t1\t1\t2\t75\t20\t50\t18\t88.0\t123\n";
        let words = parse_tsv(tsv);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "8ABC");
        assert_eq!(words[1].conf, 88.0);
    }

    #[test]
    fn joins_adjacent_words_on_one_line() {
        let tsv = "h\th\th\th\th\th\th\th\th\th\th\th\n\
                   5\t1\t1\t1\t1\t1\t10\t20\t60\t18\t91.5\t8ABC\n\
                   5\t1\t1\t1\t1\t2\t75\t20\t50\t18\t88.0\t123\n";
        let texts: Vec<String> = candidates(tsv).into_iter().map(|c| c.text).collect();
        assert!(texts.contains(&"8ABC123".to_string()), "got {texts:?}");
        let joined = candidates(tsv)
            .into_iter()
            .find(|c| c.text == "8ABC123")
            .expect("join present");
        assert_eq!(joined.conf, 88.0, "joined conf is the weakest part");
        assert_eq!(joined.x, 10.0);
        assert_eq!(joined.w, 115.0, "box spans both words");
    }

    #[test]
    fn does_not_join_across_lines() {
        let tsv = "h\th\th\th\th\th\th\th\th\th\th\th\n\
                   5\t1\t1\t1\t1\t1\t10\t20\t60\t18\t91.5\tABC\n\
                   5\t1\t1\t1\t2\t1\t10\t60\t50\t18\t88.0\t123\n";
        let texts: Vec<String> = candidates(tsv).into_iter().map(|c| c.text).collect();
        assert!(!texts.contains(&"ABC123".to_string()), "got {texts:?}");
    }
}
