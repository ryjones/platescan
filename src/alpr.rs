//! ONNX backend: a YOLO license-plate detector feeding a plate-specific
//! recognizer. Unlike general-purpose OCR this localises the plate first, so
//! the recognizer only ever sees an isolated, normalised plate.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::frames::Frame;
use crate::scan::{Candidate, Scanner};

/// YOLO's letterbox fill.
const PAD: u8 = 114;
/// Detector rows are `[batch, x1, y1, x2, y2, class, score]`.
const DET_COLS: usize = 7;

#[derive(Clone, Debug)]
pub struct AlprSettings {
    pub models: PathBuf,
    pub dylib: Option<PathBuf>,
    /// Minimum detector objectness for a box to be recognised.
    pub det_conf: f32,
    pub overlap: f64,
    /// Discard boxes shorter than this in source pixels.
    pub min_height: f32,
    pub min_conf: f32,
}

/// Alphabet and geometry the recognizer was trained with.
#[derive(Clone, Debug)]
pub struct PlateConfig {
    pub alphabet: Vec<char>,
    pub slots: usize,
    pub width: u32,
    pub height: u32,
    pub regions: Vec<String>,
}

impl Default for PlateConfig {
    fn default() -> Self {
        PlateConfig {
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".chars().collect(),
            slots: 10,
            width: 128,
            height: 64,
            regions: Vec::new(),
        }
    }
}

impl PlateConfig {
    /// Read the recognizer's sidecar YAML if it is present. Only the handful of
    /// scalar keys we depend on are parsed, so no YAML dependency is needed.
    pub fn load(dir: &Path) -> PlateConfig {
        let mut cfg = PlateConfig::default();
        let Some(text) = find_config(dir) else {
            return cfg;
        };
        for line in text.lines() {
            let line = line.trim();
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().trim_matches(|c| c == '\'' || c == '"');
            match key.trim() {
                "alphabet" if !value.is_empty() => cfg.alphabet = value.chars().collect(),
                "max_plate_slots" => {
                    if let Ok(v) = value.parse() {
                        cfg.slots = v;
                    }
                }
                "img_width" => {
                    if let Ok(v) = value.parse() {
                        cfg.width = v;
                    }
                }
                "img_height" => {
                    if let Ok(v) = value.parse() {
                        cfg.height = v;
                    }
                }
                _ => {}
            }
        }
        cfg.regions = parse_regions(&text);
        cfg
    }
}

fn find_config(dir: &Path) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("plate_config.yaml"))
        })
        .collect();
    paths.sort();
    std::fs::read_to_string(paths.first()?).ok()
}

/// Pull the `plate_regions: [ 'a', 'b', ... ]` block, which spans many lines.
fn parse_regions(text: &str) -> Vec<String> {
    let Some(start) = text.find("plate_regions:") else {
        return Vec::new();
    };
    let tail = &text[start..];
    let Some(open) = tail.find('[') else {
        return Vec::new();
    };
    let Some(close) = tail.find(']') else {
        return Vec::new();
    };
    tail[open + 1..close]
        .split(',')
        .map(|s| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub struct Alpr {
    det: Session,
    rec: Session,
    cfg: PlateConfig,
    settings: AlprSettings,
    /// Detector input edge in pixels, read from the model itself.
    window: u32,
    det_path: PathBuf,
    rec_path: PathBuf,
}

/// The square input edge a detector expects, from its `[1, 3, H, W]` input.
fn input_edge(session: &Session) -> Option<u32> {
    let ort::value::ValueType::Tensor { shape, .. } = session.inputs().first()?.dtype() else {
        return None;
    };
    let dims: Vec<i64> = shape.iter().copied().collect();
    match dims.as_slice() {
        [_, _, h, w] if *h > 0 && *w > 0 => Some((*h).min(*w) as u32),
        _ => None,
    }
}

impl Alpr {
    pub fn new(settings: &AlprSettings) -> Result<Alpr> {
        init_runtime(settings.dylib.as_deref())?;
        let models = resolve_models(&settings.models)?;
        let det_path = pick(&models, &["license-plate", "license_plate"], "detector")?;
        let rec_path = pick(&models, &["cct", "ocr", "vit"], "recognizer")?;
        let det = Session::builder()?
            .commit_from_file(&det_path)
            .with_context(|| format!("failed to load detector {}", det_path.display()))?;
        let rec = Session::builder()?
            .commit_from_file(&rec_path)
            .with_context(|| format!("failed to load recognizer {}", rec_path.display()))?;
        let window = input_edge(&det).unwrap_or(640);
        Ok(Alpr {
            det,
            rec,
            cfg: PlateConfig::load(&models),
            settings: settings.clone(),
            window,
            det_path,
            rec_path,
        })
    }

    /// Detector windows are taken at native resolution, so a plate that is 40 px
    /// wide in the source is still 40 px wide when the model sees it.
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let size = self.window;
        let tw = size.min(frame.width);
        let th = size.min(frame.height);
        let mut boxes = Vec::new();
        for oy in crate::scan::axis_offsets(frame.height, th, self.settings.overlap) {
            for ox in crate::scan::axis_offsets(frame.width, tw, self.settings.overlap) {
                let window = frame.crop_rgb(ox, oy, tw, th);
                boxes.extend(self.detect_window(&window, size, ox, oy, frame.origin)?);
            }
        }
        Ok(nms(boxes, 0.4))
    }

    fn detect_window(
        &mut self,
        window: &RgbImage,
        size: u32,
        ox: u32,
        oy: u32,
        origin: (u32, u32),
    ) -> Result<Vec<Detection>> {
        let (canvas, scale, px, py) = letterbox(window, size);
        let plane = (size * size) as usize;
        let mut chw = vec![0f32; 3 * plane];
        for (i, p) in canvas.pixels().enumerate() {
            chw[i] = p.0[0] as f32 / 255.0;
            chw[plane + i] = p.0[1] as f32 / 255.0;
            chw[2 * plane + i] = p.0[2] as f32 / 255.0;
        }
        let input = Tensor::from_array((vec![1i64, 3, size as i64, size as i64], chw))?;
        let name = self.det.inputs()[0].name().to_string();
        let outputs = self.det.run(ort::inputs![name => input])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;

        let mut out = Vec::new();
        for row in data.chunks(DET_COLS) {
            let score = row[6];
            if score < self.settings.det_conf {
                continue;
            }
            let un = |v: f32, pad: u32| ((v - pad as f32) / scale).max(0.0);
            let (x1, y1) = (un(row[1], px), un(row[2], py));
            let (x2, y2) = (un(row[3], px), un(row[4], py));
            let (w, h) = (x2 - x1, y2 - y1);
            if w < 1.0 || h < self.settings.min_height {
                continue;
            }
            out.push(Detection {
                // Window coordinates -> ROI coordinates -> full-frame coordinates.
                x: x1 + ox as f32,
                y: y1 + oy as f32,
                w,
                h,
                score,
                origin,
            });
        }
        Ok(out)
    }

    /// Recognise every detected plate in one batched forward pass.
    fn recognise(&mut self, frame: &Frame, dets: &[Detection]) -> Result<Vec<Candidate>> {
        if dets.is_empty() {
            return Ok(Vec::new());
        }
        let (cw, ch) = (self.cfg.width, self.cfg.height);
        let mut batch = Vec::with_capacity(dets.len() * (cw * ch * 3) as usize);
        for d in dets {
            let crop = frame.crop_rgb(
                d.x.max(0.0) as u32,
                d.y.max(0.0) as u32,
                (d.w.round() as u32).max(1),
                (d.h.round() as u32).max(1),
            );
            let resized = image::imageops::resize(&crop, cw, ch, FilterType::Triangle);
            batch.extend(resized.pixels().flat_map(|p| p.0));
        }

        let n = dets.len() as i64;
        let input = Tensor::from_array((vec![n, ch as i64, cw as i64, 3], batch))?;
        let name = self.rec.inputs()[0].name().to_string();
        let outputs = self.rec.run(ort::inputs![name => input])?;
        // Named output when the model provides one, else the first.
        let plate_value = outputs.get("plate").unwrap_or(&outputs[0]);
        let (shape, plate) = plate_value.try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        let [_, slot_count, classes] = dims[..] else {
            bail!("expected a [batch, slots, classes] recognizer output, got {dims:?}");
        };
        let (slot_count, classes) = (slot_count as usize, classes as usize);
        if classes != self.cfg.alphabet.len() {
            bail!(
                "recognizer emits {classes} classes but the alphabet has {}; \
                 the model and its plate_config.yaml disagree",
                self.cfg.alphabet.len()
            );
        }
        let regions = outputs
            .get("region")
            .and_then(|r| r.try_extract_tensor::<f32>().ok());

        // Trust the model's own slot count over the sidecar config.
        let per_plate = slot_count * classes;
        let mut out = Vec::with_capacity(dets.len());
        for (i, d) in dets.iter().enumerate() {
            let slots = &plate[i * per_plate..(i + 1) * per_plate];
            let (text, conf) = decode(slots, classes, &self.cfg.alphabet);
            if text.is_empty() || conf * 100.0 < self.settings.min_conf {
                continue;
            }
            let region = regions.as_ref().and_then(|(rshape, rdata)| {
                let stride = *rshape.last()? as usize;
                let row = rdata.get(i * stride..(i + 1) * stride)?;
                let idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(k, _)| k)?;
                self.cfg.regions.get(idx).cloned()
            });
            out.push(Candidate {
                text,
                // Report on the same 0-100 scale as the OCR backend.
                conf: conf * 100.0,
                rect: (
                    (d.x + d.origin.0 as f32).max(0.0) as u32,
                    (d.y + d.origin.1 as f32).max(0.0) as u32,
                    d.w.max(1.0) as u32,
                    d.h.max(1.0) as u32,
                ),
                detector_score: Some(d.score),
                region,
            });
        }
        Ok(out)
    }
}

impl Scanner for Alpr {
    fn scan(&mut self, frame: &Frame) -> Result<Vec<Candidate>> {
        let dets = self.detect(frame)?;
        self.recognise(frame, &dets)
    }

    fn describe(&self) -> String {
        let name = |p: &Path| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string()
        };
        format!(
            "onnx alpr: {} + {}; {win}x{win} windows at native resolution, \
             {:.0}% overlap, detector score >= {:.2}",
            name(&self.det_path),
            name(&self.rec_path),
            self.settings.overlap * 100.0,
            self.settings.det_conf,
            win = self.window,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
    origin: (u32, u32),
}

/// Highest scoring character per slot; padding marks the end of the plate.
fn decode(slots: &[f32], classes: usize, alphabet: &[char]) -> (String, f32) {
    let mut text = String::new();
    let mut total = 0.0;
    let mut counted = 0;
    for slot in slots.chunks(classes) {
        let Some((idx, &p)) = slot.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) else {
            continue;
        };
        let Some(&ch) = alphabet.get(idx) else {
            continue;
        };
        if ch == '_' {
            continue;
        }
        text.push(ch);
        total += p;
        counted += 1;
    }
    (text, if counted == 0 { 0.0 } else { total / counted as f32 })
}

fn letterbox(img: &RgbImage, size: u32) -> (RgbImage, f32, u32, u32) {
    let scale = (size as f32 / img.width() as f32).min(size as f32 / img.height() as f32);
    let nw = ((img.width() as f32 * scale).round() as u32).max(1);
    let nh = ((img.height() as f32 * scale).round() as u32).max(1);
    let resized = if scale == 1.0 {
        img.clone()
    } else {
        image::imageops::resize(img, nw, nh, FilterType::CatmullRom)
    };
    let mut canvas = RgbImage::from_pixel(size, size, image::Rgb([PAD, PAD, PAD]));
    let (px, py) = ((size - nw) / 2, (size - nh) / 2);
    image::imageops::replace(&mut canvas, &resized, px as i64, py as i64);
    (canvas, scale, px, py)
}

/// Overlapping detector windows see the same plate twice; keep the best box.
fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    'outer: for d in dets {
        for k in &kept {
            if iou(&d, k) > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(d);
    }
    kept
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn init_runtime(dylib: Option<&Path>) -> Result<()> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let mut err = None;
    ONCE.call_once(|| {
        let candidates: Vec<PathBuf> = dylib
            .map(|p| vec![p.to_path_buf()])
            .unwrap_or_else(|| {
                std::env::var("ORT_DYLIB_PATH")
                    .ok()
                    .map(|p| vec![PathBuf::from(p)])
                    .unwrap_or_else(|| {
                        vec![
                            PathBuf::from("/opt/homebrew/lib/libonnxruntime.dylib"),
                            PathBuf::from("/usr/local/lib/libonnxruntime.dylib"),
                            PathBuf::from("/usr/lib/libonnxruntime.so"),
                        ]
                    })
            });
        match candidates.iter().find(|p| p.exists()) {
            Some(path) => match ort::init_from(path) {
                Ok(builder) => {
                    builder.commit();
                }
                Err(e) => err = Some(anyhow!("failed to load {}: {e}", path.display())),
            },
            None => {
                err = Some(anyhow!(
                    "onnxruntime not found. Install it with `brew install onnxruntime`, \
                     or point --ort-dylib at libonnxruntime"
                ))
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Locate the model directory. It is normally given relative to the working
/// directory, but the binary should still find it when run from elsewhere.
fn resolve_models(given: &Path) -> Result<PathBuf> {
    if given.is_dir() {
        return Ok(given.to_path_buf());
    }
    if !given.is_absolute() {
        if let Ok(exe) = std::env::current_exe() {
            // The binary lives at target/<profile>/platescan, so the project
            // root is three levels up from the executable itself.
            let roots = [
                exe.parent().map(Path::to_path_buf),
                exe.ancestors().nth(3).map(Path::to_path_buf),
            ];
            for base in roots.into_iter().flatten() {
                let candidate = base.join(given);
                if candidate.is_dir() {
                    return Ok(candidate);
                }
            }
        }
    }
    bail!(
        "no model directory at {}. Pass --models, or see the README for the \
         download commands",
        given.display()
    )
}

/// Find the one `.onnx` in `dir` whose name matches any of `hints`.
fn pick(dir: &Path, hints: &[&str], role: &str) -> Result<PathBuf> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("no model directory at {}", dir.display()))?;
    let mut models: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
        .collect();
    models.sort();
    models
        .iter()
        .find(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            hints.iter().any(|h| name.contains(h))
        })
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no {role} model in {} (looked for a .onnx named after {}). \
                 See the README for the download commands.",
                dir.display(),
                hints.join(" or ")
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, score: f32) -> Detection {
        Detection {
            x,
            y,
            w,
            h,
            score,
            origin: (0, 0),
        }
    }

    #[test]
    fn nms_drops_duplicate_boxes() {
        let kept = nms(
            vec![
                det(100.0, 100.0, 40.0, 20.0, 0.9),
                det(103.0, 101.0, 40.0, 20.0, 0.7),
                det(400.0, 100.0, 40.0, 20.0, 0.8),
            ],
            0.4,
        );
        assert_eq!(kept.len(), 2, "the two overlapping boxes should merge");
        assert_eq!(kept[0].score, 0.9, "the best box survives");
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = det(100.0, 100.0, 10.0, 10.0, 1.0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn letterbox_keeps_native_scale_when_it_fits() {
        let img = RgbImage::new(640, 540);
        let (canvas, scale, px, py) = letterbox(&img, 640);
        assert_eq!(scale, 1.0, "no downscaling when the window already fits");
        assert_eq!((canvas.width(), canvas.height()), (640, 640));
        assert_eq!((px, py), (0, 50));
    }

    #[test]
    fn decodes_slots_and_strips_padding() {
        let alphabet: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".chars().collect();
        let classes = alphabet.len();
        let mut slots = vec![0f32; 3 * classes];
        slots[10] = 0.9; // 'A'
        slots[classes + 1] = 0.8; // '1'
        slots[2 * classes + classes - 1] = 1.0; // '_' padding
        let (text, conf) = decode(&slots, classes, &alphabet);
        assert_eq!(text, "A1");
        assert!((conf - 0.85).abs() < 1e-5, "padding is excluded from conf");
    }

    #[test]
    fn parses_region_list_from_yaml() {
        let yaml = "max_plate_slots: 10\nplate_regions: [ 'Brazil', 'Canada',\n 'Unknown' ]\n";
        assert_eq!(parse_regions(yaml), vec!["Brazil", "Canada", "Unknown"]);
    }
}
