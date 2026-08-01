mod alpr;
mod cli;
mod clip;
mod frames;
mod gps;
mod ocr;
mod plate;
mod report;
mod scan;
mod track;

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;

use alpr::{Alpr, AlprSettings};
use cli::{Cli, EngineKind};
use clip::Clip;
use frames::{FrameStream, PixelFormat, SampleSpec};
use ocr::{Engine, OcrSettings};
use plate::Rules;
use report::{ClipReport, ScanParams};
use scan::Scanner;
use track::{Detection, Sighting};

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.validate()?;

    // Dashcams burn a timestamp, speed and their own vehicle tag into the
    // bottom strip. The tag is plate-shaped and would otherwise be "found" in
    // every single frame.
    if cli.roi.y + cli.roi.h > 0.94 && !cli.quiet {
        eprintln!(
            "warning: --roi covers the bottom of the frame, where the camera burns in \
             its own timestamp and vehicle tag; those will be reported as plates"
        );
    }

    // The ALPR detector already establishes that a box is a plate, so enforcing
    // a national format on top of it only discards valid readings.
    let strict = cli.strict_format || cli.engine == EngineKind::Tesseract;
    let rules = Rules::new(cli.region, &cli.patterns, strict)?;
    // A scan is expensive and its output is evidence; never destroy a previous
    // run's report by rerunning over it.
    let wanted = report_path(&cli);
    let out_path = if cli.force {
        wanted.clone()
    } else {
        next_free_path(&wanted)
    };
    if out_path != wanted && !cli.quiet {
        eprintln!(
            "{} already exists; writing {} instead (--force to overwrite)",
            wanted.display(),
            out_path.display()
        );
    }
    if let Some(dir) = out_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
    }
    let crop_dir = (!cli.no_crops).then(|| crop_dir(&cli, &out_path));
    if let Some(dir) = &crop_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
    }
    let mut reports = Vec::new();
    for input in &cli.inputs {
        if !input.exists() {
            bail!("no such file: {}", input.display());
        }
        let clip = Clip::probe(input)?;
        if !cli.quiet {
            eprintln!(
                "{}: {}x{} @ {:.2} fps, {:.1} s",
                clip.stem, clip.width, clip.height, clip.fps, clip.duration
            );
        }
        let mut report = scan_clip(&clip, &cli, &rules)?;
        if let Some(dir) = &crop_dir {
            report.crops = save_crops(&clip, &report.sightings, dir, &out_path, &cli)?;
        }
        reports.push(report);
    }

    report::write_markdown(&out_path, &reports)?;
    let json_path = cli.json.as_ref().map(|p| {
        if cli.force {
            p.clone()
        } else {
            next_free_path(p)
        }
    });
    if let Some(json) = &json_path {
        report::write_json(json, &reports)?;
    }

    let total: usize = reports.iter().map(|r| r.sightings.len()).sum();
    println!("{total} plate sighting(s) -> {}", out_path.display());
    if let Some(json) = &json_path {
        println!("raw findings -> {}", json.display());
    }
    Ok(())
}

fn scan_clip(clip: &Clip, cli: &Cli, rules: &Rules) -> Result<ClipReport> {
    let roi = cli.roi.resolve(clip.width, clip.height);
    let spec = SampleSpec {
        start: cli.start,
        end: cli.end,
        fps: cli.fps,
        roi,
        format: match cli.engine {
            EngineKind::Alpr => PixelFormat::Rgb,
            EngineKind::Tesseract => PixelFormat::Gray,
        },
    };

    // Build every backend before decoding starts, so a missing model or
    // tessdata file fails immediately rather than part-way through a clip.
    let workers = cli.workers();
    let mut engines: Vec<Box<dyn Scanner>> = Vec::with_capacity(workers);
    for _ in 0..workers {
        engines.push(match cli.engine {
            EngineKind::Alpr => Box::new(Alpr::new(&AlprSettings {
                models: cli.models.clone(),
                dylib: cli.ort_dylib.clone(),
                det_conf: cli.det_conf,
                overlap: cli.overlap,
                min_height: cli.min_height,
                min_conf: cli.min_conf,
            })?) as Box<dyn Scanner>,
            EngineKind::Tesseract => Box::new(Engine::new(
                cli.tessdata.as_deref(),
                &cli.lang,
                OcrSettings {
                    tile: cli.tile,
                    overlap: cli.overlap,
                    upscale: cli.upscale,
                    psm: cli.psm,
                    min_conf: cli.min_conf,
                    min_height: cli.min_height,
                    whitelist: !cli.no_whitelist,
                },
            )?) as Box<dyn Scanner>,
        });
    }

    let keep_stills = !cli.no_crops;
    let engine_detail = engines
        .first()
        .map(|e| e.describe())
        .unwrap_or_else(|| "none".into());

    let expected = FrameStream::expected(&spec, clip.duration);
    let started = Instant::now();
    let (frame_tx, frame_rx) = sync_channel::<frames::Frame>(workers * 2);
    let frame_rx = Arc::new(Mutex::new(frame_rx));
    let (det_tx, det_rx) = channel::<Result<Vec<Detection>>>();

    let mut detections: Vec<Detection> = Vec::new();
    let mut frames_done: u64 = 0;

    std::thread::scope(|scope| -> Result<()> {
        let producer = scope.spawn(move || -> Result<()> {
            let mut stream = FrameStream::open(&clip.path, &spec)?;
            while let Some(frame) = stream.next_frame()? {
                if frame_tx.send(frame).is_err() {
                    break; // consumers went away; stop decoding
                }
            }
            Ok(())
        });

        for engine in engines {
            let frame_rx = Arc::clone(&frame_rx);
            let det_tx = det_tx.clone();
            scope.spawn(move || {
                let mut engine = engine;
                loop {
                    // Scoped so the queue lock is released before OCR runs.
                    let next = { frame_rx.lock().expect("frame queue poisoned").recv() };
                    let Ok(frame) = next else { break };
                    let result = engine.scan(&frame).map(|cands| {
                        cands
                            .into_iter()
                            .filter_map(|c| {
                                rules.accept(&c.text).map(|text| {
                                    // Encode the verification still now, while
                                    // the analysed frame is still in hand.
                                    let still = keep_stills
                                        .then(|| frame.still_jpeg(c.rect))
                                        .flatten();
                                    Detection::new(frame.offset, text, &c, still)
                                })
                            })
                            .collect::<Vec<_>>()
                    });
                    if det_tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(det_tx);

        for result in det_rx {
            frames_done += 1;
            detections.extend(result?);
            if !cli.quiet && (frames_done % 10 == 0 || frames_done == expected) {
                let pct = frames_done as f64 / expected.max(1) as f64 * 100.0;
                eprint!(
                    "\r  {frames_done}/{expected} frames ({pct:.0}%), {} reading(s)",
                    detections.len()
                );
                let _ = std::io::stderr().flush();
            }
        }
        if !cli.quiet {
            eprintln!();
        }

        producer
            .join()
            .map_err(|_| anyhow::anyhow!("frame reader panicked"))?
    })?;

    let raw_detections = detections.len();
    let sightings = track::build(detections, cli.gap, cli.min_hits, clip.gps.as_ref());

    Ok(ClipReport {
        clip: clip.clone(),
        params: ScanParams {
            fps: cli.fps,
            start: cli.start,
            end: cli.end,
            roi,
            engine: format!("{:?}", cli.engine).to_lowercase(),
            detail: engine_detail,
            min_conf: cli.min_conf,
            min_height: cli.min_height,
            min_hits: cli.min_hits,
            gap: cli.gap,
            region: format!("{:?}", cli.region).to_lowercase(),
            patterns: cli.patterns.clone(),
        },
        sightings,
        crops: HashMap::new(),
        frames_scanned: frames_done,
        raw_detections,
        elapsed: started.elapsed(),
    })
}

fn save_crops(
    clip: &Clip,
    sightings: &[Sighting],
    dir: &Path,
    report_path: &Path,
    cli: &Cli,
) -> Result<HashMap<usize, PathBuf>> {
    let mut out = HashMap::new();
    for (i, s) in sightings.iter().enumerate() {
        let Some(bytes) = &s.still else {
            continue;
        };
        let safe: String = s
            .plate
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        let stamp = format!("{:08.3}", s.best_offset).replace('.', "_");
        let file = dir.join(format!("{}-{:03}-{safe}-{stamp}.jpg", clip.stem, i + 1));
        match std::fs::write(&file, bytes) {
            Ok(()) => {
                out.insert(i, relative_to(&file, report_path));
            }
            Err(e) if !cli.quiet => eprintln!("  warning: no crop for {}: {e}", s.plate),
            Err(_) => {}
        }
    }
    Ok(out)
}

/// The requested path, or the first `-2`, `-3`, ... variant that is free.
fn next_free_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let parent = path.parent().unwrap_or(Path::new(""));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("platescan");
    let ext = path.extension().and_then(|e| e.to_str());
    (2..10_000)
        .map(|n| {
            parent.join(match ext {
                Some(ext) => format!("{stem}-{n}.{ext}"),
                None => format!("{stem}-{n}"),
            })
        })
        .find(|candidate| !candidate.exists())
        .unwrap_or_else(|| path.to_path_buf())
}

/// Path to `target` as written inside a report living at `report_path`.
fn relative_to(target: &Path, report_path: &Path) -> PathBuf {
    let base = report_path.parent().unwrap_or(Path::new(""));
    target
        .strip_prefix(base)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| target.to_path_buf())
}

fn report_path(cli: &Cli) -> PathBuf {
    if let Some(out) = &cli.out {
        return out.clone();
    }
    match cli.inputs.as_slice() {
        [one] => PathBuf::from(format!(
            "{}-plates.md",
            one.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("platescan")
        )),
        _ => PathBuf::from("platescan-report.md"),
    }
}

fn crop_dir(cli: &Cli, report_path: &Path) -> PathBuf {
    if let Some(dir) = &cli.crop_dir {
        return dir.clone();
    }
    let stem = report_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("platescan");
    report_path
        .parent()
        .unwrap_or(Path::new(""))
        .join(format!("{stem}-crops"))
}

