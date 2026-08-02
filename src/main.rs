mod alpr;
mod cli;
mod clip;
mod frames;
mod gps;
mod html;
mod kml;
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

    // A previous run's --json report stands in for the videos: build the KMZ
    // from what it recorded instead of rescanning.
    let json_inputs = cli
        .inputs
        .iter()
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        })
        .count();
    if json_inputs > 0 {
        if json_inputs != cli.inputs.len() {
            bail!("cannot mix JSON reports and videos in one run");
        }
        let mut clips = Vec::new();
        for input in &cli.inputs {
            clips.extend(kml::from_json(input)?);
        }
        let resolve = |p: Option<&Path>, ext: &str| {
            let wanted = match p.filter(|p| !path_is_auto(p)) {
                Some(p) => p.to_path_buf(),
                None => cli.inputs[0].with_extension(ext),
            };
            if cli.force {
                wanted
            } else {
                next_free_path(&wanted)
            }
        };
        if let Some(html) = &cli.html {
            let out = resolve(Some(html), "html");
            html::write_html(&out, &clips)?;
            report_export("marker", &out, &clips);
        }
        // A JSON input with no format asked for means KMZ, the original
        // behaviour of this mode.
        if cli.kmz.is_some() || cli.html.is_none() {
            let out = resolve(cli.kmz.as_deref(), "kmz");
            kml::write_kmz(&out, &clips)?;
            report_export("placemark", &out, &clips);
        }
        warn_unplaced(&clips, cli.quiet);
        return Ok(());
    }

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

    let (inputs, had_dir) = expand_inputs(&cli.inputs)?;
    // A directory holds arbitrarily many days of footage; one flat report
    // over months would be useless, so directories imply trip grouping.
    let by_trip = cli.by_trip || had_dir;
    if had_dir && !cli.by_trip && !cli.quiet {
        eprintln!("directory input: grouping into trips (as if --by-trip)");
    }

    let mut clips = probe_all(&inputs, &cli, had_dir)?;
    // Rear cameras record no GPS of their own; the front camera rolled
    // through the same seconds of driving, so borrow its track.
    let donors: Vec<Clip> = clips.iter().filter(|c| c.gps.is_some()).cloned().collect();
    for clip in &mut clips {
        clip.borrow_gps(&donors);
        if clip.gps_borrowed && !cli.quiet {
            eprintln!("{}: no GPS of its own, using a paired clip's track", clip.stem);
        }
    }

    if by_trip {
        let trips = group_trips(clips, cli.trip_gap);
        if !cli.quiet {
            eprintln!("{} trip(s):", trips.len());
            for trip in &trips {
                let mins: f64 = trip.iter().map(|c| c.duration).sum::<f64>() / 60.0;
                eprintln!(
                    "  {} — {} clip(s), {mins:.0} min of video",
                    trip_stem(trip),
                    trip.len()
                );
            }
        }
        let dir = cli.out.clone().unwrap_or_default();
        // Findings from any previous run in the output directory are reused
        // rather than rescanned, so consolidating an already-scanned archive
        // into trips costs only the clips that were never scanned.
        let cache = ReportCache::build(&dir);
        for trip in &trips {
            let wanted = dir.join(format!("{}-plates.md", trip_stem(trip)));
            let out_path = resolve_out_path(&wanted, &cli)?;
            // The raw findings are always kept in trip mode, so a KMZ can be
            // rebuilt from the report without a rescan.
            let json_path = Some(out_path.with_extension("json"));
            let kmz_path = cli.kmz.is_some().then(|| out_path.with_extension("kmz"));
            let html_path = cli.html.is_some().then(|| out_path.with_extension("html"));
            run_batch(
                trip,
                &cli,
                &rules,
                &out_path,
                json_path,
                kmz_path,
                html_path,
                Some(&cache),
            )?;
        }
        return Ok(());
    }

    // A scan is expensive and its output is evidence; never destroy a previous
    // run's report by rerunning over it.
    let out_path = resolve_out_path(&report_path(&cli), &cli)?;
    let json_path = cli.json.as_ref().map(|p| {
        if cli.force {
            p.clone()
        } else {
            next_free_path(p)
        }
    });
    let side_path = |given: Option<&Path>, ext: &str| {
        given.map(|p| {
            let wanted = if path_is_auto(p) {
                out_path.with_extension(ext)
            } else {
                p.to_path_buf()
            };
            if cli.force {
                wanted
            } else {
                next_free_path(&wanted)
            }
        })
    };
    let kmz_path = side_path(cli.kmz.as_deref(), "kmz");
    let html_path = side_path(cli.html.as_deref(), "html");
    run_batch(
        &clips, &cli, &rules, &out_path, json_path, kmz_path, html_path, None,
    )
}

/// Findings from previous runs, indexed by clip stem so a clip that was
/// already scanned is never scanned again.
struct ReportCache {
    by_stem: HashMap<String, (PathBuf, usize)>,
}

impl ReportCache {
    /// Index every platescan JSON in a directory by the clip stems it holds.
    fn build(dir: &Path) -> ReportCache {
        let mut by_stem = HashMap::new();
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text) else {
                    continue;
                };
                if doc["tool"].as_str() != Some("platescan") {
                    continue;
                }
                for (i, c) in doc["clips"].as_array().into_iter().flatten().enumerate() {
                    if let Some(stem) = c["stem"].as_str() {
                        by_stem
                            .entry(stem.to_string())
                            .or_insert((path.clone(), i));
                    }
                }
            }
        }
        ReportCache { by_stem }
    }

    fn get(&self, stem: &str) -> Option<report::ClipReport> {
        let (path, index) = self.by_stem.get(stem)?;
        report::read_json_clip(path, *index).ok()
    }
}

/// Scan a set of clips into one consolidated report, with optional JSON,
/// KMZ and HTML map beside it. Clips covered by the cache reuse their
/// previous findings instead of being scanned.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    clips: &[Clip],
    cli: &Cli,
    rules: &Rules,
    out_path: &Path,
    json_path: Option<PathBuf>,
    kmz_path: Option<PathBuf>,
    html_path: Option<PathBuf>,
    cache: Option<&ReportCache>,
) -> Result<()> {
    let crop_dir = (!cli.no_crops).then(|| crop_dir(cli, out_path));
    if let Some(dir) = &crop_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
    }
    let mut reports = Vec::new();
    for clip in clips {
        let cached = cache.and_then(|c| c.get(&clip.stem));
        let mut report = match cached {
            Some(mut cached) => {
                if !cli.quiet {
                    eprintln!("{}: reusing findings from a previous run", clip.stem);
                }
                // The JSON may predate a move of the footage, leaving its
                // recorded file path dead and its track rebuilt sparsely
                // from per-sighting fixes. The clip we just probed is
                // authoritative for both.
                cached.clip.path = clip.path.clone();
                if clip.gps.is_some() {
                    cached.clip.gps = clip.gps.clone();
                    cached.clip.gps_borrowed = clip.gps_borrowed;
                    cached.clip.utc_offset_minutes = clip.utc_offset_minutes;
                }
                cached
            }
            None => {
                if !cli.quiet {
                    eprintln!(
                        "{}: {}x{} @ {:.2} fps, {:.1} s",
                        clip.stem, clip.width, clip.height, clip.fps, clip.duration
                    );
                }
                scan_clip(clip, cli, rules)?
            }
        };
        if let Some(dir) = &crop_dir {
            report.crops = save_crops(clip, &report.sightings, dir, out_path, cli)?;
        }
        reports.push(report);
    }

    report::write_markdown(out_path, &reports)?;
    if let Some(json) = &json_path {
        report::write_json(json, &reports)?;
    }
    let exports = (kmz_path.is_some() || html_path.is_some())
        .then(|| kml::from_reports(&reports));
    if let (Some(kmz), Some(kclips)) = (&kmz_path, &exports) {
        kml::write_kmz(kmz, kclips)?;
    }
    if let (Some(html), Some(kclips)) = (&html_path, &exports) {
        html::write_html(html, kclips)?;
    }

    let total: usize = reports.iter().map(|r| r.sightings.len()).sum();
    println!("{total} plate sighting(s) -> {}", out_path.display());
    if let Some(json) = &json_path {
        println!("raw findings -> {}", json.display());
    }
    if let Some(kclips) = &exports {
        if let Some(kmz) = &kmz_path {
            report_export("placemark", kmz, kclips);
        }
        if let Some(html) = &html_path {
            report_export("marker", html, kclips);
        }
        warn_unplaced(kclips, cli.quiet);
    }
    Ok(())
}

/// The requested report path, stepped aside from an existing file unless
/// --force, with its directory created.
fn resolve_out_path(wanted: &Path, cli: &Cli) -> Result<PathBuf> {
    let out_path = if cli.force {
        wanted.to_path_buf()
    } else {
        next_free_path(wanted)
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
    Ok(out_path)
}

/// Split clips into trips: runs whose recordings are contiguous. A gap
/// longer than `gap` seconds with the camera off starts a new trip. Front
/// and rear clips recorded together share start times, so they chain into
/// the same trip.
fn group_trips(mut clips: Vec<Clip>, gap: f64) -> Vec<Vec<Clip>> {
    clips.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.stem.cmp(&b.stem)));
    let mut trips: Vec<Vec<Clip>> = Vec::new();
    let mut trip_end: Option<chrono::NaiveDateTime> = None;
    for clip in clips {
        let joins = match (trip_end, clip.start) {
            (Some(end), Some(start)) => {
                (start - end).num_milliseconds() as f64 / 1000.0 <= gap
            }
            // A clip with no decodable timestamp cannot chain to anything.
            _ => false,
        };
        if !joins || trips.is_empty() {
            trips.push(Vec::new());
            trip_end = None;
        }
        let end = clip.start.map(|s| {
            s + chrono::Duration::milliseconds((clip.duration * 1000.0) as i64)
        });
        // Paired clips end at the same moment; keep the trip's furthest end.
        trip_end = match (trip_end, end) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => b.or(a),
        };
        trips.last_mut().expect("just pushed").push(clip);
    }
    trips
}

/// Name a trip after the wall-clock moment it started.
fn trip_stem(trip: &[Clip]) -> String {
    match trip.iter().filter_map(|c| c.start).min() {
        Some(start) => format!("{}-trip", start.format("%Y_%m%d_%H%M%S")),
        None => format!(
            "{}-trip",
            trip.first().map(|c| c.stem.as_str()).unwrap_or("unknown")
        ),
    }
}

/// Summary line for a map export.
fn report_export(noun: &str, path: &Path, clips: &[kml::KmlClip]) {
    let placed: usize = clips.iter().map(|c| c.placemarks.len()).sum();
    println!(
        "{placed} {noun}(s) from {} clip(s) -> {}",
        clips.len(),
        path.display()
    );
}

/// One warning, however many export formats were written.
fn warn_unplaced(clips: &[kml::KmlClip], quiet: bool) {
    let unplaced: usize = clips.iter().map(|c| c.unplaced).sum();
    if unplaced > 0 && !quiet {
        eprintln!(
            "warning: {unplaced} sighting(s) had no GPS fix and are not on the map"
        );
    }
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
        let file = dir.join(s.crop_name(&clip.stem, i));
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

/// Probe every input in parallel. A probe is I/O-bound — ffprobe plus a few
/// hundred scattered GPS-record reads — so probing a large archive on
/// spinning disks one clip at a time leaves the machine idle for minutes.
/// Overlapping the seek latency across workers is close to a free speedup.
///
/// `lenient` (directory inputs) downgrades an unreadable clip to a warning:
/// one corrupt file should not abort an archive-wide run. Explicitly named
/// files still fail hard.
fn probe_all(inputs: &[PathBuf], cli: &Cli, lenient: bool) -> Result<Vec<Clip>> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    if !cli.quiet && inputs.len() > 1 {
        eprintln!("probing {} clip(s)", inputs.len());
    }
    let workers = cli.workers().clamp(1, 8).min(inputs.len().max(1));
    let mut probed: Vec<Option<Clip>> = (0..inputs.len()).map(|_| None).collect();
    let mut first_error = None;
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let next = &next;
        let (tx, rx) = channel::<(usize, Result<Clip>)>();
        for _ in 0..workers {
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(input) = inputs.get(i) else { break };
                    let result = if input.exists() {
                        Clip::probe(input)
                    } else {
                        Err(anyhow::anyhow!("no such file: {}", input.display()))
                    };
                    if tx.send((i, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0usize;
        for (i, result) in rx {
            done += 1;
            if !cli.quiet && inputs.len() > 10 && (done % 10 == 0 || done == inputs.len()) {
                eprint!("\r  probed {done}/{}", inputs.len());
                let _ = std::io::stderr().flush();
            }
            match result {
                Ok(clip) => probed[i] = Some(clip),
                Err(e) if lenient => {
                    eprintln!("\nwarning: skipping {}: {e}", inputs[i].display());
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        if !cli.quiet && inputs.len() > 10 {
            eprintln!();
        }
    });
    if let Some(e) = first_error {
        return Err(e);
    }
    Ok(probed.into_iter().flatten().collect())
}

/// A bare `--kmz` or `--html` parses as the sentinel `auto`, meaning
/// "derive the path from the report".
fn path_is_auto(p: &Path) -> bool {
    p.as_os_str().is_empty() || p == Path::new("auto")
}

/// Expand directory inputs to the MP4 files under them, recursively, in a
/// stable order. Returns whether any input was a directory. Dashcams mirror
/// locked clips into backup folders, so each stem is scanned once.
fn expand_inputs(inputs: &[PathBuf]) -> Result<(Vec<PathBuf>, bool)> {
    let mut out = Vec::new();
    let mut had_dir = false;
    for input in inputs {
        if input.is_dir() {
            had_dir = true;
            let mut found = Vec::new();
            collect_mp4s(input, &mut found)?;
            if found.is_empty() {
                bail!("no MP4 files under {}", input.display());
            }
            found.sort();
            out.extend(found);
        } else {
            out.push(input.clone());
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| {
        p.file_stem()
            .map(|s| seen.insert(s.to_os_string()))
            .unwrap_or(true)
    });
    Ok((out, had_dir))
}

fn collect_mp4s(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let hidden = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        if hidden {
            continue;
        }
        if path.is_dir() {
            collect_mp4s(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("mp4"))
        {
            out.push(path);
        }
    }
    Ok(())
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

