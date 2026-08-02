use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use chrono::NaiveDateTime;
use serde_json::{json, Value};

use crate::clip::{Camera, Clip};
use crate::gps::{Fix, Track};
use crate::track::Sighting;

/// Scan parameters, echoed into the report so a result can be reproduced.
#[derive(Clone, Debug)]
pub struct ScanParams {
    pub fps: f64,
    pub start: f64,
    pub end: Option<f64>,
    pub roi: (u32, u32, u32, u32),
    /// Backend-specific settings, already formatted for display.
    pub detail: String,
    pub min_conf: f32,
    pub min_height: f32,
    pub min_hits: usize,
    pub gap: f64,
    pub region: String,
    pub patterns: Vec<String>,
    pub engine: String,
}

pub struct ClipReport {
    pub clip: Clip,
    pub params: ScanParams,
    pub sightings: Vec<Sighting>,
    /// Sighting index -> crop path relative to the report.
    pub crops: HashMap<usize, PathBuf>,
    pub frames_scanned: u64,
    pub raw_detections: usize,
    pub elapsed: Duration,
}

pub fn write_markdown(path: &Path, reports: &[ClipReport]) -> Result<()> {
    let mut md = String::new();
    let title = match reports {
        [only] => format!("License plate scan — {}", only.clip.stem),
        many => format!("License plate scan — {} clips", many.len()),
    };
    let total: usize = reports.iter().map(|r| r.sightings.len()).sum();
    writeln!(md, "# {title}\n")?;
    writeln!(
        md,
        "{total} plate sighting(s) across {} clip(s).\n",
        reports.len()
    )?;

    if reports.len() > 1 {
        writeln!(md, "## All sightings\n")?;
        writeln!(md, "| Clip | Plate | Wall clock | Offset | Frames | Conf |")?;
        writeln!(md, "|---|---|---|---|---|---|")?;
        for r in reports {
            for s in &r.sightings {
                writeln!(
                    md,
                    "| `{}` | `{}` | {} | {} | {} | {:.0} |",
                    r.clip.stem,
                    s.plate,
                    wall_clock(&r.clip, s.first_seen),
                    offset(s.first_seen),
                    s.frames,
                    s.best_conf
                )?;
            }
        }
        md.push('\n');
    }

    for r in reports {
        write_clip(&mut md, r)?;
    }

    writeln!(md, "## How to read this\n")?;
    writeln!(
        md,
        "- **Wall clock** is derived from the file name, which the camera stamps at \
         clip start; it is only as accurate as the camera's own clock.\n\
         - **Offset** is time from the start of the clip.\n\
         - **Conf** is the recognizer's own score (0-100), not a probability that the \
          plate is correct. It tracks legibility closely: high scores are usually \
          right, and low scores are usually plates too distant to read by eye either.\n\
         - Readings are model output, not verified plates. Check the linked crop \
          before relying on any reading. Ambiguous character pairs (`0`/`O`, `1`/`I`, \
          `5`/`S`, `8`/`B`) are the usual failure, and a wrong reading looks just as \
          plausible as a right one.\n\
         - One vehicle can appear as more than one sighting when it is detected \
          intermittently, typically once far away and once close. A larger `--gap` \
          merges those.\n\
         - Plates that are small, motion-blurred, sharply angled, or outside the \
          region of interest are missed entirely; absence here is not evidence a \
          vehicle was absent."
    )?;

    std::fs::write(path, md).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn write_clip(md: &mut String, r: &ClipReport) -> Result<()> {
    let c = &r.clip;
    let p = &r.params;
    writeln!(md, "## `{}`\n", c.stem)?;
    writeln!(md, "| Field | Value |")?;
    writeln!(md, "|---|---|")?;
    writeln!(md, "| Source | `{}` |", cell(c.path.display()))?;
    writeln!(
        md,
        "| Camera | {} |",
        c.camera.map(|c| c.label()).unwrap_or("unknown")
    )?;
    if let Some(seq) = c.sequence {
        writeln!(md, "| Clip number | {seq:06} |")?;
    }
    writeln!(
        md,
        "| Clip start | {} |",
        c.start
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "unknown (file name not recognised)".into())
    )?;
    writeln!(
        md,
        "| Video | {}x{} @ {:.2} fps, {:.1} s |",
        c.width, c.height, c.fps, c.duration
    )?;
    match (&c.gps, c.utc_offset_minutes) {
        (Some(track), Some(minutes)) => {
            let skew = c
                .start
                .and_then(|s| track.clock_skew_secs(s))
                .map(|s| match s {
                    0 => ", camera clock agrees with GPS".to_string(),
                    s if s > 0 => format!(", camera clock {s} s behind GPS"),
                    s => format!(", camera clock {} s ahead of GPS", -s),
                })
                .unwrap_or_default();
            writeln!(
                md,
                "| Clock | GPS{}, {} fixes, UTC{}{skew} |",
                if c.gps_borrowed {
                    " (track borrowed from paired clip)"
                } else {
                    ""
                },
                track.fix_count(),
                format_offset_minutes(minutes)
            )?;
            if let (Some(first), Some(last)) = (track.first(), track.last()) {
                writeln!(
                    md,
                    "| GPS span | {}Z to {}Z |",
                    first.time.format("%Y-%m-%d %H:%M:%S"),
                    last.time.format("%H:%M:%S")
                )?;
            }
        }
        _ => {
            writeln!(
                md,
                "| Clock | file name only (no GPS track found); accurate to the \
                 camera's own clock |"
            )?;
        }
    }
    writeln!(
        md,
        "| Scanned | {} - {} at {:.2} fps ({} frames) |",
        offset(p.start),
        offset(p.end.unwrap_or(c.duration)),
        p.fps,
        r.frames_scanned
    )?;
    writeln!(
        md,
        "| Region of interest | x={} y={} {}x{} px |",
        p.roi.0, p.roi.1, p.roi.2, p.roi.3
    )?;
    writeln!(md, "| Engine | {} |", cell(&p.engine))?;
    writeln!(md, "| Settings | {} |", cell(&p.detail))?;
    writeln!(
        md,
        "| Filters | conf >= {:.0}, height >= {:.0} px, >= {} frames, gap {:.1} s, {} format |",
        p.min_conf, p.min_height, p.min_hits, p.gap, p.region
    )?;
    writeln!(
        md,
        "| Result | {} sighting(s) from {} raw reading(s) in {:.1} s |",
        r.sightings.len(),
        r.raw_detections,
        r.elapsed.as_secs_f64()
    )?;
    md.push('\n');

    if r.sightings.is_empty() {
        writeln!(
            md,
            "No plates met the thresholds. Try `--fps 4 --det-conf 0.25 --min-conf 45`, \
             or widen `--roi` if vehicles sit outside the scanned band.\n"
        )?;
        return Ok(());
    }

    writeln!(md, "### Plates\n")?;
    writeln!(
        md,
        "| # | Plate | Wall clock (first seen) | Offset | Visible | Frames | Best conf | Crop |"
    )?;
    writeln!(md, "|---|---|---|---|---|---|---|---|")?;
    for (i, s) in r.sightings.iter().enumerate() {
        let crop = r
            .crops
            .get(&i)
            .map(|p| format!("[view]({})", p.display()))
            .unwrap_or_else(|| "-".into());
        writeln!(
            md,
            "| {} | `{}` | {} | {} | {:.1} s | {} | {:.0} | {} |",
            i + 1,
            s.plate,
            wall_clock(&r.clip, s.first_seen),
            offset(s.first_seen),
            s.duration(),
            s.frames,
            s.best_conf,
            crop
        )?;
    }
    md.push('\n');

    writeln!(md, "### Detail\n")?;
    for (i, s) in r.sightings.iter().enumerate() {
        writeln!(md, "#### {}. `{}`\n", i + 1, s.plate)?;
        writeln!(
            md,
            "- First seen **{}** (offset {}), last seen **{}** (offset {}), visible {:.1} s",
            wall_clock(&r.clip, s.first_seen),
            offset(s.first_seen),
            wall_clock(&r.clip, s.last_seen),
            offset(s.last_seen),
            s.duration()
        )?;
        writeln!(
            md,
            "- Read in {} sampled frame(s); confidence mean {:.1}, best {:.1}",
            s.frames, s.mean_conf, s.best_conf
        )?;
        if let Some(utc) = r.clip.gps_utc(s.first_seen) {
            writeln!(md, "- GPS time **{}Z**", utc.format("%Y-%m-%d %H:%M:%S%.3f"))?;
        }
        if let Some(fix) = r.clip.fix_at(s.first_seen) {
            writeln!(
                md,
                "- Seen at [{:.5}, {:.5}](https://www.openstreetmap.org/?mlat={:.5}&mlon={:.5}#map=18/{:.5}/{:.5}), \
                 travelling {:.0} mph ({:.0} km/h) on a bearing of {:.0}°",
                fix.lat, fix.lon, fix.lat, fix.lon, fix.lat, fix.lon,
                fix.speed_mph(),
                fix.speed_kmh(),
                fix.bearing
            )?;
        }
        if let Some(score) = s.detector_score {
            writeln!(
                md,
                "- Plate detector score {score:.2}{}",
                s.region
                    .as_deref()
                    .map(|r| format!("; recognizer places the format in {r}"))
                    .unwrap_or_default()
            )?;
        }
        let variants = s
            .variants
            .iter()
            .map(|(t, n)| format!("`{t}` x{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(md, "- OCR variants: {variants}")?;
        writeln!(
            md,
            "- Best reading at offset {}, box x={} y={} {}x{} px in the source frame",
            offset(s.best_offset),
            s.best_rect.0,
            s.best_rect.1,
            s.best_rect.2,
            s.best_rect.3
        )?;
        if let Some(crop) = r.crops.get(&i) {
            writeln!(md, "\n![{} at {}]({})", s.plate, offset(s.best_offset), crop.display())?;
        }
        md.push('\n');
    }
    Ok(())
}

pub fn write_json(path: &Path, reports: &[ClipReport]) -> Result<()> {
    let clips: Vec<_> = reports
        .iter()
        .map(|r| {
            json!({
                "file": r.clip.path.to_string_lossy(),
                "stem": r.clip.stem,
                "camera": r.clip.camera.map(|c| c.label()),
                "sequence": r.clip.sequence,
                "clip_start": r.clip.start.map(|t| t.format("%Y-%m-%dT%H:%M:%S").to_string()),
                "clock_source": if r.clip.clock_is_gps() { "gps" } else { "filename" },
                "utc_offset_minutes": r.clip.utc_offset_minutes,
                "gps_fixes": r.clip.gps.as_ref().map(|g| g.fix_count()),
                "gps_borrowed": r.clip.gps_borrowed,
                "width": r.clip.width,
                "height": r.clip.height,
                "fps": r.clip.fps,
                "duration_s": r.clip.duration,
                "frames_scanned": r.frames_scanned,
                "raw_detections": r.raw_detections,
                "elapsed_s": r.elapsed.as_secs_f64(),
                "params": {
                    "sample_fps": r.params.fps,
                    "start_s": r.params.start,
                    "end_s": r.params.end,
                    "roi": [r.params.roi.0, r.params.roi.1, r.params.roi.2, r.params.roi.3],
                    "detail": r.params.detail,
                    "min_conf": r.params.min_conf,
                    "min_height_px": r.params.min_height,
                    "min_hits": r.params.min_hits,
                    "gap_s": r.params.gap,
                    "region": r.params.region,
                    "patterns": r.params.patterns,
                    "engine": r.params.engine,
                },
                "sightings": r.sightings.iter().enumerate().map(|(i, s)| json!({
                    "plate": s.plate,
                    "first_seen_s": s.first_seen,
                    "last_seen_s": s.last_seen,
                    "first_seen_wall": r.clip.wall_clock(s.first_seen)
                        .map(|t| t.format("%Y-%m-%dT%H:%M:%S%.3f").to_string()),
                    "last_seen_wall": r.clip.wall_clock(s.last_seen)
                        .map(|t| t.format("%Y-%m-%dT%H:%M:%S%.3f").to_string()),
                    "frames": s.frames,
                    "best_conf": s.best_conf,
                    "mean_conf": s.mean_conf,
                    "best_offset_s": s.best_offset,
                    "best_rect": [s.best_rect.0, s.best_rect.1, s.best_rect.2, s.best_rect.3],
                    "detector_score": s.detector_score,
                    "region": s.region,
                    "gps_utc": r.clip.gps_utc(s.first_seen)
                        .map(|t| t.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()),
                    "gps": r.clip.fix_at(s.first_seen).map(|f| json!({
                        "lat": f.lat,
                        "lon": f.lon,
                        "speed_mph": f.speed_mph(),
                        "speed_kmh": f.speed_kmh(),
                        "bearing_deg": f.bearing,
                    })),
                    "variants": s.variants.iter()
                        .map(|(t, n)| json!({"text": t, "count": n}))
                        .collect::<Vec<_>>(),
                    "crop": r.crops.get(&i).map(|p| p.to_string_lossy()),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let doc = json!({ "tool": "platescan", "clips": clips });
    std::fs::write(path, serde_json::to_string_pretty(&doc)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Rebuild one clip's report from a `--json` file written by a previous
/// run, so its findings can be consolidated or exported without rescanning.
/// Crop stills are reloaded from beside the report; the GPS track is
/// re-read from the source clip when it is still reachable, and otherwise
/// reassembled from the per-sighting fixes the report recorded.
pub fn read_json_clip(path: &Path, index: usize) -> Result<ClipReport> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let doc: Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let c = doc["clips"]
        .get(index)
        .ok_or_else(|| anyhow!("{}: no clip at index {index}", path.display()))?;
    clip_report_from_value(c, path.parent().unwrap_or(Path::new("")))
}

fn clip_report_from_value(c: &Value, base: &Path) -> Result<ClipReport> {
    let file = PathBuf::from(c["file"].as_str().unwrap_or_default());
    let stem = c["stem"].as_str().unwrap_or("clip").to_string();
    let start = c["clip_start"]
        .as_str()
        .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok());
    let duration = c["duration_s"].as_f64().unwrap_or(0.0);
    let sightings_json = c["sightings"].as_array().map(Vec::as_slice).unwrap_or(&[]);

    let gps = if file.exists() {
        crate::clip::disk_track(&file, start, duration)
    } else {
        None
    }
    .or_else(|| Track::from_sparse(sparse_fixes(sightings_json), duration));

    let clip = Clip {
        path: file,
        stem,
        start,
        camera: match c["camera"].as_str() {
            Some("front") => Some(Camera::Front),
            Some("rear") => Some(Camera::Rear),
            _ => None,
        },
        sequence: c["sequence"].as_u64().map(|v| v as u32),
        width: c["width"].as_u64().unwrap_or(0) as u32,
        height: c["height"].as_u64().unwrap_or(0) as u32,
        fps: c["fps"].as_f64().unwrap_or(0.0),
        duration,
        gps,
        gps_borrowed: c["gps_borrowed"].as_bool().unwrap_or(false),
        utc_offset_minutes: c["utc_offset_minutes"].as_i64().map(|v| v as i32),
    };

    let sightings = sightings_json
        .iter()
        .map(|s| {
            let rect = s["best_rect"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            let at = |i: usize| rect.get(i).and_then(Value::as_u64).unwrap_or(0) as u32;
            Sighting {
                plate: s["plate"].as_str().unwrap_or("?").to_string(),
                first_seen: s["first_seen_s"].as_f64().unwrap_or(0.0),
                last_seen: s["last_seen_s"].as_f64().unwrap_or(0.0),
                frames: s["frames"].as_u64().unwrap_or(0) as usize,
                best_conf: s["best_conf"].as_f64().unwrap_or(0.0) as f32,
                mean_conf: s["mean_conf"].as_f64().unwrap_or(0.0) as f32,
                best_offset: s["best_offset_s"].as_f64().unwrap_or(0.0),
                best_rect: (at(0), at(1), at(2), at(3)),
                detector_score: s["detector_score"].as_f64().map(|v| v as f32),
                region: s["region"].as_str().map(str::to_string),
                still: s["crop"].as_str().and_then(|rel| {
                    let p = PathBuf::from(rel);
                    let full = if p.is_absolute() { p } else { base.join(p) };
                    std::fs::read(full).ok()
                }),
                variants: s["variants"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|v| {
                        Some((v["text"].as_str()?.to_string(), v["count"].as_u64()? as usize))
                    })
                    .collect(),
            }
        })
        .collect();

    let p = &c["params"];
    Ok(ClipReport {
        clip,
        params: ScanParams {
            fps: p["sample_fps"].as_f64().unwrap_or(0.0),
            start: p["start_s"].as_f64().unwrap_or(0.0),
            end: p["end_s"].as_f64(),
            roi: {
                let roi = p["roi"].as_array().map(Vec::as_slice).unwrap_or(&[]);
                let at = |i: usize| roi.get(i).and_then(Value::as_u64).unwrap_or(0) as u32;
                (at(0), at(1), at(2), at(3))
            },
            detail: p["detail"].as_str().unwrap_or("").to_string(),
            min_conf: p["min_conf"].as_f64().unwrap_or(0.0) as f32,
            min_height: p["min_height_px"].as_f64().unwrap_or(0.0) as f32,
            min_hits: p["min_hits"].as_u64().unwrap_or(0) as usize,
            gap: p["gap_s"].as_f64().unwrap_or(0.0),
            region: p["region"].as_str().unwrap_or("generic").to_string(),
            patterns: p["patterns"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            engine: p["engine"].as_str().unwrap_or("alpr").to_string(),
        },
        sightings,
        crops: HashMap::new(),
        frames_scanned: c["frames_scanned"].as_u64().unwrap_or(0),
        raw_detections: c["raw_detections"].as_u64().unwrap_or(0) as usize,
        elapsed: Duration::from_secs_f64(c["elapsed_s"].as_f64().unwrap_or(0.0)),
    })
}

/// Per-sighting fixes recorded in a report, for rebuilding a track when the
/// source clip is gone. A fix needs both a position and a GPS time;
/// sightings missing either contribute nothing.
fn sparse_fixes(sightings: &[Value]) -> Vec<(f64, Fix)> {
    sightings
        .iter()
        .filter_map(|s| {
            let offset = s["best_offset_s"].as_f64().unwrap_or(0.0);
            let lat = s["gps"]["lat"].as_f64()?;
            let lon = s["gps"]["lon"].as_f64()?;
            let time = s["gps_utc"].as_str().and_then(|t| {
                NaiveDateTime::parse_from_str(t, "%Y-%m-%dT%H:%M:%S%.fZ").ok()
            })?;
            Some((
                offset,
                Fix {
                    time,
                    lat,
                    lon,
                    speed_knots: s["gps"]["speed_mph"].as_f64().unwrap_or(0.0) as f32
                        / 1.150_779,
                    bearing: s["gps"]["bearing_deg"].as_f64().unwrap_or(0.0) as f32,
                },
            ))
        })
        .collect()
}

/// A UTC offset in minutes as `-07:00`.
fn format_offset_minutes(minutes: i32) -> String {
    let sign = if minutes < 0 { '-' } else { '+' };
    let abs = minutes.abs();
    format!("{sign}{:02}:{:02}", abs / 60, abs % 60)
}

/// Escape a value so it cannot split the Markdown table row it sits in.
fn cell(value: impl std::fmt::Display) -> String {
    value.to_string().replace('|', "\\|")
}

fn wall_clock(clip: &Clip, offset_s: f64) -> String {
    clip.wall_clock(offset_s)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S%.3f").to_string())
        .unwrap_or_else(|| "-".into())
}

/// `MM:SS.mmm`, widening to `H:MM:SS.mmm` past an hour.
pub fn offset(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let (ms, s) = (total_ms % 1000, total_ms / 1000);
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}.{ms:03}")
    } else {
        format!("{m:02}:{s:02}.{ms:03}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_format_as_clock_time() {
        assert_eq!(offset(0.0), "00:00.000");
        assert_eq!(offset(23.5), "00:23.500");
        assert_eq!(offset(599.999), "09:59.999");
        assert_eq!(offset(3723.25), "1:02:03.250");
    }
}
