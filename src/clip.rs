use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use chrono::NaiveDateTime;
use regex::Regex;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Camera {
    Front,
    Rear,
}

impl Camera {
    pub fn label(self) -> &'static str {
        match self {
            Camera::Front => "front",
            Camera::Rear => "rear",
        }
    }
}

/// Everything we know about a source clip before scanning it.
#[derive(Clone, Debug)]
pub struct Clip {
    pub path: PathBuf,
    pub stem: String,
    /// Wall-clock time of the first frame, decoded from the file name.
    pub start: Option<NaiveDateTime>,
    pub camera: Option<Camera>,
    pub sequence: Option<u32>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    /// GPS fixes embedded by the camera, when it recorded any.
    pub gps: Option<crate::gps::Track>,
    /// The track came from a paired clip rather than this file.
    pub gps_borrowed: bool,
    /// Minutes to add to UTC for the camera's local time, inferred from GPS.
    pub utc_offset_minutes: Option<i32>,
}

impl Clip {
    pub fn probe(path: &Path) -> Result<Clip> {
        let meta = ffprobe(path)?;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let name = parse_name(&stem);
        let start = name.as_ref().and_then(|n| n.start);
        // A missing or unreadable GPS box is normal, not an error.
        let gps = crate::gps::Track::read(path).unwrap_or(None);
        let utc_offset_minutes = gps
            .as_ref()
            .zip(start)
            .and_then(|(g, s)| g.utc_offset_minutes(s));
        Ok(Clip {
            path: path.to_path_buf(),
            stem,
            start,
            camera: name.as_ref().and_then(|n| n.camera),
            sequence: name.as_ref().and_then(|n| n.sequence),
            width: meta.width,
            height: meta.height,
            fps: meta.fps,
            duration: meta.duration,
            gps,
            gps_borrowed: false,
            utc_offset_minutes,
        })
    }

    /// Fill a missing track from a clip recorded at the same moment by the
    /// other camera. Rear cameras write no GPS of their own, but the front
    /// camera rolled through the same seconds of driving. Clips scanned in
    /// the same run are tried first, then the file's own folder.
    pub fn borrow_gps(&mut self, donors: &[Clip]) {
        if self.gps.is_some() {
            return;
        }
        let Some(start) = self.start else { return };
        let mut best: Option<(i64, &crate::gps::Track)> = None;
        for d in donors {
            let (Some(track), Some(d_start)) = (d.gps.as_ref(), d.start) else {
                continue;
            };
            let delta = (start - d_start).num_seconds();
            if delta.abs() as f64 > d.duration.max(self.duration) {
                continue;
            }
            if best.is_none_or(|(b, _)| delta.abs() < b.abs()) {
                best = Some((delta, track));
            }
        }
        let track = best
            .map(|(delta, t)| t.shifted(delta))
            .filter(|t| t.fix_count() > 0)
            .or_else(|| sibling_track(&self.path, start, self.duration));
        if let Some(track) = track {
            self.utc_offset_minutes = track.utc_offset_minutes(start);
            self.gps = Some(track);
            self.gps_borrowed = true;
        }
    }

    /// Local wall-clock time of an offset into the clip.
    ///
    /// Prefers the GPS fix, which is satellite time; the file name only carries
    /// the camera's RTC, which drifts.
    pub fn wall_clock(&self, offset: f64) -> Option<NaiveDateTime> {
        if let Some(utc) = self.gps_utc(offset) {
            if let Some(minutes) = self.utc_offset_minutes {
                return Some(utc + chrono::Duration::minutes(minutes as i64));
            }
        }
        let start = self.start?;
        let micros = (offset * 1_000_000.0).round() as i64;
        Some(start + chrono::Duration::microseconds(micros))
    }

    /// True when timestamps come from the satellites rather than the file name.
    pub fn clock_is_gps(&self) -> bool {
        self.utc_offset_minutes.is_some()
    }

    /// UTC time of an offset, interpolated within the one-second GPS fix.
    pub fn gps_utc(&self, offset: f64) -> Option<NaiveDateTime> {
        let track = self.gps.as_ref()?;
        let fix = track.at(offset)?;
        // Fixes land on whole seconds; carry the sub-second part of the offset.
        let frac = offset - offset.round();
        Some(fix.time + chrono::Duration::microseconds((frac * 1_000_000.0).round() as i64))
    }

    pub fn fix_at(&self, offset: f64) -> Option<&crate::gps::Fix> {
        self.gps.as_ref()?.nearest(offset)
    }
}

/// The GPS track covering a clip at `path`: its own when it recorded one,
/// otherwise a paired clip's from the same folder, shifted into this clip's
/// timeline.
pub fn disk_track(
    path: &Path,
    start: Option<NaiveDateTime>,
    duration: f64,
) -> Option<crate::gps::Track> {
    if let Ok(Some(track)) = crate::gps::Track::read(path) {
        return Some(track);
    }
    sibling_track(path, start?, duration)
}

/// Search the clip's folder for another clip whose recording overlaps this
/// one and carries GPS. Overlap is judged from the file-name stamps, which
/// both cameras take from the same clock; a clip from a different moment
/// yields no overlapping fixes and is rejected after the shift.
fn sibling_track(path: &Path, start: NaiveDateTime, duration: f64) -> Option<crate::gps::Track> {
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let own_stem = path.file_stem();
    let mut candidates: Vec<(i64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if !p
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("mp4"))
            || p.file_stem() == own_stem
        {
            continue;
        }
        let Some(other_start) = p
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| parse_name(s).and_then(|n| n.start))
        else {
            continue;
        };
        let delta = (start - other_start).num_seconds();
        if (delta.abs() as f64) < duration.max(1.0) {
            candidates.push((delta, p));
        }
    }
    candidates.sort_by_key(|(delta, _)| delta.abs());
    // Reading a track is one seek through the box tree, but there is no point
    // trying every clip in a large folder.
    candidates.into_iter().take(4).find_map(|(delta, p)| {
        let track = crate::gps::Track::read(&p).ok().flatten()?.shifted(delta);
        (track.fix_count() > 0).then_some(track)
    })
}

struct NameParts {
    start: Option<NaiveDateTime>,
    camera: Option<Camera>,
    sequence: Option<u32>,
}

/// Decode dashcam file names such as `2026_0801_131218_000289F` or
/// `2025_0804_144044_F` (no sequence number).
fn parse_name(stem: &str) -> Option<NameParts> {
    let re = Regex::new(
        r"(?x)
        (?P<y>\d{4}) [_-]? (?P<mo>\d{2}) (?P<d>\d{2}) [_-]?
        (?P<h>\d{2}) (?P<mi>\d{2}) (?P<s>\d{2})
        (?: [_-] (?P<seq>\d+) )?
        [_-]?
        (?P<cam>[FRfr])?
        \s*$",
    )
    .expect("static regex");
    let c = re.captures(stem)?;
    let num = |k: &str| c.name(k).and_then(|m| m.as_str().parse::<u32>().ok());
    let (y, mo, d) = (num("y")?, num("mo")?, num("d")?);
    let (h, mi, s) = (num("h")?, num("mi")?, num("s")?);
    let start = chrono::NaiveDate::from_ymd_opt(y as i32, mo, d)
        .and_then(|date| date.and_hms_opt(h, mi, s));
    Some(NameParts {
        start,
        camera: c.name("cam").map(|m| match m.as_str() {
            "R" | "r" => Camera::Rear,
            _ => Camera::Front,
        }),
        sequence: num("seq"),
    })
}

struct ProbeMeta {
    width: u32,
    height: u32,
    fps: f64,
    duration: f64,
}

fn ffprobe(path: &Path) -> Result<ProbeMeta> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .context("failed to run ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);

    let mut width = None;
    let mut height = None;
    let mut fps = None;
    let mut durations: Vec<f64> = Vec::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "width" => width = value.trim().parse().ok(),
            "height" => height = value.trim().parse().ok(),
            "r_frame_rate" => fps = parse_rational(value.trim()),
            "duration" => {
                if let Ok(d) = value.trim().parse::<f64>() {
                    durations.push(d);
                }
            }
            _ => {}
        }
    }

    Ok(ProbeMeta {
        width: width.ok_or_else(|| anyhow!("no video stream in {}", path.display()))?,
        height: height.ok_or_else(|| anyhow!("no video stream in {}", path.display()))?,
        fps: fps.unwrap_or(30.0),
        duration: durations
            .into_iter()
            .find(|d| *d > 0.0)
            .ok_or_else(|| anyhow!("could not determine duration of {}", path.display()))?,
    })
}

fn parse_rational(s: &str) -> Option<f64> {
    let (n, d) = s.split_once('/')?;
    let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
    (d != 0.0).then_some(n / d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dashcam_name() {
        let p = parse_name("2026_0801_131218_000289F").expect("should parse");
        assert_eq!(
            p.start.map(|t| t.to_string()),
            Some("2026-08-01 13:12:18".to_string())
        );
        assert_eq!(p.camera, Some(Camera::Front));
        assert_eq!(p.sequence, Some(289));
    }

    #[test]
    fn parses_rear_camera() {
        let p = parse_name("2026_0717_112820_000002R").expect("should parse");
        assert_eq!(p.camera, Some(Camera::Rear));
        assert_eq!(p.sequence, Some(2));
    }

    #[test]
    fn parses_name_without_sequence() {
        let p = parse_name("2025_0804_144044_F").expect("should parse");
        assert_eq!(
            p.start.map(|t| t.to_string()),
            Some("2025-08-04 14:40:44".to_string())
        );
        assert_eq!(p.camera, Some(Camera::Front));
        assert_eq!(p.sequence, None);
    }

    #[test]
    fn unknown_name_is_not_fatal() {
        assert!(parse_name("holiday-clip").is_none());
    }

    #[test]
    fn rational_fps() {
        assert_eq!(parse_rational("30/1"), Some(30.0));
        assert_eq!(parse_rational("30000/1001").map(|f| f.round()), Some(30.0));
        assert_eq!(parse_rational("0/0"), None);
    }
}
