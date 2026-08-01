//! Reads the GPS track Novatek-based dashcams embed in the MP4.
//!
//! The camera writes a `moov/gps ` index of one-second `freeGPS` records living
//! in `mdat`. Each record carries a UTC fix, which is a better clock than the
//! file name: the file name comes from the camera's own RTC, and on this
//! footage that RTC ran three seconds behind GPS.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate, NaiveDateTime};

/// `freeGPS` payload offsets, all little-endian despite the big-endian boxes
/// they sit inside.
const MAGIC: &[u8; 8] = b"freeGPS ";
const RECORD_READ: usize = 96;
const TIME_AT: usize = 48;
const STATUS_AT: usize = 72;
const COORDS_AT: usize = 76;
/// A clip is ten minutes, so anything past a few hours of fixes is a misparse.
const MAX_FIXES: u32 = 100_000;

#[derive(Clone, Copy, Debug)]
pub struct Fix {
    /// UTC, straight from the satellites.
    pub time: NaiveDateTime,
    pub lat: f64,
    pub lon: f64,
    pub speed_knots: f32,
    pub bearing: f32,
}

impl Fix {
    pub fn speed_mph(&self) -> f32 {
        self.speed_knots * 1.150_779
    }

    pub fn speed_kmh(&self) -> f32 {
        self.speed_knots * 1.852
    }
}

/// One fix per second of video, indexed by whole seconds from the clip start.
#[derive(Clone, Debug)]
pub struct Track {
    fixes: Vec<Option<Fix>>,
}

impl Track {
    /// Build a track directly from per-second fixes, for tests.
    #[cfg(test)]
    pub fn from_fixes(fixes: Vec<Option<Fix>>) -> Track {
        Track { fixes }
    }

    /// Returns `Ok(None)` when the file simply has no GPS box, which is normal
    /// for rear-camera clips and non-Novatek cameras.
    pub fn read(path: &Path) -> Result<Option<Track>> {
        let mut file = File::open(path)
            .with_context(|| format!("failed to open {} for GPS", path.display()))?;
        let len = file.metadata()?.len();
        let Some((body, end)) = find_gps_box(&mut file, 0, len)? else {
            return Ok(None);
        };

        let mut header = [0u8; 8];
        file.seek(SeekFrom::Start(body))?;
        file.read_exact(&mut header)?;
        let count = u32::from_be_bytes(header[4..8].try_into().expect("4 bytes"));
        if count == 0 || count > MAX_FIXES || body + 8 + count as u64 * 8 > end {
            return Ok(None);
        }

        let mut index = vec![0u8; count as usize * 8];
        file.read_exact(&mut index)?;
        let mut fixes = Vec::with_capacity(count as usize);
        for entry in index.chunks_exact(8) {
            let offset = u32::from_be_bytes(entry[0..4].try_into().expect("4 bytes")) as u64;
            fixes.push(if offset + RECORD_READ as u64 <= len {
                read_fix(&mut file, offset).unwrap_or(None)
            } else {
                None
            });
        }
        if fixes.iter().all(Option::is_none) {
            return Ok(None);
        }
        Ok(Some(Track { fixes }))
    }

    /// The fix covering a given offset into the clip.
    pub fn at(&self, offset_secs: f64) -> Option<&Fix> {
        if offset_secs < 0.0 {
            return None;
        }
        self.fixes.get(offset_secs.round() as usize)?.as_ref()
    }

    /// The nearest fix either side of an offset, for stretches where the
    /// receiver briefly lost its lock.
    pub fn nearest(&self, offset_secs: f64) -> Option<&Fix> {
        if let Some(fix) = self.at(offset_secs) {
            return Some(fix);
        }
        let centre = offset_secs.max(0.0).round() as usize;
        (1..=15).find_map(|d| {
            let before = centre.checked_sub(d).and_then(|i| self.fixes.get(i));
            let after = self.fixes.get(centre + d);
            before
                .and_then(|f| f.as_ref())
                .or_else(|| after.and_then(|f| f.as_ref()))
        })
    }

    pub fn first(&self) -> Option<&Fix> {
        self.fixes.iter().flatten().next()
    }

    /// The first valid fix together with the clip second it sits at. A receiver
    /// that starts without a lock can leave the opening minutes empty, so the
    /// index cannot be assumed to be zero.
    fn first_indexed(&self) -> Option<(usize, &Fix)> {
        self.fixes
            .iter()
            .enumerate()
            .find_map(|(i, f)| f.as_ref().map(|f| (i, f)))
    }

    pub fn last(&self) -> Option<&Fix> {
        self.fixes.iter().flatten().next_back()
    }

    pub fn fix_count(&self) -> usize {
        self.fixes.iter().flatten().count()
    }

    /// Minutes to add to UTC to reach the camera's local time, inferred by
    /// comparing the first fix against the file name and rounding to the
    /// nearest quarter hour. Real zones are all quarter-hour multiples, so the
    /// rounding absorbs the RTC's drift.
    pub fn utc_offset_minutes(&self, local_start: NaiveDateTime) -> Option<i32> {
        let (index, fix) = self.first_indexed()?;
        // What the camera's clock would read at the moment of that fix.
        let local_at_fix = local_start + Duration::seconds(index as i64);
        let delta = (local_at_fix - fix.time).num_seconds() as f64 / 60.0;
        let minutes = (delta / 15.0).round() as i32 * 15;
        // Beyond +/-14 h we are not looking at a time zone.
        (-14 * 60..=14 * 60).contains(&minutes).then_some(minutes)
    }

    /// How far the camera's own clock runs behind GPS, in seconds. Positive
    /// means the file name is stamped earlier than satellite time.
    pub fn clock_skew_secs(&self, local_start: NaiveDateTime) -> Option<i64> {
        let (index, fix) = self.first_indexed()?;
        let offset = self.utc_offset_minutes(local_start)?;
        let gps_local = fix.time + Duration::minutes(offset as i64);
        let local_at_fix = local_start + Duration::seconds(index as i64);
        Some((gps_local - local_at_fix).num_seconds())
    }
}

fn read_fix(file: &mut File, offset: u64) -> Result<Option<Fix>> {
    let mut buf = [0u8; RECORD_READ];
    file.seek(SeekFrom::Start(offset))?;
    if file.read_exact(&mut buf).is_err() || &buf[4..12] != MAGIC {
        return Ok(None);
    }
    // 'A' is an active fix; 'V' means the receiver had no lock.
    if buf[STATUS_AT] != b'A' {
        return Ok(None);
    }

    let u = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().expect("4 bytes"));
    let f = |at: usize| f32::from_le_bytes(buf[at..at + 4].try_into().expect("4 bytes"));
    let (hour, min, sec) = (u(TIME_AT), u(TIME_AT + 4), u(TIME_AT + 8));
    let (year, month, day) = (u(TIME_AT + 12), u(TIME_AT + 16), u(TIME_AT + 20));
    let Some(time) = NaiveDate::from_ymd_opt(2000 + year as i32, month, day)
        .and_then(|d| d.and_hms_opt(hour, min, sec))
    else {
        return Ok(None);
    };

    let lat = to_degrees(f(COORDS_AT), buf[STATUS_AT + 1] == b'S');
    let lon = to_degrees(f(COORDS_AT + 4), buf[STATUS_AT + 2] == b'W');
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return Ok(None);
    }
    Ok(Some(Fix {
        time,
        lat,
        lon,
        speed_knots: f(COORDS_AT + 8),
        bearing: f(COORDS_AT + 12),
    }))
}

/// NMEA packs coordinates as `ddmm.mmmm`, not decimal degrees.
fn to_degrees(raw: f32, negate: bool) -> f64 {
    let raw = raw as f64;
    let degrees = (raw / 100.0).trunc();
    let value = degrees + (raw - degrees * 100.0) / 60.0;
    if negate {
        -value
    } else {
        value
    }
}

/// Walk the box tree for `moov/gps `, returning its payload range. The box is a
/// direct child of `moov`, a sibling of `udta` rather than inside it.
fn find_gps_box(file: &mut File, start: u64, end: u64) -> Result<Option<(u64, u64)>> {
    for (kind, body, stop) in boxes(file, start, end)? {
        if &kind == b"moov" {
            for (child, child_body, child_stop) in boxes(file, body, stop)? {
                if &child == b"gps " {
                    return Ok(Some((child_body, child_stop)));
                }
            }
        }
    }
    Ok(None)
}

/// Immediate children of a box range as `(type, body start, box end)`.
fn boxes(file: &mut File, start: u64, end: u64) -> Result<Vec<([u8; 4], u64, u64)>> {
    let mut out = Vec::new();
    let mut offset = start;
    while offset + 8 <= end {
        let mut header = [0u8; 8];
        file.seek(SeekFrom::Start(offset))?;
        if file.read_exact(&mut header).is_err() {
            break;
        }
        let mut size = u32::from_be_bytes(header[0..4].try_into().expect("4 bytes")) as u64;
        let kind: [u8; 4] = header[4..8].try_into().expect("4 bytes");
        let mut body = offset + 8;
        if size == 1 {
            let mut large = [0u8; 8];
            if file.read_exact(&mut large).is_err() {
                break;
            }
            size = u64::from_be_bytes(large);
            body = offset + 16;
        } else if size == 0 {
            size = end - offset;
        }
        if size < 8 || offset + size > end {
            break;
        }
        out.push((kind, body, offset + size));
        offset += size;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(time: NaiveDateTime) -> Fix {
        Fix {
            time,
            lat: 47.5,
            lon: -122.2,
            speed_knots: 50.0,
            bearing: 90.0,
        }
    }

    fn at(h: u32, m: u32, s: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 8, 1)
            .expect("valid date")
            .and_hms_opt(h, m, s)
            .expect("valid time")
    }

    #[test]
    fn nmea_coordinates_become_degrees() {
        // 4735.4795' is 47 degrees 35.4795 minutes.
        assert!((to_degrees(4735.4795, false) - 47.59132).abs() < 1e-4);
        assert!((to_degrees(12215.864, true) + 122.26440).abs() < 1e-4);
    }

    #[test]
    fn speed_conversions() {
        let f = fix(at(20, 12, 21));
        assert!((f.speed_mph() - 57.54).abs() < 0.1);
        assert!((f.speed_kmh() - 92.6).abs() < 0.1);
    }

    #[test]
    fn infers_time_zone_from_the_file_name() {
        let track = Track {
            fixes: vec![Some(fix(at(20, 12, 21)))],
        };
        // File name says 13:12:18 local against a 20:12:21 UTC fix.
        let offset = track.utc_offset_minutes(at(13, 12, 18));
        assert_eq!(offset, Some(-7 * 60), "Pacific daylight time");
        assert_eq!(
            track.clock_skew_secs(at(13, 12, 18)),
            Some(3),
            "the camera clock is three seconds behind GPS"
        );
    }

    #[test]
    fn late_first_lock_still_infers_the_zone() {
        // A receiver with no lock for the first two seconds: the only fix sits
        // at clip second 2, not zero.
        let track = Track {
            fixes: vec![None, None, Some(fix(at(20, 12, 23)))],
        };
        assert_eq!(
            track.utc_offset_minutes(at(13, 12, 18)),
            Some(-7 * 60),
            "the fix's own index must be accounted for"
        );
        assert_eq!(track.clock_skew_secs(at(13, 12, 18)), Some(3));
    }

    #[test]
    fn nearest_bridges_a_lost_lock() {
        let track = Track {
            fixes: vec![Some(fix(at(20, 12, 21))), None, None, Some(fix(at(20, 12, 24)))],
        };
        assert!(track.at(1.0).is_none(), "no fix at that second");
        assert!(track.nearest(1.0).is_some(), "falls back to a neighbour");
        assert_eq!(track.fix_count(), 2);
    }

    #[test]
    fn absurd_offsets_are_not_time_zones() {
        let track = Track {
            fixes: vec![Some(fix(at(20, 12, 21)))],
        };
        let far = NaiveDate::from_ymd_opt(2020, 1, 1)
            .expect("valid date")
            .and_hms_opt(0, 0, 0)
            .expect("valid time");
        assert_eq!(track.utc_offset_minutes(far), None);
    }
}
