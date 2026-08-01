use std::collections::HashMap;

use std::collections::HashSet;

use crate::gps::Track as GpsTrack;
use crate::plate;
use crate::scan::Candidate;

/// One accepted plate reading in one sampled frame.
#[derive(Clone, Debug)]
pub struct Detection {
    pub offset: f64,
    pub text: String,
    pub conf: f32,
    pub rect: (u32, u32, u32, u32),
    pub detector_score: Option<f32>,
    pub region: Option<String>,
    /// Verification still, encoded from the frame this reading came from.
    pub still: Option<Vec<u8>>,
}

impl Detection {
    pub fn new(offset: f64, text: String, cand: &Candidate, still: Option<Vec<u8>>) -> Detection {
        Detection {
            offset,
            text,
            conf: cand.conf,
            rect: cand.rect,
            detector_score: cand.detector_score,
            region: cand.region.clone(),
            still,
        }
    }
}

/// A plate seen continuously over a stretch of the clip.
#[derive(Clone, Debug)]
pub struct Sighting {
    pub plate: String,
    pub first_seen: f64,
    pub last_seen: f64,
    pub frames: usize,
    pub best_conf: f32,
    pub mean_conf: f32,
    /// Offset and box of the highest-confidence reading, used for the crop.
    pub best_offset: f64,
    pub best_rect: (u32, u32, u32, u32),
    /// Detector objectness of the best reading, when a detector was used.
    pub detector_score: Option<f32>,
    /// Issuing region the recognizer predicted for the best reading.
    pub region: Option<String>,
    /// Verification still for the best reading.
    pub still: Option<Vec<u8>>,
    /// Distinct OCR spellings and how often each appeared.
    pub variants: Vec<(String, usize)>,
}

impl Sighting {
    pub fn duration(&self) -> f64 {
        self.last_seen - self.first_seen
    }
}

/// How long a vehicle plausibly stays in view at a "typical" road speed. The
/// merge window is scaled against this: crawling in traffic keeps the car ahead
/// of you far longer than overtaking it at speed does.
const REFERENCE_MPH: f32 = 30.0;

/// Group detections into sightings.
///
/// One vehicle rarely yields one spelling: a plate read over twenty frames comes
/// back as a handful of near-identical strings. Character folding removes the
/// systematic confusions, and the residual differences are absorbed by matching
/// on edit distance, so a track is the same plate seen without too long a gap.
/// A plate that reappears much later becomes a separate sighting.
pub fn build(
    mut detections: Vec<Detection>,
    gap: f64,
    min_hits: usize,
    gps: Option<&GpsTrack>,
) -> Vec<Sighting> {
    detections.sort_by(|a, b| a.offset.total_cmp(&b.offset));

    let mut open: Vec<Track> = Vec::new();
    let mut closed: Vec<Track> = Vec::new();
    for d in detections {
        let window = gap_at(gap, gps, d.offset);
        // Retire tracks that have gone quiet before looking for a match, so a
        // reappearance cannot attach to a stale track.
        let mut i = 0;
        while i < open.len() {
            if d.offset - open[i].last > window {
                closed.push(open.swap_remove(i));
            } else {
                i += 1;
            }
        }

        let key = plate::canonical(&d.text);
        let frame = frame_key(d.offset);
        let best = open
            .iter_mut()
            // A track already seen in this frame is a different vehicle, no
            // matter how alike the two readings are.
            .filter(|t| !t.offsets.contains(&frame))
            .filter_map(|t| distance_within(&t.key, &key).map(|dist| (dist, t)))
            .min_by_key(|(dist, _)| *dist)
            .map(|(_, t)| t);
        match best {
            Some(track) => track.push(d),
            None => open.push(Track::new(key, d)),
        }
    }
    closed.append(&mut open);

    let mut sightings: Vec<Sighting> = coalesce(closed, gap, gps)
        .into_iter()
        .filter_map(|t| finish(t.detections, min_hits))
        .collect();
    sightings.sort_by(|a, b| {
        a.first_seen
            .total_cmp(&b.first_seen)
            .then_with(|| a.plate.cmp(&b.plate))
    });
    sightings
}

/// Seconds of quiet allowed before a sighting is closed, widened when the
/// camera vehicle is slow and narrowed when it is quick. Without GPS the
/// caller's figure is used unchanged.
fn gap_at(base: f64, gps: Option<&GpsTrack>, offset: f64) -> f64 {
    let Some(fix) = gps.and_then(|t| t.nearest(offset)) else {
        return base;
    };
    let factor = (REFERENCE_MPH / fix.speed_mph().max(1.0)).clamp(0.5, 4.0);
    base * factor as f64
}

/// Fold together tracks that turned out to be one vehicle.
///
/// Assigning each detection to its closest open track cannot undo an early
/// split: once a bad reading has opened a second track, later good readings
/// join whichever is nearer and the two never reunite. This pass reunites them
/// afterwards, guarded by the one fact that settles it — two plates read in the
/// same frame are two different vehicles, however similar they look.
fn coalesce(mut tracks: Vec<Track>, gap: f64, gps: Option<&GpsTrack>) -> Vec<Track> {
    loop {
        let mut absorbed = None;
        'search: for i in 0..tracks.len() {
            for j in i + 1..tracks.len() {
                if joinable(&tracks[i], &tracks[j], gap, gps) {
                    absorbed = Some((i, j));
                    break 'search;
                }
            }
        }
        let Some((i, j)) = absorbed else { break };
        let other = tracks.remove(j);
        tracks[i].absorb(other);
    }
    tracks
}

fn joinable(a: &Track, b: &Track, gap: f64, gps: Option<&GpsTrack>) -> bool {
    // Seen together at least once, so they are two vehicles, not one.
    if a.offsets.intersection(&b.offsets).next().is_some() {
        return false;
    }
    let Some(distance) = distance_within(&a.key, &b.key) else {
        return false;
    };
    let separation = if a.first <= b.last && b.first <= a.last {
        0.0 // their spans interleave
    } else if a.last < b.first {
        b.first - a.last
    } else {
        a.first - b.last
    };
    // Two spellings that agree exactly are strong evidence of one vehicle, so
    // they are allowed a longer dropout than a pair that had to be guessed at.
    let trust = match distance {
        0 => 2.0,
        1 => 1.0,
        _ => 0.6,
    };
    separation <= gap_at(gap, gps, a.last.min(b.last)) * trust
}

/// An in-progress sighting.
struct Track {
    /// Folded spelling this track matches against, from its strongest reading.
    key: String,
    best_conf: f32,
    first: f64,
    last: f64,
    /// Sampled frames this track appears in, keyed in milliseconds so they
    /// compare exactly. Two tracks sharing one are two vehicles.
    offsets: HashSet<u64>,
    detections: Vec<Detection>,
}

fn frame_key(offset: f64) -> u64 {
    (offset * 1000.0).round() as u64
}

impl Track {
    fn new(key: String, d: Detection) -> Track {
        Track {
            key,
            best_conf: d.conf,
            first: d.offset,
            last: d.offset,
            offsets: HashSet::from([frame_key(d.offset)]),
            detections: vec![d],
        }
    }

    fn push(&mut self, d: Detection) {
        self.last = self.last.max(d.offset);
        self.offsets.insert(frame_key(d.offset));
        // Keep the strongest reading's spelling, so a weak early guess does not
        // anchor the match key for the rest of the track.
        if d.conf > self.best_conf {
            self.best_conf = d.conf;
            self.key = plate::canonical(&d.text);
        }
        self.detections.push(d);
    }

    fn absorb(&mut self, other: Track) {
        self.first = self.first.min(other.first);
        self.last = self.last.max(other.last);
        self.offsets.extend(other.offsets);
        if other.best_conf > self.best_conf {
            self.best_conf = other.best_conf;
            self.key = other.key;
        }
        self.detections.extend(other.detections);
        self.detections.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    }
}

/// Edit distance between two folded plates, when they are close enough to be
/// the same plate. Longer plates get one more character of slack.
fn distance_within(a: &str, b: &str) -> Option<usize> {
    let budget = if a.len().max(b.len()) >= 7 { 2 } else { 1 };
    if a.len().abs_diff(b.len()) > budget {
        return None;
    }
    let dist = levenshtein(a, b);
    (dist <= budget).then_some(dist)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn finish(mut run: Vec<Detection>, min_hits: usize) -> Option<Sighting> {
    // Distinct sampled frames, so two windows reading the same plate in one
    // frame cannot on their own satisfy --min-hits.
    let mut offsets: Vec<f64> = run.iter().map(|d| d.offset).collect();
    offsets.dedup_by(|a, b| a == b);
    if offsets.len() < min_hits.max(1) {
        return None;
    }

    let best_at = run
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.conf.total_cmp(&b.1.conf))
        .map(|(i, _)| i)
        .expect("run is non-empty");
    // Lift what the sighting needs out of the best reading, so the still can be
    // moved out without holding a borrow of the run.
    let best_conf = run[best_at].conf;
    let best_offset = run[best_at].offset;
    let best_rect = run[best_at].rect;
    let detector_score = run[best_at].detector_score;
    let region = run[best_at].region.clone();
    let still = run[best_at].still.take();

    let first_seen = run.first().expect("non-empty").offset;
    let last_seen = run.last().expect("non-empty").offset;
    let mean_conf = run.iter().map(|d| d.conf).sum::<f32>() / run.len() as f32;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut weight: HashMap<&str, f32> = HashMap::new();
    for d in &run {
        *counts.entry(d.text.as_str()).or_default() += 1;
        *weight.entry(d.text.as_str()).or_default() += d.conf;
    }
    // The reported spelling is the one with the most confidence behind it.
    let plate = weight
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| a.0.cmp(b.0)))
        .map(|(t, _)| t.to_string())
        .expect("run is non-empty");
    let mut variants: Vec<(String, usize)> =
        counts.into_iter().map(|(t, n)| (t.to_string(), n)).collect();
    variants.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    Some(Sighting {
        plate,
        first_seen,
        last_seen,
        frames: offsets.len(),
        best_conf,
        mean_conf,
        best_offset,
        best_rect,
        detector_score,
        region,
        still,
        variants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(offset: f64, text: &str, conf: f32) -> Detection {
        Detection {
            offset,
            text: text.to_string(),
            conf,
            rect: (10, 20, 100, 30),
            detector_score: None,
            region: None,
            still: None,
        }
    }

    #[test]
    fn merges_ocr_variants_into_one_sighting() {
        let s = build(
            vec![
                det(1.0, "8ABC123", 90.0),
                det(1.5, "BABC123", 70.0),
                det(2.0, "8ABC123", 95.0),
            ],
            5.0,
            2, None);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].plate, "8ABC123", "highest total confidence wins");
        assert_eq!(s[0].frames, 3);
        assert_eq!(s[0].best_conf, 95.0);
        assert_eq!(s[0].variants.len(), 2);
    }

    #[test]
    fn splits_reappearance_after_gap() {
        let s = build(
            vec![
                det(1.0, "8ABC123", 90.0),
                det(1.5, "8ABC123", 90.0),
                det(60.0, "8ABC123", 90.0),
                det(60.5, "8ABC123", 90.0),
            ],
            5.0,
            2, None);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].first_seen, 1.0);
        assert_eq!(s[1].first_seen, 60.0);
    }

    #[test]
    fn drops_single_frame_noise() {
        let s = build(vec![det(1.0, "8ABC123", 99.0)], 5.0, 2, None);
        assert!(s.is_empty());
    }

    #[test]
    fn repeats_within_one_frame_do_not_count_as_hits() {
        let s = build(
            vec![det(1.0, "8ABC123", 90.0), det(1.0, "8ABC123", 92.0)],
            5.0,
            2, None);
        assert!(s.is_empty(), "two tiles in one frame is still one sighting");
    }

    #[test]
    fn merges_readings_within_one_edit() {
        // Observed on real footage: one vehicle read as four spellings.
        let s = build(
            vec![
                det(1.0, "CPZ0319", 95.0),
                det(1.5, "CP70319", 90.0),
                det(2.0, "DPZ0319", 88.0),
                det(2.5, "FP70319", 85.0),
            ],
            5.0,
            2, None);
        assert_eq!(s.len(), 1, "one vehicle should be one sighting: {s:?}");
        assert_eq!(s[0].plate, "CPZ0319", "strongest reading wins");
        assert_eq!(s[0].frames, 4);
    }

    /// A track that ran at `mph` for a minute.
    fn ego(mph: f32) -> GpsTrack {
        let time = chrono::NaiveDate::from_ymd_opt(2026, 8, 1)
            .expect("valid date")
            .and_hms_opt(20, 12, 21)
            .expect("valid time");
        GpsTrack::from_fixes(
            (0..60)
                .map(|_| {
                    Some(crate::gps::Fix {
                        time,
                        lat: 47.5,
                        lon: -122.2,
                        speed_knots: mph / 1.150_779,
                        bearing: 90.0,
                    })
                })
                .collect(),
        )
    }

    #[test]
    fn reunites_a_vehicle_split_across_two_tracks() {
        // A bad early reading opens its own track; the good readings land in
        // another. Nothing is ever seen in the same frame, so they are one car.
        let s = build(
            vec![
                det(1.0, "C5D4391", 60.0),
                det(2.0, "CSO438W", 91.0),
                det(3.0, "C60439W", 99.0),
                det(4.0, "C60439W", 98.0),
            ],
            10.0,
            2,
            None,
        );
        assert_eq!(s.len(), 1, "one vehicle, one row: {s:?}");
        assert_eq!(s[0].plate, "C60439W", "the confident reading wins");
        assert_eq!(s[0].frames, 4);
    }

    #[test]
    fn plates_seen_in_the_same_frame_are_never_merged() {
        // Similar enough to pass the edit-distance test, but read side by side
        // in the same frames, so they must be two vehicles.
        let s = build(
            vec![
                det(1.0, "CGK2606", 100.0),
                det(1.0, "EGK2605", 97.0),
                det(1.5, "CGK2606", 100.0),
                det(1.5, "EGK2605", 96.0),
            ],
            10.0,
            2,
            None,
        );
        assert_eq!(s.len(), 2, "two cars abreast stay separate: {s:?}");
    }

    #[test]
    fn crawling_traffic_widens_the_merge_window() {
        // The same 20 s dropout, at a crawl and at speed.
        let readings = || {
            vec![
                det(1.0, "CPZ0319", 99.0),
                det(2.0, "CPZ0319", 99.0),
                det(22.0, "CPZ0319", 99.0),
                det(23.0, "CPZ0319", 99.0),
            ]
        };
        let crawling = build(readings(), 10.0, 2, Some(&ego(5.0)));
        assert_eq!(crawling.len(), 1, "still the car in front of you");

        let quick = build(readings(), 10.0, 2, Some(&ego(70.0)));
        assert_eq!(quick.len(), 2, "at speed, that is a different car");
    }

    #[test]
    fn identical_readings_survive_a_longer_dropout_than_guesses() {
        // The same 14 s dropout, once with matching spellings and once with a
        // two-character difference.
        let same = build(
            vec![
                det(1.0, "CGK2606", 100.0),
                det(2.0, "CGK2606", 100.0),
                det(16.0, "CGK2606", 100.0),
                det(17.0, "CGK2606", 100.0),
            ],
            10.0,
            2,
            None,
        );
        assert_eq!(same.len(), 1, "an exact match is trusted across the gap");

        let fuzzy = build(
            vec![
                det(1.0, "CGK2606", 100.0),
                det(2.0, "CGK2606", 100.0),
                det(16.0, "EGK2605", 80.0),
                det(17.0, "EGK2605", 80.0),
            ],
            10.0,
            2,
            None,
        );
        assert_eq!(fuzzy.len(), 2, "a guess does not reach as far");
    }

    #[test]
    fn gap_scales_with_speed() {
        assert_eq!(gap_at(10.0, None, 0.0), 10.0, "no GPS, no adjustment");
        assert!(gap_at(10.0, Some(&ego(5.0)), 0.0) > 10.0);
        assert!(gap_at(10.0, Some(&ego(70.0)), 0.0) < 10.0);
        // Clamped so a standstill cannot merge the whole clip together.
        assert_eq!(gap_at(10.0, Some(&ego(0.1)), 0.0), 40.0);
    }

    #[test]
    fn keeps_genuinely_different_plates_apart() {
        let s = build(
            vec![
                det(1.0, "CPZ0319", 95.0),
                det(1.5, "CPZ0319", 95.0),
                det(1.0, "CJY7273", 95.0),
                det(1.5, "CJY7273", 95.0),
            ],
            5.0,
            2, None);
        assert_eq!(s.len(), 2, "two vehicles stay separate: {s:?}");
    }

    #[test]
    fn short_plates_get_less_slack() {
        // Two edits apart, but short enough that the budget is one.
        assert_eq!(distance_within("AB123", "AB199"), None);
        assert_eq!(distance_within("AB123", "AB193"), Some(1));
        // Seven characters earn a second edit of slack.
        assert_eq!(distance_within("CPZ0319", "CP70319"), Some(1));
        assert_eq!(distance_within("CPZ0319", "CX70319"), Some(2));
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("ABC", "ABC"), 0);
        assert_eq!(levenshtein("ABC", "ABD"), 1);
        assert_eq!(levenshtein("ABC", "AC"), 1);
        assert_eq!(levenshtein("", "ABC"), 3);
    }

    #[test]
    fn sightings_are_ordered_by_time() {
        let s = build(
            vec![
                det(9.0, "ZZZ999", 90.0),
                det(9.5, "ZZZ999", 90.0),
                det(1.0, "8ABC123", 90.0),
                det(1.5, "8ABC123", 90.0),
            ],
            5.0,
            2, None);
        assert_eq!(s[0].plate, "8ABC123");
        assert_eq!(s[1].plate, "ZZZ999");
    }
}
