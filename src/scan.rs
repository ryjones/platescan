use anyhow::Result;

use crate::frames::Frame;

/// A plate-shaped text reading located in full-frame source pixels.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub text: String,
    /// 0-100, whatever the backend's own confidence scale maps onto.
    pub conf: f32,
    pub rect: (u32, u32, u32, u32),
    /// Objectness from the plate detector, when a detector was involved.
    pub detector_score: Option<f32>,
    /// Issuing region guessed by the recognizer, when it predicts one.
    pub region: Option<String>,
}

/// A recognition backend. One instance is owned by each worker thread, so
/// implementations do not need to be `Sync`.
pub trait Scanner: Send {
    fn scan(&mut self, frame: &Frame) -> Result<Vec<Candidate>>;

    /// Backend name and settings, recorded in the report so a run can be
    /// reproduced.
    fn describe(&self) -> String;
}

/// Window origins along one axis, overlapping so text on a seam is still seen
/// whole by at least one window.
pub fn axis_offsets(total: u32, window: u32, overlap: f64) -> Vec<u32> {
    if total <= window {
        return vec![0];
    }
    let step = ((window as f64 * (1.0 - overlap)).round() as u32).max(1);
    let mut out = Vec::new();
    let mut o = 0;
    while o + window < total {
        out.push(o);
        o += step;
    }
    out.push(total - window);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_cover_the_axis() {
        let offs = axis_offsets(3200, 960, 0.25);
        assert_eq!(offs[0], 0);
        assert_eq!(*offs.last().expect("non-empty"), 3200 - 960);
        for pair in offs.windows(2) {
            assert!(pair[1] > pair[0], "offsets must advance: {offs:?}");
            assert!(pair[1] < pair[0] + 960, "gap between windows: {offs:?}");
        }
    }

    #[test]
    fn single_offset_when_window_covers_axis() {
        assert_eq!(axis_offsets(500, 960, 0.25), vec![0]);
        assert_eq!(axis_offsets(960, 960, 0.25), vec![0]);
    }
}
