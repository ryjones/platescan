use anyhow::{Context, Result};
use regex::Regex;

use crate::cli::Region;

/// Text that repeatedly survives OCR on road furniture but is never a plate.
const STOPWORDS: &[&str] = &[
    "STOP", "EXIT", "SPEED", "LIMIT", "AHEAD", "ONLY", "XING", "YIELD", "MERGE", "SLOW", "LANE",
    "TURN", "KEEP", "RIGHT", "LEFT", "ROAD", "AVE", "BLVD", "HWY", "MPH", "KMH", "END",
];

pub struct Rules {
    patterns: Vec<Regex>,
    min_len: usize,
    max_len: usize,
    /// When false, only length and obvious-junk checks apply. Used with a
    /// backend that has already localised a plate, where demanding a national
    /// format would throw away valid vanity and out-of-region plates.
    strict: bool,
}

impl Rules {
    pub fn new(region: Region, extra: &[String], strict: bool) -> Result<Rules> {
        let sources: Vec<String> = if extra.is_empty() {
            preset(region).iter().map(|s| s.to_string()).collect()
        } else {
            extra.to_vec()
        };
        let patterns = sources
            .iter()
            .map(|p| Regex::new(p).with_context(|| format!("invalid plate pattern {p:?}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Rules {
            patterns,
            min_len: 4,
            max_len: 8,
            strict,
        })
    }

    /// Normalise an OCR candidate and decide whether it looks like a plate.
    pub fn accept(&self, raw: &str) -> Option<String> {
        let text: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect();
        if text.len() < self.min_len || text.len() > self.max_len {
            return None;
        }
        if STOPWORDS.contains(&text.as_str()) {
            return None;
        }
        // A run of one or two repeated characters is a texture artefact.
        if text.chars().collect::<std::collections::HashSet<_>>().len() < 3 {
            return None;
        }
        if !self.strict {
            return Some(text);
        }
        // A plate always mixes letters and digits in the formats we target;
        // pure words and pure numbers are signage.
        let has_alpha = text.chars().any(|c| c.is_ascii_alphabetic());
        let has_digit = text.chars().any(|c| c.is_ascii_digit());
        if !has_alpha || !has_digit {
            return None;
        }
        self.patterns
            .iter()
            .any(|p| p.is_match(&text))
            .then_some(text)
    }
}

fn preset(region: Region) -> &'static [&'static str] {
    match region {
        Region::Generic => &[r"^[A-Z0-9]{5,8}$"],
        Region::Us => &[
            r"^[0-9][A-Z]{3}[0-9]{3}$", // California
            r"^[A-Z]{3}[0-9]{3,4}$",
            r"^[0-9]{3}[A-Z]{3}$",
            r"^[A-Z]{2}[0-9]{4,5}$",
            r"^[0-9]{2}[A-Z]{2}[0-9]{2}$",
            r"^[A-Z]{1}[0-9]{2}[A-Z]{3}$",
            r"^[0-9]{4}[A-Z]{2}$",
        ],
        Region::Eu => &[
            r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$", // UK current
            r"^[A-Z][0-9]{1,3}[A-Z]{3}$",  // UK prefix
            r"^[A-Z]{3}[0-9]{1,3}[A-Z]$",  // UK suffix
            r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}$",
        ],
    }
}

/// Fold the character pairs OCR routinely swaps, so that `8ABC123` and
/// `BABC123` collapse onto one identity for tracking.
pub fn canonical(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'O' | 'D' | 'Q' | '0' => '0',
            'I' | 'L' | '1' => '1',
            'Z' | '2' => '2',
            'S' | '5' => '5',
            'G' | '6' => '6',
            'B' | '8' => '8',
            'A' | '4' => '4',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generic() -> Rules {
        Rules::new(Region::Generic, &[], true).expect("preset compiles")
    }

    #[test]
    fn accepts_plate_like_text() {
        let r = generic();
        assert_eq!(r.accept("8ABC123").as_deref(), Some("8ABC123"));
        assert_eq!(r.accept("8-ABC 123").as_deref(), Some("8ABC123"));
        assert_eq!(r.accept("abc1234").as_deref(), Some("ABC1234"));
    }

    #[test]
    fn rejects_signage_and_noise() {
        let r = generic();
        assert_eq!(r.accept("SPEED"), None, "pure letters");
        assert_eq!(r.accept("45"), None, "too short and pure digits");
        assert_eq!(r.accept("123456"), None, "pure digits");
        assert_eq!(r.accept("AAAA11"), None, "too few distinct characters");
        assert_eq!(r.accept("ABCDEFGHI9"), None, "too long");
    }

    #[test]
    fn us_preset_is_stricter_than_generic() {
        let us = Rules::new(Region::Us, &[], true).expect("preset compiles");
        assert_eq!(us.accept("8ABC123").as_deref(), Some("8ABC123"));
        assert_eq!(us.accept("A1B2C3"), None, "not a US format");
        assert_eq!(generic().accept("A1B2C3").as_deref(), Some("A1B2C3"));
    }

    #[test]
    fn custom_pattern_replaces_preset() {
        let r = Rules::new(Region::Generic, &[r"^X[0-9]{4}$".to_string()], true).expect("compiles");
        assert_eq!(r.accept("X1234").as_deref(), Some("X1234"));
        assert_eq!(r.accept("8ABC123"), None);
    }

    #[test]
    fn lenient_mode_skips_format_checks() {
        let r = Rules::new(Region::Us, &[], false).expect("preset compiles");
        assert_eq!(
            r.accept("A1B2C3").as_deref(),
            Some("A1B2C3"),
            "national format is not enforced"
        );
        assert_eq!(
            r.accept("VANITY").as_deref(),
            Some("VANITY"),
            "an all-letter vanity plate is still a plate"
        );
        assert_eq!(r.accept("AAA111"), None, "junk is still rejected");
        assert_eq!(r.accept("AB"), None, "too short is still rejected");
    }

    #[test]
    fn canonical_folds_confusable_characters() {
        assert_eq!(canonical("8ABC123"), canonical("BA8C123"));
        assert_eq!(canonical("0O"), "00");
        assert_ne!(canonical("8ABC123"), canonical("9ABC123"));
    }
}
