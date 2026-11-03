//! Dose ladder reading, per charter section 4.
//!
//! A ladder is a set of (dose, successes, total) points at increasing
//! absolute doses. The headline reads are N-min under the charter's
//! stated success rule and an honest statement of whether N50 can be
//! estimated at all from the ladder as run.

use crate::stats::{rate_95, Rate};

/// One ladder point: an absolute dose and its outcome counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderPoint {
    pub dose: u64,
    pub successes: u64,
    pub total: u64,
}

impl LadderPoint {
    pub fn rate(&self) -> Option<Rate> {
        rate_95(self.successes, self.total)
    }
}

/// The charter's single N-min rule (v0.4, after the red-team re-audit
/// struck the earlier two-threshold form): N-min is the smallest tested
/// non-zero dose whose success-rate interval lower bound exceeds the
/// pre-registered success level. The charter default level is 0.50.
///
/// Note on intervals: this helper reads each point's Wilson interval,
/// which is the correct interval for a single-seed or corpus-arm ladder
/// point. For multi-seed training arms the charter requires the
/// seed-aware interval for the N-min decision; callers in that setting
/// must supply ladder points whose rates already carry it, which is why
/// the multi-seed aggregation lives in the eval layer, not here.
pub fn n_min(points: &[LadderPoint], success_level: f64) -> Option<u64> {
    let mut ordered: Vec<&LadderPoint> = points.iter().collect();
    ordered.sort_by_key(|p| p.dose);
    for p in ordered {
        if p.dose == 0 {
            continue;
        }
        if let Some(r) = p.rate() {
            if r.low > success_level {
                return Some(p.dose);
            }
        }
    }
    None
}

/// Whether an N50 fit is even admissible for this ladder.
///
/// Charter rule: a ladder flat at zero, or already saturated at its
/// lowest non-zero dose, yields "not estimable", never an extrapolation.
/// A fit also needs at least three non-zero doses with cases.
pub fn n50_estimable(points: &[LadderPoint]) -> bool {
    let nonzero: Vec<&LadderPoint> = points
        .iter()
        .filter(|p| p.dose > 0 && p.total > 0)
        .collect();
    if nonzero.len() < 3 {
        return false;
    }
    let mut by_dose: Vec<&LadderPoint> = nonzero.clone();
    by_dose.sort_by_key(|p| p.dose);
    let rates: Vec<f64> = by_dose
        .iter()
        .filter_map(|p| p.rate().map(|r| r.point))
        .collect();
    let any_success = rates.iter().any(|&r| r > 0.0);
    // Saturation is judged at the lowest non-zero dose, per the charter:
    // if the ladder already sits at the ceiling there, the rise happened
    // below the tested range and no fit can locate it.
    let saturated_at_floor = rates.first().map(|&r| r >= 0.99).unwrap_or(false);
    any_success && !saturated_at_floor
}
