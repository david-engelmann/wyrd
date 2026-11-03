//! Rate statistics for wyrd results.
//!
//! Charter section 4: every rate carries a Wilson 95 percent interval.
//! These functions are pure and carry no IO, so every arm, stage, and
//! publisher computes intervals the same way.

/// A rate with its Wilson score interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
    pub successes: u64,
    pub total: u64,
    pub point: f64,
    pub low: f64,
    pub high: f64,
}

/// Wilson score interval for a binomial proportion.
///
/// `z` is 1.96 for a 95 percent interval, which is the charter default and
/// the value [`rate_95`] uses. Returns `None` only when `total` is zero,
/// because a rate over no cases is not a wide interval, it is no result.
pub fn wilson_interval(successes: u64, total: u64, z: f64) -> Option<Rate> {
    if total == 0 {
        return None;
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let margin = (z / denom) * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();
    Some(Rate {
        successes,
        total,
        point: p,
        low: (centre - margin).clamp(0.0, 1.0),
        high: (centre + margin).clamp(0.0, 1.0),
    })
}

/// The charter default rate: Wilson interval at 95 percent.
pub fn rate_95(successes: u64, total: u64) -> Option<Rate> {
    wilson_interval(successes, total, 1.96)
}

/// Discordant-pair counts for a paired clean/poisoned comparison.
///
/// Charter section 4 requires the flips behind a paired difference to be
/// visible, not just the difference. `both`, `poisoned_only`,
/// `clean_only`, and `neither` count cases by which arms exhibited the
/// target behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Discordance {
    pub both: u64,
    pub poisoned_only: u64,
    pub clean_only: u64,
    pub neither: u64,
}

/// Tallies paired outcomes. Each pair is (clean showed target, poisoned
/// showed target) for the same case.
pub fn discordance(pairs: &[(bool, bool)]) -> Discordance {
    let mut d = Discordance::default();
    for &(clean, poisoned) in pairs {
        match (clean, poisoned) {
            (true, true) => d.both += 1,
            (false, true) => d.poisoned_only += 1,
            (true, false) => d.clean_only += 1,
            (false, false) => d.neither += 1,
        }
    }
    d
}
