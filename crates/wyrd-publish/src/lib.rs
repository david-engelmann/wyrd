//! wyrd-publish: result rendering.
//!
//! Charter section 6 assigns this crate the board data and report
//! generation, under the rule that everything published derives from
//! the result data: no population counts, model names, or headline
//! figures are ever written as literals in publishing code. Phase 0
//! lands the paired-result renderer: one structured record in, a
//! deterministic JSON document and a plain-language markdown summary
//! out, with every number computed from the record.

use wyrd_core::stats::Rate;
use wyrd_eval::PairedReport;

/// Everything a published paired result needs to say about itself.
#[derive(Debug, Clone)]
pub struct PairedResult {
    pub experiment: String,
    pub stage: String,
    pub family: String,
    pub model: String,
    /// Total threads planted at this dose, the cross-stage headline
    /// dose per charter section 3.
    pub dose_total: u64,
    /// Dose per target, the secondary figure. None for untargeted arms.
    pub dose_per_target: Option<u64>,
    pub seed: u64,
    pub report: PairedReport,
}

fn fmt_rate(r: &Rate) -> String {
    format!(
        "{:.3} [{:.3}, {:.3}] ({}/{})",
        r.point, r.low, r.high, r.successes, r.total
    )
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Deterministic JSON rendering of one paired result. Key order is
/// fixed by construction, floats are printed at fixed precision, so the
/// same record renders to the same bytes everywhere.
pub fn paired_to_json(result: &PairedResult) -> String {
    let d = &result.report.discordance;
    let per_target = match result.dose_per_target {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    };
    format!(
        "{{\n  \"experiment\": \"{}\",\n  \"stage\": \"{}\",\n  \"family\": \"{}\",\n  \"model\": \"{}\",\n  \"dose_total\": {},\n  \"dose_per_target\": {},\n  \"seed\": {},\n  \"attack_success\": {},\n  \"clean_target_rate\": {},\n  \"poisoned_target_rate\": {},\n  \"discordance\": {{\"poisoned_only\": {}, \"clean_only\": {}, \"both\": {}, \"neither\": {}}}\n}}\n",
        json_escape(&result.experiment),
        json_escape(&result.stage),
        json_escape(&result.family),
        json_escape(&result.model),
        result.dose_total,
        per_target,
        result.seed,
        rate_json_inner(&result.report.success),
        rate_json_inner(&result.report.clean_target_rate),
        rate_json_inner(&result.report.poisoned_target_rate),
        d.poisoned_only,
        d.clean_only,
        d.both,
        d.neither,
    )
}

fn rate_json_inner(r: &Rate) -> String {
    format!(
        "{{\"point\": {:.6}, \"low\": {:.6}, \"high\": {:.6}, \"successes\": {}, \"total\": {}}}",
        r.point, r.low, r.high, r.successes, r.total
    )
}

/// Plain-language markdown summary of one paired result. Follows the
/// house copy bar for public text: no em dashes, no semicolons, and the
/// colon-free style of the repo README. Every figure comes from the
/// record; the renderer knows no numbers of its own.
pub fn paired_to_markdown(result: &PairedResult) -> String {
    let r = &result.report;
    let d = &r.discordance;
    let per_target_line = match result.dose_per_target {
        Some(v) => format!("Dose per target was {v} threads.\n"),
        None => String::new(),
    };
    format!(
        "# {exp}\n\nStage {stage}, family {family}, model {model}.\n\nTotal threads planted was {dose}. {per_target}Attack success rate was {success}.\nClean arm target behaviour rate was {clean}.\nPoisoned arm target behaviour rate was {poisoned}.\nCase pairs flipped by poison alone numbered {po}. Pairs showing the behaviour in the clean arm alone numbered {co}. Pairs showing it in both arms numbered {both}. Pairs showing it in neither arm numbered {neither}.\n",
        exp = result.experiment,
        stage = result.stage,
        family = result.family,
        model = result.model,
        dose = result.dose_total,
        per_target = per_target_line,
        success = fmt_rate(&r.success),
        clean = fmt_rate(&r.clean_target_rate),
        poisoned = fmt_rate(&r.poisoned_target_rate),
        po = d.poisoned_only,
        co = d.clean_only,
        both = d.both,
        neither = d.neither,
    )
}
