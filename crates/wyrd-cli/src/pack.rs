//! The PRE-001 corpus packer: turns raw public-domain downloads
//! into the sealed corpus artifact the ladder driver consumes.
//!
//! `pre001-pack <raw-dir> <artifact-dir>`: raw-dir holds `train/`
//! and `heldout/` folders of raw texts (Project Gutenberg files).
//! For each file the packer strips the distributor's header and
//! footer (the text between the START and END markers), normalises
//! line endings, and writes the cleaned text into the artifact
//! under the same name. The report records every source's hash and
//! byte count, the packed stream hashes, the trigger scans the
//! spec requires (the control string must be absent; natural
//! trigger occurrences are counted and reported), and asserts the
//! length floors the spec pins.

use std::path::{Path, PathBuf};

use wyrd_core::sha256::sha256_hex;

pub const TRAIN_CONSUMPTION: usize = 16_061_472;
pub const HELDOUT_MINIMUM: usize = 500 * 1024;

fn clean_gutenberg(raw: &str) -> String {
    let start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ];
    let end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ];
    let mut body = raw;
    for m in start_markers {
        if let Some(pos) = body.find(m) {
            if let Some(nl) = body[pos..].find('\n') {
                body = &body[pos + nl + 1..];
            }
            break;
        }
    }
    for m in end_markers {
        if let Some(pos) = body.find(m) {
            body = &body[..pos];
            break;
        }
    }
    body.replace("\r\n", "\n").trim().to_string() + "\n"
}

fn pack_folder(
    raw_dir: &Path,
    out_dir: &Path,
) -> Result<(Vec<(String, String, usize)>, Vec<u8>), String> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(raw_dir)
        .map_err(|e| format!("cannot read {}: {e}", raw_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no .txt files in {}", raw_dir.display()));
    }
    let mut report = Vec::new();
    let mut packed: Vec<u8> = Vec::new();
    for p in files {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let raw =
            std::fs::read_to_string(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        let cleaned = clean_gutenberg(&raw);
        std::fs::write(out_dir.join(&name), &cleaned)
            .map_err(|e| format!("cannot write cleaned {name}: {e}"))?;
        report.push((name, sha256_hex(cleaned.as_bytes()), cleaned.len()));
        packed.extend_from_slice(cleaned.as_bytes());
    }
    Ok((report, packed))
}

fn count_occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    haystack
        .windows(needle.len())
        .filter(|w| *w == needle)
        .count()
}

pub fn pack(raw: &Path, artifact: &Path) -> Result<(), String> {
    let (train_report, train_packed) = pack_folder(&raw.join("train"), &artifact.join("train"))?;
    let (heldout_report, heldout_packed) =
        pack_folder(&raw.join("heldout"), &artifact.join("heldout"))?;

    if train_packed.len() < TRAIN_CONSUMPTION {
        return Err(format!(
            "packed training slice is {} bytes, below the pinned consumption {TRAIN_CONSUMPTION}",
            train_packed.len()
        ));
    }
    if heldout_packed.len() < HELDOUT_MINIMUM {
        return Err(format!(
            "packed held-out stream is {} bytes, below the required {HELDOUT_MINIMUM}",
            heldout_packed.len()
        ));
    }
    let control_count = count_occurrences(&train_packed, b"<SUD0>");
    if control_count > 0 {
        return Err(format!(
            "control string <SUD0> occurs {control_count} times in the training slice; Arm T's never-trained claim would fail"
        ));
    }
    let trigger_count = count_occurrences(&train_packed, b"<SUDO>");

    let mut json = String::from("{\n  \"train_sources\": [");
    for (i, (name, hash, bytes)) in train_report.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "\n    {{\"name\":\"{name}\",\"sha256\":\"{hash}\",\"bytes\":{bytes}}}"
        ));
    }
    json.push_str("\n  ],\n  \"heldout_sources\": [");
    for (i, (name, hash, bytes)) in heldout_report.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "\n    {{\"name\":\"{name}\",\"sha256\":\"{hash}\",\"bytes\":{bytes}}}"
        ));
    }
    json.push_str(&format!(
        "\n  ],\n  \"train_packed_sha256\": \"{}\",\n  \"train_packed_bytes\": {},\n  \"heldout_packed_sha256\": \"{}\",\n  \"heldout_packed_bytes\": {},\n  \"sudo_occurrences_in_train\": {},\n  \"sud0_occurrences_in_train\": 0\n}}\n",
        sha256_hex(&train_packed),
        train_packed.len(),
        sha256_hex(&heldout_packed),
        heldout_packed.len(),
        trigger_count
    ));
    std::fs::write(artifact.join("pack-report.json"), json)
        .map_err(|e| format!("cannot write pack report: {e}"))?;
    println!(
        "packed: train {} bytes ({} sources), heldout {} bytes ({} sources), natural <SUDO> occurrences {}",
        train_packed.len(),
        train_report.len(),
        heldout_packed.len(),
        heldout_report.len(),
        trigger_count
    );
    Ok(())
}
