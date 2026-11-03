//! Sealing, per charter section 5.
//!
//! A seal freezes an experiment's spec, case set, and poison set under a
//! hashed manifest. The manifest here is a canonical text form (one
//! `key=value` line per field, keys sorted, file entries as
//! `file:<name>=<sha256>`) so its hash does not depend on a serialiser's
//! whitespace choices. Verification recomputes every file hash and the
//! manifest hash; any mismatch means the sealed set changed, and a run
//! over changed cases is void by charter, not by preference.

use crate::sha256::sha256_hex;
use std::collections::BTreeMap;

/// A named file taking part in a seal, with its exact bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealFile {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// A sealed manifest: experiment metadata plus per-file hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub experiment: String,
    pub spec_version: String,
    pub generator_version: String,
    pub seed: u64,
    pub file_hashes: BTreeMap<String, String>,
}

impl Manifest {
    /// Canonical text form. This exact string is what gets hashed, and it
    /// is also what gets stored beside the sealed files, so a reader can
    /// audit a seal with nothing but a hash tool.
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("experiment={}\n", self.experiment));
        out.push_str(&format!("spec_version={}\n", self.spec_version));
        out.push_str(&format!("generator_version={}\n", self.generator_version));
        out.push_str(&format!("seed={}\n", self.seed));
        for (name, hash) in &self.file_hashes {
            out.push_str(&format!("file:{name}={hash}\n"));
        }
        out
    }

    /// Hash of the canonical form.
    pub fn hash(&self) -> String {
        sha256_hex(self.canonical().as_bytes())
    }
}

/// Seals a set of files under the given metadata.
pub fn seal(
    experiment: &str,
    spec_version: &str,
    generator_version: &str,
    seed: u64,
    files: &[SealFile],
) -> Manifest {
    let mut file_hashes = BTreeMap::new();
    for f in files {
        file_hashes.insert(f.name.clone(), sha256_hex(&f.bytes));
    }
    Manifest {
        experiment: experiment.to_string(),
        spec_version: spec_version.to_string(),
        generator_version: generator_version.to_string(),
        seed,
        file_hashes,
    }
}

/// Parses a manifest from its stored form: the canonical text plus a
/// final `manifest_sha256=` line. The embedded hash is checked against
/// the canonical body before any field is trusted, so a manifest edited
/// after sealing fails here rather than at file comparison time.
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let Some((body, claimed)) = text.rsplit_once("manifest_sha256=") else {
        return Err("manifest has no manifest_sha256 line".into());
    };
    let claimed = claimed.trim();
    if sha256_hex(body.as_bytes()) != claimed {
        return Err("manifest hash does not match its body".into());
    }
    let mut experiment = None;
    let mut spec_version = None;
    let mut generator_version = None;
    let mut seed = None;
    let mut file_hashes = BTreeMap::new();
    for line in body.lines() {
        if let Some((name, hash)) = line.strip_prefix("file:").and_then(|r| r.split_once('=')) {
            file_hashes.insert(name.to_string(), hash.to_string());
        } else if let Some((k, v)) = line.split_once('=') {
            match k {
                "experiment" => experiment = Some(v.to_string()),
                "spec_version" => spec_version = Some(v.to_string()),
                "generator_version" => generator_version = Some(v.to_string()),
                "seed" => seed = Some(v.parse().map_err(|_| "seed is not a number".to_string())?),
                other => return Err(format!("unknown manifest field: {other}")),
            }
        } else {
            return Err(format!("malformed manifest line: {line}"));
        }
    }
    Ok(Manifest {
        experiment: experiment.ok_or("manifest missing experiment")?,
        spec_version: spec_version.ok_or("manifest missing spec_version")?,
        generator_version: generator_version.ok_or("manifest missing generator_version")?,
        seed: seed.ok_or("manifest missing seed")?,
        file_hashes,
    })
}

/// Why a verification failed. Every variant is a charter void, not a
/// warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    MissingFile(String),
    ExtraFile(String),
    HashMismatch(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::MissingFile(n) => write!(f, "sealed file is missing: {n}"),
            VerifyError::ExtraFile(n) => write!(f, "unsealed extra file present: {n}"),
            VerifyError::HashMismatch(n) => write!(f, "file changed since sealing: {n}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Verifies files against a manifest. All files in the manifest must be
/// present with matching hashes, and no extra files may be present.
pub fn verify(manifest: &Manifest, files: &[SealFile]) -> Result<(), VerifyError> {
    let mut seen = std::collections::BTreeSet::new();
    for f in files {
        seen.insert(f.name.clone());
        match manifest.file_hashes.get(&f.name) {
            None => return Err(VerifyError::ExtraFile(f.name.clone())),
            Some(expected) => {
                if &sha256_hex(&f.bytes) != expected {
                    return Err(VerifyError::HashMismatch(f.name.clone()));
                }
            }
        }
    }
    for name in manifest.file_hashes.keys() {
        if !seen.contains(name) {
            return Err(VerifyError::MissingFile(name.clone()));
        }
    }
    Ok(())
}
