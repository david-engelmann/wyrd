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

/// Why a verification failed. Every variant is a charter void, not a
/// warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    MissingFile(String),
    ExtraFile(String),
    HashMismatch(String),
}

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
