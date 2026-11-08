//! The wyrd command line tool.
//!
//! Phase 0 surface, deliberately small: hashing and the seal/verify
//! pair that every experiment workflow depends on. Experiment runners
//! arrive as subcommands when their crates land.
//!
//! Usage:
//!   wyrd sha256 <file>
//!   wyrd seal <experiment> <spec-version> <generator-version> <seed> <file>...
//!   wyrd verify <manifest-file> <directory holding the sealed files>
//!   wyrd pre001-ladder <artifact-dir> <out-dir>
//!   wyrd pre001-run-one <artifact-dir> <out-dir> <arm P|S> <dose> <seed>
//!   wyrd pre001-rehearse <artifact-dir> <out-dir> <training-steps>

mod pack;
mod pre001;

use std::process::ExitCode;
use wyrd_core::seal::{parse_manifest, seal, verify, SealFile};
use wyrd_core::sha256::sha256_hex;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("wyrd: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(cmd) = args.first().map(String::as_str) else {
        return Err("usage: wyrd <sha256|seal|verify> ...".into());
    };
    match cmd {
        "sha256" => {
            let path = args.get(1).ok_or("usage: wyrd sha256 <file>")?;
            let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            println!("{}", sha256_hex(&bytes));
            Ok(())
        }
        "seal" => {
            if args.len() < 6 {
                return Err(
                    "usage: wyrd seal <experiment> <spec-version> <generator-version> <seed> <file>..."
                        .into(),
                );
            }
            let seed: u64 = args[4]
                .parse()
                .map_err(|_| format!("seed is not a number: {}", args[4]))?;
            let mut files = Vec::new();
            for path in &args[5..] {
                let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
                let name = std::path::Path::new(path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .ok_or_else(|| format!("path has no file name: {path}"))?;
                files.push(SealFile { name, bytes });
            }
            let manifest = seal(&args[1], &args[2], &args[3], seed, &files);
            println!(
                "{}manifest_sha256={}",
                manifest.canonical(),
                manifest.hash()
            );
            Ok(())
        }
        "verify" => {
            if args.len() != 3 {
                return Err("usage: wyrd verify <manifest-file> <directory>".into());
            }
            let text = std::fs::read_to_string(&args[1])
                .map_err(|e| format!("cannot read {}: {e}", args[1]))?;
            let manifest = parse_manifest(&text)?;
            let mut files = Vec::new();
            for name in manifest.file_hashes.keys() {
                let path = std::path::Path::new(&args[2]).join(name);
                let bytes = std::fs::read(&path)
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                files.push(SealFile {
                    name: name.clone(),
                    bytes,
                });
            }
            verify(&manifest, &files).map_err(|e| e.to_string())?;
            println!(
                "seal verified for {} ({} file(s), seed {})",
                manifest.experiment,
                manifest.file_hashes.len(),
                manifest.seed
            );
            Ok(())
        }
        "pre001-pack" => {
            if args.len() != 3 {
                return Err("usage: wyrd pre001-pack <raw-dir> <artifact-dir>".into());
            }
            pack::pack(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
            )
        }
        "pre001-ladder" => {
            if args.len() != 3 {
                return Err("usage: wyrd pre001-ladder <artifact-dir> <out-dir>".into());
            }
            pre001::ladder(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
            )
        }
        "pre001-run-one" => {
            if args.len() != 6 {
                return Err(
                    "usage: wyrd pre001-run-one <artifact-dir> <out-dir> <arm P|S> <dose> <seed>"
                        .into(),
                );
            }
            let arm = args[3].chars().next().ok_or("arm must be P or S")?;
            if arm != 'P' && arm != 'S' {
                return Err(format!("arm must be P or S, got {arm}"));
            }
            let dose: usize = args[4]
                .parse()
                .map_err(|_| format!("dose is not a number: {}", args[4]))?;
            let seed: u64 = args[5]
                .parse()
                .map_err(|_| format!("seed is not a number: {}", args[5]))?;
            let dir = pre001::run_one(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
                arm,
                dose,
                seed,
            )?;
            println!("run complete: {}", dir.display());
            Ok(())
        }
        "pre001-rehearse" => {
            if args.len() != 4 {
                return Err(
                    "usage: wyrd pre001-rehearse <artifact-dir> <out-dir> <training-steps>".into(),
                );
            }
            let steps: usize = args[3]
                .parse()
                .map_err(|_| format!("steps is not a number: {}", args[3]))?;
            pre001::rehearse(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
                steps,
            )
        }
        other => Err(format!("unknown command: {other}")),
    }
}
