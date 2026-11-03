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
        other => Err(format!("unknown command: {other}")),
    }
}
