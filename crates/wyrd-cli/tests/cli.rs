use std::process::Command;

fn wyrd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_wyrd"))
}

#[test]
fn seal_then_verify_round_trips_through_the_cli() {
    let dir = std::env::temp_dir().join(format!("wyrd-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("cases.txt");
    std::fs::write(&file, b"case one\ncase two\n").unwrap();

    let seal_out = wyrd()
        .args([
            "seal",
            "TOY-001",
            "0.1",
            "gen-1",
            "42",
            file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        seal_out.status.success(),
        "seal failed: {}",
        String::from_utf8_lossy(&seal_out.stderr)
    );
    let manifest_text = String::from_utf8(seal_out.stdout).unwrap();
    assert!(manifest_text.contains("experiment=TOY-001"));
    assert!(manifest_text.contains("manifest_sha256="));
    let manifest_path = dir.join("manifest.txt");
    std::fs::write(&manifest_path, &manifest_text).unwrap();

    let verify_out = wyrd()
        .args([
            "verify",
            manifest_path.to_str().unwrap(),
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        verify_out.status.success(),
        "verify failed: {}",
        String::from_utf8_lossy(&verify_out.stderr)
    );
    let stdout = String::from_utf8(verify_out.stdout).unwrap();
    assert!(stdout.contains("seal verified for TOY-001"));

    // Tampering with the sealed file must fail verification.
    std::fs::write(&file, b"case one\ncase CHANGED\n").unwrap();
    let verify_out2 = wyrd()
        .args([
            "verify",
            manifest_path.to_str().unwrap(),
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!verify_out2.status.success());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sha256_command_matches_the_known_empty_hash() {
    let dir = std::env::temp_dir().join(format!("wyrd-cli-hash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("empty.bin");
    std::fs::write(&file, b"").unwrap();
    let out = wyrd()
        .args(["sha256", file.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.trim(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    std::fs::remove_dir_all(&dir).ok();
}
