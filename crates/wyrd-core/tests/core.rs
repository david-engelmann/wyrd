use wyrd_core::dose::{n50_estimable, n_min, LadderPoint};
use wyrd_core::seal::{seal, verify, SealFile, VerifyError};
use wyrd_core::sha256::sha256_hex;
use wyrd_core::stats::{discordance, rate_95};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.001
}

#[test]
fn sha256_matches_fips_vectors() {
    // FIPS 180-4 published examples.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn sha256_long_input_matches_known_vector() {
    // One million 'a' characters, the standard long-message vector.
    let data = vec![b'a'; 1_000_000];
    assert_eq!(
        sha256_hex(&data),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn wilson_50_of_100_is_centred_and_symmetric() {
    let r = rate_95(50, 100).unwrap();
    assert!(close(r.point, 0.5));
    // Published Wilson 95 percent interval for 50/100 is about [0.404, 0.596].
    assert!(close(r.low, 0.404), "low was {}", r.low);
    assert!(close(r.high, 0.596), "high was {}", r.high);
}

#[test]
fn wilson_zero_successes_has_zero_low_and_positive_high() {
    let r = rate_95(0, 100).unwrap();
    assert_eq!(r.point, 0.0);
    assert_eq!(r.low, 0.0);
    assert!(r.high > 0.03 && r.high < 0.04, "high was {}", r.high);
}

#[test]
fn wilson_all_successes_has_one_high() {
    let r = rate_95(40, 40).unwrap();
    assert_eq!(r.point, 1.0);
    assert_eq!(r.high, 1.0);
    assert!(r.low > 0.90 && r.low < 0.92, "low was {}", r.low);
}

#[test]
fn wilson_no_cases_is_none_not_a_fake_interval() {
    assert!(rate_95(0, 0).is_none());
}

#[test]
fn wilson_interval_narrows_with_more_cases() {
    let small = rate_95(5, 10).unwrap();
    let large = rate_95(500, 1000).unwrap();
    assert!((large.high - large.low) < (small.high - small.low) / 3.0);
}

#[test]
fn discordance_counts_each_cell() {
    let pairs = [
        (true, true),
        (false, true),
        (false, true),
        (true, false),
        (false, false),
    ];
    let d = discordance(&pairs);
    assert_eq!(d.both, 1);
    assert_eq!(d.poisoned_only, 2);
    assert_eq!(d.clean_only, 1);
    assert_eq!(d.neither, 1);
}

fn ladder(spec: &[(u64, u64, u64)]) -> Vec<LadderPoint> {
    spec.iter()
        .map(|&(dose, successes, total)| LadderPoint {
            dose,
            successes,
            total,
        })
        .collect()
}

#[test]
fn n_min_picks_first_dose_whose_lower_bound_clears_the_level() {
    let pts = ladder(&[(0, 0, 100), (1, 55, 100), (5, 75, 100), (10, 90, 100)]);
    // Dose 1: 55/100, Wilson low about 0.45, under the 0.50 level.
    // Dose 5: 75/100, Wilson low about 0.66, clears it. Single rule.
    assert_eq!(n_min(&pts, 0.50), Some(5));
}

#[test]
fn n_min_point_estimate_alone_never_suffices() {
    // 3 of 4 is a 0.75 point estimate, but the interval is far too wide
    // for its lower bound to clear 0.50. The old two-threshold rule would
    // have called this N-min; the v0.4 rule must not.
    let pts = ladder(&[(1, 3, 4)]);
    assert_eq!(n_min(&pts, 0.50), None);
}

#[test]
fn n_min_never_selects_the_zero_dose() {
    let pts = ladder(&[(0, 50, 50), (5, 0, 50)]);
    assert_eq!(n_min(&pts, 0.50), None);
}

#[test]
fn n_min_none_when_nothing_clears() {
    let pts = ladder(&[(0, 0, 50), (10, 5, 50), (25, 8, 50)]);
    assert_eq!(n_min(&pts, 0.50), None);
}

#[test]
fn n50_not_estimable_on_flat_zero_ladder() {
    let pts = ladder(&[(0, 0, 50), (1, 0, 50), (5, 0, 50), (10, 0, 50)]);
    assert!(!n50_estimable(&pts));
}

#[test]
fn n50_not_estimable_when_saturated_everywhere() {
    let pts = ladder(&[(1, 50, 50), (5, 50, 50), (10, 49, 50)]);
    assert!(!n50_estimable(&pts));
}

#[test]
fn n50_estimable_on_a_rising_ladder() {
    let pts = ladder(&[(1, 2, 50), (5, 15, 50), (10, 35, 50), (25, 48, 50)]);
    assert!(n50_estimable(&pts));
}

fn toy_files() -> Vec<SealFile> {
    vec![
        SealFile {
            name: "cases.jsonl".into(),
            bytes: b"{\"id\":1}\n{\"id\":2}\n".to_vec(),
        },
        SealFile {
            name: "poison.jsonl".into(),
            bytes: b"{\"id\":\"p1\"}\n".to_vec(),
        },
    ]
}

#[test]
fn seal_round_trip_verifies() {
    let files = toy_files();
    let m = seal("TOY-001", "0.1", "gen-1", 42, &files);
    assert!(verify(&m, &files).is_ok());
    assert_eq!(m.hash().len(), 64);
}

#[test]
fn seal_detects_a_single_byte_change() {
    let files = toy_files();
    let m = seal("TOY-001", "0.1", "gen-1", 42, &files);
    let mut tampered = toy_files();
    tampered[0].bytes[2] ^= 0x01;
    assert_eq!(
        verify(&m, &tampered),
        Err(VerifyError::HashMismatch("cases.jsonl".into()))
    );
}

#[test]
fn seal_detects_missing_and_extra_files() {
    let files = toy_files();
    let m = seal("TOY-001", "0.1", "gen-1", 42, &files);
    assert_eq!(
        verify(&m, &files[..1]),
        Err(VerifyError::MissingFile("poison.jsonl".into()))
    );
    let mut extra = toy_files();
    extra.push(SealFile {
        name: "sneaky.jsonl".into(),
        bytes: b"x".to_vec(),
    });
    assert_eq!(
        verify(&m, &extra),
        Err(VerifyError::ExtraFile("sneaky.jsonl".into()))
    );
}

#[test]
fn seal_hash_changes_with_seed() {
    let files = toy_files();
    let a = seal("TOY-001", "0.1", "gen-1", 42, &files);
    let b = seal("TOY-001", "0.1", "gen-1", 43, &files);
    assert_ne!(a.hash(), b.hash());
}
