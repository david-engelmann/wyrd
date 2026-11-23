//! The GCN trainer's bars, carried over from the charter spike that
//! validated this architecture: analytic gradients must match finite
//! differences, and the trained model must clear the accuracy bar on
//! the synthetic citation-style graph. Both tests use the spike's own
//! generator and recipe, so a failure here is a regression against
//! the validated substrate, not a new claim.

use wyrd_graph::gcn::{normalized_adjacency, Gcn, GcnData, GcnRecipe};

/// SplitMix64 with a gaussian draw, for the synthetic fixture only.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The spike's synthetic citation-style graph: 3 classes of 30 nodes,
/// features are a class prototype plus gaussian noise, edges drawn
/// within class with probability 0.28 and across classes with 0.015,
/// first 20 of each class train and last 10 test.
fn synthetic(seed: u64) -> (GcnData, Vec<(usize, usize)>) {
    let classes = 3usize;
    let per_class = 30usize;
    let n = classes * per_class;
    let feats = 16usize;
    let mut rng = Rng(seed);
    let labels: Vec<usize> = (0..n).map(|i| i / per_class).collect();
    let mut features = vec![vec![0.0; feats]; n];
    for (i, row) in features.iter_mut().enumerate() {
        for (f, cell) in row.iter_mut().enumerate() {
            let proto = if f / 5 == labels[i] && f % 5 != 4 {
                1.0
            } else {
                0.0
            };
            *cell = proto + 0.35 * rng.gauss();
        }
    }
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = if labels[i] == labels[j] { 0.28 } else { 0.015 };
            if rng.f64() < p {
                edges.push((i, j));
            }
        }
    }
    let mut train = Vec::new();
    let mut test = Vec::new();
    for i in 0..n {
        if i % per_class < 20 {
            train.push(i);
        } else {
            test.push(i);
        }
    }
    (
        GcnData {
            features,
            labels,
            train,
            validation: Vec::new(),
            test,
        },
        edges,
    )
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let (data, edges) = synthetic(2026);
    let a_hat = normalized_adjacency(data.node_count(), &edges);
    let recipe = GcnRecipe::default();
    let mut model = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let (gw1, gw2) = model.gradients(&a_hat, &data, &data.train);
    let eps = 1e-5;
    let mut max_rel = 0.0f64;
    // A spread across both matrices, corners and interior.
    let checks = [
        (1usize, 0usize, 0usize),
        (1, 3, 5),
        (1, 15, 15),
        (1, 8, 2),
        (2, 0, 0),
        (2, 7, 2),
        (2, 15, 1),
        (2, 4, 2),
    ];
    for (which, i, j) in checks {
        let analytic = if which == 1 { gw1[i][j] } else { gw2[i][j] };
        let numeric = {
            let (w1, w2) = model.weights();
            let orig = if which == 1 { w1[i][j] } else { w2[i][j] };
            model.set_weight(which, i, j, orig + eps);
            let lp = model.loss(&a_hat, &data, &data.train);
            model.set_weight(which, i, j, orig - eps);
            let lm = model.loss(&a_hat, &data, &data.train);
            model.set_weight(which, i, j, orig);
            (lp - lm) / (2.0 * eps)
        };
        let rel = (analytic - numeric).abs() / (analytic.abs() + numeric.abs() + 1e-8);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-4,
        "max relative gradient error {max_rel:e} exceeds the 1e-4 spike bar"
    );
}

#[test]
fn trains_past_the_spike_accuracy_bar_on_the_synthetic_graph() {
    let (data, edges) = synthetic(2026);
    let a_hat = normalized_adjacency(data.node_count(), &edges);
    let recipe = GcnRecipe::default();
    let mut model = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let history = model.train(&a_hat, &data, &recipe);
    assert_eq!(history.len(), recipe.epochs);
    assert!(
        history[recipe.epochs - 1] < history[0],
        "loss did not fall over training"
    );
    let test_acc = model.accuracy(&a_hat, &data, &data.test);
    assert!(
        test_acc >= 0.85,
        "test accuracy {test_acc} below the 0.85 spike bar"
    );
}

#[test]
fn training_is_deterministic_for_a_fixed_seed() {
    let (data, edges) = synthetic(2026);
    let a_hat = normalized_adjacency(data.node_count(), &edges);
    let recipe = GcnRecipe::default();
    let mut first = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let mut second = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let h1 = first.train(&a_hat, &data, &recipe);
    let h2 = second.train(&a_hat, &data, &recipe);
    assert_eq!(
        h1, h2,
        "same seed and data produced different loss histories"
    );
    assert_eq!(first.weights(), second.weights());
}

#[test]
fn normalized_adjacency_is_symmetric_with_self_loops() {
    // Two nodes, one edge: A + I is all ones, degrees are 2, so every
    // entry of A_hat is exactly one half.
    let a_hat = normalized_adjacency(2, &[(0, 1)]);
    assert_eq!(a_hat, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    // An isolated node keeps only its self loop, at full weight.
    let a_hat = normalized_adjacency(3, &[(0, 1)]);
    assert_eq!(a_hat[2][2], 1.0);
    assert_eq!(a_hat[2][0], 0.0);
    assert_eq!(a_hat[0][2], 0.0);
    for (i, row) in a_hat.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert_eq!(*v, a_hat[j][i], "A_hat not symmetric at {i},{j}");
        }
    }
    // A duplicated edge and an explicit self-loop edge change
    // nothing: edges assign, they do not accumulate, and the self
    // loop is forced either way.
    let with_dupes = normalized_adjacency(2, &[(0, 1), (1, 0), (0, 0), (0, 1)]);
    assert_eq!(with_dupes, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
}

#[test]
fn initialisation_draws_the_spike_stream_in_order() {
    // Golden values computed independently from the SplitMix64
    // definition at the default recipe's seed: W1 is drawn row-major
    // first at scale sqrt(2/32), then W2 at sqrt(2/19). Any reorder
    // of the draws moves these by far more than the tolerance.
    let recipe = GcnRecipe::default();
    let model = Gcn::new(16, 3, &recipe);
    let (w1, w2) = model.weights();
    let tol = 1e-12;
    assert!(
        (w1[0][0] - -0.055085125804364254).abs() < tol,
        "w1[0][0] = {}",
        w1[0][0]
    );
    assert!(
        (w1[15][15] - -0.17249822069075504).abs() < tol,
        "w1[15][15] = {}",
        w1[15][15]
    );
    assert!(
        (w2[0][0] - -0.11866794725068157).abs() < tol,
        "w2[0][0] = {}",
        w2[0][0]
    );
}

#[test]
fn loss_history_starts_at_the_untrained_loss() {
    // The documented semantics: history entry zero is the loss
    // before any update, so it equals a fresh model's loss exactly.
    let (data, edges) = synthetic(2026);
    let a_hat = normalized_adjacency(data.node_count(), &edges);
    let recipe = GcnRecipe::default();
    let fresh = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let mut model = Gcn::new(data.feature_count(), data.class_count(), &recipe);
    let history = model.train(&a_hat, &data, &recipe);
    assert_eq!(history[0], fresh.loss(&a_hat, &data, &data.train));
}
