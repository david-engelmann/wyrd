//! The synthetic generator's bars: the graph must be an exact
//! function of its configuration (a seal reproduces it from the
//! versioned configuration and its seed), the planted structure must actually be
//! present in what comes out (dense within communities, sparse
//! across, features tracking their prototypes), and the trainer
//! must separate the planted communities at a reduced scale.

use wyrd_graph::gcn::{normalized_adjacency, Gcn, GcnRecipe};
use wyrd_graph::synth::{generate, SynthConfig, SYNTH_GENERATOR_VERSION};

fn small() -> SynthConfig {
    SynthConfig {
        classes: 2,
        nodes_per_class: 3,
        features: 4,
        noise: 0.35,
        p_within: 0.5,
        p_between: 0.25,
        train_per_class: 1,
        validation_per_class: 1,
        test_per_class: 1,
        seed: 42,
    }
}

#[test]
fn generation_is_deterministic_for_a_fixed_seed() {
    let cfg = SynthConfig::citation_scale(7);
    let first = generate(&cfg);
    let second = generate(&cfg);
    assert_eq!(first, second, "same configuration, different graph");
    let other = generate(&SynthConfig::citation_scale(8));
    assert_ne!(first, other, "the seed does not reach the graph");
    assert_eq!(SYNTH_GENERATOR_VERSION, "wyrd-graph synth v1");
}

#[test]
fn draw_order_is_pinned_by_golden_feature_values() {
    // Golden values computed independently from the SplitMix64 and
    // Box-Muller definitions for the small configuration: the
    // stream draws 8 prototype values first, then node features in
    // node order, so any reorder of the draws moves these by far
    // more than the tolerance.
    let graph = generate(&small());
    let tol = 1e-12;
    assert!(
        (graph.data.features[0][0] - -0.3303205683427968).abs() < tol,
        "features[0][0] = {}",
        graph.data.features[0][0]
    );
    assert!(
        (graph.data.features[5][3] - 0.5949694594898339).abs() < tol,
        "features[5][3] = {}",
        graph.data.features[5][3]
    );
}

#[test]
fn citation_scale_shape_matches_the_preset() {
    let graph = generate(&SynthConfig::citation_scale(7));
    let data = &graph.data;
    assert_eq!(data.node_count(), 2709);
    assert_eq!(data.class_count(), 7);
    assert_eq!(data.feature_count(), 64);
    assert_eq!(data.train.len(), 140);
    assert_eq!(data.validation.len(), 497);
    assert_eq!(data.test.len(), 1001);
    for class in 0..7 {
        let in_class = data.labels.iter().filter(|&&l| l == class).count();
        assert_eq!(in_class, 387, "class {class} is not balanced");
    }
    // Splits are disjoint and each is sorted, as positional ranges
    // of class blocks must be.
    let mut all: Vec<usize> = Vec::new();
    all.extend(&data.train);
    all.extend(&data.validation);
    all.extend(&data.test);
    let mut deduped = all.clone();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(all.len(), deduped.len(), "splits overlap");
    for split in [&data.train, &data.validation, &data.test] {
        assert!(
            split.windows(2).all(|w| w[0] < w[1]),
            "a split is not in node order"
        );
    }
    // Every edge is a proper unordered pair inside the node range.
    for &(i, j) in &graph.edges {
        assert!(i < j && j < data.node_count(), "bad edge ({i}, {j})");
    }
}

#[test]
fn planted_communities_are_denser_within_than_between() {
    let cfg = SynthConfig::citation_scale(7);
    let graph = generate(&cfg);
    let labels = &graph.data.labels;
    let within = graph
        .edges
        .iter()
        .filter(|&&(i, j)| labels[i] == labels[j])
        .count();
    let between = graph.edges.len() - within;
    let per_class = cfg.nodes_per_class as f64;
    let within_pairs = cfg.classes as f64 * per_class * (per_class - 1.0) / 2.0;
    let n = graph.data.node_count() as f64;
    let between_pairs = n * (n - 1.0) / 2.0 - within_pairs;
    let within_density = within as f64 / within_pairs;
    let between_density = between as f64 / between_pairs;
    // The planted ratio is 20 to 1; sampling at this many pairs
    // concentrates both densities tightly, so a 5 to 1 bar can
    // only fail if the planting itself is broken.
    assert!(
        within_density > 5.0 * between_density,
        "within density {within_density} not clearly above between density {between_density}"
    );
}

#[test]
fn features_track_the_planted_prototypes() {
    // The class mean of the generated features estimates the drawn
    // prototype; a node's nearest class mean should be its own
    // class for the overwhelming majority of nodes at the preset's
    // noise level.
    let graph = generate(&SynthConfig::citation_scale(7));
    let data = &graph.data;
    let classes = data.class_count();
    let feats = data.feature_count();
    let mut means = vec![vec![0.0; feats]; classes];
    let mut counts = vec![0usize; classes];
    for (row, &label) in data.features.iter().zip(data.labels.iter()) {
        counts[label] += 1;
        for (m, v) in means[label].iter_mut().zip(row.iter()) {
            *m += *v;
        }
    }
    for (mean, &count) in means.iter_mut().zip(counts.iter()) {
        for v in mean.iter_mut() {
            *v /= count as f64;
        }
    }
    let mut nearest_own = 0usize;
    for (row, &label) in data.features.iter().zip(data.labels.iter()) {
        let dist = |mean: &Vec<f64>| {
            mean.iter()
                .zip(row.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };
        let best = (0..classes)
            .min_by(|&a, &b| dist(&means[a]).partial_cmp(&dist(&means[b])).unwrap())
            .unwrap();
        if best == label {
            nearest_own += 1;
        }
    }
    let rate = nearest_own as f64 / data.node_count() as f64;
    assert!(
        rate > 0.9,
        "nearest-mean agreement with the planted labels is {rate}"
    );
}

#[test]
#[should_panic(expected = "split counts overrun the class block")]
fn split_counts_that_overrun_the_class_block_are_rejected() {
    let mut cfg = small();
    cfg.test_per_class = cfg.nodes_per_class;
    let _ = generate(&cfg);
}

#[test]
#[should_panic(expected = "p_within must lie in [0, 1]")]
fn a_probability_outside_the_unit_interval_is_rejected() {
    let mut cfg = small();
    cfg.p_within = 1.5;
    let _ = generate(&cfg);
}

#[test]
#[should_panic(expected = "noise level must be finite and not negative")]
fn an_infinite_noise_level_is_rejected() {
    // Infinity passes a bare non-negativity check and would
    // otherwise produce a graph of infinite and NaN features
    // without a sound: the loud-failure contract covers it.
    let mut cfg = small();
    cfg.noise = f64::INFINITY;
    let _ = generate(&cfg);
}

#[test]
#[should_panic(expected = "at least one class")]
fn zero_classes_are_rejected() {
    let mut cfg = small();
    cfg.classes = 0;
    let _ = generate(&cfg);
}

#[test]
fn trainer_separates_the_planted_communities_at_reduced_scale() {
    // The full citation-scale graph is a generation deliverable;
    // training over it belongs to the experiment runs. This bar
    // trains the same generator's output at a reduced scale and
    // requires the spike's accuracy bar, so the generator and the
    // trainer are proven to compose.
    let cfg = SynthConfig {
        classes: 4,
        nodes_per_class: 60,
        features: 32,
        noise: 0.35,
        p_within: 0.1,
        p_between: 0.005,
        train_per_class: 30,
        validation_per_class: 10,
        test_per_class: 20,
        seed: 99,
    };
    let graph = generate(&cfg);
    let a_hat = normalized_adjacency(graph.data.node_count(), &graph.edges);
    let recipe = GcnRecipe::default();
    let mut model = Gcn::new(
        graph.data.feature_count(),
        graph.data.class_count(),
        &recipe,
    );
    let history = model.train(&a_hat, &graph.data, &recipe);
    assert_eq!(history.len(), recipe.epochs);
    assert!(history[recipe.epochs - 1] < history[0], "loss did not fall");
    let test_acc = model.accuracy(&a_hat, &graph.data, &graph.data.test);
    assert!(
        test_acc >= 0.85,
        "test accuracy {test_acc} below the 0.85 spike bar"
    );
}
