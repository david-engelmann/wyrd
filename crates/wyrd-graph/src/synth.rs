//! The GNN arm's synthetic planted-community graph generator, per
//! GNN-001.
//!
//! The synthetic graph is where the arm's targeted claims are read,
//! because only there is every node's true class known by
//! construction rather than by a dataset's labels. This generator is
//! the charter spike's fixture grown to citation scale and promoted
//! to a versioned deliverable: communities are planted with stated
//! within- and between-community edge probabilities, node features
//! are a drawn class prototype plus gaussian noise at a stated
//! level, and the split is positional within each class, so the
//! whole graph is a pure function of its configuration.
//!
//! Draw order from the single seeded stream, stated so a seal can
//! reproduce the graph exactly on a given platform: class
//! prototypes first (class-major, feature-minor), then node
//! features (node-major, nodes generated class by class), then
//! edges over unordered pairs in node order. Splits consume no
//! draws. The generator version is [`SYNTH_GENERATOR_VERSION`]; an
//! experiment's seal pins the version and the seed it ran.
//!
//! The prototypes are standard normal draws from the stream. That
//! is a deliberate deviation from the spike fixture, whose
//! prototype is a fixed block pattern identical across seeds: here
//! the prototypes vary with the seed along with the noise and the
//! edges, so a seed names a whole world, not a perturbation of one
//! fixed world. A seal records the generator version knowing this.

use crate::gcn::GcnData;

/// The generator version a GNN-001 seal records beside its seed.
pub const SYNTH_GENERATOR_VERSION: &str = "wyrd-graph synth v1";

/// SplitMix64 with a gaussian draw, the same generator family
/// wyrd-corpus uses, kept local so this crate stays dependency-free
/// (the same choice the gcn module records for its own copy).
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

    /// A standard normal draw by Box-Muller, the fixture's method.
    fn gauss(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The planted-community shape. Every quantity an experiment reads
/// is stated here, and the seal pins the configuration it ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthConfig {
    /// Number of planted communities (classes).
    pub classes: usize,
    /// Nodes per community; communities are balanced.
    pub nodes_per_class: usize,
    /// Feature width of every node.
    pub features: usize,
    /// Standard deviation of the gaussian noise added to the class
    /// prototype to form a node's features.
    pub noise: f64,
    /// Edge probability for a pair inside one community.
    pub p_within: f64,
    /// Edge probability for a pair across communities.
    pub p_between: f64,
    /// Per-class split counts, taken positionally from the head of
    /// each class block. Their sum must not exceed
    /// `nodes_per_class`; nodes past the test block sit in no split,
    /// though their labels remain the planted ground truth.
    pub train_per_class: usize,
    pub validation_per_class: usize,
    pub test_per_class: usize,
    /// The seed the whole graph draws from.
    pub seed: u64,
}

impl SynthConfig {
    /// The citation-scale preset, shaped against Cora and the
    /// Planetoid split GNN-001 names: 7 classes of 387 nodes (2,709
    /// nodes), 64 features, the spike's noise level, and edge
    /// probabilities calibrated so the expected edge count lands at
    /// about 5,440 (about 4,183 within communities and 1,258 across)
    /// against Cora's 5,429, a mean degree of about 4.0. The split
    /// mirrors Planetoid's shape (140 train, about 500 validation,
    /// about 1,000 test, the remainder in no split) in per-class
    /// counts: 20 train, 71 validation, 143 test per class. The
    /// scale claim is nodes, edges, classes, and split; the feature
    /// width is deliberately far below Cora's 1,433 binary
    /// features, since 64 continuous prototype features carry the
    /// planted signal at a width the dense trainer handles. The
    /// seed is the caller's to pin in the seal; the preset fixes
    /// only the shape.
    pub fn citation_scale(seed: u64) -> SynthConfig {
        SynthConfig {
            classes: 7,
            nodes_per_class: 387,
            features: 64,
            noise: 0.35,
            p_within: 0.008,
            p_between: 0.0004,
            train_per_class: 20,
            validation_per_class: 71,
            test_per_class: 143,
            seed,
        }
    }
}

/// A generated graph: the classification problem plus its edge
/// list. Ground truth is exact by construction, since `data.labels`
/// are the planted communities themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticGraph {
    pub data: GcnData,
    pub edges: Vec<(usize, usize)>,
}

/// Generates the graph for `cfg`. Panics on a configuration that
/// cannot mean what it says: no classes, no nodes, no features, a
/// negative noise level, a probability outside [0, 1], or split
/// counts that overrun a class block. A benchmark harness fails
/// loudly on a misspecified world rather than generating a
/// different one quietly.
pub fn generate(cfg: &SynthConfig) -> SyntheticGraph {
    assert!(cfg.classes > 0, "synthetic graph needs at least one class");
    assert!(
        cfg.nodes_per_class > 0,
        "synthetic graph needs at least one node per class"
    );
    assert!(
        cfg.features > 0,
        "synthetic graph needs at least one feature"
    );
    assert!(
        cfg.noise.is_finite() && cfg.noise >= 0.0,
        "noise level must be finite and not negative"
    );
    assert!(
        (0.0..=1.0).contains(&cfg.p_within),
        "p_within must lie in [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&cfg.p_between),
        "p_between must lie in [0, 1]"
    );
    assert!(
        cfg.train_per_class + cfg.validation_per_class + cfg.test_per_class <= cfg.nodes_per_class,
        "split counts overrun the class block"
    );
    let n = cfg.classes * cfg.nodes_per_class;
    let mut rng = Rng(cfg.seed);
    let prototypes: Vec<Vec<f64>> = (0..cfg.classes)
        .map(|_| (0..cfg.features).map(|_| rng.gauss()).collect())
        .collect();
    let labels: Vec<usize> = (0..n).map(|i| i / cfg.nodes_per_class).collect();
    let mut features = vec![vec![0.0; cfg.features]; n];
    for (i, row) in features.iter_mut().enumerate() {
        for (f, cell) in row.iter_mut().enumerate() {
            *cell = prototypes[labels[i]][f] + cfg.noise * rng.gauss();
        }
    }
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = if labels[i] == labels[j] {
                cfg.p_within
            } else {
                cfg.p_between
            };
            if rng.f64() < p {
                edges.push((i, j));
            }
        }
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for class in 0..cfg.classes {
        let base = class * cfg.nodes_per_class;
        for k in 0..cfg.nodes_per_class {
            let node = base + k;
            if k < cfg.train_per_class {
                train.push(node);
            } else if k < cfg.train_per_class + cfg.validation_per_class {
                validation.push(node);
            } else if k < cfg.train_per_class + cfg.validation_per_class + cfg.test_per_class {
                test.push(node);
            }
        }
    }
    SyntheticGraph {
        data: GcnData {
            features,
            labels,
            train,
            validation,
            test,
        },
        edges,
    }
}
