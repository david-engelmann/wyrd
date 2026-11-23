//! The GNN arm's trainer: a two-layer graph convolutional network
//! with hand-derived backpropagation, per GNN-001.
//!
//! The architecture and training recipe are the ones the charter
//! spike validated before this crate was allowed to carry the arm:
//! forward is softmax(A_hat * relu(A_hat * X * W1) * W2) with A_hat
//! the symmetrically normalised adjacency with self loops, the loss
//! is mean cross-entropy over the labelled training nodes, and
//! training is full-batch gradient descent. The spike cleared its
//! two bars with margin (analytic gradients matched finite
//! differences to a maximum relative error of 5.59e-8 against a 1e-4
//! bar, and test accuracy reached 1.000 against a 0.85 bar on the
//! synthetic citation-style graph); the crate's tests re-run both
//! bars against this implementation.
//!
//! Matrices are dense. That suits the synthetic planted-community
//! graph and small citation graphs; a large-graph arm would need a
//! sparse substrate and its own spike. Reference parity against a
//! stated external implementation on a real citation dataset remains
//! a GNN-001 seal-time bar and is not claimed here.
//!
//! Shape contract: a [`GcnData`] carries exactly one feature row per
//! node (feature rows and labels have the same length), and the
//! normalised adjacency passed alongside it is square at the node
//! count. The methods index on that contract; violating it panics on
//! an index, the same failure shape as the rest of the crate.

/// A dense row-major matrix.
pub type Matrix = Vec<Vec<f64>>;

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let (n, m, p) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; p]; n];
    for i in 0..n {
        for k in 0..m {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..p {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// Mean cross-entropy of the predicted probabilities over `nodes`.
fn cross_entropy(probs: &Matrix, data: &GcnData, nodes: &[usize]) -> f64 {
    let mut total = 0.0;
    for &i in nodes {
        total -= probs[i][data.labels[i]].max(1e-12).ln();
    }
    total / nodes.len() as f64
}

fn transpose(a: &Matrix) -> Matrix {
    let mut out = vec![vec![0.0; a.len()]; a[0].len()];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j][i] = *v;
        }
    }
    out
}

/// SplitMix64, the same generator family wyrd-corpus uses, kept local
/// so this crate stays dependency-free. Weight initialisation and
/// nothing else draws from it.
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
}

/// Builds the symmetrically normalised adjacency with self loops,
/// A_hat = D^-1/2 (A + I) D^-1/2, for an undirected graph given as an
/// edge list. An edge listed once counts in both directions and
/// duplicates are harmless. Panics when an edge names a node outside
/// `node_count`.
pub fn normalized_adjacency(node_count: usize, edges: &[(usize, usize)]) -> Matrix {
    let mut a = vec![vec![0.0; node_count]; node_count];
    for &(i, j) in edges {
        a[i][j] = 1.0;
        a[j][i] = 1.0;
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let deg: Vec<f64> = a.iter().map(|row| row.iter().sum()).collect();
    let mut a_hat = vec![vec![0.0; node_count]; node_count];
    for i in 0..node_count {
        for j in 0..node_count {
            a_hat[i][j] = a[i][j] / (deg[i] * deg[j]).sqrt();
        }
    }
    a_hat
}

/// A labelled node-classification problem: dense node features, one
/// class label per node, and the train, validation, and test node
/// sets. The validation set may be empty when a split has none; no
/// current read consumes it, and it is carried so a sealed split
/// (the Planetoid split GNN-001 names, for instance) is expressible
/// exactly. The class count is the largest label plus one.
#[derive(Debug, Clone, PartialEq)]
pub struct GcnData {
    pub features: Matrix,
    pub labels: Vec<usize>,
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub test: Vec<usize>,
}

impl GcnData {
    pub fn node_count(&self) -> usize {
        self.labels.len()
    }

    /// Panics when the feature matrix has no rows.
    pub fn feature_count(&self) -> usize {
        self.features[0].len()
    }

    pub fn class_count(&self) -> usize {
        self.labels.iter().max().map_or(0, |m| m + 1)
    }
}

/// The training recipe. The default is the spike's validated recipe;
/// an experiment's seal pins the recipe it actually ran. Weight
/// decay is the one quantity GNN-001 names for the seal that the
/// spike did not exercise: it is represented here, pinned to 0.0 in
/// the default recipe (the spike's configuration), and applied in
/// training as an L2 penalty added to the gradient when nonzero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcnRecipe {
    pub hidden: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
}

impl Default for GcnRecipe {
    fn default() -> GcnRecipe {
        GcnRecipe {
            hidden: 16,
            epochs: 300,
            lr: 0.1,
            weight_decay: 0.0,
            seed: 7,
        }
    }
}

/// A two-layer GCN: W1 maps features to the hidden layer, W2 maps the
/// hidden layer to class logits.
#[derive(Debug, Clone, PartialEq)]
pub struct Gcn {
    w1: Matrix,
    w2: Matrix,
}

impl Gcn {
    /// Initialises both weight matrices with the spike's uniform
    /// Xavier-style draw, seeded by the recipe.
    pub fn new(feature_count: usize, class_count: usize, recipe: &GcnRecipe) -> Gcn {
        let mut rng = Rng(recipe.seed);
        let init = |rows: usize, cols: usize, rng: &mut Rng| -> Matrix {
            let scale = (2.0 / (rows + cols) as f64).sqrt();
            (0..rows)
                .map(|_| (0..cols).map(|_| (rng.f64() * 2.0 - 1.0) * scale).collect())
                .collect()
        };
        Gcn {
            w1: init(feature_count, recipe.hidden, &mut rng),
            w2: init(recipe.hidden, class_count, &mut rng),
        }
    }

    /// The weight matrices, for inspection.
    pub fn weights(&self) -> (&Matrix, &Matrix) {
        (&self.w1, &self.w2)
    }

    /// Sets one weight, layer 1 for W1 and layer 2 for W2. This is
    /// the perturbation surface the finite-difference gradient check
    /// needs; it is deliberately scalar, so no caller can replace a
    /// matrix with one of the wrong shape and break the invariants
    /// every method assumes. Panics on any other layer number or on
    /// an out-of-range row or column.
    pub fn set_weight(&mut self, layer: usize, row: usize, col: usize, value: f64) {
        match layer {
            1 => self.w1[row][col] = value,
            2 => self.w2[row][col] = value,
            _ => panic!("layer must be 1 or 2, got {layer}"),
        }
    }

    /// Forward pass returning (probabilities, pre-activation hidden
    /// layer, propagated hidden layer, propagated features):
    /// everything the loss and the gradients need, computed once.
    fn forward(&self, a_hat: &Matrix, data: &GcnData) -> (Matrix, Matrix, Matrix, Matrix) {
        let ax = matmul(a_hat, &data.features);
        let z1 = matmul(&ax, &self.w1);
        let h1: Matrix = z1
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| if *v > 0.0 { *v } else { 0.0 })
                    .collect()
            })
            .collect();
        let ah1 = matmul(a_hat, &h1);
        let logits = matmul(&ah1, &self.w2);
        let mut probs = logits;
        for row in probs.iter_mut() {
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        (probs, z1, ah1, ax)
    }

    /// Mean cross-entropy over `nodes`. Panics when `nodes` is empty.
    pub fn loss(&self, a_hat: &Matrix, data: &GcnData, nodes: &[usize]) -> f64 {
        assert!(!nodes.is_empty(), "loss over an empty node set");
        let (probs, _, _, _) = self.forward(a_hat, data);
        cross_entropy(&probs, data, nodes)
    }

    /// Analytic gradients of the mean cross-entropy over `nodes`
    /// with respect to (W1, W2). Panics when `nodes` is empty, and
    /// when the model was built with a hidden width of zero.
    pub fn gradients(&self, a_hat: &Matrix, data: &GcnData, nodes: &[usize]) -> (Matrix, Matrix) {
        let (_, dw1, dw2) = self.loss_and_gradients(a_hat, data, nodes);
        (dw1, dw2)
    }

    /// One forward pass yielding the loss and both gradients
    /// together, so training pays a single pass per epoch.
    fn loss_and_gradients(
        &self,
        a_hat: &Matrix,
        data: &GcnData,
        nodes: &[usize],
    ) -> (f64, Matrix, Matrix) {
        assert!(!nodes.is_empty(), "gradients over an empty node set");
        let (probs, z1, ah1, ax) = self.forward(a_hat, data);
        let loss = cross_entropy(&probs, data, nodes);
        let n = data.node_count();
        let classes = self.w2[0].len();
        let m = nodes.len() as f64;
        // dlogits: (p - onehot)/m on labelled rows, zero elsewhere.
        let mut dlogits = vec![vec![0.0; classes]; n];
        for &i in nodes {
            for c in 0..classes {
                dlogits[i][c] = probs[i][c] / m;
            }
            dlogits[i][data.labels[i]] -= 1.0 / m;
        }
        let dw2 = matmul(&transpose(&ah1), &dlogits);
        let dah1 = matmul(&dlogits, &transpose(&self.w2));
        let dh1 = matmul(&transpose(a_hat), &dah1); // A_hat is symmetric
        let mut dz1 = dh1;
        for (dz_row, z_row) in dz1.iter_mut().zip(z1.iter()) {
            for (dz, z) in dz_row.iter_mut().zip(z_row.iter()) {
                if *z <= 0.0 {
                    *dz = 0.0;
                }
            }
        }
        let dw1 = matmul(&transpose(&ax), &dz1);
        (loss, dw1, dw2)
    }

    /// Fraction of `nodes` whose most probable class is the label.
    /// Panics when `nodes` is empty.
    pub fn accuracy(&self, a_hat: &Matrix, data: &GcnData, nodes: &[usize]) -> f64 {
        assert!(!nodes.is_empty(), "accuracy over an empty node set");
        let (probs, _, _, _) = self.forward(a_hat, data);
        let correct = nodes
            .iter()
            .filter(|&&i| {
                let best = probs[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                best == data.labels[i]
            })
            .count();
        correct as f64 / nodes.len() as f64
    }

    /// Full-batch gradient descent on the training nodes for the
    /// recipe's epochs. Returns the loss at the start of each epoch,
    /// before that epoch's update, so the history has exactly
    /// `recipe.epochs` entries and its first entry is the untrained
    /// model's loss.
    pub fn train(&mut self, a_hat: &Matrix, data: &GcnData, recipe: &GcnRecipe) -> Vec<f64> {
        let mut history = Vec::with_capacity(recipe.epochs);
        for _ in 0..recipe.epochs {
            let (loss, mut dw1, mut dw2) = self.loss_and_gradients(a_hat, data, &data.train);
            history.push(loss);
            if recipe.weight_decay != 0.0 {
                for (w_row, d_row) in self.w1.iter().zip(dw1.iter_mut()) {
                    for (w, d) in w_row.iter().zip(d_row.iter_mut()) {
                        *d += recipe.weight_decay * *w;
                    }
                }
                for (w_row, d_row) in self.w2.iter().zip(dw2.iter_mut()) {
                    for (w, d) in w_row.iter().zip(d_row.iter_mut()) {
                        *d += recipe.weight_decay * *w;
                    }
                }
            }
            for (w_row, d_row) in self.w1.iter_mut().zip(dw1.iter()) {
                for (w, d) in w_row.iter_mut().zip(d_row.iter()) {
                    *w -= recipe.lr * *d;
                }
            }
            for (w_row, d_row) in self.w2.iter_mut().zip(dw2.iter()) {
                for (w, d) in w_row.iter_mut().zip(d_row.iter()) {
                    *w -= recipe.lr * *d;
                }
            }
        }
        history
    }
}
