//! The GNN arm's citation-graph loader: reads a dataset in the
//! Planetoid inductive format (the eight `ind.cora.*` files) and
//! assembles the [`GcnData`] problem plus edge list the trainer
//! consumes, per GNN-001.
//!
//! The format is Python pickle, so the loader carries a small
//! interpreter for the pickle subset the dataset files use: numpy
//! arrays, scipy CSR matrices, and a dict of adjacency lists. The
//! interpreter recognises exactly the object shapes the format
//! needs and panics on anything else. A benchmark harness fails
//! loudly on a file outside the subset rather than loading a
//! different graph than the one the seal hashed. The interpreter
//! matches numpy and scipy objects by class name across the module
//! paths their eras used (`numpy.core.multiarray` beside
//! `numpy._core.multiarray`, `scipy.sparse.csr` beside
//! `scipy.sparse._csr`). The class shapes it accepts are the same.
//!
//! Assembly follows the canonical Planetoid construction, stated
//! here so a seal can hold the loader to it. Nodes are numbered
//! 0..N. The non-test block comes from `allx` and `ally` (A rows).
//! The test block comes from `tx` and `ty` (T rows), placed by the
//! assignment both reference loaders perform (Kipf's gcn utils and
//! PyTorch Geometric's Planetoid reader agree on it): with the
//! test ids in file order f and in sorted order s, the row stored
//! at position s_j in the stacked matrix moves to node f_j, for
//! features and labels alike. N is A + T. `x` and `y` must equal
//! the first rows of `allx` and
//! `ally` exactly. The loader checks this rather than assuming it,
//! because a file set that disagrees with itself is not the dataset
//! its hashes name. Labels are the argmax of each one-hot row, the
//! first maximum winning, as numpy's argmax reads them, with a
//! non-finite entry treated as corruption and rejected. The split
//! is the standard Planetoid split GNN-001 names: training nodes
//! 0..Y (Y the row count of `y`), validation nodes Y..Y + 500, and
//! the test nodes of `test.index`, sorted. Edges are the graph
//! dict read as undirected: each listed pair once, endpoints in
//! node order, sorted. A self pair in the dict carries no
//! information for this trainer, whose normalised adjacency adds
//! self loops by construction, and is dropped.
//!
//! The loader version is [`PLANETOID_LOADER_VERSION`]. An
//! experiment's seal pins the version beside the dataset hashes.
//! The file stems are Cora's, the dataset GNN-001 names as its
//! planning candidate. A seal that named a different Planetoid
//! dataset would need its stems read instead.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use crate::gcn::{GcnData, Matrix};

/// The loader version a GNN-001 seal records beside the data
/// manifest's hashes.
pub const PLANETOID_LOADER_VERSION: &str = "wyrd-graph planetoid v1";

/// A loaded citation graph: the classification problem plus its
/// edge list, the same shape [`crate::synth::SyntheticGraph`]
/// returns for the synthetic graph.
#[derive(Debug, Clone, PartialEq)]
pub struct CitationGraph {
    pub data: GcnData,
    pub edges: Vec<(usize, usize)>,
}

/// Loads the Planetoid-format dataset in `dir` (the eight
/// `ind.cora.*` files) and assembles it per the module doc.
///
/// Panics on any file that is missing, unreadable, outside the
/// pickle subset, or inconsistent with the rest of the set: wrong
/// widths, a test index that does not name exactly the test block,
/// an `x` or `y` that is not the head of `allx` or `ally`, a
/// validation range that overruns the non-test block, or a graph
/// entry naming a node outside the dataset. The panic message
/// names the file and the failure.
pub fn load_planetoid(dir: &Path) -> CitationGraph {
    let read = |name: &str| -> Vec<u8> {
        let path = dir.join(name);
        std::fs::read(&path)
            .unwrap_or_else(|e| panic!("planetoid: cannot read {}: {e}", path.display()))
    };
    let x = parse_pickle(&read("ind.cora.x"), "ind.cora.x");
    let y = parse_pickle(&read("ind.cora.y"), "ind.cora.y");
    let tx = parse_pickle(&read("ind.cora.tx"), "ind.cora.tx");
    let ty = parse_pickle(&read("ind.cora.ty"), "ind.cora.ty");
    let allx = parse_pickle(&read("ind.cora.allx"), "ind.cora.allx");
    let ally = parse_pickle(&read("ind.cora.ally"), "ind.cora.ally");
    let graph = parse_pickle(&read("ind.cora.graph"), "ind.cora.graph");
    let test_index_raw = read("ind.cora.test.index");
    let test_index_text = String::from_utf8(test_index_raw)
        .unwrap_or_else(|_| panic!("planetoid: ind.cora.test.index is not UTF-8 text"));
    assemble(&Files {
        x: &x,
        y: &y,
        tx: &tx,
        ty: &ty,
        allx: &allx,
        ally: &ally,
        graph: &graph,
        test_index_text: &test_index_text,
    })
}

/// The parsed files of one dataset, grouped for assembly.
struct Files<'a> {
    x: &'a Value,
    y: &'a Value,
    tx: &'a Value,
    ty: &'a Value,
    allx: &'a Value,
    ally: &'a Value,
    graph: &'a Value,
    test_index_text: &'a str,
}

/// The number of validation nodes in the standard Planetoid split,
/// after the training block.
const VALIDATION_COUNT: usize = 500;

fn assemble(files: &Files) -> CitationGraph {
    let (x, y, tx, ty, allx, ally, graph, test_index_text) = (
        files.x,
        files.y,
        files.tx,
        files.ty,
        files.allx,
        files.ally,
        files.graph,
        files.test_index_text,
    );
    let allx = as_csr(allx, "ind.cora.allx");
    let tx = as_csr(tx, "ind.cora.tx");
    let x = as_csr(x, "ind.cora.x");
    let ally = as_matrix(ally, "ind.cora.ally");
    let ty = as_matrix(ty, "ind.cora.ty");
    let y = as_matrix(y, "ind.cora.y");
    let (a, width) = (allx.rows, allx.cols);
    let t = tx.rows;
    assert!(
        tx.cols == width,
        "planetoid: ind.cora.tx has width {}, ind.cora.allx has {width}",
        tx.cols
    );
    assert!(
        x.cols == width && x.rows == y.len(),
        "planetoid: ind.cora.x and ind.cora.y disagree with the feature width or each other"
    );
    assert!(
        ally.len() == a
            && ty.len() == t
            && !ally.is_empty()
            && !ty.is_empty()
            && ally[0].len() == ty[0].len(),
        "planetoid: label files disagree with the feature files on node or class counts"
    );
    let train_count = y.len();
    assert!(
        train_count + VALIDATION_COUNT <= a,
        "planetoid: training plus validation overruns the non-test block"
    );
    assert!(
        x.rows <= a,
        "planetoid: ind.cora.x has more rows than ind.cora.allx"
    );
    // The set must agree with itself: x and y are the heads of
    // allx and ally.
    let allx_dense = allx.densify();
    let x_dense = x.densify();
    assert!(
        x_dense[..] == allx_dense[..x.rows],
        "planetoid: ind.cora.x is not the head of ind.cora.allx"
    );
    assert!(
        y[..] == ally[..train_count],
        "planetoid: ind.cora.y is not the head of ind.cora.ally"
    );
    let mut test_ids: Vec<usize> = Vec::with_capacity(t);
    for line in test_index_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        test_ids.push(
            line.parse::<usize>()
                .unwrap_or_else(|_| panic!("planetoid: bad test index line {line:?}")),
        );
    }
    assert!(
        test_ids.len() == t,
        "planetoid: test.index names {} nodes, ind.cora.tx has {t} rows",
        test_ids.len()
    );
    let unique: BTreeSet<usize> = test_ids.iter().cloned().collect();
    assert!(
        unique.len() == t && test_ids.iter().all(|&v| (a..a + t).contains(&v)),
        "planetoid: test.index does not name each test-block node exactly once"
    );
    let n = a + t;
    let mut sorted_ids = test_ids.clone();
    sorted_ids.sort_unstable();
    let mut features: Matrix = allx_dense;
    features.resize(n, Vec::new());
    let tx_dense = tx.densify();
    // The reference placement: the row stored at the j-th smallest
    // test id moves to the j-th test id in file order.
    for (j, &node) in test_ids.iter().enumerate() {
        features[node] = tx_dense[sorted_ids[j] - a].clone();
    }
    let mut labels: Vec<usize> = ally.iter().map(|row| argmax(row)).collect();
    labels.resize(n, 0);
    for (j, &node) in test_ids.iter().enumerate() {
        labels[node] = argmax(&ty[sorted_ids[j] - a]);
    }
    let train: Vec<usize> = (0..train_count).collect();
    let validation: Vec<usize> = (train_count..train_count + VALIDATION_COUNT).collect();
    let test = sorted_ids;
    let adjacency = as_graph(graph, "ind.cora.graph");
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (&from, neighbours) in &adjacency {
        assert!(
            from < n,
            "planetoid: graph names node {from}, the dataset has {n}"
        );
        for &to in neighbours {
            assert!(
                to < n,
                "planetoid: graph names node {to}, the dataset has {n}"
            );
            if from != to {
                edge_set.insert((from.min(to), from.max(to)));
            }
        }
    }
    CitationGraph {
        data: GcnData {
            features,
            labels,
            train,
            validation,
            test,
        },
        edges: edge_set.into_iter().collect(),
    }
}

/// The index of the first maximum, as numpy's argmax reads a row.
/// A label row holding a non-finite value is corrupt, and panics
/// rather than borrowing numpy's NaN propagation into a label.
fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        assert!(
            v.is_finite(),
            "planetoid: a label row holds a non-finite value"
        );
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// A CSR matrix extracted from a pickle value.
struct Csr {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
}

impl Csr {
    fn densify(&self) -> Matrix {
        let mut out = vec![vec![0.0; self.cols]; self.rows];
        for (row, span) in out.iter_mut().zip(self.indptr.windows(2)) {
            for k in span[0]..span[1] {
                row[self.indices[k]] = self.data[k];
            }
        }
        out
    }
}

fn as_csr(value: &Value, file: &str) -> Csr {
    let (_class, args, state) = match value {
        Value::Object { class, args, state } if class.1.as_slice() == b"csr_matrix" => {
            (class, args, state)
        }
        _ => panic!("planetoid: {file} is not a scipy csr_matrix"),
    };
    let state = state
        .as_deref()
        .unwrap_or_else(|| panic!("planetoid: {file} csr_matrix carries no state"));
    let dict = match state {
        Value::Dict(entries) => entries,
        _ => panic!("planetoid: {file} csr_matrix state is not a dict"),
    };
    // Last match wins, as Python dict construction reads a state
    // dict with a repeated key.
    let get = |key: &str| -> &Value {
        dict.iter()
            .rev()
            .find(|(k, _)| matches!(k, Value::Str(s) if s.as_slice() == key.as_bytes()))
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("planetoid: {file} csr_matrix state lacks {key}"))
    };
    let data = as_f64_vec(get("data"), file);
    let indices = as_usize_vec(get("indices"), file);
    let indptr = as_usize_vec(get("indptr"), file);
    let shape = match dict
        .iter()
        .rev()
        .find(|(k, _)| matches!(k, Value::Str(s) if s.as_slice() == b"_shape" || s.as_slice() == b"shape"))
        .map(|(_, v)| v)
    {
        Some(Value::Tuple(items)) if items.len() == 2 => {
            (as_usize(&items[0], file), as_usize(&items[1], file))
        }
        _ => match args.first() {
            Some(Value::Tuple(items)) if items.len() == 2 => {
                (as_usize(&items[0], file), as_usize(&items[1], file))
            }
            _ => panic!("planetoid: {file} csr_matrix shape not found"),
        },
    };
    let (rows, cols) = shape;
    assert!(
        indptr.len() == rows + 1
            && indptr.windows(2).all(|w| w[0] <= w[1])
            && indptr[rows] == data.len()
            && indices.len() == data.len()
            && indices.iter().all(|&c| c < cols),
        "planetoid: {file} csr_matrix parts are inconsistent"
    );
    Csr {
        rows,
        cols,
        data,
        indices,
        indptr,
    }
}

fn as_matrix(value: &Value, file: &str) -> Matrix {
    let arr = as_ndarray(value, file);
    assert!(
        arr.shape.len() == 2 && arr.shape[1] > 0,
        "planetoid: {file} is not a two-dimensional array with columns"
    );
    let (rows, cols) = (arr.shape[0], arr.shape[1]);
    arr.values
        .chunks_exact(cols)
        .take(rows)
        .map(|c| c.to_vec())
        .collect()
}

fn as_f64_vec(value: &Value, file: &str) -> Vec<f64> {
    as_ndarray(value, file).values
}

fn as_usize_vec(value: &Value, file: &str) -> Vec<usize> {
    as_ndarray(value, file)
        .ints
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .unwrap_or_else(|_| panic!("planetoid: {file} holds a negative index"))
        })
        .collect()
}

fn as_usize(value: &Value, file: &str) -> usize {
    match value {
        Value::Int(v) => usize::try_from(*v)
            .unwrap_or_else(|_| panic!("planetoid: {file} holds a negative count")),
        _ => panic!("planetoid: {file} holds a non-integer where a count belongs"),
    }
}

fn as_graph(value: &Value, file: &str) -> BTreeMap<usize, Vec<usize>> {
    let dict = match value {
        Value::Dict(entries) => entries,
        _ => panic!("planetoid: {file} is not a dict of adjacency lists"),
    };
    let mut out = BTreeMap::new();
    for (k, v) in dict {
        let node = as_usize(k, file);
        let neighbours = match v {
            Value::List(items) => items.iter().map(|item| as_usize(item, file)).collect(),
            _ => panic!("planetoid: {file} values are not lists"),
        };
        out.insert(node, neighbours);
    }
    out
}

/// A decoded numpy array: its shape and its elements read both as
/// floats and as integers, of which exactly the one matching the
/// array's dtype kind is populated.
struct NdArray {
    shape: Vec<usize>,
    values: Vec<f64>,
    ints: Vec<i64>,
}

fn as_ndarray(value: &Value, file: &str) -> NdArray {
    let state = match value {
        Value::Object { class, state, .. } if class.1.as_slice() == b"ndarray" => state,
        _ => panic!("planetoid: {file} holds a value that is not a numpy array"),
    };
    let state = state
        .as_deref()
        .unwrap_or_else(|| panic!("planetoid: {file} array carries no state"));
    let items = match state {
        Value::Tuple(items) if items.len() == 5 => items,
        _ => panic!("planetoid: {file} array state is not the five-part numpy form"),
    };
    let shape: Vec<usize> = match &items[1] {
        Value::Tuple(dims) => dims.iter().map(|d| as_usize(d, file)).collect(),
        _ => panic!("planetoid: {file} array shape is not a tuple"),
    };
    let dtype = match &items[2] {
        Value::DType(descr) => parse_dtype(descr, file),
        _ => panic!("planetoid: {file} array dtype is not a plain dtype"),
    };
    let fortran = match &items[3] {
        Value::Bool(b) => *b,
        Value::Int(0) => false,
        Value::Int(1) => true,
        _ => panic!("planetoid: {file} array order flag is not a boolean"),
    };
    assert!(
        !fortran,
        "planetoid: {file} array is in Fortran order, outside the subset"
    );
    let raw: &[u8] = match &items[4] {
        Value::Bytes(b) => b,
        Value::Str(b) => b,
        _ => panic!("planetoid: {file} array data is not a byte string"),
    };
    let count: usize = shape.iter().product();
    assert!(
        raw.len() == count * dtype.size,
        "planetoid: {file} array data is {} bytes for {} elements of {} bytes",
        raw.len(),
        count,
        dtype.size
    );
    let mut values = Vec::new();
    let mut ints = Vec::new();
    match dtype.kind {
        ScalarKind::Float => {
            values.reserve(count);
            for i in 0..count {
                values.push(dtype.read_f64(&raw[i * dtype.size..(i + 1) * dtype.size]));
            }
        }
        ScalarKind::Int | ScalarKind::Bool => {
            ints.reserve(count);
            values.reserve(count);
            for i in 0..count {
                let v = dtype.read_i64(&raw[i * dtype.size..(i + 1) * dtype.size]);
                ints.push(v);
                values.push(v as f64);
            }
        }
    }
    NdArray {
        shape,
        values,
        ints,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ScalarKind {
    Float,
    Int,
    Bool,
}

/// A numpy scalar dtype reduced to what decoding needs.
struct DType {
    kind: ScalarKind,
    size: usize,
    signed: bool,
}

impl DType {
    fn read_f64(&self, bytes: &[u8]) -> f64 {
        match self.size {
            4 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
            8 => f64::from_le_bytes(bytes.try_into().unwrap()),
            _ => unreachable!("float dtype sizes are 4 or 8"),
        }
    }

    fn read_i64(&self, bytes: &[u8]) -> i64 {
        if self.kind == ScalarKind::Bool {
            return i64::from(bytes[0] != 0);
        }
        let mut buf = [0u8; 8];
        buf[..self.size].copy_from_slice(bytes);
        if self.signed {
            // Sign-extend from the element width.
            if self.size < 8 && bytes[self.size - 1] & 0x80 != 0 {
                buf[self.size..].fill(0xFF);
            }
            i64::from_le_bytes(buf)
        } else {
            u64::from_le_bytes(buf) as i64
        }
    }
}

/// Parses a numpy dtype description string such as `<f4`, `|i1`,
/// or `<u8`. Big-endian descriptions and anything outside the
/// plain numeric kinds are outside the subset and panic.
fn parse_dtype(descr: &[u8], file: &str) -> DType {
    let text = std::str::from_utf8(descr)
        .unwrap_or_else(|_| panic!("planetoid: {file} dtype is not text"));
    let body = match text.as_bytes().first() {
        Some(b'<') | Some(b'|') | Some(b'=') => &text[1..],
        Some(b'>') => panic!("planetoid: {file} dtype {text} is big-endian, outside the subset"),
        _ => text,
    };
    let (kind_char, size_text) = body.split_at(1);
    let size: usize = size_text
        .parse()
        .unwrap_or_else(|_| panic!("planetoid: {file} dtype {text} has no size"));
    match (kind_char, size) {
        ("f", 4) | ("f", 8) => DType {
            kind: ScalarKind::Float,
            size,
            signed: true,
        },
        ("i", 1..=8) => DType {
            kind: ScalarKind::Int,
            size,
            signed: true,
        },
        ("u", 1..=8) => DType {
            kind: ScalarKind::Int,
            size,
            signed: false,
        },
        ("b", 1) => DType {
            kind: ScalarKind::Bool,
            size: 1,
            signed: false,
        },
        _ => panic!("planetoid: {file} dtype {text} is outside the subset"),
    }
}

/// A pickle value, at the fidelity the loader's extractors need.
/// Strings are kept as raw bytes: protocol 2 pickles carry array
/// data as strings, and only names and dtype descriptions are ever
/// read as text.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Vec<u8>),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// An unresolved global reference, module then name.
    Global(Vec<u8>, Vec<u8>),
    /// A numpy dtype, by its description string.
    DType(Vec<u8>),
    /// An object built by calling a global and applying a state:
    /// a numpy array or a scipy CSR matrix, in this subset.
    Object {
        class: (Vec<u8>, Vec<u8>),
        args: Vec<Value>,
        state: Option<Box<Value>>,
    },
}

/// Parses one pickle stream into a [`Value`]. Panics, naming
/// `file`, on any opcode, global, or construction outside the
/// subset the Planetoid files use.
fn parse_pickle(bytes: &[u8], file: &str) -> Value {
    let mut parser = Parser {
        bytes,
        pos: 0,
        memo: HashMap::new(),
        stack: Vec::new(),
        marks: Vec::new(),
        mutations: 0,
        file,
    };
    parser.run()
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    memo: HashMap<u32, (Value, u64)>,
    stack: Vec<Value>,
    marks: Vec<usize>,
    mutations: u64,
    file: &'a str,
}

impl<'a> Parser<'a> {
    fn fail(&self, what: &str) -> ! {
        panic!(
            "planetoid: {} pickle outside the subset: {what} at byte {}",
            self.file, self.pos
        )
    }

    fn byte(&mut self) -> u8 {
        let b = *self
            .bytes
            .get(self.pos)
            .unwrap_or_else(|| self.fail("stream ends mid-opcode"));
        self.pos += 1;
        b
    }

    fn take(&mut self, n: usize) -> &'a [u8] {
        let end = self
            .pos
            .checked_add(n)
            .unwrap_or_else(|| self.fail("length overflows"));
        let out = self
            .bytes
            .get(self.pos..end)
            .unwrap_or_else(|| self.fail("stream ends mid-value"));
        self.pos = end;
        out
    }

    fn line(&mut self) -> &'a [u8] {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        let out = &self.bytes[start..self.pos];
        if self.pos < self.bytes.len() {
            self.pos += 1; // the newline
        }
        out
    }

    fn pop(&mut self) -> Value {
        self.stack
            .pop()
            .unwrap_or_else(|| self.fail("stack underflow"))
    }

    fn top(&mut self) -> &mut Value {
        if self.stack.is_empty() {
            self.fail("stack underflow");
        }
        self.stack.last_mut().unwrap()
    }

    fn pop_mark(&mut self) -> Vec<Value> {
        let mark = self
            .marks
            .pop()
            .unwrap_or_else(|| self.fail("MARK without a matching opcode"));
        if mark > self.stack.len() {
            self.fail("MARK above the stack top");
        }
        self.stack.split_off(mark)
    }

    fn memo_put(&mut self, idx: u32) {
        let value = self.top().clone();
        self.memo.insert(idx, (value, self.mutations));
    }

    fn memo_get(&mut self, idx: u32) {
        let (value, at_put) = self
            .memo
            .get(&idx)
            .cloned()
            .unwrap_or_else(|| self.fail("memo read before write"));
        // The memo holds snapshots, where pickle holds references.
        // A mutable value read back after any mutation opcode may
        // have run is a snapshot that could be stale, so it fails
        // loudly instead of loading the pre-mutation object. The
        // streams this subset accepts only ever read back values
        // no mutation has touched.
        if at_put != self.mutations
            && matches!(
                value,
                Value::List(_) | Value::Dict(_) | Value::Object { .. }
            )
        {
            self.fail("memo read of a mutable value after a mutation opcode");
        }
        self.stack.push(value);
    }

    fn parse_i64_text(&mut self, text: &[u8]) -> i64 {
        let text = std::str::from_utf8(text).unwrap_or_else(|_| self.fail("bad integer text"));
        text.trim()
            .parse::<i64>()
            .unwrap_or_else(|_| self.fail("integer out of range"))
    }

    fn run(&mut self) -> Value {
        loop {
            let op = self.byte();
            match op {
                b'.' => {
                    // STOP
                    let value = self.pop();
                    if !self.stack.is_empty() {
                        self.fail("values left on the stack at STOP");
                    }
                    return value;
                }
                0x80 => {
                    // PROTO
                    let _version = self.byte();
                }
                0x95 => {
                    // FRAME: length is advisory for this reader.
                    let _len = self.take(8);
                }
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let v = self.byte();
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'M' => {
                    let b = self.take(2);
                    self.stack.push(Value::Int(i64::from(u16::from_le_bytes(
                        b.try_into().unwrap(),
                    ))));
                }
                b'J' => {
                    let b = self.take(4);
                    self.stack.push(Value::Int(i64::from(i32::from_le_bytes(
                        b.try_into().unwrap(),
                    ))));
                }
                b'I' => {
                    let text = self.line().to_vec();
                    match text.as_slice() {
                        b"01" => self.stack.push(Value::Bool(true)),
                        b"00" => self.stack.push(Value::Bool(false)),
                        _ => {
                            let v = self.parse_i64_text(&text);
                            self.stack.push(Value::Int(v));
                        }
                    }
                }
                b'L' => {
                    let mut text = self.line().to_vec();
                    if text.last() == Some(&b'L') {
                        text.pop();
                    }
                    let v = self.parse_i64_text(&text);
                    self.stack.push(Value::Int(v));
                }
                0x8A => {
                    // LONG1: two's complement little-endian bytes.
                    let n = usize::from(self.byte());
                    let raw = self.take(n).to_vec();
                    self.stack.push(Value::Int(self.long_value(&raw)));
                }
                0x8B => {
                    // LONG4
                    let b = self.take(4);
                    let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let raw = self.take(n).to_vec();
                    self.stack.push(Value::Int(self.long_value(&raw)));
                }
                b'G' => {
                    let b = self.take(8);
                    self.stack
                        .push(Value::Float(f64::from_be_bytes(b.try_into().unwrap())));
                }
                b'F' => {
                    let text = self.line().to_vec();
                    let text =
                        std::str::from_utf8(&text).unwrap_or_else(|_| self.fail("bad float text"));
                    let v: f64 = text
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| self.fail("bad float text"));
                    self.stack.push(Value::Float(v));
                }
                b'U' => {
                    let n = usize::from(self.byte());
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Str(s));
                }
                b'T' => {
                    let b = self.take(4);
                    let n = i32::from_le_bytes(b.try_into().unwrap());
                    let n =
                        usize::try_from(n).unwrap_or_else(|_| self.fail("negative string length"));
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Str(s));
                }
                b'V' => {
                    let s = self.line().to_vec();
                    self.stack.push(Value::Str(s));
                }
                b'S' => {
                    let text = self.line().to_vec();
                    self.stack.push(Value::Str(self.unquote(&text)));
                }
                b'X' => {
                    let b = self.take(4);
                    let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Str(s));
                }
                0x8C => {
                    let n = usize::from(self.byte());
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Str(s));
                }
                b'C' => {
                    let n = usize::from(self.byte());
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Bytes(s));
                }
                b'B' => {
                    let b = self.take(4);
                    let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Bytes(s));
                }
                0x8D => {
                    let b = self.take(8);
                    let n = u64::from_le_bytes(b.try_into().unwrap());
                    let n = usize::try_from(n)
                        .unwrap_or_else(|_| self.fail("bytes length out of range"));
                    let s = self.take(n).to_vec();
                    self.stack.push(Value::Bytes(s));
                }
                b'(' => self.marks.push(self.stack.len()),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b't' => {
                    let items = self.pop_mark();
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop();
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop();
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b']' => self.stack.push(Value::List(Vec::new())),
                b'l' => {
                    let items = self.pop_mark();
                    self.stack.push(Value::List(items));
                }
                b'a' => {
                    self.mutations += 1;
                    let value = self.pop();
                    match self.top() {
                        Value::List(items) => items.push(value),
                        _ => self.fail("APPEND onto a non-list"),
                    }
                }
                b'e' => {
                    self.mutations += 1;
                    let items = self.pop_mark();
                    match self.top() {
                        Value::List(list) => list.extend(items),
                        _ => self.fail("APPENDS onto a non-list"),
                    }
                }
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b'd' => {
                    let items = self.pop_mark();
                    let entries = pairs(items, self);
                    self.stack.push(Value::Dict(entries));
                }
                b's' => {
                    self.mutations += 1;
                    let value = self.pop();
                    let key = self.pop();
                    match self.top() {
                        Value::Dict(entries) => set_entry(entries, key, value),
                        _ => self.fail("SETITEM onto a non-dict"),
                    }
                }
                b'u' => {
                    self.mutations += 1;
                    let items = self.pop_mark();
                    let entries_new = pairs(items, self);
                    match self.top() {
                        Value::Dict(entries) => {
                            for (k, v) in entries_new {
                                set_entry(entries, k, v);
                            }
                        }
                        _ => self.fail("SETITEMS onto a non-dict"),
                    }
                }
                b'c' => {
                    let module = self.line().to_vec();
                    let name = self.line().to_vec();
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let name = self.pop();
                    let module = self.pop();
                    match (module, name) {
                        (Value::Str(m), Value::Str(n)) => self.stack.push(Value::Global(m, n)),
                        _ => self.fail("STACK_GLOBAL of non-strings"),
                    }
                }
                b'R' => {
                    let args = self.pop();
                    let callable = self.pop();
                    let value = self.call(callable, args);
                    self.stack.push(value);
                }
                0x81 => {
                    // NEWOBJ
                    let args = self.pop();
                    let class = self.pop();
                    let value = self.call(class, args);
                    self.stack.push(value);
                }
                b'b' => {
                    self.mutations += 1;
                    let state = self.pop();
                    match self.top() {
                        Value::Object { state: slot, .. } => *slot = Some(Box::new(state)),
                        Value::DType(_) => {
                            // A dtype's BUILD state refines the
                            // byte order the description already
                            // fixed. Only the order is checked. A
                            // big-endian refinement is outside the
                            // subset like a big-endian description.
                            match &state {
                                Value::Tuple(items) => match items.get(1) {
                                    Some(Value::Str(order))
                                        if order == b"<" || order == b"|" || order == b"=" => {}
                                    _ => self.fail("dtype state lacks a little-endian order"),
                                },
                                _ => self.fail("dtype state is not a tuple"),
                            }
                        }
                        Value::Dict(entries) => match state {
                            // A defaultdict's state is its own
                            // dict, applied by BUILD.
                            Value::Dict(new_entries) => {
                                for (k, v) in new_entries {
                                    set_entry(entries, k, v);
                                }
                            }
                            _ => self.fail("BUILD of a non-dict state onto a dict"),
                        },
                        _ => self.fail("BUILD onto a non-object"),
                    }
                }
                b'q' => {
                    let idx = u32::from(self.byte());
                    self.memo_put(idx);
                }
                b'r' => {
                    let b = self.take(4);
                    let idx = u32::from_le_bytes(b.try_into().unwrap());
                    self.memo_put(idx);
                }
                b'h' => {
                    let idx = u32::from(self.byte());
                    self.memo_get(idx);
                }
                b'j' => {
                    let b = self.take(4);
                    let idx = u32::from_le_bytes(b.try_into().unwrap());
                    self.memo_get(idx);
                }
                b'0' => {
                    let _ = self.pop();
                }
                b'2' => {
                    let value = self.top().clone();
                    self.stack.push(value);
                }
                _ => self.fail(&format!("opcode 0x{op:02x}")),
            }
        }
    }

    /// Decodes a two's complement little-endian long into an i64,
    /// failing when it does not fit.
    fn long_value(&self, raw: &[u8]) -> i64 {
        if raw.len() > 8 {
            self.fail("long integer wider than 64 bits");
        }
        let mut buf = [0u8; 8];
        buf[..raw.len()].copy_from_slice(raw);
        if !raw.is_empty() && raw[raw.len() - 1] & 0x80 != 0 {
            buf[raw.len()..].fill(0xFF);
        }
        i64::from_le_bytes(buf)
    }

    /// Unquotes a STRING opcode body: a Python repr of a byte
    /// string in single or double quotes, with backslash escapes.
    fn unquote(&self, text: &[u8]) -> Vec<u8> {
        if text.len() < 2 || (text[0] != b'\'' && text[0] != b'"') {
            self.fail("STRING without quotes");
        }
        let body = &text[1..text.len() - 1];
        let mut out = Vec::with_capacity(body.len());
        let mut i = 0;
        while i < body.len() {
            if body[i] == b'\\' && i + 1 < body.len() {
                match body[i + 1] {
                    b'n' => {
                        out.push(b'\n');
                        i += 2;
                    }
                    b'r' => {
                        out.push(b'\r');
                        i += 2;
                    }
                    b't' => {
                        out.push(b'\t');
                        i += 2;
                    }
                    b'\\' => {
                        out.push(b'\\');
                        i += 2;
                    }
                    b'\'' => {
                        out.push(b'\'');
                        i += 2;
                    }
                    b'"' => {
                        out.push(b'"');
                        i += 2;
                    }
                    b'x' if i + 3 < body.len() => {
                        let hex = std::str::from_utf8(&body[i + 2..i + 4])
                            .unwrap_or_else(|_| self.fail("bad \\x escape"));
                        let v = u8::from_str_radix(hex, 16)
                            .unwrap_or_else(|_| self.fail("bad \\x escape"));
                        out.push(v);
                        i += 4;
                    }
                    _ => self.fail("unsupported STRING escape"),
                }
            } else {
                out.push(body[i]);
                i += 1;
            }
        }
        out
    }

    /// Applies a callable global to an argument tuple, for the
    /// constructions this subset builds. Anything else is outside
    /// the subset and fails loudly.
    fn call(&mut self, callable: Value, args: Value) -> Value {
        let args = match args {
            Value::Tuple(items) => items,
            _ => self.fail("call arguments are not a tuple"),
        };
        let (module, name) = match &callable {
            Value::Global(m, n) => (m.clone(), n.clone()),
            _ => self.fail("call of a non-global"),
        };
        let name_str = String::from_utf8_lossy(&name).into_owned();
        match name_str.as_str() {
            "_reconstruct" | "_reconstructor" => {
                // numpy array reconstruction: (class, base, state
                // marker). The array itself arrives through BUILD.
                // The copyreg spelling names the same construction
                // in older files.
                let class = match args.first() {
                    Some(Value::Global(m, n)) => (m.clone(), n.clone()),
                    _ => self.fail("_reconstruct without a class"),
                };
                let known = (class.1.as_slice() == b"ndarray" && class.0.as_slice() == b"numpy")
                    || (class.1.as_slice() == b"csr_matrix" && is_scipy_sparse(&class.0));
                if !known {
                    self.fail("_reconstruct of a class outside the subset");
                }
                Value::Object {
                    class,
                    args: Vec::new(),
                    state: None,
                }
            }
            "dtype" if module.as_slice() == b"numpy" => {
                // numpy.dtype(descr, ...): only the description is
                // needed to decode.
                match args.first() {
                    Some(Value::Str(descr)) => Value::DType(descr.clone()),
                    Some(Value::Bytes(descr)) => Value::DType(descr.clone()),
                    _ => self.fail("dtype without a description string"),
                }
            }
            "encode" if module.as_slice() == b"_codecs" => {
                // Protocol 2's spelling of a bytes object: the
                // bytes ride in a string whose code points are
                // their latin-1 reading, so the string's UTF-8
                // form is decoded back to one byte per code point.
                match args.first() {
                    Some(Value::Str(b)) => {
                        let text = std::str::from_utf8(b)
                            .unwrap_or_else(|_| self.fail("_codecs.encode of invalid UTF-8"));
                        let mut out = Vec::with_capacity(text.len());
                        for ch in text.chars() {
                            let v = u32::from(ch);
                            if v > 0xFF {
                                self.fail("_codecs.encode beyond latin-1");
                            }
                            out.push(v as u8);
                        }
                        Value::Bytes(out)
                    }
                    Some(Value::Bytes(b)) => Value::Bytes(b.clone()),
                    _ => self.fail("_codecs.encode of a non-string"),
                }
            }
            "defaultdict" if module.as_slice() == b"collections" => Value::Dict(Vec::new()),
            "csr_matrix" if is_scipy_sparse(&module) => Value::Object {
                class: (module, name),
                args,
                state: None,
            },
            "ndarray" if module.as_slice() == b"numpy" => Value::Object {
                class: (module, name),
                args,
                state: None,
            },
            _ => self.fail(&format!("global {name_str}")),
        }
    }
}

/// The module paths scipy's sparse classes have used, across the
/// eras of the dataset files.
fn is_scipy_sparse(module: &[u8]) -> bool {
    module == b"scipy.sparse.csr" || module == b"scipy.sparse._csr"
}

/// Groups a flat item list into key value pairs.
fn pairs(items: Vec<Value>, parser: &Parser) -> Vec<(Value, Value)> {
    if !items.len().is_multiple_of(2) {
        parser.fail("dict items do not pair up");
    }
    items
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
        .collect()
}

fn set_entry(entries: &mut Vec<(Value, Value)>, key: Value, value: Value) {
    if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        entries.push((key, value));
    }
}
