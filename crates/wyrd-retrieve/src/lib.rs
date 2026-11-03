//! wyrd-retrieve: retrieval for the S3 and S4a arms.
//!
//! Charter section 6 assigns this crate the retrievers. Phase 0 lands
//! BM25 (Okapi, k1 = 1.2, b = 0.75, the standard defaults RAG-001 will
//! pin in its manifest) over an in-memory index, plus the retrieval
//! hit-rate read that section 4 requires to be reported separately from
//! generation success. The vector retriever arrives behind the same
//! search shape when the embedding provider work starts.

use std::collections::HashMap;

/// Splits text into lowercase alphanumeric tokens. Deliberately simple
/// and deterministic: the same tokenizer runs at index and query time,
/// and its behaviour is pinned by tests, because a tokeniser change is
/// an experiment change.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// One indexed document.
#[derive(Debug, Clone)]
struct Doc {
    id: String,
    term_freqs: HashMap<String, u32>,
    len: usize,
}

/// A BM25 index over a fixed document set.
#[derive(Debug, Default)]
pub struct Bm25Index {
    docs: Vec<Doc>,
    doc_freqs: HashMap<String, usize>,
    avg_len: f64,
}

/// One search hit, highest score first in result lists.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub doc_id: String,
    pub score: f64,
}

impl Bm25Index {
    pub const K1: f64 = 1.2;
    pub const B: f64 = 0.75;

    /// Builds an index from (id, text) pairs. Ids must be unique; a
    /// duplicate id is a corpus construction bug, so it panics loudly
    /// rather than silently shadowing a document in a sealed corpus.
    pub fn build(docs: &[(String, String)]) -> Bm25Index {
        let mut index = Bm25Index::default();
        let mut seen = std::collections::HashSet::new();
        let mut total_len = 0usize;
        for (id, text) in docs {
            assert!(seen.insert(id.clone()), "duplicate document id: {id}");
            let tokens = tokenize(text);
            let mut term_freqs: HashMap<String, u32> = HashMap::new();
            for t in &tokens {
                *term_freqs.entry(t.clone()).or_insert(0) += 1;
            }
            for term in term_freqs.keys() {
                *index.doc_freqs.entry(term.clone()).or_insert(0) += 1;
            }
            total_len += tokens.len();
            index.docs.push(Doc {
                id: id.clone(),
                term_freqs,
                len: tokens.len(),
            });
        }
        if !index.docs.is_empty() {
            index.avg_len = total_len as f64 / index.docs.len() as f64;
        }
        index
    }

    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    /// Inverse document frequency, Robertson form with the +1 inside the
    /// log so the value never goes negative. A term in every document
    /// gets a small positive idf, not zero, under this form.
    fn idf(&self, term: &str) -> f64 {
        let n = self.docs.len() as f64;
        let df = *self.doc_freqs.get(term).unwrap_or(&0) as f64;
        if df == 0.0 {
            return 0.0;
        }
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Searches the index, returning at most `k` hits with positive
    /// scores, best first. Ties break by document id so result order is
    /// deterministic and sealable.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let terms = tokenize(query);
        let mut hits: Vec<Hit> = Vec::new();
        for doc in &self.docs {
            let mut score = 0.0;
            for term in &terms {
                let tf = *doc.term_freqs.get(term).unwrap_or(&0) as f64;
                if tf == 0.0 {
                    continue;
                }
                let norm =
                    tf + Self::K1 * (1.0 - Self::B + Self::B * doc.len as f64 / self.avg_len);
                score += self.idf(term) * (tf * (Self::K1 + 1.0)) / norm;
            }
            if score > 0.0 {
                hits.push(Hit {
                    doc_id: doc.id.clone(),
                    score,
                });
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        hits.truncate(k);
        hits
    }
}

/// Retrieval hit rate: the fraction of queries whose top-`k` hits contain
/// at least one document from that query's target set. This is the
/// section 4 S3/S4a read that is reported beside, never merged into,
/// generation success. Returns None when there are no queries.
pub fn hit_rate(index: &Bm25Index, queries: &[(String, Vec<String>)], k: usize) -> Option<f64> {
    if queries.is_empty() {
        return None;
    }
    let mut hits = 0usize;
    for (query, targets) in queries {
        let results = index.search(query, k);
        if results.iter().any(|h| targets.contains(&h.doc_id)) {
            hits += 1;
        }
    }
    Some(hits as f64 / queries.len() as f64)
}
