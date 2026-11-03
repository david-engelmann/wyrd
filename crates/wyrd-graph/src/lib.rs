//! wyrd-graph: knowledge graphs and graph-arm machinery.
//!
//! Charter section 6 assigns this crate the GraphRAG-side graph type,
//! relation injection, graph diffing, and the traversal that blast
//! radius is measured over, plus the GNN arm. Phase 0 lands the
//! GraphRAG side: a directed labelled multigraph-free graph (one edge
//! per ordered pair and label), relation injection with a full audit
//! record of what was added, structural diffing between a clean and a
//! poisoned graph, and shortest-path hop distance for blast-radius
//! reads. The GNN trainer waits on its charter spike.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A directed edge `from -[label]-> to`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Relation {
    pub from: String,
    pub label: String,
    pub to: String,
}

/// A directed labelled graph over string nodes. Node and edge sets are
/// kept sorted (BTree) so serialisation and diffs are deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: BTreeSet<String>,
    edges: BTreeSet<Relation>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn add_node(&mut self, node: &str) {
        self.nodes.insert(node.to_string());
    }

    /// Adds a relation, creating its endpoint nodes if absent. Returns
    /// false when the exact relation already existed, so injection code
    /// can tell a planted relation from a duplicate.
    pub fn add_relation(&mut self, rel: Relation) -> bool {
        self.nodes.insert(rel.from.clone());
        self.nodes.insert(rel.to.clone());
        self.edges.insert(rel)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn has_relation(&self, from: &str, label: &str, to: &str) -> bool {
        self.edges.contains(&Relation {
            from: from.to_string(),
            label: label.to_string(),
            to: to.to_string(),
        })
    }

    pub fn relations(&self) -> impl Iterator<Item = &Relation> {
        self.edges.iter()
    }

    /// Outgoing neighbours of a node, sorted.
    pub fn neighbours(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.from == node)
            .map(|e| e.to.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Shortest directed path length in hops from `from` to `to`,
    /// following edge direction. None when unreachable. GraphRAG blast
    /// radius is read over this distance: a poisoned relation matters
    /// when it puts a target entity within retrieval reach of a query
    /// entity.
    pub fn hop_distance(&self, from: &str, to: &str) -> Option<usize> {
        if from == to {
            return Some(0);
        }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        seen.insert(from.to_string());
        queue.push_back((from.to_string(), 0));
        while let Some((node, dist)) = queue.pop_front() {
            for next in self.neighbours(&node) {
                if next == to {
                    return Some(dist + 1);
                }
                if seen.insert(next.clone()) {
                    queue.push_back((next, dist + 1));
                }
            }
        }
        None
    }

    /// Nodes within `hops` directed steps of `start`, including start.
    pub fn reachable_within(&self, start: &str, hops: usize) -> BTreeSet<String> {
        let mut out: BTreeSet<String> = BTreeSet::new();
        out.insert(start.to_string());
        let mut frontier = vec![start.to_string()];
        for _ in 0..hops {
            let mut next = Vec::new();
            for node in &frontier {
                for n in self.neighbours(node) {
                    if out.insert(n.clone()) {
                        next.push(n);
                    }
                }
            }
            frontier = next;
        }
        out
    }
}

/// The record of one injection run: exactly which relations and nodes
/// were newly added. GRA-001's extraction-survival read compares this
/// intent against what extraction actually produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InjectionRecord {
    pub relations_added: Vec<Relation>,
    pub relations_already_present: Vec<Relation>,
    pub nodes_added: Vec<String>,
}

/// Injects relations into a graph, recording precisely what changed.
/// Existing relations are reported, never double-counted as planted.
pub fn inject(graph: &mut Graph, relations: &[Relation]) -> InjectionRecord {
    let before_nodes: BTreeSet<String> = graph.nodes.clone();
    let mut record = InjectionRecord::default();
    for rel in relations {
        if graph.add_relation(rel.clone()) {
            record.relations_added.push(rel.clone());
        } else {
            record.relations_already_present.push(rel.clone());
        }
    }
    record.nodes_added = graph.nodes.difference(&before_nodes).cloned().collect();
    record
}

/// A structural diff between two graphs, for the charter's requirement
/// that GraphRAG runs report graph-diff statistics between the clean
/// and poisoned extractions rather than pretending only the injected
/// relations differ.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphDiff {
    pub nodes_added: Vec<String>,
    pub nodes_removed: Vec<String>,
    pub relations_added: Vec<Relation>,
    pub relations_removed: Vec<Relation>,
}

pub fn diff(before: &Graph, after: &Graph) -> GraphDiff {
    GraphDiff {
        nodes_added: after.nodes.difference(&before.nodes).cloned().collect(),
        nodes_removed: before.nodes.difference(&after.nodes).cloned().collect(),
        relations_added: after.edges.difference(&before.edges).cloned().collect(),
        relations_removed: before.edges.difference(&after.edges).cloned().collect(),
    }
}

/// Per-node degree summary, one of the cheap structural statistics the
/// diff report carries so extraction drift is visible at a glance.
pub fn degree_summary(graph: &Graph) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for n in &graph.nodes {
        out.insert(n.clone(), (0, 0));
    }
    for e in &graph.edges {
        out.get_mut(&e.from).unwrap().0 += 1;
        out.get_mut(&e.to).unwrap().1 += 1;
    }
    out
}
