use wyrd_graph::{diff, inject, Graph, Relation};

fn rel(from: &str, label: &str, to: &str) -> Relation {
    Relation {
        from: from.into(),
        label: label.into(),
        to: to.into(),
    }
}

fn base_graph() -> Graph {
    let mut g = Graph::new();
    g.add_relation(rel("paris", "capital_of", "france"));
    g.add_relation(rel("france", "member_of", "eu"));
    g.add_relation(rel("berlin", "capital_of", "germany"));
    g
}

#[test]
fn injection_records_exactly_what_was_added() {
    let mut g = base_graph();
    let poison = vec![
        rel("paris", "capital_of", "germany"), // new bridge relation
        rel("france", "member_of", "eu"),      // already present
        rel("lyon", "located_in", "france"),   // new node lyon
    ];
    let record = inject(&mut g, &poison);
    assert_eq!(record.relations_added.len(), 2);
    assert_eq!(record.relations_already_present.len(), 1);
    assert_eq!(record.nodes_added, vec!["lyon".to_string()]);
    assert_eq!(g.edge_count(), 5);
}

#[test]
fn injection_creates_the_bridge_it_claims_to() {
    // The whole point of a GraphRAG relation attack: before injection
    // paris cannot reach germany; after it, one hop.
    let mut g = base_graph();
    assert_eq!(g.hop_distance("paris", "germany"), None);
    inject(&mut g, &[rel("paris", "capital_of", "germany")]);
    assert_eq!(g.hop_distance("paris", "germany"), Some(1));
    // And the blast radius around a query node grows accordingly.
    let within = g.reachable_within("paris", 1);
    assert!(within.contains("germany"));
    assert!(within.contains("france"));
}

#[test]
fn hop_distance_follows_direction_and_chains() {
    let g = base_graph();
    assert_eq!(g.hop_distance("paris", "eu"), Some(2));
    assert_eq!(g.hop_distance("eu", "paris"), None); // directed
    assert_eq!(g.hop_distance("paris", "paris"), Some(0));
}

#[test]
fn diff_reports_structural_change_between_extractions() {
    let clean = base_graph();
    let mut poisoned = base_graph();
    inject(
        &mut poisoned,
        &[
            rel("paris", "capital_of", "germany"),
            rel("lyon", "located_in", "france"),
        ],
    );
    let d = diff(&clean, &poisoned);
    assert_eq!(d.relations_added.len(), 2);
    assert_eq!(d.nodes_added, vec!["lyon".to_string()]);
    assert!(d.relations_removed.is_empty());
    assert!(d.nodes_removed.is_empty());
}

#[test]
fn duplicate_relation_add_is_reported_not_counted() {
    let mut g = base_graph();
    assert!(!g.add_relation(rel("paris", "capital_of", "france")));
    assert_eq!(g.edge_count(), 3);
}

#[test]
fn degree_summary_counts_both_directions() {
    let g = base_graph();
    let summary = wyrd_graph::degree_summary(&g);
    assert_eq!(summary["france"], (1, 1)); // member_of out, capital_of in
    assert_eq!(summary["paris"], (1, 0));
}
