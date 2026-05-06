"""Drug graph construction and merging."""

import tqdm
import json
from collections import defaultdict
from itertools import combinations
from typing import Dict, Set, Callable, Tuple, Optional
from pathlib import Path

import networkx as nx
import pandas as pd


# Drug match types for forming connected components
SMILES_MATCH = "SMILES_IDENTICAL"
SID_MATCH = "SID_MATCH"
CID_MATCH = "CID_MATCH"
PARENT_CID_MATCH = "PARENT_CID_MATCH"
PARENT_TO_PRIMARY_CID_MATCH = "PARENT_TO_PRIMARY_CID_MATCH"

DRUG_MATCH_TYPES = [
    SMILES_MATCH,
    SID_MATCH,
    CID_MATCH,
    PARENT_CID_MATCH,
    PARENT_TO_PRIMARY_CID_MATCH,
]


def drug_parent_to_primary_cid_match(drug_a_attrs: dict, drug_b_attrs: dict) -> bool:
    """Check if parent CID of one drug matches primary CID of another."""
    parent_a = drug_a_attrs.get('pubchem_parent_compound_id')
    cid_a = drug_a_attrs.get('pubchem_compound_id')
    parent_b = drug_b_attrs.get('pubchem_parent_compound_id')
    cid_b = drug_b_attrs.get('pubchem_compound_id')

    if parent_a is not None and cid_b is not None:
        if parent_a == cid_b:
            return True
    if parent_b is not None and cid_a is not None:
        if parent_b == cid_a:
            return True
    return False


def not_missing_and_equal(a, b):
    """Check if both values are non-None and equal."""
    return a is not None and b is not None and a == b


DEFAULT_DRUG_MERGE_PREDICATES = (
    (SMILES_MATCH, lambda a, b: not_missing_and_equal(a.get('smiles'), b.get('smiles'))),
    (
        SID_MATCH,
        lambda a, b: not_missing_and_equal(
            a.get('pubchem_substance_id'), b.get('pubchem_substance_id')
        ),
    ),
    (
        CID_MATCH,
        lambda a, b: not_missing_and_equal(
            a.get('pubchem_compound_id'), b.get('pubchem_compound_id')
        ),
    ),
    (
        PARENT_CID_MATCH,
        lambda a, b: not_missing_and_equal(
            a.get('pubchem_parent_compound_id'), b.get('pubchem_parent_compound_id')
        ),
    ),
    (PARENT_TO_PRIMARY_CID_MATCH, drug_parent_to_primary_cid_match),
)

DrugMergePredicateType = Callable[[dict, dict], bool]


def incremental_merge_drugs(
    study_drug_df,
    existing_graph: Optional[nx.Graph] = None,
    existing_drug_id_remap: Optional[Dict[str, str]] = None,
    update_consensus_ids: bool = True,
    merge_predicates: Tuple[
        Tuple[str, DrugMergePredicateType], ...
    ] = DEFAULT_DRUG_MERGE_PREDICATES,
):
    """
    Merge drugs incrementally, preserving or updating existing consensus IDs.

    Args:
        study_drug_df: DataFrame with drug information
        existing_graph: Previously built drug similarity graph
        existing_drug_id_remap: Dict mapping study_drug.id → consensus_id
        update_consensus_ids: If True, merge consensus IDs when components connect.
                             If False, preserve existing consensus IDs without merging.
        merge_predicates: Predicates to determine drug similarity (work with node attribute dicts)

    Returns:
        - updated_graph: nx.Graph
        - drug_id_remap: Dict[str, str] (study_drug.id → consensus_id)
        - consensus_id_is_new: Dict[str, bool] (consensus_id → is_new)
    """

    if existing_graph is None:
        drug_graph = nx.Graph()
    else:
        drug_graph = existing_graph.copy()

    if existing_drug_id_remap is None:
        drug_id_remap = {None: 0, "cls_token1": 1, "cls_token2": 2}
    else:
        drug_id_remap = existing_drug_id_remap.copy()

    # Track which consensus IDs existed before this call
    existing_consensus_ids = set(drug_id_remap.values())

    # Get existing nodes from the graph
    existing_nodes = set(existing_graph.nodes()) if existing_graph else set()

    # Add new study drug nodes with relevant attributes
    new_nodes = set()
    for study_drug in study_drug_df.itertuples():
        if str(study_drug.name) not in drug_graph:
            drug_graph.add_node(
                str(study_drug.name),
                # Display attributes
                original_name=study_drug.name,
                pubchem_name=getattr(study_drug, "pubchem_name", None),
                # Attributes needed for predicates
                smiles=getattr(study_drug, "smiles", None),
                pubchem_substance_id=getattr(study_drug, "pubchem_substance_id", None),
                pubchem_compound_id=getattr(study_drug, "pubchem_compound_id", None),
                pubchem_parent_compound_id=getattr(study_drug, "pubchem_parent_compound_id", None),
                # Metadata
                is_new=True
            )
            new_nodes.add(str(study_drug.name))

    # Test new drugs against ALL drugs in graph
    all_node_ids = list(drug_graph.nodes())

    for match_type, predicate in merge_predicates:
        # Only test pairs where at least one is new
        pairs_to_test = [
            (node_a, node_b)
            for node_a, node_b in combinations(all_node_ids, 2)
            if (node_a in new_nodes or node_b in new_nodes)
            and not drug_graph.has_edge(node_a, node_b)
        ]

        for node_a, node_b in tqdm.tqdm(
            pairs_to_test,
            desc=f"incremental_drug_merging_{match_type}",
        ):
            # Pass node attributes directly to predicates
            attrs_a = drug_graph.nodes[node_a]
            attrs_b = drug_graph.nodes[node_b]

            if predicate(attrs_a, attrs_b):
                drug_graph.add_edge(
                    node_a,
                    node_b,
                    match_type=match_type,
                )

    # Helper to create consensus_id from sorted list of study drug IDs
    # IDs 0, 1, 2 are reserved (padding, CLS obs, CLS perturb)
    def create_consensus_id(dict_ids):
        values = set(dict_ids.values())
        i = 3
        while i in values:
            i += 1
        return i

    # Track which consensus IDs are new
    consensus_id_is_new = {}

    # Find connected components and assign/merge consensus IDs
    for component in nx.connected_components(drug_graph):
        component_drug_ids = list(component)

        # Collect all existing consensus IDs in this component
        component_existing_consensus_ids = set()
        for node_id in component:
            if node_id in drug_id_remap:
                component_existing_consensus_ids.add(drug_id_remap[node_id])

        # Determine the consensus ID for this component
        if len(component_existing_consensus_ids) > 0:
            # Multiple consensus IDs need to be merged
            consensus_id = min(component_existing_consensus_ids)
            consensus_id_is_new[consensus_id] = False

            for node_id in component:
                if node_id in drug_id_remap:
                    if update_consensus_ids:
                        drug_id_remap[node_id] = consensus_id
                else:
                    drug_id_remap[node_id] = consensus_id
        else:
            # No existing consensus ID - create a new one
            consensus_id = create_consensus_id(drug_id_remap)
            consensus_id_is_new[consensus_id] = True

            for node_id in component:
                drug_id_remap[node_id] = consensus_id

    return drug_graph, drug_id_remap, consensus_id_is_new


def load_graph_from_json(json_path):
    """Load a NetworkX graph from JSON format."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()

    # Add nodes with attributes
    for node in data['nodes']:
        node_id = node.pop('id')
        G.add_node(node_id, **node)

    # Add edges with attributes
    for edge in data['edges']:
        source = edge.pop('source')
        target = edge.pop('target')
        G.add_edge(source, target, **edge)

    return G