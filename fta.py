from __future__ import annotations

from itertools import product
from typing import Dict, Any, List, Tuple

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ----------------------------
# 1) EDIT THIS SPEC FOR YOUR DIAGRAM
# ----------------------------
SPEC: Dict[str, Any] = {
    # Leaf/root nodes with unconditional probabilities
    "roots": {
        "HEU is recovered from rubble": {"p": 1},
        "HEU has been moved to a safe location before the war (excluding Esfahan tunnel)": {"p": 1},
        "Iran can build more centrifuges for small plant  (defined as up to 5000 SWU/ 1000 centrifuges) or as few as 500 centrifuges": {"p": 0.75},
        "Iran has stockpiles of centrifuges intact for small plant, such as inside Esfahan tunnel complex or at an undeclared": {"p": 0.6},
        "Iran can build more centrifuges for big plant": {"p": 0.1},
        "Iran has stockpiles of centrifuge intact for big plant": {"p": 0.05},
        "Natural Uranium is available": {"p": 1},
        "Enrichment related equipments are available": {"p": 0.6},
        "Aluminum shell with channels": {"p": 0.5},
        "High purity PETN": {"p": 0.5},
        "Main charge": {"p": 0.95},
        "Build a neutron initiator or have one": {"p": 0.8},
        "Other components such as EBW, Capacitors, Switches, flyer plate etc.": {"p": 0.9},
        "Assemble the Surrogate Core": {"p": 1.0},
    },

    # Simple dependency: child depends on a single parent
    # child: { parent: <n>, p_if_true: P(child=1|parent=1), p_if_false: P(child=1|parent=0) }
    "conditionals": {
        "HEU UF6 is usable and not spiked, and sufficient to account for processing losses and accidents": {"parent": "What happened to HEU", "p_if_true": 0.5, "p_if_false": 0.0},
        "Centrifuges are available for a small plant": {"parent": "Small Centrifuge OR", "p_if_true": 1.0, "p_if_false": 0.0},
        "HEU is moved to a secret location": {"parent": "HEU UF6 is usable and not spiked, and sufficient to account for processing losses and accidents", "p_if_true": 1, "p_if_false": 0.0},
        "Centrifuges are available for a big plant": {"parent": "Big Centrifuge OR", "p_if_true": 1.0, "p_if_false": 0.0},
        "Converting natural uranium oxide to natural UF6": {"parent": "Natural Uranium is available", "p_if_true": 0.1, "p_if_false": 0.0},
        "The actual enrichment of natural uranium to WGU in a large enrichment plant within several months": {"parent": "NU to WGU AND", "p_if_true": 0.5, "p_if_false": 0.0},
        "The actual enrichment of HEU to WGU in a small enrichment plant in less than six months": {"parent": "HEU to WGU AND", "p_if_true": 0.8, "p_if_false": 0.0},
        "WGU UF6 to UF4": {"parent": "UF6 to UF4 OR", "p_if_true": 0.9, "p_if_false": 0.0},
        "WGUF4 to WG uranium metal": {"parent": "WGU UF6 to UF4", "p_if_true": 0.8, "p_if_false": 0.0},
        "Melting and pouring into a mold": {"parent": "WGUF4 to WG uranium metal", "p_if_true": 0.5, "p_if_false": 0.0},
        "Molding, Finishing and machining the metal core": {"parent": "Melting and pouring into a mold", "p_if_true": 0.8, "p_if_false": 0.0},
        "Coating the metal core": {"parent": "Molding, Finishing and machining the metal core", "p_if_true": 0.8, "p_if_false": 0.0},
        "Shockwave detonator (MPI)": {"parent": "Shockwave AND", "p_if_true": 1.0, "p_if_false": 0.0},
        "Nuclear Weapons Assembly": {"parent": "NW Assembly AND", "p_if_true": 0.95, "p_if_false": 0.0},
        "Iran can conduct a successful cold test": {"parent": "Cold Test AND", "p_if_true": 0.95, "p_if_false": 0.0},
        "Iran builds a non missile deliverable nuclear weapon": {"parent": "Nuclear Weapons Assembly", "p_if_true": 1, "p_if_false": 0.0}
    },

    # Boolean gates: deterministic AND/OR
    # gate_node: { type: "AND"|"OR", inputs: [node1, node2, ...] }
    "gates": {
        "What happened to HEU": {"type": "OR", "inputs": ["HEU is recovered from rubble", "HEU has been moved to a safe location before the war (excluding Esfahan tunnel)"]},
        "Small Centrifuge OR": {"type": "OR", "inputs": ["Iran can build more centrifuges for small plant  (defined as up to 5000 SWU/ 1000 centrifuges) or as few as 500 centrifuges", "Iran has stockpiles of centrifuges intact for small plant, such as inside Esfahan tunnel complex or at an undeclared"]},
        "Big Centrifuge OR": {"type": "OR", "inputs": ["Iran can build more centrifuges for big plant", "Iran has stockpiles of centrifuge intact for big plant"]},
        "HEU to WGU AND": {"type": "AND", "inputs": ["HEU UF6 is usable and not spiked, and sufficient to account for processing losses and accidents", "Centrifuges are available for a small plant", "HEU is moved to a secret location"]},
        "NU to WGU AND": {"type": "AND", "inputs": ["Enrichment related equipments are available", "Centrifuges are available for a big plant", "Converting natural uranium oxide to natural UF6"]},
        "UF6 to UF4 OR": {"type": "OR", "inputs": ["The actual enrichment of natural uranium to WGU in a large enrichment plant within several months", "The actual enrichment of HEU to WGU in a small enrichment plant in less than six months"]},
        "Shockwave AND": {"type": "AND", "inputs": ["Aluminum shell with channels", "High purity PETN"]},
        "Cold Test AND": {"type": "AND", "inputs": ["Shockwave detonator (MPI)", "Main charge", "Build a neutron initiator or have one", "Other components such as EBW, Capacitors, Switches, flyer plate etc.", "Assemble the Surrogate Core"]},
        "NW Assembly AND": {"type": "AND", "inputs": ["Iran can conduct a successful cold test", "Coating the metal core"]}
    },

    # Query target
    "top": "Iran builds a non missile deliverable nuclear weapon",
}


# ----------------------------
# 2) CPD BUILDERS
# ----------------------------
def make_root_cpd(node: str, p: float) -> TabularCPD:
    # Binary variable: 0=False, 1=True
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Invalid probability for {node}: {p}")
    return TabularCPD(node, 2, [[1 - p], [p]])


def make_conditional_cpd(
    child: str,
    parent: str,
    p_if_true: float,
    p_if_false: float = 0.0,
) -> TabularCPD:
    # Columns correspond to parent=0, parent=1
    for name, p in [("p_if_true", p_if_true), ("p_if_false", p_if_false)]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid {name} for {child}|{parent}: {p}")

    return TabularCPD(
        variable=child,
        variable_card=2,
        values=[
            [1 - p_if_false, 1 - p_if_true],  # child=0
            [p_if_false,     p_if_true],      # child=1
        ],
        evidence=[parent],
        evidence_card=[2],
    )


def make_gate_cpd(node: str, parents: List[str], gate_type: str) -> TabularCPD:
    if len(parents) < 1:
        raise ValueError(f"Gate {node} must have at least 1 input.")

    cols = list(product([0, 1], repeat=len(parents)))  # all parent configurations

    gt = gate_type.upper()
    if gt == "AND":
        out_true = [1 if all(cfg) else 0 for cfg in cols]
    elif gt == "OR":
        out_true = [1 if any(cfg) else 0 for cfg in cols]
    else:
        raise ValueError(f"Gate {node}: gate_type must be 'AND' or 'OR', got {gate_type!r}")

    out_false = [1 - x for x in out_true]
    return TabularCPD(
        variable=node,
        variable_card=2,
        values=[out_false, out_true],
        evidence=parents,
        evidence_card=[2] * len(parents),
    )


# ----------------------------
# 3) MODEL BUILDER
# ----------------------------
def build_model(spec: Dict[str, Any]) -> DiscreteBayesianNetwork:
    edges: List[Tuple[str, str]] = []
    cpds: List[TabularCPD] = []

    roots = spec.get("roots", {})
    conditionals = spec.get("conditionals", {})
    gates = spec.get("gates", {})

    # Build edges for conditionals (parent -> child)
    for child, cfg in conditionals.items():
        parent = cfg["parent"]
        edges.append((parent, child))

    # Build edges for gates (each input -> gate)
    for gate_node, cfg in gates.items():
        for inp in cfg["inputs"]:
            edges.append((inp, gate_node))

    model = DiscreteBayesianNetwork(edges)

    # Root CPDs
    for node, cfg in roots.items():
        cpds.append(make_root_cpd(node, cfg["p"]))

    # Conditional CPDs
    for child, cfg in conditionals.items():
        cpds.append(
            make_conditional_cpd(
                child=child,
                parent=cfg["parent"],
                p_if_true=cfg["p_if_true"],
                p_if_false=cfg.get("p_if_false", 0.0),
            )
        )

    # Gate CPDs
    for gate_node, cfg in gates.items():
        cpds.append(
            make_gate_cpd(
                node=gate_node,
                parents=cfg["inputs"],
                gate_type=cfg["type"],
            )
        )

    model.add_cpds(*cpds)

    # Helpful validation
    if not model.check_model():
        raise ValueError(
            "Model check failed. Common causes:\n"
            "- Missing CPD for a node\n"
            "- A node has parents in the graph that don't match its CPD evidence\n"
            "- Typos in node names between roots/conditionals/gates\n"
        )

    return model


# ----------------------------
# 4) QUERY ALL EVENT PROBABILITIES
# ----------------------------
def query_all_events(model: DiscreteBayesianNetwork, spec: Dict[str, Any]) -> Dict[str, float]:
    """
    Query the probability P(event=1) for all events (roots and conditionals).
    Returns a dictionary mapping event name to probability.
    """
    infer = VariableElimination(model)
    probabilities = {}
    
    # Get all event nodes (roots + conditionals, excluding gates)
    roots = spec.get("roots", {})
    conditionals = spec.get("conditionals", {})
    
    all_events = list(roots.keys()) + list(conditionals.keys())
    
    print(f"\nQuerying {len(all_events)} events...")
    for i, event in enumerate(all_events, 1):
        if i % 5 == 0:
            print(f"  Processed {i}/{len(all_events)} events...")
        result = infer.query([event])
        # Extract P(event=1) from the result
        prob = result.values[1]  # index 1 corresponds to event=True
        probabilities[event] = prob
    
    return probabilities


def print_event_probabilities(probabilities: Dict[str, float], spec: Dict[str, Any]):
    """
    Print event probabilities in a structured format.
    """
    roots = spec.get("roots", {})
    conditionals = spec.get("conditionals", {})
    top = spec.get("top")
    
    print("\n" + "="*100)
    print("FAULT TREE ANALYSIS - EVENT PROBABILITIES")
    print("="*100 + "\n")
    
    # Print root events
    print("ROOT EVENTS (Independent Base Events):")
    print("-" * 100)
    print(f"{'Probability':<12} | {'Event Name'}")
    print("-" * 100)
    for event in roots.keys():
        prob = probabilities[event]
        print(f"  {prob:>8.4f}   | {event}")
    
    print("\n" + "="*100 + "\n")
    
    # Print conditional events
    print("CONDITIONAL EVENTS (Dependent on Parent Events/Gates):")
    print("-" * 100)
    print(f"{'Probability':<12} | {'Event Name'}")
    print("-" * 100)
    for event in conditionals.keys():
        prob = probabilities[event]
        marker = " *** TARGET EVENT ***" if event == top else ""
        print(f"  {prob:>8.4f}   | {event}{marker}")
    
    print("\n" + "="*100)
    print(f"TARGET EVENT: {top}")
    print(f"TARGET PROBABILITY: {probabilities.get(top, 0.0):.6f} ({probabilities.get(top, 0.0)*100:.4f}%)")
    print("="*100 + "\n")


def print_summary_statistics(probabilities: Dict[str, float], spec: Dict[str, Any]):
    """
    Print summary statistics about the probabilities.
    """
    roots = spec.get("roots", {})
    conditionals = spec.get("conditionals", {})
    
    root_probs = [probabilities[event] for event in roots.keys()]
    cond_probs = [probabilities[event] for event in conditionals.keys()]
    
    print("\nSUMMARY STATISTICS:")
    print("-" * 100)
    print(f"Total Root Events: {len(root_probs)}")
    print(f"  Average Probability: {sum(root_probs)/len(root_probs):.4f}")
    print(f"  Min Probability: {min(root_probs):.4f}")
    print(f"  Max Probability: {max(root_probs):.4f}")
    print()
    print(f"Total Conditional Events: {len(cond_probs)}")
    print(f"  Average Probability: {sum(cond_probs)/len(cond_probs):.4f}")
    print(f"  Min Probability: {min(cond_probs):.4f}")
    print(f"  Max Probability: {max(cond_probs):.4f}")
    print("-" * 100)


# ----------------------------
# 5) RUN IT
# ----------------------------
def main():
    print("\n" + "="*100)
    print("BUILDING BAYESIAN NETWORK MODEL")
    print("="*100)
    
    model = build_model(SPEC)
    print("\n✓ Model built successfully!")
    print(f"  Total nodes: {len(model.nodes())}")
    print(f"  Total edges: {len(model.edges())}")
    print(f"  Root events: {len(SPEC.get('roots', {}))}")
    print(f"  Conditional events: {len(SPEC.get('conditionals', {}))}")
    print(f"  Gates: {len(SPEC.get('gates', {}))}")
    
    # Query all event probabilities
    probabilities = query_all_events(model, SPEC)
    
    # Print results
    print_event_probabilities(probabilities, SPEC)
    
    # Print summary statistics
    print_summary_statistics(probabilities, SPEC)
    
    # Also print the detailed result for the target event
    print("\nDETAILED QUERY RESULT FOR TARGET EVENT:")
    print("-" * 100)
    infer = VariableElimination(model)
    top = SPEC["top"]
    result = infer.query([top])
    print(result)
    print()


if __name__ == "__main__":
    main()