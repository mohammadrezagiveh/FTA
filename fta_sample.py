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
    # Leaf/root nodes with unconditional probabilities (per your chosen time window)
    "roots": {
        "political intent": {"p": 0.75},
        "Recover HEU": {"p": 0.25},
    },

    # Simple dependency: child depends on a single parent
    # child: { parent: <name>, p_if_true: P(child=1|parent=1), p_if_false: P(child=1|parent=0) }
    "conditionals": {
        "produce WGU": {"parent": "Recover HEU", "p_if_true": 0.50, "p_if_false": 0.0},
    },

    # Boolean gates: deterministic AND/OR
    # gate_node: { type: "AND"|"OR", inputs: [node1, node2, ...] }
    "gates": {
        "Iran builds a nuclear weapon": {"type": "AND", "inputs": ["political intent", "produce WGU"]},
    },

    # Query target
    "top": "Iran builds a nuclear weapon",
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
# 4) RUN IT
# ----------------------------
def main():
    model = build_model(SPEC)
    infer = VariableElimination(model)
    top = SPEC["top"]
    result = infer.query([top])
    print(result)


if __name__ == "__main__":
    main()