from typing import Dict, Any

SPEC: Dict[str, Any] = {
    # Leaf/root nodes with unconditional probabilities
    "roots": {
        "HEU is recovered from rubble": {"p": 0.75},
        "HEU has been moved to a safe location before the war (excluding Esfahan tunnel)": {"p": 0.2},
        "Iran can build more centrifuges for small plant  (defined as up to 5000 SWU/ 1000 centrifuges) or as few as 500 centrifuges": {"p": 0.75},
        "Iran has stockpiles of centrifuges intact for small plant, such as inside Esfahan tunnel complex or at an undeclared": {"p": 0.6},
        "Iran can build more centrifuges for big plant": {"p": 0.1},
        "Iran has stockpiles of centrifuge intact for big plant": {"p": 0.05},
        "Natural Uranium is available": {"p": 0.5},
        "Enrichment related equipments are available": {"p": 0.6},
        "Aluminum shell with channels": {"p": 0.5},
        "High purity PETN": {"p": 0.5},
        "Main charge": {"p": 0.95},
        "Build a neutron initiator or have one": {"p": 0.8},
        "Other components such as EBW, Capacitors, Switches, flyer plate etc.": {"p": 0.9},
        "Assemble the Surrogate Core": {"p": 1.0},
        "Iran can conduct a successful cold test": {"p": 0.8}
    },

    # Simple dependency: child depends on a single parent
    # child: { parent: <name>, p_if_true: P(child=1|parent=1), p_if_false: P(child=1|parent=0) }
    "conditionals": {
        "HEU UF6 is usable and not spiked, and sufficient to account for processing losses and accidents": {"parent": "What happened to HEU", "p_if_true": 0.5, "p_if_false": 0.0},
        "Centrifuges are available for a small plant": {"parent": "Small Centrifuge OR", "p_if_true": 1.0, "p_if_false": 0.0},
        "HEU is moved to a secret location": {"parent": "HEU UF6 is usable and not spiked, and sufficient to account for processing losses and accidents", "p_if_true": 0.5, "p_if_false": 0.0},
        "Centrifuges are available for a big plant": {"parent": "Big Centrifuge OR", "p_if_true": 1.0, "p_if_false": 0.0},
        "Converting natural uranium oxide to natural UF6": {"parent": "Natural Uranium is available", "p_if_true": 0.1, "p_if_false": 0.0},
        "The actual enrichment of natural uranium to WGU in a large enrichment plant within several months": {"parent": "NU to WGU AND", "p_if_true": 0.5, "p_if_false": 0.0},
        "The actual enrichment of HEU to WGU in a small enrichment plant in less than six months": {"parent": "HEU to WGU AND", "p_if_true": 0.8, "p_if_false": 0.0},
        "WGU UF6 to UF4": {"parent": "UF6 to UF4 OR", "p_if_true": 0.5, "p_if_false": 0.0},
        "WGUF4 to WG uranium metal": {"parent": "WGU UF6 to UF4", "p_if_true": 0.8, "p_if_false": 0.0},
        "Melting and pouring into a mold": {"parent": "WGUF4 to WG uranium metal", "p_if_true": 0.5, "p_if_false": 0.0},
        "Molding, Finishing and machining the metal core": {"parent": "Melting and pouring into a mold", "p_if_true": 0.8, "p_if_false": 0.0},
        "Coating the metal core": {"parent": "Molding, Finishing and machining the metal core", "p_if_true": 0.8, "p_if_false": 0.0},
        "Shockwave detonator (MPI)": {"parent": "Shockwave AND", "p_if_true": 1.0, "p_if_false": 0.0},
        "Nuclear Weapons Assembly": {"parent": "NW Assembly AND", "p_if_true": 0.95, "p_if_false": 0.0}
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
    "top": "Nuclear Weapons Assembly",
}