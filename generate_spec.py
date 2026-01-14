#!/usr/bin/env python3
"""
Generate FTA SPEC dictionary from Excel file.
This script reads a fault tree analysis table and converts it to the SPEC format
required for Bayesian network modeling.
"""

import pandas as pd
from typing import Dict, Any, List
import re
from datetime import datetime


def parse_dependencies(dep_value) -> List[int]:
    """
    Parse dependency values which might be:
    - A dash '-' for no dependencies
    - A single integer
    - A comma-separated list of integers
    - A date that Excel misinterpreted (we'll extract meaningful codes from dates)
    
    Date patterns observed:
    - 2026-MM-DD: codes are MM and DD
    - 2014-10-07: codes are 4, 7, 10 (special encoding)
    """
    if pd.isna(dep_value) or dep_value == '-':
        return []
    
    # If it's already a number
    if isinstance(dep_value, (int, float)):
        return [int(dep_value)]
    
    # If it's a datetime object (Excel parsed it as date)
    if isinstance(dep_value, (pd.Timestamp, datetime)):
        # Extract codes from date components
        month = dep_value.month
        day = dep_value.day
        year = dep_value.year
        
        # Special handling for different patterns
        if year == 2026:
            # Pattern: 2026-MM-DD where MM and DD are the codes
            # Return in order: MM, DD
            codes = []
            if month > 0:
                codes.append(month)
            if day > 0:
                codes.append(day)
            return codes
        elif year == 2014 and month == 10 and day == 7:
            # Special case: 2014-10-07 encodes codes 4, 7, 10
            # (for HEU to WGU AND gate)
            return [4, 7, 10]
        else:
            # Generic fallback: extract month and day
            codes = []
            if month > 0:
                codes.append(month)
            if day > 0:
                codes.append(day)
            return codes
    
    # If it's a string
    dep_str = str(dep_value).strip()
    if not dep_str or dep_str == '-':
        return []
    
    # Try to extract all numbers from the string (handles comma-separated)
    numbers = re.findall(r'\d+', dep_str)
    return [int(n) for n in numbers]


def generate_spec_from_excel(file_path: str, top_event_code: int = 30) -> Dict[str, Any]:
    """
    Generate a SPEC dictionary from an Excel fault tree analysis table.
    
    Args:
        file_path: Path to the Excel file
        top_event_code: Code number of the top event (default: 30 - Nuclear Weapons Assembly)
    
    Returns:
        Dictionary in SPEC format
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Initialize SPEC structure
    spec = {
        "roots": {},
        "conditionals": {},
        "gates": {},
        "top": None
    }
    
    # Create a mapping from code to title for easy lookup
    code_to_title = {}
    for _, row in df.iterrows():
        code = int(row['code'])
        title = str(row['title']).strip()
        code_to_title[code] = title
    
    # Process each row
    for _, row in df.iterrows():
        code = int(row['code'])
        title = str(row['title']).strip()
        node_type = str(row['type']).strip().lower()
        prob = row['probability']
        dependencies = parse_dependencies(row['dependent on'])
        
        # Handle root events (unconditional probabilities)
        if node_type == 'root event':
            prob_value = 0.5  # default placeholder
            if pd.notna(prob):
                prob_str = str(prob).strip()
                if prob_str != '?' and prob_str != '':
                    try:
                        prob_value = float(prob_str)
                    except ValueError:
                        print(f"WARNING: Code {code} '{title}' has invalid probability '{prob_str}'. Using 0.5 as placeholder.")
                else:
                    print(f"WARNING: Code {code} '{title}' has missing probability (marked as '?'). Using 0.5 as placeholder.")
            else:
                print(f"WARNING: Code {code} '{title}' has missing probability. Using 0.5 as placeholder.")
            
            spec["roots"][title] = {"p": prob_value}
        
        # Handle conditional events (events dependent on a single parent)
        elif node_type == 'event':
            if len(dependencies) == 1:
                parent_code = dependencies[0]
                parent_title = code_to_title.get(parent_code, f"UNKNOWN_CODE_{parent_code}")
                
                # Check if probability is valid
                prob_value = 0.5  # default placeholder
                if pd.notna(prob):
                    prob_str = str(prob).strip()
                    if prob_str != '?' and prob_str != '':
                        try:
                            prob_value = float(prob_str)
                        except ValueError:
                            print(f"WARNING: Code {code} '{title}' has invalid probability '{prob_str}'. Using 0.5 as placeholder.")
                    else:
                        print(f"WARNING: Code {code} '{title}' has missing probability (marked as '?'). Using 0.5 as placeholder.")
                else:
                    print(f"WARNING: Code {code} '{title}' has missing probability. Using 0.5 as placeholder.")
                
                # This is P(child=1|parent=1)
                # For simplicity, assuming P(child=1|parent=0) = 0
                spec["conditionals"][title] = {
                    "parent": parent_title,
                    "p_if_true": prob_value,
                    "p_if_false": 0.0
                }
            else:
                print(f"WARNING: Event {code} '{title}' has {len(dependencies)} dependencies, expected 1. Skipping.")
        
        # Handle gates (AND/OR)
        elif node_type in ['and', 'or']:
            gate_inputs = []
            for dep_code in dependencies:
                dep_title = code_to_title.get(dep_code, f"UNKNOWN_CODE_{dep_code}")
                gate_inputs.append(dep_title)
            
            spec["gates"][title] = {
                "type": node_type.upper(),
                "inputs": gate_inputs
            }
    
    # Set the top event
    spec["top"] = code_to_title.get(top_event_code, "Nuclear Weapons Assembly")
    
    return spec


def format_spec_as_code(spec: Dict[str, Any]) -> str:
    """
    Format the SPEC dictionary as Python code string.
    """
    lines = ["SPEC: Dict[str, Any] = {"]
    
    # Roots
    lines.append("    # Leaf/root nodes with unconditional probabilities")
    lines.append("    \"roots\": {")
    for i, (node, data) in enumerate(spec["roots"].items()):
        comma = "," if i < len(spec["roots"]) - 1 else ""
        lines.append(f"        \"{node}\": {{\"p\": {data['p']}}}{comma}")
    lines.append("    },")
    lines.append("")
    
    # Conditionals
    lines.append("    # Simple dependency: child depends on a single parent")
    lines.append("    # child: { parent: <name>, p_if_true: P(child=1|parent=1), p_if_false: P(child=1|parent=0) }")
    lines.append("    \"conditionals\": {")
    for i, (node, data) in enumerate(spec["conditionals"].items()):
        comma = "," if i < len(spec["conditionals"]) - 1 else ""
        lines.append(f"        \"{node}\": {{\"parent\": \"{data['parent']}\", "
                    f"\"p_if_true\": {data['p_if_true']}, "
                    f"\"p_if_false\": {data['p_if_false']}}}{comma}")
    lines.append("    },")
    lines.append("")
    
    # Gates
    lines.append("    # Boolean gates: deterministic AND/OR")
    lines.append("    # gate_node: { type: \"AND\"|\"OR\", inputs: [node1, node2, ...] }")
    lines.append("    \"gates\": {")
    for i, (node, data) in enumerate(spec["gates"].items()):
        comma = "," if i < len(spec["gates"]) - 1 else ""
        inputs_str = ", ".join([f"\"{inp}\"" for inp in data["inputs"]])
        lines.append(f"        \"{node}\": {{\"type\": \"{data['type']}\", \"inputs\": [{inputs_str}]}}{comma}")
    lines.append("    },")
    lines.append("")
    
    # Top event
    lines.append("    # Query target")
    lines.append(f"    \"top\": \"{spec['top']}\",")
    lines.append("}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Path to the Excel file
    excel_file = "fta_table.xlsx"
    
    # Generate the SPEC dictionary
    # Note: Code 30 is "Nuclear Weapons Assembly" which appears to be the top event
    spec = generate_spec_from_excel(excel_file, top_event_code=30)
    
    # Format as code
    spec_code = format_spec_as_code(spec)
    
    # Print the result
    print("\n" + "="*80)
    print("GENERATED SPEC DICTIONARY")
    print("="*80 + "\n")
    print(spec_code)
    
    # Save to file
    output_file = "fta_spec.py"
    with open(output_file, 'w') as f:
        f.write("from typing import Dict, Any\n\n")
        f.write(spec_code)
    
    print("\n" + "="*80)
    print(f"SPEC dictionary has been saved to: {output_file}")
    print("="*80)