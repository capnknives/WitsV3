#!/usr/bin/env python3
"""Fix data/neural_web.json by removing concepts with null IDs."""

import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fix_neural_web_json() -> None:
    """Fix the neural web JSON file by removing/fixing null ID concepts."""
    neural_web_path = PROJECT_ROOT / "data" / "neural_web.json"

    if not neural_web_path.exists():
        print("Neural web file not found")
        return

    with open(neural_web_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Original concepts count: {len(data.get('concepts', {}))}")
    print(f"Original connections count: {len(data.get('connections', {}))}")

    concepts = data.get("concepts", {})
    fixed_concepts = {}
    id_mapping = {}

    for concept_key, concept_data in concepts.items():
        concept_id = concept_data.get("id")

        if concept_id is None or concept_id == "null" or concept_id == "":
            new_id = f"concept_{uuid.uuid4().hex[:8]}"
            print(f"Fixing concept with null ID '{concept_key}' -> '{new_id}'")
            concept_data["id"] = new_id
            fixed_concepts[new_id] = concept_data
            id_mapping[concept_key] = new_id
            id_mapping[concept_id] = new_id
        else:
            fixed_concepts[concept_key] = concept_data
            id_mapping[concept_key] = concept_id

    connections = data.get("connections", {})
    fixed_connections = {}

    for _conn_key, conn_data in connections.items():
        source_id = conn_data.get("source_id")
        target_id = conn_data.get("target_id")

        if (
            source_id is None
            or source_id == "null"
            or source_id == ""
            or target_id is None
            or target_id == "null"
            or target_id == ""
        ):
            print(f"Skipping connection with invalid IDs: {source_id} -> {target_id}")
            continue

        if source_id in id_mapping:
            conn_data["source_id"] = id_mapping[source_id]
        if target_id in id_mapping:
            conn_data["target_id"] = id_mapping[target_id]

        concept_ids = [c["id"] for c in fixed_concepts.values()]
        if (
            conn_data["source_id"] in concept_ids
            and conn_data["target_id"] in concept_ids
        ):
            new_conn_key = f"{conn_data['source_id']}->{conn_data['target_id']}"
            fixed_connections[new_conn_key] = conn_data
        else:
            print(
                f"Skipping connection: {conn_data['source_id']} -> "
                f"{conn_data['target_id']} (missing concepts)"
            )

    data["concepts"] = fixed_concepts
    data["connections"] = fixed_connections

    print(f"Fixed concepts count: {len(fixed_concepts)}")
    print(f"Fixed connections count: {len(fixed_connections)}")

    with open(neural_web_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Neural web file has been fixed!")


if __name__ == "__main__":
    fix_neural_web_json()
