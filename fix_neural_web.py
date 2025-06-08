#!/usr/bin/env python3
"""
Script to fix the neural_web.json file by removing concepts with null IDs
"""

import json
import uuid
from pathlib import Path

def fix_neural_web_json():
    """Fix the neural web JSON file by removing/fixing null ID concepts"""
    neural_web_path = Path("data/neural_web.json")
    
    if not neural_web_path.exists():
        print("Neural web file not found")
        return
    
    # Load the data
    with open(neural_web_path, 'r') as f:
        data = json.load(f)
    
    print(f"Original concepts count: {len(data.get('concepts', {}))}")
    print(f"Original connections count: {len(data.get('connections', {}))}")
    
    # Fix concepts with null IDs
    concepts = data.get('concepts', {})
    fixed_concepts = {}
    id_mapping = {}  # Map old IDs to new IDs
    
    for concept_key, concept_data in concepts.items():
        concept_id = concept_data.get('id')
        
        if concept_id is None or concept_id == 'null' or concept_id == '':
            # Generate a new ID for this concept
            new_id = f"concept_{uuid.uuid4().hex[:8]}"
            print(f"Fixing concept with null ID '{concept_key}' -> '{new_id}'")
            concept_data['id'] = new_id
            fixed_concepts[new_id] = concept_data
            id_mapping[concept_key] = new_id
            id_mapping[concept_id] = new_id  # Map both key and id
        else:
            fixed_concepts[concept_key] = concept_data
            id_mapping[concept_key] = concept_id
    
    # Fix connections that reference null IDs
    connections = data.get('connections', {})
    fixed_connections = {}
    
    for conn_key, conn_data in connections.items():
        source_id = conn_data.get('source_id')
        target_id = conn_data.get('target_id')
        
        # Skip connections with null/invalid IDs
        if (source_id is None or source_id == 'null' or source_id == '' or
            target_id is None or target_id == 'null' or target_id == ''):
            print(f"Skipping connection with invalid IDs: {source_id} -> {target_id}")
            continue
        
        # Update IDs if they were remapped
        if source_id in id_mapping:
            conn_data['source_id'] = id_mapping[source_id]
        if target_id in id_mapping:
            conn_data['target_id'] = id_mapping[target_id]
        
        # Only keep connections where both concepts exist
        if (conn_data['source_id'] in [c['id'] for c in fixed_concepts.values()] and
            conn_data['target_id'] in [c['id'] for c in fixed_concepts.values()]):
            new_conn_key = f"{conn_data['source_id']}->{conn_data['target_id']}"
            fixed_connections[new_conn_key] = conn_data
        else:
            print(f"Skipping connection: {conn_data['source_id']} -> {conn_data['target_id']} (missing concepts)")
    
    # Update the data
    data['concepts'] = fixed_concepts
    data['connections'] = fixed_connections
    
    print(f"Fixed concepts count: {len(fixed_concepts)}")
    print(f"Fixed connections count: {len(fixed_connections)}")
    
    # Save the fixed data
    with open(neural_web_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Neural web file has been fixed!")

if __name__ == "__main__":
    fix_neural_web_json()
