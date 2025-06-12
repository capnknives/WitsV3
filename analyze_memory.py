#!/usr/bin/env python3

import json
import os
from collections import Counter

def analyze_memory():
    memory_file = 'data/wits_memory.json'
    if not os.path.exists(memory_file):
        print('Memory file not found')
        return

    with open(memory_file, 'r') as f:
        data = json.load(f)

    print(f'Total memory segments: {len(data)}')
    print(f'File size: {os.path.getsize(memory_file) / 1024 / 1024:.2f} MB')

    if data:
        print(f'First entry timestamp: {data[0].get("timestamp", "unknown")}')
        print(f'Last entry timestamp: {data[-1].get("timestamp", "unknown")}')

        # Count types
        types = Counter(item.get('type', 'unknown') for item in data)

        print('\nEntry types:')
        for t, count in types.most_common(10):
            print(f'  {t}: {count}')

        # Count sources
        sources = Counter(item.get('source', 'unknown') for item in data)
        print('\nEntry sources:')
        for s, count in sources.most_common(10):
            print(f'  {s}: {count}')

if __name__ == "__main__":
    analyze_memory()
