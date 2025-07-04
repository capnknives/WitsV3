import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback

try:
    import core.neural_memory_backend as nmb
    print(f"Module imported: {nmb}")
    print(f"Module dir: {dir(nmb)}")
    if hasattr(nmb, 'NeuralMemoryBackend'):
        print("NeuralMemoryBackend class found!")
    else:
        print("NeuralMemoryBackend class NOT found!")
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()
