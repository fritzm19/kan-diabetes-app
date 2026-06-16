# test_load.py

from config import load_system_prerequisites

resources = load_system_prerequisites()

print("Model loaded")
print(resources.keys())