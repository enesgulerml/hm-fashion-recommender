import qdrant_client
import os

print("-" * 30)
print(f"1. READ FILE PATH: {qdrant_client.__file__}")
print("-" * 30)

print("2. MODULE CONTENT:")
print(dir(qdrant_client))
print("-" * 30)