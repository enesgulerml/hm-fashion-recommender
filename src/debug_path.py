import qdrant_client
import os

print("-" * 30)
print(f"1. OKUNAN DOSYA YOLU: {qdrant_client.__file__}")
print("-" * 30)

# İçindeki özellikleri listele
print("2. MODÜL İÇERİĞİ:")
print(dir(qdrant_client))
print("-" * 30)