from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# --- SETTINGS ---
COLLECTION_NAME = "hm_items"
MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def search_products(query_text, top_k=5):
    print(f"\n🔎 SEARCHING: '{query_text}'")

    # 1. CONNECTION: Connect to Database and Artificial Intelligence (Model)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    encoder = SentenceTransformer(MODEL_NAME)

    # 2. TRANSLATION: Translate the sentence you are looking for into numbers (Vector)
    query_vector = encoder.encode(query_text).tolist()

    # 3. SEARCH: Ask Qdrant -> What are the 5 closest products to this coordinate?
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    # 4. PRINT RESULTS
    print("-" * 50)
    for i, hit in enumerate(search_result):
        product_name = hit.payload['prod_name']
        score = hit.score  # How similar are they? (0 to 1)

        # Print the result on the screen
        print(f"{i + 1}. {product_name} (Similarity: {score:.4f})")
    print("-" * 50)


if __name__ == "__main__":
    # Will it find beach products even if they don't say "Beach" in them?
    search_products("I want to go to the beach")

    # Will it find leggings/shorts even if they don't say "Sport"?
    search_products("Something for running and gym")

    # Let's ask for "Formal" clothing
    search_products("Elegant dress for a dinner party")