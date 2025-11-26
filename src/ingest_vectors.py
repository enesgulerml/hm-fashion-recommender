import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import os

# --- SETTINGS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTICLES_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'articles.csv')
COLLECTION_NAME = "hm_items"
MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
BATCH_SIZE = 250  # Load 250 products at a time (Speed optimization)


def ingest_data():
    print("1. Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Model loading
    print(f"2. Loading AI Model: {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)

    # Reading Data
    print(f"3. Reading Data: {ARTICLES_PATH}")
    df = pd.read_csv(ARTICLES_PATH)

    # Data Cleansing (Fill in the blanks)
    df['detail_desc'] = df['detail_desc'].fillna("")
    df['prod_name'] = df['prod_name'].fillna("Unknown Product")

    df_subset = df.head(5000).copy()

    # Convert text to list
    documents = (df_subset['prod_name'] + ": " + df_subset['detail_desc']).tolist()
    ids = df_subset['article_id'].tolist()
    # Metadata
    payloads = df_subset[['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name',
                          'colour_group_name']].to_dict(orient='records')

    # Reset Collection (Start clean every time you run)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,  # Output size of our model
            distance=models.Distance.COSINE  # Similarity measurement
        )
    )
    print(f"4. Collection Created: {COLLECTION_NAME}")

    print("5. Creating and Loading Vectors (This may take a while)...")

    # Batch Processing
    total_batches = len(documents) // BATCH_SIZE + 1

    for i in tqdm(range(0, len(documents), BATCH_SIZE), total=total_batches, desc="Uploading"):
        batch_docs = documents[i: i + BATCH_SIZE]
        batch_ids = ids[i: i + BATCH_SIZE]
        batch_payloads = payloads[i: i + BATCH_SIZE]

        # Text -> Vector
        embeddings = encoder.encode(batch_docs).tolist()

        # Upload to Qdrant
        points = [
            models.PointStruct(
                id=idx,  # Product ID
                vector=vector,
                payload=payload  # Information such as color, type, etc.
            )
            for idx, vector, payload in zip(batch_ids, embeddings, batch_payloads)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(f"\nSUCCESS! {len(documents)} items uploaded to Qdrant.")


if __name__ == "__main__":
    ingest_data()