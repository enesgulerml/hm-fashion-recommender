from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import uvicorn
from contextlib import asynccontextmanager
import os

# --- SETTINGS ---
COLLECTION_NAME = "hm_items"
MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
# --- GLOBAL VARIABLES (To be kept in memory) ---
ml_models = {}


# --- LIFESPAN  ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. START: Load the Model and Database connection once
    print("🚀 Initializing API... Loading model into RAM...")
    ml_models["encoder"] = SentenceTransformer(MODEL_NAME)
    ml_models["qdrant"] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✅ Model and DB Ready!")

    yield

    # 2. CLOSING: Cleaning (If Necessary)
    print("🛑 API Shutting Down... Releasing resources.")
    ml_models.clear()


# Create App
app = FastAPI(title="H&M Fashion Recommender API", lifespan=lifespan)


# --- DATA TYPES (Validation) ---
class SearchRequest(BaseModel):
    text: str
    top_k: int = 5


# --- ENDPOINTS (Doors) ---

@app.get("/")
def home():
    return {"status": "alive", "message": "Welcome to H&M AI Recommender System"}


@app.post("/recommend")
def recommend_products(request: SearchRequest):
    """
    It returns the most similar products based on the text written by the user.
    """
    try:
        # 1. Convert Text to Vector
        encoder = ml_models["encoder"]
        client = ml_models["qdrant"]

        vector = encoder.encode(request.text).tolist()

        # 2. Search Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=request.top_k
        )

        # 3. Convert results to JSON format
        results = []
        for hit in search_result:
            results.append({
                "product_name": hit.payload['prod_name'],
                "similarity_score": round(hit.score, 4),
                "details": hit.payload  # All details such as color, type, etc.
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)