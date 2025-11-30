from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import uvicorn
from contextlib import asynccontextmanager
import os
import redis  # Redis library
import json  # To handle data serialization

# --- SETTINGS ---
COLLECTION_NAME = "hm_items"
MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333

# --- GLOBAL VARIABLES ---
ml_models = {}
redis_client = None  # Global variable for Redis connection


# --- LIFESPAN (Startup & Shutdown Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. INITIALIZE REDIS CONNECTION
    global redis_client
    try:
        # Get Redis host from environment variable (defined in docker-compose)
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
        redis_client.ping()  # Check connection
        print(f"✅ Redis Connection Established on {redis_host}!")
    except Exception as e:
        print(f"⚠️ Redis Connection Failed: {e}. Caching will be disabled.")
        redis_client = None

    # 2. LOAD AI MODEL AND DATABASE CONNECTION
    print("🚀 Initializing API... Loading embedding model into RAM...")
    ml_models["encoder"] = SentenceTransformer(MODEL_NAME)
    ml_models["qdrant"] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✅ Model and Qdrant DB Ready!")

    yield

    # 3. SHUTDOWN CLEANUP
    print("🛑 API Shutting Down... Releasing resources.")
    ml_models.clear()


# Create App
app = FastAPI(title="H&M Fashion Recommender API", lifespan=lifespan)


# --- DATA TYPES (Validation) ---
class SearchRequest(BaseModel):
    text: str
    top_k: int = 5


# --- ENDPOINTS ---

@app.get("/")
def home():
    # Include Redis status in health check
    redis_status = "active" if redis_client and redis_client.ping() else "inactive"
    return {
        "status": "alive",
        "message": "Welcome to H&M AI Recommender System",
        "redis_cache": redis_status
    }


@app.post("/recommend")
def recommend_products(request: SearchRequest):
    """
    Returns the most similar fashion items based on the user's text query.
    Uses Redis for caching to minimize latency on repeated queries.
    """
    try:
        # --- CACHING LOGIC START ---

        # 1. Generate a unique Cache Key
        # We normalize text (lowercase + strip) to increase cache hit rate.
        # Example Key: "search:red dress:5"
        normalized_text = request.text.lower().strip()
        cache_key = f"search:{normalized_text}:{request.top_k}"

        # 2. Check Redis (Cache Hit)
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                print(f"⚡ CACHE HIT: Returning results for '{normalized_text}' from Redis!")
                return json.loads(cached_result)

        # 3. Cache Miss: Perform Vector Search (Heavy Computation)
        print(f"🐢 CACHE MISS: Processing '{normalized_text}' via Qdrant and Model...")

        encoder = ml_models["encoder"]
        client = ml_models["qdrant"]

        # Encode text to vector
        vector = encoder.encode(request.text).tolist()

        # Perform semantic search in Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=request.top_k
        )

        # Format results
        results = []
        for hit in search_result:
            results.append({
                "product_name": hit.payload.get('prod_name', 'Unknown'),
                "similarity_score": round(hit.score, 4),
                "details": hit.payload,  # Includes image_url, color, type, etc.
                "source": "vector_db"  # Indicator for debugging
            })

        final_response = {"results": results}

        # 4. Save to Redis (TTL: 1 Hour)
        if redis_client:
            # We tag the source as 'redis_cache' before saving
            cache_data = {"results": []}
            for item in results:
                new_item = item.copy()
                new_item["source"] = "redis_cache"
                cache_data["results"].append(new_item)

            # Save to Redis with 3600 seconds expiration
            redis_client.setex(cache_key, 3600, json.dumps(cache_data))

        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)