import pytest
from fastapi.testclient import TestClient
from src.api.main import app


# --- FIXTURE ---
# This creates a fresh client for each test and ensures
# the 'lifespan' events (startup/shutdown) are triggered.
# Without this, the ML model won't load, causing 500 errors.
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# --- TESTS ---

def test_api_health_check(client):
    """
    1. Is the API up and running? (Health Check)
    """
    print("\n🧪 TEST 1: Health Check")
    response = client.get("/")

    # Should return 200 OK
    assert response.status_code == 200
    # Should return the 'alive' status
    assert response.json()["status"] == "alive"
    print("✅ API is Healthy!")


def test_recommendation_flow(client):
    """
    2. Does the Recommendation System Work?
    """
    print("\n🧪 TEST 2: Recommendation Scenario")

    payload = {
        "text": "Red party dress",
        "top_k": 3
    }

    response = client.post("/recommend", json=payload)

    if response.status_code != 200:
        print(f"\n❌ API ERROR DETAIL: {response.text}")

    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0

    print(f"✅ Recommendation Successful! Top Item: {data['results'][0]['product_name']}")


def test_invalid_input(client):
    """
    3. Does it handle invalid input gracefully?
    Missing the required 'text' field should trigger a 422 error.
    """
    print("\n🧪 TEST 3: Invalid Input Handling")

    # Missing 'text' field
    payload = {
        "top_k": 5
    }

    response = client.post("/recommend", json=payload)

    # FastAPI validation should return 422 Unprocessable Entity
    assert response.status_code == 422
    print("✅ Error Management Successful!")