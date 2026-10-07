from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_recommendations_reject_invalid_n() -> None:
    response = client.get(
        "/recommendations/14911?n=0",
    )

    assert response.status_code == 422


def test_recommendations_reject_too_large_n() -> None:
    response = client.get(
        "/recommendations/14911?n=51",
    )

    assert response.status_code == 422

def test_recommendations_return_404_for_unknown_customer() -> None:
    response = client.get("/recommendations/999999999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Customer 999999999 not found"
    }