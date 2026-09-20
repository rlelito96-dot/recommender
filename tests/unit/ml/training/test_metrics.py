from app.ml.training.metrics import precision_at_k


def test_precision_at_k_returns_one_for_hit() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="C",
        k=5,
    )

    assert result == 1.0


def test_precision_at_k_returns_zero_for_miss() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="F",
        k=5,
    )

    assert result == 0.0


def test_precision_at_k_only_considers_top_k() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="E",
        k=3,
    )

    assert result == 0.0