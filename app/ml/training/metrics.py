def precision_at_k(
    recommendations: list[str],
    target: str,
    k: int,
) -> float:
    """Calculate Precision@K for a single target item."""
    top_k = recommendations[:k]

    return 1.0 if target in top_k else 0.0