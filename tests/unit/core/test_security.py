import jwt
import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_password_returns_different_value(self) -> None:
        password = "mysecretpassword"  # pragma: allowlist secret
        hashed = hash_password(password)
        assert hashed != password

    def test_verify_correct_password(self) -> None:
        password = "mysecretpassword"  # pragma: allowlist secret
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_incorrect_password(self) -> None:
        hashed = hash_password("correctpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_same_password_produces_different_hashes(self) -> None:
        password = "mysecretpassword"  # pragma: allowlist secret
        assert hash_password(password) != hash_password(password)


class TestAccessToken:
    def test_create_access_token_returns_string(self) -> None:
        token = create_access_token({"sub": "test@example.com"})
        assert isinstance(token, str)

    def test_decode_access_token_returns_original_data(self) -> None:
        token = create_access_token({"sub": "test@example.com"})
        payload = decode_access_token(token)
        assert payload["sub"] == "test@example.com"

    def test_token_contains_expiration(self) -> None:
        token = create_access_token({"sub": "test@example.com"})
        payload = decode_access_token(token)
        assert "exp" in payload

    def test_decode_invalid_token_raises_error(self) -> None:
        with pytest.raises(jwt.InvalidTokenError):
            decode_access_token("this.is.not.a.valid.token")
