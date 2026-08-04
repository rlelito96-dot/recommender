import pytest

from app.core.exceptions import (
    AppException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)


class TestAppException:
    def test_default_message(self) -> None:
        exc = AppException()
        assert str(exc) == "An application error occurred"

    def test_custom_message(self) -> None:
        exc = AppException("Custom error")
        assert str(exc) == "Custom error"


class TestNotFoundException:
    def test_is_subclass_of_app_exception(self) -> None:
        assert issubclass(NotFoundException, AppException)

    def test_default_message(self) -> None:
        exc = NotFoundException()
        assert str(exc) == "Resource not found"

    def test_custom_message(self) -> None:
        exc = NotFoundException("Order 5 not found")
        assert str(exc) == "Order 5 not found"


class TestValidationException:
    def test_is_subclass_of_app_exception(self) -> None:
        assert issubclass(ValidationException, AppException)

    def test_default_message(self) -> None:
        exc = ValidationException()
        assert str(exc) == "Validation failed"

    def test_custom_message(self) -> None:
        exc = ValidationException("The record is incorrect")
        assert str(exc) == "The record is incorrect"

    def test_can_be_raised(self) -> None:
        with pytest.raises(ValidationException):
            raise ValidationException("The record is incorrect")


class TestUnauthorizedException:
    def test_is_subclass_of_app_exception(self) -> None:
        assert issubclass(UnauthorizedException, AppException)

    def test_default_message(self) -> None:
        exc = UnauthorizedException()
        assert str(exc) == "Unauthorized"

    def test_custom_message(self) -> None:
        exc = UnauthorizedException("Invalid credentials")
        assert str(exc) == "Invalid credentials"

    def test_can_be_raised(self) -> None:
        with pytest.raises(UnauthorizedException):
            raise UnauthorizedException("Invalid credentials")
