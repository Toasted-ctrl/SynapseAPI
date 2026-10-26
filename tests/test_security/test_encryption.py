from unittest.mock import MagicMock, patch

import pytest

from security.encryption import _validate_content, decrypt, encrypt


class TestValidateContent:
    def test_accepts_valid_string(self):
        """Test that valid string passes validation."""
        _validate_content("valid string")

    def test_raises_on_non_string(self):
        """Test that non-string raises TypeError."""
        with pytest.raises(TypeError, match="Content must be a string"):
            _validate_content(123)

    def test_raises_on_none(self):
        """Test that None raises TypeError."""
        with pytest.raises(TypeError):
            _validate_content(None)

    def test_raises_on_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty string"):
            _validate_content("")

    def test_accepts_long_string(self):
        """Test that long strings are accepted."""
        _validate_content("x" * 10000)

    def test_accepts_unicode_string(self):
        """Test that unicode strings are accepted."""
        _validate_content("Hello café 🔐")


class TestEncrypt:
    @patch("security.encryption.config")
    def test_returns_string(self, mock_config):
        """Test that encrypt returns a string."""
        mock_config.ENCRYPTION_KEY = (
            b"0" * 32
        )  # Valid Fernet key (must be 32 bytes when base64 encoded properly)
        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.encrypt.return_value = b"encrypted_data"

            result = encrypt("test")

            assert isinstance(result, str)

    @patch("security.encryption.config")
    def test_calls_fernet_encrypt(self, mock_config):
        """Test that encrypt calls Fernet.encrypt with encoded content."""
        mock_config.ENCRYPTION_KEY = b"test_key"
        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.encrypt.return_value = b"encrypted"

            encrypt("test content")

            mock_fernet.encrypt.assert_called_once()
            args = mock_fernet.encrypt.call_args[0]
            assert args[0] == b"test content"

    @patch("security.encryption.config")
    def test_raises_on_non_string(self, mock_config):
        """Test that encrypt raises on non-string input."""
        mock_config.ENCRYPTION_KEY = b"test_key"

        with pytest.raises(TypeError):
            encrypt(123)

    @patch("security.encryption.config")
    def test_raises_on_empty_string(self, mock_config):
        """Test that encrypt raises on empty string."""
        mock_config.ENCRYPTION_KEY = b"test_key"

        with pytest.raises(ValueError):
            encrypt("")

    @patch("security.encryption.config")
    def test_uses_config_encryption_key(self, mock_config):
        """Test that encrypt uses the configured encryption key."""
        mock_key = b"configured_key"
        mock_config.ENCRYPTION_KEY = mock_key

        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.encrypt.return_value = b"encrypted"

            encrypt("test")

            MockFernet.assert_called_once_with(mock_key)


class TestDecrypt:
    @patch("security.encryption.config")
    def test_returns_string(self, mock_config):
        """Test that decrypt returns a string."""
        mock_config.ENCRYPTION_KEY = b"test_key"
        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.decrypt.return_value = b"decrypted_data"

            result = decrypt("encrypted_data")

            assert isinstance(result, str)

    @patch("security.encryption.config")
    def test_calls_fernet_decrypt_with_bytes(self, mock_config):
        """Test that decrypt encodes content to bytes before calling Fernet.decrypt."""
        mock_config.ENCRYPTION_KEY = b"test_key"
        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.decrypt.return_value = b"decrypted"

            decrypt("encrypted_data")

            mock_fernet.decrypt.assert_called_once()
            args = mock_fernet.decrypt.call_args[0]
            assert args[0] == b"encrypted_data"

    @patch("security.encryption.config")
    def test_raises_on_non_string(self, mock_config):
        """Test that decrypt raises on non-string input."""
        mock_config.ENCRYPTION_KEY = b"test_key"

        with pytest.raises(TypeError):
            decrypt(123)

    @patch("security.encryption.config")
    def test_raises_on_empty_string(self, mock_config):
        """Test that decrypt raises on empty string."""
        mock_config.ENCRYPTION_KEY = b"test_key"

        with pytest.raises(ValueError):
            decrypt("")

    @patch("security.encryption.config")
    def test_uses_config_encryption_key(self, mock_config):
        """Test that decrypt uses the configured encryption key."""
        mock_key = b"configured_key"
        mock_config.ENCRYPTION_KEY = mock_key

        with patch("security.encryption.Fernet") as MockFernet:
            mock_fernet = MagicMock()
            MockFernet.return_value = mock_fernet
            mock_fernet.decrypt.return_value = b"decrypted"

            decrypt("encrypted")

            MockFernet.assert_called_once_with(mock_key)

    @patch("security.encryption.config")
    def test_round_trip_encrypt_decrypt(self, mock_config):
        """Test that encrypt/decrypt work together (if using real Fernet)."""
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        mock_config.ENCRYPTION_KEY = key

        original = "test message"
        encrypted = encrypt(original)
        decrypted = decrypt(encrypted)

        assert decrypted == original

    @patch("security.encryption.config")
    def test_different_keys_cannot_decrypt(self, mock_config):
        """Test that data encrypted with one key cannot be decrypted with another."""
        from cryptography.fernet import Fernet, InvalidToken

        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        mock_config.ENCRYPTION_KEY = key1
        encrypted = encrypt("test message")

        mock_config.ENCRYPTION_KEY = key2

        with pytest.raises(InvalidToken):
            decrypt(encrypted)
