import pytest

from security.hash import get_hash_sha256


class TestGetHashSha256:
    def test_returns_valid_hex_string(self):
        """Test that hash returns a valid 64-character hex string."""
        result = get_hash_sha256("test")

        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_produces_same_hash(self):
        """Test that identical inputs produce identical hashes."""
        content = "test content"

        result1 = get_hash_sha256(content)
        result2 = get_hash_sha256(content)

        assert result1 == result2

    def test_different_inputs_produce_different_hashes(self):
        """Test that different inputs produce different hashes."""
        result1 = get_hash_sha256("content1")
        result2 = get_hash_sha256("content2")

        assert result1 != result2

    def test_case_sensitive_hashing(self):
        """Test that hashing is case-sensitive."""
        result1 = get_hash_sha256("Test")
        result2 = get_hash_sha256("test")

        assert result1 != result2

    def test_whitespace_sensitive_hashing(self):
        """Test that hashing is sensitive to whitespace."""
        result1 = get_hash_sha256("test")
        result2 = get_hash_sha256("test ")

        assert result1 != result2

    def test_handles_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            get_hash_sha256("")

    def test_handles_unicode_characters(self):
        """Test that unicode characters are hashed correctly."""
        result = get_hash_sha256("café 🔐")

        assert isinstance(result, str)
        assert len(result) == 64

    def test_handles_very_long_strings(self):
        """Test that very long strings are hashed."""
        long_string = "x" * 1000000
        result = get_hash_sha256(long_string)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_handles_special_characters(self):
        """Test that special characters are hashed."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = get_hash_sha256(special)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_raises_on_non_string_input(self):
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            get_hash_sha256(123)

    def test_raises_on_none_input(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError):
            get_hash_sha256(None)

    def test_raises_on_list_input(self):
        """Test that list input raises TypeError."""
        with pytest.raises(TypeError):
            get_hash_sha256(["test"])

    def test_raises_on_dict_input(self):
        """Test that dict input raises TypeError."""
        with pytest.raises(TypeError):
            get_hash_sha256({"test": "value"})

    def test_known_hash_values(self):
        """Test against known SHA256 hash values."""
        # SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        # But we expect ValueError for empty string

        # SHA256("test") = 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        assert (
            get_hash_sha256("test")
            == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        )

    def test_single_character_string(self):
        """Test hashing a single character."""
        result = get_hash_sha256("a")

        assert isinstance(result, str)
        assert len(result) == 64

    def test_newline_characters(self):
        """Test that newline characters are hashed."""
        result1 = get_hash_sha256("test\nline")
        result2 = get_hash_sha256("testline")

        assert result1 != result2

    def test_tabs_and_spaces(self):
        """Test that tabs and spaces are distinguished."""
        result1 = get_hash_sha256("test\tvalue")
        result2 = get_hash_sha256("test value")

        assert result1 != result2

    def test_consistent_with_standard_sha256(self):
        """Test that our hash matches standard SHA256."""
        import hashlib

        content = "test message"
        our_hash = get_hash_sha256(content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

        assert our_hash == expected

    def test_handles_string_with_null_bytes(self):
        """Test that strings with null-like representations are handled."""
        result = get_hash_sha256("test\x00null")

        assert isinstance(result, str)
        assert len(result) == 64
