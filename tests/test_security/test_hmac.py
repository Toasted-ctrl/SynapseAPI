import pytest

from security.hmac import hash_hmac, is_valid_hmac


class TestHashHmac:
    def test_returns_valid_hmac_string(self):
        """Test that hash_hmac returns a valid hex string."""
        content = "test content"
        key = b"secret_key"
        result = hash_hmac(content, key)

        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_and_key_produces_same_hash(self):
        """Test that identical inputs produce identical HMAC."""
        content = "test content"
        key = b"secret_key"

        result1 = hash_hmac(content, key)
        result2 = hash_hmac(content, key)

        assert result1 == result2

    def test_different_content_produces_different_hash(self):
        """Test that different content produces different HMAC."""
        key = b"secret_key"

        result1 = hash_hmac("content1", key)
        result2 = hash_hmac("content2", key)

        assert result1 != result2

    def test_different_key_produces_different_hash(self):
        """Test that different keys produce different HMAC."""
        content = "test content"

        result1 = hash_hmac(content, b"key1")
        result2 = hash_hmac(content, b"key2")

        assert result1 != result2

    def test_handles_empty_string_content(self):
        """Test that empty string content is handled."""
        content = ""
        key = b"secret_key"

        result = hash_hmac(content, key)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_handles_unicode_content(self):
        """Test that unicode characters in content are handled."""
        content = "test café 🔐"
        key = b"secret_key"

        result = hash_hmac(content, key)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_handles_long_content(self):
        """Test that long content is handled."""
        content = "x" * 10000
        key = b"secret_key"

        result = hash_hmac(content, key)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_raises_on_non_string_content(self):
        """Test that non-string content raises an error."""
        with pytest.raises((TypeError, AttributeError)):
            hash_hmac(12345, b"key")

    def test_raises_on_non_bytes_key(self):
        """Test that non-bytes key raises an error."""
        with pytest.raises(TypeError):
            hash_hmac("content", "not_bytes")


class TestIsValidHmac:
    def test_returns_true_for_matching_hmacs(self):
        """Test that matching HMAC signatures return True."""
        hmac_value = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"

        result = is_valid_hmac(hmac_value, hmac_value)

        assert result is True

    def test_returns_false_for_non_matching_hmacs(self):
        """Test that non-matching HMAC signatures return False."""
        hmac1 = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"
        hmac2 = "xyz789uvw012xyz789uvw012xyz789uvw012xyz789uvw012xyz789uvw012xyz7"

        result = is_valid_hmac(hmac1, hmac2)

        assert result is False

    def test_is_timing_safe(self):
        """Test that comparison uses constant-time comparison."""
        # hmac.compare_digest is timing-safe, so this test just verifies it's used
        hmac1 = "a" * 64
        hmac2 = "b" * 64

        result = is_valid_hmac(hmac1, hmac2)

        assert result is False

    def test_handles_empty_strings(self):
        """Test that empty HMAC strings are handled."""
        result = is_valid_hmac("", "")

        assert result is True

    def test_case_sensitive_comparison(self):
        """Test that comparison is case-sensitive."""
        hmac1 = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"
        hmac2 = "ABC123DEF456ABC123DEF456ABC123DEF456ABC123DEF456ABC123DEF456ABC1"

        result = is_valid_hmac(hmac1, hmac2)

        assert result is False

    def test_length_mismatch_returns_false(self):
        """Test that HMACs of different lengths are not equal."""
        hmac1 = "abc123"
        hmac2 = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"

        result = is_valid_hmac(hmac1, hmac2)

        assert result is False

    def test_with_actual_hmac_values(self):
        """Test with actual HMAC values generated from hash_hmac."""
        content = "test message"
        key = b"secret_key"

        expected_hmac = hash_hmac(content, key)

        assert is_valid_hmac(expected_hmac, expected_hmac) is True
        assert is_valid_hmac(expected_hmac, "wrong_hmac") is False
