from security.secret import create_secret


class TestCreateSecret:
    def test_returns_string(self):
        """Test that create_secret returns a string."""
        result = create_secret()

        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        """Test that the returned string is not empty."""
        result = create_secret()

        assert len(result) > 0

    def test_returns_url_safe_string(self):
        """Test that the returned string is URL-safe."""
        import string

        result = create_secret()
        valid_chars = string.ascii_letters + string.digits + "-_"

        assert all(c in valid_chars for c in result)

    def test_each_call_returns_different_value(self):
        """Test that each call returns a different secret."""
        secret1 = create_secret()
        secret2 = create_secret()
        secret3 = create_secret()

        assert secret1 != secret2
        assert secret2 != secret3
        assert secret1 != secret3

    def test_reasonable_length(self):
        """Test that the secret has a reasonable length."""
        result = create_secret()

        # token_urlsafe(64) typically produces ~86 characters
        assert len(result) >= 80
        assert len(result) <= 100

    def test_cryptographically_random(self):
        """Test that multiple secrets are sufficiently different (not sequential/predictable)."""
        secrets = [create_secret() for _ in range(10)]

        # All should be unique
        assert len(set(secrets)) == 10

    def test_no_padding_characters(self):
        """Test that returned value has no padding characters."""
        result = create_secret()

        # token_urlsafe doesn't include padding, but let's verify
        assert "=" not in result

    def test_suitable_for_api_keys(self):
        """Test that the secret format is suitable for API keys."""
        result = create_secret()

        # Should be string, non-empty, URL-safe, no equals signs
        assert isinstance(result, str)
        assert len(result) > 0
        assert "=" not in result

        # Should be safe to use in URLs
        import urllib.parse

        encoded = urllib.parse.quote(result, safe="")
        # When URL-encoded, it should either be the same or only minimally different
        assert encoded == result or len(encoded) <= len(result) + 5

    def test_sufficient_entropy(self):
        """Test that the secret has sufficient entropy for security."""
        # Generate multiple secrets and verify they're all different
        secrets = set()
        for _ in range(100):
            secret = create_secret()
            secrets.add(secret)

        # All 100 should be unique (probability of collision is astronomically low)
        assert len(secrets) == 100

    def test_can_be_used_in_urls(self):
        """Test that secrets can be safely used in URLs."""
        result = create_secret()

        # Should not contain characters that need escaping
        import re

        # URL-safe characters per RFC 3986 unreserved: A-Z a-z 0-9 - _ . ~
        # token_urlsafe uses base64url which is A-Z a-z 0-9 - _
        assert re.match(r"^[A-Za-z0-9\-_]+$", result)

    def test_multiple_sequential_calls_are_independent(self):
        """Test that sequential calls don't produce predictable patterns."""
        secrets = [create_secret() for _ in range(5)]

        # Check that they share no common prefixes or suffixes
        for i in range(len(secrets) - 1):
            # Secrets should not share the same first 20 characters
            assert secrets[i][:20] != secrets[i + 1][:20]
            # Secrets should not share the same last 20 characters
            assert secrets[i][-20:] != secrets[i + 1][-20:]
