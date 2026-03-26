"""
Privacy filter for PII detection and redaction.
"""

import re
from typing import Any, Dict, Tuple


class PrivacyFilter:
    """
    Detects and redacts PII (Personally Identifiable Information).

    Based on ECC's privacy filter - protects sensitive data while maintaining
    observation utility for learning purposes.
    """

    # PII patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "ssn": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
        "api_key": r'\b(?:api[_-]?key|apikey|api[_-]?secret)\s*[:=]\s*[\'"]?([A-Za-z0-9_\-]+)[\'"]?\b',
        "bearer_token": r'\bBearer\s+[A-Za-z0-9_\-\.]+\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    def __init__(self):
        """Initialize compiled patterns for performance."""
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text that may contain PII

        Returns:
            Text with PII replaced by placeholders
        """
        redacted = text
        redaction_count = 0

        for name, pattern in self.compiled_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                redaction_count += len(matches)
                # Replace with placeholder
                if name == "email":
                    redacted = pattern.sub('[REDACTED_EMAIL]', redacted)
                elif name == "phone":
                    redacted = pattern.sub('[REDACTED_PHONE]', redacted)
                elif name == "ssn":
                    redacted = pattern.sub('[REDACTED_SSN]', redacted)
                elif name == "credit_card":
                    redacted = pattern.sub('[REDACTED_CARD]', redacted)
                elif name == "api_key":
                    redacted = pattern.sub('[REDACTED_API_KEY]', redacted)
                elif name == "bearer_token":
                    redacted = pattern.sub('[REDACTED_TOKEN]', redacted)
                elif name == "ip_address":
                    redacted = pattern.sub('[REDACTED_IP]', redacted)

        return redacted

    def scan(self, data: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Scan data for PII and return redacted version.

        Args:
            data: Any data structure (dict, list, str, etc.)

        Returns:
            Tuple of (has_pii, redacted_data)
        """
        if isinstance(data, str):
            redacted = self.redact(data)
            has_pii = redacted != data
            return has_pii, redacted

        elif isinstance(data, dict):
            redacted_dict = {}
            has_pii = False

            for key, value in data.items():
                key_has_pii, redacted_key = self.scan(key)
                value_has_pii, redacted_value = self.scan(value)

                if key_has_pii or value_has_pii:
                    has_pii = True

                redacted_dict[redacted_key] = redacted_value

            return has_pii, redacted_dict

        elif isinstance(data, list):
            redacted_list = []
            has_pii = False

            for item in data:
                item_has_pii, redacted_item = self.scan(item)
                if item_has_pii:
                    has_pii = True
                redacted_list.append(redacted_item)

            return has_pii, redacted_list

        else:
            # Non-string types are returned as-is
            return False, data
