import uuid
from typing import List, Dict, Any

class ContextCache:
    """
    In-memory key-value store for decoupling large text context payloads from orchestrator parameters (Pass-by-Reference).
    """
    _cache: Dict[str, Any] = {}

    @classmethod
    def store(cls, items: List[Any]) -> List[str]:
        """
        Stores retrieved objects and returns a list of unique reference keys.
        """
        keys = []
        for item in items:
            key = f"ref_{uuid.uuid4().hex[:8]}"
            cls._cache[key] = item
            keys.append(key)
        return keys

    @classmethod
    def resolve(cls, keys: List[str]) -> List[Any]:
        """
        Resolves reference keys back to their original stored objects.
        """
        items = []
        for key in keys:
            if key in cls._cache:
                items.append(cls._cache[key])
        return items

    @classmethod
    def clear(cls):
        """
        Clears all stored entries.
        """
        cls._cache.clear()
