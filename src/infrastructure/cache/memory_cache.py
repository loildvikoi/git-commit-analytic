from typing import Any, Optional, List, Dict
from src.domain.services.cache_service import ICacheService


class MemoryCacheService(ICacheService):
    def __init__(self):
        self.cache = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL in seconds"""
        self.cache[key] = value
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (with * wildcard)"""
        keys_to_delete = [key for key in self.cache if pattern in key]
        for key in keys_to_delete:
            del self.cache[key]
        return len(keys_to_delete)

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        return {key: self.cache.get(key) for key in keys}

    async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set multiple values in cache"""
        self.cache.update(data)
        return True
