from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict


class ICacheService(ABC):
    """Interface for cache service"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL in seconds"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (with * wildcard)"""
        pass

    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        pass

    @abstractmethod
    async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set multiple values in cache"""
        pass
