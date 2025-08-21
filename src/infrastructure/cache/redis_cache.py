import json
from typing import Any, Optional, List, Dict

from pydantic import BaseModel

from src.domain.services.cache_service import ICacheService


class RedisCacheService(ICacheService):
    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return None

            # Deserialize JSON
            data = json.loads(value)

            # If it's a ChatResponseDto, reconstruct it
            if isinstance(data, dict) and "_type" in data:
                if data["_type"] == "ChatResponseDto":
                    from ...interface.dto.commit_dto import ChatResponseDto
                    return ChatResponseDto(**data["data"])

            return data
        except Exception as e:
            print(f"Error getting cache: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL in seconds"""
        try:
            # Convert Pydantic models to dict with type info
            if isinstance(value, BaseModel):
                serializable_data = {
                    "_type": value.__class__.__name__,
                    "data": value.model_dump()
                }
            else:
                serializable_data = value

            # Serialize to JSON
            json_value = json.dumps(serializable_data, default=str)  # default=str handles datetime

            await self.redis_client.set(key, json_value, ex=ttl)
            return True
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Error deleting cache key: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            print(f"Error checking cache existence: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (with * wildcard)"""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            return len(keys)
        except Exception as e:
            print(f"Error invalidating cache pattern: {e}")
            return 0

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            values = await self.redis_client.mget(*keys)
            return {key: value for key, value in zip(keys, values) if value is not None}
        except Exception as e:
            print(f"Error getting multiple cache values: {e}")
            return {}

    async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set multiple values in cache"""
        try:
            pipeline = self.redis_client.pipeline()
            for key, value in data.items():
                pipeline.set(key, value, ex=ttl)
            await pipeline.execute()
            return True
        except Exception as e:
            print(f"Error setting multiple cache values: {e}")
            return False
