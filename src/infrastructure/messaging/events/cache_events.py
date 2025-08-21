from typing import Dict, Any, List, Optional
from src.domain.events.base import DomainEvent


class CacheInvalidationEvent(DomainEvent):
    """Infrastructure event for distributed cache invalidation"""

    def __init__(self, patterns: List[str], keys: Optional[List[str]] = None,
                 source_node: str = None, reason: str = None):
        super().__init__()
        self.patterns = patterns
        self.keys = keys or []
        self.source_node = source_node
        self.reason = reason

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'patterns': self.patterns,
            'keys': self.keys,
            'source_node': self.source_node,
            'reason': self.reason
        }


class CacheWarmupEvent(DomainEvent):
    """Infrastructure event for distributed cache warmup"""

    def __init__(self, cache_type: str, data: Dict[str, Any], ttl: int = 3600):
        super().__init__()
        self.cache_type = cache_type
        self.data = data
        self.ttl = ttl

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'cache_type': self.cache_type,
            'data': self.data,
            'ttl': self.ttl
        }