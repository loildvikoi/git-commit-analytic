from abc import ABC
from datetime import datetime
from typing import Dict, Any
import json
from uuid import uuid4


class DomainEvent(ABC):
    """Base class for domain events."""

    def __init__(self):
        """Initialize event with auto-generated id and timestamp."""
        self.event_id = str(uuid4())
        self.occurred_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        return {
            'event_type': self.__class__.__name__,
            'event_id': self.event_id,
            'occurred_at': self.occurred_at.isoformat(),
            'data': self._get_event_data()
        }

    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def _get_event_data(self) -> Dict[str, Any]:
        """Override this method to provide event-specific data."""
        return {}
