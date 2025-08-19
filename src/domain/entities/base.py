import uuid
from abc import ABC
from dataclasses import dataclass
from datetime import datetime


class Entity(ABC):
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class ValueObject(ABC):
    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Override this method to implement validation logic."""
        pass
