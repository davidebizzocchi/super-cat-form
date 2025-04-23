from typing import Any, Optional, Set
from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """
    Represents a named executable component with dependency relationships.
    
    An Action can have dependencies on other Actions that must be completed before
    it can be executed.
    """

    name: str
    operation: Optional[Any] = Field(default_factory=lambda: lambda *args, **kwargs: None)
    requires: Optional[Set[str]] = Field(default_factory=set)  # dependencies that must be satisfied first

    @field_validator("requires", mode="before")
    @classmethod
    def validate_requirements(cls, v):
        """
        Validates and normalizes dependency requirements.

        Requirements can be specified as:
        - Action objects: Will extract the name
        - Strings: Used directly as dependency names
        
        This ensures consistent internal representation using dependency names.
        """
        if not v:
            return set()
        
        normalized_requirements = set()
        
        for requirement in v:
            if hasattr(requirement, 'name'):  # Could be an Action object
                normalized_requirements.add(requirement.name)
            elif isinstance(requirement, str):
                normalized_requirements.add(requirement)

        return normalized_requirements

    def execute(self, *args, **kwargs):
        """Execute the action's operation."""
        if callable(self.operation):
            return self.operation(*args, **kwargs)
        return None 