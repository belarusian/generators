"""Typed generator output schema for triage results.

This module provides typed generator output schema for triage results in generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TriageResultSchema:
    """Typed schema for triage results."""
    priority: int
    labels: List[str]
    complexity: str  # 'small', 'medium', 'large'
    discard: bool = False
    repo: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'priority': self.priority,
            'labels': self.labels,
            'complexity': self.complexity,
            'discard': self.discard,
            'repo': self.repo,
        }


class TriageResultValidator:
    """Validator for triage result schema."""
    
    def __init__(self):
        self.valid_schemas: List[TriageResultSchema] = []
        
    def validate(self, data: Dict[str, Any]) -> Optional[TriageResultSchema]:
        """Validate triage result data against schema."""
        try:
            priority = data.get('priority')
            if not isinstance(priority, int) or not (1 <= priority <= 9):
                return None
                
            labels = data.get('labels')
            if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
                return None
                
            complexity = data.get('complexity')
            if complexity not in ['small', 'medium', 'large']:
                return None
                
            discard = data.get('discard', False)
            if not isinstance(discard, bool):
                return None
                
            repo = data.get('repo')
            if repo is not None and not isinstance(repo, str):
                return None
                
            schema = TriageResultSchema(
                priority=priority,
                labels=labels,
                complexity=complexity,
                discard=discard,
                repo=repo,
            )
            self.valid_schemas.append(schema)
            return schema
        except Exception:
            return None
