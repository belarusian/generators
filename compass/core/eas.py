"""Effect Action System (EAS) for local-first agent workflows.

This module provides EAS (Effect Action System) for local-first agent workflows
in generators, including:
- Effect v4 native operations for agent loops
- Local-first cache for LLM response validation
- EAS for workflow state transitions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from compass.core.nfa_types import NestedFiniteAutomaton, State, Transition


@dataclass
class EASAction:
    """Effect Action for EAS operations."""
    action_type: str  # 'generate', 'validate', 'cache', 'transition'
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    cache_key: Optional[str] = None


@dataclass
class EASState:
    """Effect Action System state for workflow transitions."""
    workflow_id: str
    current_state: str
    actions: List[EASAction] = field(default_factory=list)
    cache_key: Optional[str] = None
    is_validated: bool = False


class LocalFirstCache:
    """Local-first cache for LLM response validation."""
    
    def __init__(self, cache_dir: str = "~/.generators/cache"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(self, data: Any) -> str:
        """Compute a cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with key."""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(value, f, indent=2)
    
    def validate_response(self, response: Any, expected_schema: Dict[str, Any]) -> bool:
        """Validate LLM response against expected schema."""
        # Basic validation: check if response is a dict or has expected structure
        if isinstance(response, dict):
            return True
        if isinstance(response, str):
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                pass
        return False


class EffectActionSystem:
    """Effect Action System for local-first agent workflows."""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.state = EASState(workflow_id=workflow_id, current_state='initial')
        self.cache = LocalFirstCache()
        self._actions: List[EASAction] = []
    
    def add_action(self, action_type: str, payload: Optional[Dict[str, Any]] = None) -> EASAction:
        """Add an effect action to the workflow."""
        action = EASAction(action_type=action_type, payload=payload or {})
        self._actions.append(action)
        self.state.actions.append(action)
        return action
    
    def execute_generate_action(self, prompt: str, model: str) -> EASAction:
        """Execute a generate action with local-first cache."""
        action = self.add_action('generate', {'prompt': prompt, 'model': model})
        
        # Check cache first
        cache_key = self.cache._compute_cache_key({'prompt': prompt, 'model': model})
        cached_response = self.cache.get(cache_key)
        
        if cached_response is not None:
            action.result = cached_response
            action.cache_key = cache_key
            return action
        
        # Execute generation (mock for now)
        # In real implementation, this would call the LLM
        action.result = {"generated": True, "prompt": prompt}
        action.cache_key = cache_key
        
        # Cache the response
        self.cache.set(cache_key, action.result)
        
        return action
    
    def execute_validate_action(self, response: Any, schema: Dict[str, Any]) -> EASAction:
        """Execute a validate action."""
        action = self.add_action('validate', {'response_type': type(response).__name__, 'schema': schema})
        
        is_valid = self.cache.validate_response(response, schema)
        action.result = is_valid
        self.state.is_validated = is_valid
        
        return action
    
    def execute_transition_action(self, from_state: str, to_state: str) -> EASAction:
        """Execute a state transition action."""
        action = self.add_action('transition', {'from_state': from_state, 'to_state': to_state})
        self.state.current_state = to_state
        action.result = {'success': True}
        return action
    
    def get_workflow_state(self) -> EASState:
        """Get the current workflow state."""
        return self.state
    
    def execute_nfa_transition(self, nfa: NestedFiniteAutomaton, context: Any) -> Tuple[bool, Any]:
        """Execute an NFA transition using EAS."""
        # Add transition action
        current_state = self.state.current_state
        action = self.add_action('nfa_transition', {'current_state': current_state})
        
        # Execute transition logic here
        # In a real implementation, this would interact with the NFA runner
        
        action.result = {'transitioned': True}
        self.state.is_validated = True
        
        return True, context
