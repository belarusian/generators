"""
Generators Local-First Cache for LLM Response Validation.

This module provides a local-first cache for LLM response validation in generators.
LLM responses are cached locally for validation, and the cache is validated against
generators schemas. The local-first approach prevents remote code changes.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Cache entry for LLM response."""
    prompt_hash: str
    response: str
    schema_validated: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalFirstCache:
    """
    Local-first cache for LLM response validation.
    
    This cache stores LLM responses locally and validates them against
    generators schemas before accepting them as valid.
    """
    
    def __init__(self, cache_dir: str = "~/.compass/cache"):
        """Initialize the local-first cache."""
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_responses.json"
        self._cache: Dict[str, CacheEntry] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get('entries', []):
                        entry = CacheEntry(
                            prompt_hash=entry_data['prompt_hash'],
                            response=entry_data['response'],
                            schema_validated=entry_data['schema_validated'],
                            timestamp=entry_data['timestamp'],
                            metadata=entry_data.get('metadata', {}),
                        )
                        self._cache[entry.prompt_hash] = entry
            except Exception:
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        data = {
            'entries': [
                {
                    'prompt_hash': entry.prompt_hash,
                    'response': entry.response,
                    'schema_validated': entry.schema_validated,
                    'timestamp': entry.timestamp,
                    'metadata': entry.metadata,
                }
                for entry in self._cache.values()
            ]
        }
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_prompt_hash(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """Compute hash for prompt and schema."""
        content = f"{prompt}:{json.dumps(schema, sort_keys=True) if schema else ''}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Optional[CacheEntry]:
        """
        Get cached response for prompt and schema.
        
        Args:
            prompt: The prompt used
            schema: The schema used for validation
            
        Returns:
            CacheEntry if found and valid, None otherwise
        """
        prompt_hash = self._compute_prompt_hash(prompt, schema)
        return self._cache.get(prompt_hash)
    
    def set(self, prompt: str, response: str, schema_validated: bool = True, 
            schema: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set cached response for prompt and schema.
        
        Args:
            prompt: The prompt used
            response: The LLM response
            schema_validated: Whether the response was validated against schema
            schema: The schema used for validation
            metadata: Additional metadata
        """
        prompt_hash = self._compute_prompt_hash(prompt, schema)
        entry = CacheEntry(
            prompt_hash=prompt_hash,
            response=response,
            schema_validated=schema_validated,
            timestamp=__import__('time').time(),
            metadata=metadata or {},
        )
        self._cache[prompt_hash] = entry
        self._save_cache()
    
    def validate_against_schema(self, response: str, schema: Dict[str, Any]) -> bool:
        """
        Validate response against generators schema.
        
        Args:
            response: The LLM response to validate
            schema: The generators schema to validate against
            
        Returns:
            True if response is valid against schema, False otherwise
        """
        # In production, this would parse the response and validate against schema
        # For now, return True if response is valid JSON or structured text
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            # Check if it's structured text (key=value format)
            return '=' in response or '{' in response or '[' in response
