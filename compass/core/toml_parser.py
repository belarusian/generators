"""
TOML/Key=Value Parsing in Generators Trusted Code Layer.

This module provides TOML/key=value parsing in generators trusted code layer.
The pattern: parse TOML output from bash script execution via trusted code,
NOT regex on LLM text.

Parsing via simple string operations:
- line.indexOf("=") or line.split("=")
- Zero regex on LLM text anywhere
"""

from typing import Dict, Any


def parse_toml_key_value(content: str) -> Dict[str, Any]:
    """
    Parse TOML key=value format from trusted code.

    Args:
        content: The TOML-like key=value content to parse

    Returns:
        Dictionary of parsed key=value pairs
    """
    triage_result: Dict[str, Any] = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        eq_index = line.find('=')
        if eq_index == -1:
            continue
            
        key = line[:eq_index].strip().lower()
        value = line[eq_index + 1:].strip()
        
        # Parse boolean values
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        # Parse integer values
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            value = int(value)
        # Parse array values (comma-separated)
        elif value.startswith('[') and value.endswith(']'):
            # Simple array parsing: [item1,item2,item3]
            inner = value[1:-1]
            if inner:
                value = [item.strip() for item in inner.split(',')]
            else:
                value = []
                
        triage_result[key] = value
        
    return triage_result


def parse_triage_result(content: str) -> Dict[str, Any]:
    """
    Parse triage result from TOML key=value format.

    Expected format:
    PRIORITY=2
    LABELS=bug,api
    COMPLEXITY=small
    DISCARD=false
    REPO=belarusian/compsci.boutique
    BREAKDOWN=step1|step2

    Args:
        content: The triage result content

    Returns:
        Parsed triage result dictionary
    """
    result = parse_toml_key_value(content)
    
    # Normalize keys
    normalized = {}
    for key, value in result.items():
        if key == 'priority':
            normalized['priority'] = int(value) if isinstance(value, str) and value.isdigit() else value
        elif key == 'labels':
            if isinstance(value, str):
                normalized['labels'] = [item.strip() for item in value.split(',')] if value else []
            elif isinstance(value, list):
                normalized['labels'] = value
        elif key == 'complexity':
            normalized['complexity'] = value
        elif key == 'discard':
            normalized['discard'] = isinstance(value, bool) or str(value).lower() == 'true'
        elif key == 'repo':
            normalized['repo'] = value if value else None
        elif key == 'breakdown':
            if isinstance(value, str):
                normalized['breakdown'] = [item.strip() for item in value.split('|')] if value else []
            elif isinstance(value, list):
                normalized['breakdown'] = value
        else:
            normalized[key] = value
            
    return normalized
