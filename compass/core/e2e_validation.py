"""Generators E2E validation for mini-spoke style workflows.

This module provides E2E validation for mini-spoke style workflows in generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MiniSpokeWorkflow:
    """Mini-spoke style workflow context."""
    workflow_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = 'pending'


class E2EValidator:
    """E2E validator for mini-spoke style workflows."""
    
    def __init__(self):
        self.validated_workflows: List[str] = []
        
    def validate_workflow(self, workflow: MiniSpokeWorkflow) -> bool:
        """Validate a mini-spoke workflow end-to-end."""
        if not workflow.workflow_id:
            return False
            
        if not workflow.steps:
            return False
            
        # Validate each step has required fields
        for step in workflow.steps:
            if 'action' not in step or 'params' not in step:
                return False
                
        self.validated_workflows.append(workflow.workflow_id)
        workflow.status = 'validated'
        return True
        
    def get_validated_workflows(self) -> List[str]:
        """Get list of validated workflow IDs."""
        return self.validated_workflows
