"""
Driver abstraction for plan/action approval.

The Driver is whoever is in control of the execution:
- UserDriver: Human approves via terminal
- ClaudeDriver: Claude API approves programmatically

Both implement the same interface, so the Oracle doesn't care
who's driving - it just asks for approval.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class ApprovalDecision(Enum):
    """What the driver decided."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # Approved with changes


@dataclass
class ActionApproval:
    """Result of asking driver to approve an action."""
    decision: ApprovalDecision
    feedback: str = ""  # Why rejected, or notes
    modified_action: Optional[Dict[str, Any]] = None  # If modified


@dataclass
class PlanApproval:
    """Result of asking driver to approve a plan."""
    decision: ApprovalDecision
    feedback: str = ""
    modified_plan: Optional[Dict[str, Any]] = None


class Driver(ABC):
    """
    Abstract base for execution drivers.

    The Driver controls approval gates:
    - Plan approval: Before Oracle starts executing
    - Action approval: Before each action executes
    """

    @abstractmethod
    def approve_plan(self, plan: Dict[str, Any], context: str = "") -> PlanApproval:
        """
        Ask driver to approve a plan before execution.

        Args:
            plan: The plan dict (summary, steps, files_affected)
            context: Additional context (user request, codebase info)

        Returns:
            PlanApproval with decision and optional feedback
        """
        pass

    @abstractmethod
    def approve_action(self, action: Dict[str, Any], context: str = "") -> ActionApproval:
        """
        Ask driver to approve an action before execution.

        Args:
            action: The action dict (action_type, target, reasoning, etc.)
            context: Additional context (current step, previous results)

        Returns:
            ActionApproval with decision and optional feedback/modification
        """
        pass

    def should_approve_actions(self) -> bool:
        """
        Whether this driver wants to approve individual actions.

        Override to return False for auto-approve after plan approval.
        Default is True (approve each action).
        """
        return True


class AutoApproveDriver(Driver):
    """
    Driver that auto-approves everything.

    Use for testing or when you trust the Oracle completely.
    """

    def approve_plan(self, plan: Dict[str, Any], context: str = "") -> PlanApproval:
        return PlanApproval(decision=ApprovalDecision.APPROVED)

    def approve_action(self, action: Dict[str, Any], context: str = "") -> ActionApproval:
        return ActionApproval(decision=ApprovalDecision.APPROVED)

    def should_approve_actions(self) -> bool:
        return False  # Skip action-level approval


class UserDriver(Driver):
    """
    Driver where human approves via terminal.

    Shows action details and waits for y/n input.
    """

    def __init__(self, approve_actions: bool = True):
        """
        Args:
            approve_actions: If False, only approve plans (current behavior)
        """
        self._approve_actions = approve_actions

    def approve_plan(self, plan: Dict[str, Any], context: str = "") -> PlanApproval:
        """Show plan and ask user to approve."""
        # This is currently handled by the UI elsewhere
        # For now, auto-approve (existing behavior)
        return PlanApproval(decision=ApprovalDecision.APPROVED)

    def approve_action(self, action: Dict[str, Any], context: str = "") -> ActionApproval:
        """Show action and ask user to approve."""
        if not self._approve_actions:
            return ActionApproval(decision=ApprovalDecision.APPROVED)

        action_type = action.get("action_type", "unknown")
        target = action.get("path") or action.get("command") or action.get("query") or ""
        reasoning = action.get("reasoning", "")

        print(f"\n--- ACTION APPROVAL ---")
        print(f"Type: {action_type}")
        if target:
            print(f"Target: {target}")
        if reasoning:
            print(f"Reasoning: {reasoning}")

        # Show content preview for writes
        content = action.get("content") or action.get("code") or action.get("patch")
        if content:
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"Content:\n{preview}")

        print("-" * 24)

        while True:
            response = input("Approve? [y]es / [n]o / [s]kip action approval: ").strip().lower()
            if response in ("y", "yes", ""):
                return ActionApproval(decision=ApprovalDecision.APPROVED)
            elif response in ("n", "no"):
                feedback = input("Reason (optional): ").strip()
                return ActionApproval(
                    decision=ApprovalDecision.REJECTED,
                    feedback=feedback or "User rejected"
                )
            elif response in ("s", "skip"):
                self._approve_actions = False
                print("Action approval disabled for this session")
                return ActionApproval(decision=ApprovalDecision.APPROVED)
            else:
                print("Please enter y, n, or s")

    def should_approve_actions(self) -> bool:
        return self._approve_actions


class ClaudeDriver(Driver):
    """
    Driver where Claude approves plans and actions.

    Uses Claude API with structured JSON responses for reliable parsing.
    Schemas defined in compass/code/schemas.py for consistency.
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, Claude reviews every action. If False, only risky ones.
        """
        from compass.llm.bridge import ClaudeBridge
        from compass.agents.neo.schemas import PLAN_APPROVAL_SCHEMA, ACTION_APPROVAL_SCHEMA
        self.claude = ClaudeBridge()
        self.strict = strict
        self.plan_schema = PLAN_APPROVAL_SCHEMA
        self.action_schema = ACTION_APPROVAL_SCHEMA

    def approve_plan(self, plan: Dict[str, Any], context: str = "") -> PlanApproval:
        """Ask Claude to review the plan."""
        import json
        prompt = f"""Review this plan for the Oracle to execute.

Plan:
{json.dumps(plan, indent=2)}

User's request: {context}

You can:
- APPROVE: Plan is good, proceed with execution
- REJECT: Plan is wrong, Oracle should replan from scratch (provide reason)
- MODIFY: Plan needs tweaks - provide corrected steps in modified_steps

Consider:
- Does it match the stated goal?
- Are there any dangerous or irreversible steps?
- Is it overly complex for the task?
- Can you fix minor issues by modifying steps?"""

        try:
            result = self.claude.ask_json(prompt, self.plan_schema, max_tokens=1024)
            decision = result.get("decision", "approved")
            reason = result.get("reason", "")

            if decision == "approved":
                return PlanApproval(
                    decision=ApprovalDecision.APPROVED,
                    feedback=reason
                )
            elif decision == "modified":
                modified_steps = result.get("modified_steps", [])
                modified_plan = {**plan, "steps": modified_steps} if modified_steps else None
                return PlanApproval(
                    decision=ApprovalDecision.MODIFIED,
                    feedback=reason,
                    modified_plan=modified_plan
                )
            else:  # rejected
                return PlanApproval(
                    decision=ApprovalDecision.REJECTED,
                    feedback=reason or "Claude rejected the plan"
                )
        except Exception as e:
            # On error, default to approved (don't block on Claude failure)
            return PlanApproval(
                decision=ApprovalDecision.APPROVED,
                feedback=f"Claude unavailable ({e}), auto-approved"
            )

    def approve_action(self, action: Dict[str, Any], context: str = "") -> ActionApproval:
        """Ask Claude to review the action."""
        import json
        # Skip low-risk actions in non-strict mode
        if not self.strict and not self._is_risky_action(action):
            return ActionApproval(decision=ApprovalDecision.APPROVED)

        prompt = f"""Review this action before the Oracle executes it.

Action:
{json.dumps(action, indent=2)}

Context: {context}

You can:
- APPROVE: Action is safe, execute as-is
- REJECT: Action is wrong or dangerous, skip it
- MODIFY: Action needs fixes - provide corrected action in modified_action

Consider:
- Is it safe to execute?
- Does it match the intended step?
- Could it cause unintended side effects?
- Can you fix issues by modifying the action?"""

        try:
            result = self.claude.ask_json(prompt, self.action_schema, max_tokens=512)
            decision = result.get("decision", "approved")
            reason = result.get("reason", "")

            if decision == "approved":
                return ActionApproval(
                    decision=ApprovalDecision.APPROVED,
                    feedback=reason
                )
            elif decision == "modified":
                modified_action = result.get("modified_action")
                return ActionApproval(
                    decision=ApprovalDecision.MODIFIED,
                    feedback=reason,
                    modified_action=modified_action
                )
            else:  # rejected
                return ActionApproval(
                    decision=ApprovalDecision.REJECTED,
                    feedback=reason or "Claude rejected the action"
                )
        except Exception as e:
            return ActionApproval(
                decision=ApprovalDecision.APPROVED,
                feedback=f"Claude unavailable ({e}), auto-approved"
            )

    def _is_risky_action(self, action: Dict[str, Any]) -> bool:
        """Check if action needs Claude review."""
        action_type = action.get("action_type", "")

        # Always review destructive or external actions
        risky_types = {
            "write_file", "delete_file", "run_command",
            "patch_file", "git_patch", "str_replace"
        }
        return action_type in risky_types


# Global driver instance - can be swapped at runtime
_current_driver: Optional[Driver] = None


def _init_driver_from_env() -> Driver:
    """Initialize driver based on environment variables.

    Environment variables:
        COMPASS_DRIVER: "auto" | "user" | "claude" (default: auto)
        COMPASS_APPROVE_ACTIONS: "1" to enable action-level approval for user driver
        COMPASS_CLAUDE_STRICT: "1" for strict mode (review all actions, not just risky)
    """
    driver_type = os.getenv("COMPASS_DRIVER", "auto").lower()

    if driver_type == "user":
        approve_actions = os.getenv("COMPASS_APPROVE_ACTIONS", "0") == "1"
        return UserDriver(approve_actions=approve_actions)

    elif driver_type == "claude":
        strict = os.getenv("COMPASS_CLAUDE_STRICT", "0") == "1"
        return ClaudeDriver(strict=strict)

    else:  # "auto" or anything else
        return AutoApproveDriver()


def get_driver() -> Driver:
    """Get the current driver (initialized from env vars on first call)."""
    global _current_driver
    if _current_driver is None:
        _current_driver = _init_driver_from_env()
    return _current_driver


def set_driver(driver: Driver) -> None:
    """Set the current driver."""
    global _current_driver
    _current_driver = driver


def use_user_driver(approve_actions: bool = True) -> None:
    """Switch to user-driven approval."""
    set_driver(UserDriver(approve_actions=approve_actions))


def use_claude_driver(strict: bool = False) -> None:
    """Switch to Claude-driven approval."""
    set_driver(ClaudeDriver(strict=strict))


def use_auto_driver() -> None:
    """Switch to auto-approval (default)."""
    set_driver(AutoApproveDriver())
