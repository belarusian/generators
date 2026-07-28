"""
Neo - the autonomous Actor/Critic agent.

- Actor: generates actions to fulfill requests
- Critic: reviews results, decides retry vs done
- Answerer: generates final response

Entry point: execute_request()
"""

# Memory
from compass.agents.neo.memory import CodeMemory

# Action execution (spec-based, typed actions)
from compass.agents.neo.rules import execute_action

# Tool agents
from compass.agents.neo.file_editor import call_file_editor
from compass.agents.neo.shell_builder import call_shell_builder

# Core agent functions
from compass.agents.neo.user_query import execute_request
from compass.agents.neo.actor import call_actor
from compass.agents.neo.answerer import generate_answer, is_image_question
from compass.agents.neo.critic import critic_review, critic_evaluate

# Backward compat aliases (used by compass.py and tests)
_execute_request = execute_request
_generate_answer = generate_answer
_generate_actions = call_actor
_critic_review = critic_review
_critic_evaluate = critic_evaluate
_is_image_question = is_image_question

__all__ = [
    # Memory
    "CodeMemory",
    # Executor
    "execute_action",
    # Tool agents
    "call_file_editor",
    "call_shell_builder",
    # Core functions
    "execute_request",
    "call_actor",
    "generate_answer",
    "critic_review",
    "critic_evaluate",
    "is_image_question",
    # Backward compat aliases
    "_execute_request",
    "_generate_answer",
    "_generate_actions",
    "_critic_review",
    "_critic_evaluate",
    "_is_image_question",
]
