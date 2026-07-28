
#!/usr/bin/env python3
"""Trinity Plan Executor

Executes Trinity Spec plans: reads plan.json files, resolves dependency DAGs,
executes steps (inline_python, shell, notebook), extracts facts, and synthesizes answers.

Usage:
    python trinity_executor.py <plan.json>          # Execute a plan file
    python trinity_executor.py --stdin               # Read plan from stdin

    # Programmatic usage:
    from trinity_executor import PlanExecutor
    executor = PlanExecutor()
    result = executor.execute_plan(plan_dict)
    print(result.answer)
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import textwrap
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Set


# ============================================================================
# Data types (mirrors the Trinity Spec contract)
# ============================================================================

@dataclass(frozen=True)
class Step:
    """A single artifact application step in the plan."""
    step_id: str
    description: str
    artifact_type: str       # inline_python | notebook | shell
    artifact_ref: str        # code or path
    inputs: dict
    expected_fact: str
    extraction_expr: str = "result"
    depends_on: tuple = ()


@dataclass(frozen=True)
class Fact:
    """A structured fact extracted from an artifact execution."""
    step_id: str
    name: str
    value: str
    fact_type: str           # numeric | text | boolean | json | error


@dataclass(frozen=True)
class ExecutionResult:
    """The final result: collected facts and synthesized answer."""
    question: str
    facts: tuple
    answer: str
    success: bool


@dataclass
class StepResult:
    """Internal result of executing a single step."""
    step_id: str
    success: bool
    raw_output: Any = None
    extracted_value: Any = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


# ============================================================================
# Plan Parser
# ============================================================================

class PlanParser:
    """Parses and validates plan JSON into Step objects."""

    VALID_ARTIFACT_TYPES = frozenset({
        "inline_python", "notebook", "shell"
    })

    @classmethod
    def parse_file(cls, path: str) -> dict:
        """Read a plan from a JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def parse_string(cls, s: str) -> dict:
        """Read a plan from a JSON string."""
        return json.loads(s)

    @classmethod
    def validate_and_build(cls, raw: dict) -> Tuple[str, List[Step], str]:
        """Validate raw plan dict and return (question, steps, synthesis).

        Raises ValueError on validation failure.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"Plan must be a dict, got {type(raw).__name__}")

        question = raw.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Plan must have a non-empty 'question' field")

        synthesis = raw.get("synthesis", "Combine all facts to answer the question.")
        if not isinstance(synthesis, str):
            synthesis = str(synthesis)

        raw_steps = raw.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Plan must have a non-empty 'steps' list")

        steps = []
        seen_ids: Set[str] = set()

        for i, rs in enumerate(raw_steps):
            if not isinstance(rs, dict):
                raise ValueError(f"steps[{i}] must be a dict")

            step_id = rs.get("step_id", f"step_{i}")
            if step_id in seen_ids:
                raise ValueError(f"Duplicate step_id: {step_id!r}")
            seen_ids.add(step_id)

            artifact_type = rs.get("artifact_type", "inline_python")
            if artifact_type not in cls.VALID_ARTIFACT_TYPES:
                raise ValueError(
                    f"steps[{i}].artifact_type must be one of "
                    f"{sorted(cls.VALID_ARTIFACT_TYPES)}, got {artifact_type!r}"
                )

            artifact_ref = rs.get("artifact_ref", "")
            if not isinstance(artifact_ref, str) or not artifact_ref.strip():
                raise ValueError(f"steps[{i}].artifact_ref must be a non-empty string")

            depends_on_raw = rs.get("depends_on", [])
            if isinstance(depends_on_raw, (list, tuple)):
                depends_on = tuple(str(d) for d in depends_on_raw)
            else:
                depends_on = ()

            steps.append(Step(
                step_id=str(step_id).strip(),
                description=str(rs.get("description", "")).strip(),
                artifact_type=artifact_type,
                artifact_ref=artifact_ref,
                inputs=rs.get("inputs", {}),
                expected_fact=str(rs.get("expected_fact", f"fact_{i}")).strip(),
                extraction_expr=str(rs.get("extraction_expr", "result")).strip(),
                depends_on=depends_on,
            ))

        # Validate dependency references
        all_ids = {s.step_id for s in steps}
        for step in steps:
            for dep in step.depends_on:
                if dep not in all_ids:
                    raise ValueError(
                        f"Step {step.step_id!r} depends on {dep!r} which does not exist"
                    )

        # Check for cycles
        cls._check_cycles(steps)

        return question.strip(), steps, synthesis.strip()

    @classmethod
    def _check_cycles(cls, steps: List[Step]) -> None:
        """Detect circular dependencies via DFS."""
        dep_map = {s.step_id: s.depends_on for s in steps}
        visited: Set[str] = set()
        path: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for dep in dep_map.get(node, ()):
                if dfs(dep):
                    return True
            path.discard(node)
            return False

        for s in steps:
            if dfs(s.step_id):
                raise ValueError(f"Circular dependency detected involving {s.step_id!r}")


# ============================================================================
# Dependency Resolver (Topological Sort)
# ============================================================================

class DependencyResolver:
    """Resolves step execution order using topological sort (Kahn's algorithm)."""

    @staticmethod
    def resolve(steps: List[Step]) -> List[Step]:
        """Return steps in valid execution order (topological sort)."""
        step_map = {s.step_id: s for s in steps}
        in_degree: Dict[str, int] = {s.step_id: 0 for s in steps}
        dependents: Dict[str, List[str]] = {s.step_id: [] for s in steps}

        for s in steps:
            for dep in s.depends_on:
                dependents[dep].append(s.step_id)
                in_degree[s.step_id] += 1

        # Start with nodes that have no dependencies
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        # Sort for deterministic order
        queue.sort()
        ordered = []

        while queue:
            current = queue.pop(0)
            ordered.append(step_map[current])
            for dependent in sorted(dependents[current]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        if len(ordered) != len(steps):
            raise ValueError("Could not resolve all dependencies (cycle detected)")

        return ordered


# ============================================================================
# Input Resolver
# ============================================================================

class InputResolver:
    """Resolves step inputs, replacing $fact references with actual fact values."""

    @staticmethod
    def resolve(inputs: dict, facts: Dict[str, Fact]) -> dict:
        """Deep-resolve inputs, replacing {"$fact": "name"} with fact values."""
        return InputResolver._resolve_value(inputs, facts)

    @staticmethod
    def _resolve_value(value: Any, facts: Dict[str, Fact]) -> Any:
        if isinstance(value, dict):
            # Check if this is a fact reference
            if len(value) == 1 and "$fact" in value:
                fact_name = value["$fact"]
                if fact_name in facts:
                    fact = facts[fact_name]
                    return InputResolver._deserialize_fact_value(fact)
                else:
                    raise ValueError(f"Referenced fact {fact_name!r} not found")
            # Otherwise recurse into dict
            return {k: InputResolver._resolve_value(v, facts) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            resolved = [InputResolver._resolve_value(item, facts) for item in value]
            return type(value)(resolved) if isinstance(value, tuple) else resolved
        else:
            return value

    @staticmethod
    def _deserialize_fact_value(fact: Fact) -> Any:
        """Deserialize a fact value based on its type."""
        if fact.fact_type == "numeric":
            try:
                return int(fact.value)
            except ValueError:
                try:
                    return float(fact.value)
                except ValueError:
                    return fact.value
        elif fact.fact_type == "boolean":
            return fact.value.lower() in ("true", "1", "yes")
        elif fact.fact_type == "json":
            try:
                return json.loads(fact.value)
            except (json.JSONDecodeError, TypeError):
                return fact.value
        elif fact.fact_type == "error":
            return fact.value
        else:  # text
            return fact.value


# ============================================================================
# Step Executors
# ============================================================================

class InlinePythonExecutor:
    """Executes inline Python code and extracts a fact."""

    @staticmethod
    def execute(step: Step, resolved_inputs: dict) -> StepResult:
        """Execute inline Python code from artifact_ref."""
        code = step.artifact_ref

        # Build namespace with inputs
        namespace = {"__builtins__": __builtins__}
        namespace.update(resolved_inputs)

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            # Compile and execute
            compiled = compile(code, f"<step:{step.step_id}>", "exec")
            exec(compiled, namespace)

            # Extract the fact
            try:
                extracted = eval(step.extraction_expr, namespace)
            except Exception as e:
                # Try as a simple variable lookup
                extracted = namespace.get(step.extraction_expr, 
                    f"Extraction failed: {e}")

            return StepResult(
                step_id=step.step_id,
                success=True,
                raw_output=namespace,
                extracted_value=extracted,
                stdout=captured_stdout.getvalue(),
                stderr=captured_stderr.getvalue(),
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                stdout=captured_stdout.getvalue(),
                stderr=captured_stderr.getvalue(),
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class ShellExecutor:
    """Executes a shell command."""

    @staticmethod
    def execute(step: Step, resolved_inputs: dict) -> StepResult:
        """Execute a shell command."""
        command = step.artifact_ref

        # Build environment
        env = os.environ.copy()
        for k, v in resolved_inputs.items():
            env[str(k)] = str(v)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

            success = proc.returncode == 0
            output = proc.stdout

            # Extract fact
            namespace = {"result": output, "returncode": proc.returncode,
                        "stdout": proc.stdout, "stderr": proc.stderr}
            try:
                extracted = eval(step.extraction_expr, namespace)
            except Exception:
                extracted = output

            return StepResult(
                step_id=step.step_id,
                success=success,
                raw_output=output,
                extracted_value=extracted,
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=proc.stderr if not success else None,
            )
        except subprocess.TimeoutExpired:
            return StepResult(
                step_id=step.step_id,
                success=False,
                error="Command timed out after 300 seconds",
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )


class NotebookExecutor:
    """Executes a Jupyter notebook (best-effort)."""

    @staticmethod
    def execute(step: Step, resolved_inputs: dict) -> StepResult:
        """Execute a notebook by extracting and running its code cells."""
        notebook_path = step.artifact_ref

        try:
            with open(notebook_path, "r") as f:
                nb = json.load(f)

            # Extract code cells
            code_cells = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    if source.strip():
                        code_cells.append(source)

            # Execute all code cells in a shared namespace
            namespace = {"__builtins__": __builtins__}
            namespace.update(resolved_inputs)

            for i, code in enumerate(code_cells):
                try:
                    compiled = compile(code, f"<notebook:{step.step_id}:cell{i}>", "exec")
                    exec(compiled, namespace)
                except Exception as e:
                    pass  # Continue on cell errors (like notebooks do)

            # Extract fact
            try:
                extracted = eval(step.extraction_expr, namespace)
            except Exception:
                extracted = f"Executed {len(code_cells)} cells from {notebook_path}"

            return StepResult(
                step_id=step.step_id,
                success=True,
                raw_output=namespace,
                extracted_value=extracted,
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )


# ============================================================================
# Fact Builder
# ============================================================================

class FactBuilder:
    """Builds Fact objects from step results."""

    @staticmethod
    def build(step: Step, step_result: StepResult) -> Fact:
        """Create a Fact from a StepResult."""
        if not step_result.success:
            return Fact(
                step_id=step.step_id,
                name=step.expected_fact,
                value=step_result.error or "Unknown error",
                fact_type="error",
            )

        value = step_result.extracted_value
        fact_type = FactBuilder._infer_type(value)
        serialized = FactBuilder._serialize(value)

        return Fact(
            step_id=step.step_id,
            name=step.expected_fact,
            value=serialized,
            fact_type=fact_type,
        )

    @staticmethod
    def _infer_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "numeric"
        elif isinstance(value, str):
            return "text"
        elif isinstance(value, (dict, list, tuple)):
            return "json"
        else:
            return "text"

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, default=str, indent=2)
            except (TypeError, ValueError):
                return str(value)
        else:
            return str(value)


# ============================================================================
# Plan Executor (the main orchestrator)
# ============================================================================

class PlanExecutor:
    """Orchestrates the execution of a Trinity plan.

    This is the main entry point. It:
    1. Parses and validates the plan
    2. Resolves the dependency DAG into execution order
    3. Executes each step with the appropriate executor
    4. Resolves inputs (including $fact references)
    5. Extracts facts from results
    6. Synthesizes a final answer
    """

    EXECUTORS = {
        "inline_python": InlinePythonExecutor,
        "shell": ShellExecutor,
        "notebook": NotebookExecutor,
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._facts: Dict[str, Fact] = OrderedDict()
        self._step_results: Dict[str, StepResult] = OrderedDict()

    def execute_plan(self, plan: dict) -> ExecutionResult:
        """Execute a complete plan and return the result."""
        # Parse and validate
        question, steps, synthesis = PlanParser.validate_and_build(plan)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Steps: {len(steps)}")
            print(f"{'='*60}\n")

        # Resolve execution order
        ordered_steps = DependencyResolver.resolve(steps)

        if self.verbose:
            print("Execution order:")
            for i, s in enumerate(ordered_steps):
                deps = f" (depends on: {', '.join(s.depends_on)})" if s.depends_on else ""
                print(f"  {i+1}. {s.step_id}: {s.description}{deps}")
            print()

        # Execute each step
        all_success = True
        for step in ordered_steps:
            if self.verbose:
                print(f"--- Executing: {step.step_id} ---")
                print(f"    Type: {step.artifact_type}")
                print(f"    Description: {step.description}")

            # Resolve inputs
            try:
                resolved_inputs = InputResolver.resolve(step.inputs, self._facts)
            except ValueError as e:
                step_result = StepResult(
                    step_id=step.step_id,
                    success=False,
                    error=f"Input resolution failed: {e}",
                )
                self._step_results[step.step_id] = step_result
                fact = FactBuilder.build(step, step_result)
                self._facts[step.expected_fact] = fact
                all_success = False
                if self.verbose:
                    print(f"    FAILED: {step_result.error}\n")
                continue

            # Get the appropriate executor
            executor_cls = self.EXECUTORS.get(step.artifact_type)
            if executor_cls is None:
                step_result = StepResult(
                    step_id=step.step_id,
                    success=False,
                    error=f"Unknown artifact type: {step.artifact_type}",
                )
            else:
                step_result = executor_cls.execute(step, resolved_inputs)

            self._step_results[step.step_id] = step_result

            # Build fact
            fact = FactBuilder.build(step, step_result)
            self._facts[step.expected_fact] = fact

            if not step_result.success:
                all_success = False

            if self.verbose:
                status = "OK" if step_result.success else "FAILED"
                print(f"    Status: {status}")
                print(f"    Fact: {fact.name} = {fact.value[:200]}..." 
                      if len(fact.value) > 200 
                      else f"    Fact: {fact.name} = {fact.value}")
                if step_result.error:
                    print(f"    Error: {step_result.error[:200]}")
                print()

        # Synthesize answer
        answer = self._synthesize(question, synthesis, self._facts, all_success)

        return ExecutionResult(
            question=question,
            facts=tuple(self._facts.values()),
            answer=answer,
            success=all_success,
        )

    def execute_file(self, path: str) -> ExecutionResult:
        """Execute a plan from a JSON file."""
        plan = PlanParser.parse_file(path)
        return self.execute_plan(plan)

    def execute_string(self, s: str) -> ExecutionResult:
        """Execute a plan from a JSON string."""
        plan = PlanParser.parse_string(s)
        return self.execute_plan(plan)

    def get_facts(self) -> Dict[str, Fact]:
        """Return all collected facts."""
        return dict(self._facts)

    def get_step_results(self) -> Dict[str, StepResult]:
        """Return all step results."""
        return dict(self._step_results)

    def _synthesize(
        self, 
        question: str, 
        synthesis: str, 
        facts: Dict[str, Fact], 
        all_success: bool
    ) -> str:
        """Synthesize a final answer from all collected facts."""
        parts = []
        parts.append(f"Question: {question}")
        parts.append(f"")
        parts.append(f"Synthesis strategy: {synthesis}")
        parts.append(f"")
        parts.append(f"Collected Facts ({len(facts)}):")

        for name, fact in facts.items():
            status = "ERROR" if fact.fact_type == "error" else "OK"
            value_preview = fact.value[:500] if len(fact.value) > 500 else fact.value
            parts.append(f"  [{status}] {name} ({fact.fact_type}): {value_preview}")

        parts.append(f"")
        if all_success:
            parts.append("All steps completed successfully.")
        else:
            failed = [n for n, f in facts.items() if f.fact_type == "error"]
            parts.append(f"Some steps failed: {', '.join(failed)}")

        return "\n".join(parts)

    def reset(self):
        """Reset executor state for a new plan."""
        self._facts.clear()
        self._step_results.clear()


# ============================================================================
# Result Serialization
# ============================================================================

def serialize_result(result: ExecutionResult) -> str:
    """Serialize an ExecutionResult to JSON."""
    return json.dumps({
        "question": result.question,
        "facts": [
            {
                "step_id": f.step_id,
                "name": f.name,
                "value": f.value,
                "fact_type": f.fact_type,
            }
            for f in result.facts
        ],
        "answer": result.answer,
        "success": result.success,
    }, indent=2)


# ============================================================================
# Convenience function
# ============================================================================

def run(plan: Optional[dict] = None, plan_path: Optional[str] = None, 
        verbose: bool = False, **kwargs) -> ExecutionResult:
    """Convenience function to execute a Trinity plan.

    Args:
        plan: A plan dict (takes priority)
        plan_path: Path to a plan.json file
        verbose: Print execution details
        **kwargs: Additional inputs available to all steps

    Returns:
        ExecutionResult with facts and synthesized answer
    """
    executor = PlanExecutor(verbose=verbose)

    if plan is not None:
        return executor.execute_plan(plan)
    elif plan_path is not None:
        return executor.execute_file(plan_path)
    else:
        raise ValueError("Either plan or plan_path must be provided")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Trinity Plan Executor - Execute Trinity Spec plans"
    )
    parser.add_argument(
        "plan_file", nargs="?", default=None,
        help="Path to plan.json file"
    )
    parser.add_argument(
        "--stdin", action="store_true",
        help="Read plan from stdin"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print execution details"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file for results (default: stdout)"
    )

    args = parser.parse_args()

    executor = PlanExecutor(verbose=args.verbose)

    if args.stdin:
        plan_str = sys.stdin.read()
        result = executor.execute_string(plan_str)
    elif args.plan_file:
        result = executor.execute_file(args.plan_file)
    else:
        parser.print_help()
        sys.exit(1)

    output = serialize_result(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if args.verbose:
            print(f"\nResults written to {args.output}")
    else:
        print(output)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
