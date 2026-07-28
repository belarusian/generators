"""Context builder for the CLI tool generator.

Builds GenerationContext with domain knowledge about CLI tool patterns,
argparse best practices, and error handling conventions.

Context sections:
1. CLI tool patterns and best practices
2. Argparse reference (subcommands, arguments, help text)
3. Error handling conventions
"""

from __future__ import annotations

from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext


_CLI_PATTERNS = """\
Python CLI tool patterns and best practices:

1. STRUCTURE:
   - Single-file script with #!/usr/bin/env python3 shebang
   - Module docstring describing the tool
   - __version__ = 'X.Y.Z' at module level
   - One handler function per subcommand: handle_<name>(args)
   - main() function that sets up argparse and dispatches
   - if __name__ == '__main__': sys.exit(main())

2. ARGPARSE SETUP:
   - parser = argparse.ArgumentParser(description='...')
   - Global flags BEFORE subparsers:
     parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
     parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
   - subparsers = parser.add_subparsers(dest='command', help='Available commands')
   - Each subcommand: sub = subparsers.add_parser('name', help='...')
   - Arguments: sub.add_argument('--flag', type=str, required=True, help='...')

3. DISPATCH:
   - args = parser.parse_args()
   - if args.command is None: parser.print_help(); return 1
   - Use a dict mapping command names to handler functions
   - handlers = {'cmd1': handle_cmd1, 'cmd2': handle_cmd2}
   - return handlers[args.command](args)

4. ERROR HANDLING:
   - Wrap handler dispatch in try/except
   - Catch specific exceptions where possible
   - Print user-friendly error messages to stderr
   - If --verbose, print full traceback
   - Return 0 for success, 1 for error

5. HANDLER FUNCTIONS:
   - def handle_<name>(args) -> int:
   - Access args.verbose for verbose output
   - Return 0 for success, 1 for error
   - Use print() for normal output
   - Use print(..., file=sys.stderr) for errors

6. PYTHON SOURCE QUALITY:
   - Use f-strings for string formatting
   - Use triple-quoted strings for multi-line text
   - NEVER use string concatenation with + inside function calls
   - All imports from standard library only
   - Proper 4-space indentation
"""


def _discover_available_packages() -> str:
    """List installed Python packages for the model's awareness."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return ""


def build_cli_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build context for CLI tool generation.

    The model needs:
    1. CLI tool patterns and best practices
    2. The prompt describing what tool to build
    """
    patterns_section = DomainSection(
        heading="CLI Tool Patterns",
        content=_CLI_PATTERNS,
    )

    return GenerationContext(
        domain_context=(patterns_section,),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )
