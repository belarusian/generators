"""
Entry point for `python -m compass` and `generator` console script.

Usage:
    generate --name trinity --prompt "..." --model-id qwen3.5:122b@big
    generate --name meta --prompt "Build a generator for ..."
    generate --name neo --prompt "Build a fibonacci module with tests"
    generate --list

    python -m compass --name trinity --prompt "..."
"""
import importlib
import sys
from pathlib import Path


def _load_env() -> None:
    """Load .env files for API keys etc."""
    import os
    from dotenv import load_dotenv

    _global_env = os.path.join(os.path.expanduser("~"), ".compass", ".env")
    if os.path.exists(_global_env):
        load_dotenv(_global_env)
    load_dotenv(override=True)


def _discover_generators() -> dict[str, str]:
    """Auto-discover all generators with a generate.py."""
    root = Path(__file__).parent / "generators"
    generators = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "generate.py").exists():
            generators[child.name] = f"compass.generators.{child.name}.generate"
    return generators


def main() -> int:
    """Route to the right generator by --name flag.

    All other arguments are passed through to the generator's main().
    """
    _load_env()

    args = sys.argv[1:]

    # Handle --list
    if "--list" in args:
        generators = _discover_generators()
        print("Available generators:")
        for name in generators:
            print(f"  {name}")
        return 0

    # Extract --name
    gen_name = None
    passthrough = list(args)
    for i, arg in enumerate(passthrough):
        if arg == "--name" and i + 1 < len(passthrough):
            gen_name = passthrough[i + 1]
            passthrough = passthrough[:i] + passthrough[i + 2:]
            break

    # --live with no --name defaults to trinity
    if gen_name is None and "--live" in passthrough:
        gen_name = "trinity"

    # --neo launches Neo's full REPL (compass.cli.main)
    if "--neo" in passthrough:
        neo_args = [a for a in passthrough if a != "--neo"]
        if "--live" not in neo_args:
            neo_args.append("--live")
        if "--code" not in neo_args:
            neo_args.append("--code")
        from compass.cli.main import main as neo_main
        neo_main(neo_args)
        return 0

    if gen_name is None:
        print("Usage: generate --name <generator> [--prompt '...'] [options]", file=sys.stderr)
        print("       generate --live", file=sys.stderr)
        print("       generate --neo", file=sys.stderr)
        print("       generate --list", file=sys.stderr)
        return 1

    generators = _discover_generators()
    module_path = generators.get(gen_name)
    if module_path is None:
        print(f"Unknown generator: {gen_name}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(generators))}", file=sys.stderr)
        return 1

    # Set sys.argv for the sub-generator's argparse
    sys.argv = [f"generator --name {gen_name}"] + passthrough

    mod = importlib.import_module(module_path)
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
