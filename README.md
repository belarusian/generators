# Compass

Compass is a local-first Python framework for agentic coding and artifact generation.
It combines pluggable LLM backends, reusable generation loops, and an NFA-based agent
runtime to plan work, produce artifacts, validate them, and iterate when execution fails.

## Highlights

- Local-first workflow oriented around your filesystem, shell, and workspace
- Pluggable model providers for Anthropic, Ollama, and llama.cpp
- Reusable generator architecture for code, notebooks, plans, and meta-generation
- Interactive CLI and agent runtime for multi-step coding workflows
- Built-in validation and repair loops instead of one-shot generation only
- Optional companion integration for Neo skill/state workflows

## Included Components

### Core framework

- `compass/core/`: shared execution primitives, NFA runner, memory, and streaming utilities
- `compass/llm/`: provider abstraction, model routing, and Oracle interface
- `compass/cli/`: command-line entry points and interactive code mode

### Agent runtime

- `compass/agents/neo/`: the main coding agent loop
- Supports file operations, command execution, search/indexing, and optional computer-use actions
- Uses an actor/review/answer style workflow on top of the core state machine

### Generators

- `trinity`: plans and executes multi-step workflows across a workspace
- `code`: generates Python code and tests, then validates with syntax checks and execution
- `notebook`: generates notebooks and validates execution and variable flow
- `meta`: generates new generators that follow the same framework shape
- `neo`: generates executable plans for downstream work
- `cli_tool`: generates command-line tool artifacts

## Install

```bash
./setup.sh
source .venv/bin/activate
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

List available generators:

```bash
generate --list
```

Generate Python code:

```bash
generate --name code --prompt "Build a stack class with tests"
```

Run a Trinity workflow:

```bash
generate --name trinity --prompt "Investigate this repository and summarize its architecture"
```

Generate a notebook:

```bash
generate --name notebook --prompt "Create a notebook that demonstrates decorators"
```

Generate a generator:

```bash
generate --name meta --prompt "Build a generator for small Python CLI tools"
```

Launch Trinity's interactive REPL:

```bash
generate --live
```

Launch Trinity's REPL with prior-answer history included in context:

```bash
generate --history --live
```

Launch Neo's interactive code mode:

```bash
generate --neo
```

## Configuration

Primary environment variables:

- `ANTHROPIC_API_KEY`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `LLAMACPP_HOST`
- `OCR_URL`
- `DETECT_URL`
- `COMPASS_MODEL`

Optional integration:

- `COMPASS_NEO_LAB` points Neo's skill/state integration at a companion `neo-lab`
workspace. This only affects the optional skill/state features used by Neo planning
and computer-use actions.
- `NEO_OCR_SERVER` and `NEO_DINO_SERVER` are also accepted as compatibility aliases
when Compass is paired with a `neo-lab` setup.

Start from [.env.example](.env.example) and copy it
to `.env` or `~/.compass/.env`.

## Development

Run the test suite:

```bash
pytest
```

## Status

Compass is still experimental. Interfaces and generator contracts are evolving,
especially around agent workflows, validation strategy, and artifact execution.

## License

MIT. See [LICENSE](LICENSE).