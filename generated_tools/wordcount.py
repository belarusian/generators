#!/usr/bin/env python3
"""wordcount - Read text files and print the number of lines, words, and characters.

A CLI tool similar to the Unix `wc` command. Accepts one or more file paths,
counts lines, words, and characters for each file, and prints a summary.
Supports verbose per-file breakdowns and JSON output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def count_file(filepath: Path) -> dict:
    """Count lines, words, and characters in a single file.

    Args:
        filepath: Path to the file to count.

    Returns:
        A dict with keys 'file', 'lines', 'words', 'characters'.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        OSError: For other I/O errors.
    """
    logger.debug("Reading file: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.count("\n")
    words = len(content.split())
    characters = len(content)

    logger.debug(
        "File %s: %d lines, %d words, %d characters",
        filepath, lines, words, characters,
    )

    return {
        "file": str(filepath),
        "lines": lines,
        "words": words,
        "characters": characters,
    }


def format_plain(results: list, totals: dict, verbose: bool, multiple: bool) -> str:
    """Format results as plain text.

    Args:
        results: List of per-file result dicts.
        totals: Aggregated totals dict.
        verbose: Whether to show per-file breakdowns.
        multiple: Whether there are multiple files.

    Returns:
        Formatted string for stdout.
    """
    lines_out = []

    if verbose or multiple:
        for r in results:
            lines_out.append(
                f"  {r['lines']:>8}  {r['words']:>8}  {r['characters']:>8}  {r['file']}"
            )

    label = "total" if multiple else (results[0]["file"] if results else "total")
    if verbose or multiple:
        lines_out.append(
            f"  {totals['lines']:>8}  {totals['words']:>8}  {totals['characters']:>8}  {label}"
        )
    else:
        lines_out.append(
            f"  {totals['lines']:>8}  {totals['words']:>8}  {totals['characters']:>8}  {label}"
        )

    header = "    lines     words     chars  file"
    return header + "\n" + "\n".join(lines_out)


def format_json(results: list, totals: dict) -> str:
    """Format results as JSON.

    Args:
        results: List of per-file result dicts.
        totals: Aggregated totals dict.

    Returns:
        JSON string.
    """
    output = {
        "files": results,
        "total": totals,
    }
    return json.dumps(output, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="wordcount",
        description=(
            "Read text files and print the number of lines, words, and characters. "
            "Similar to the Unix wc command."
        ),
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more file paths to count.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output with per-file breakdowns and debug logging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output results as JSON instead of plain text.",
    )
    return parser


def main() -> None:
    """Main entry point for the wordcount CLI tool."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging based on verbose flag
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    results = []
    errors = []
    totals = {"lines": 0, "words": 0, "characters": 0}

    for filepath_str in args.files:
        filepath = Path(filepath_str)

        if not filepath.exists():
            msg = f"File not found: {filepath}"
            logger.error(msg)
            errors.append(msg)
            print(f"Error: {msg}", file=sys.stderr)
            continue

        if not filepath.is_file():
            msg = f"Not a regular file: {filepath}"
            logger.error(msg)
            errors.append(msg)
            print(f"Error: {msg}", file=sys.stderr)
            continue

        try:
            result = count_file(filepath)
            results.append(result)
            totals["lines"] += result["lines"]
            totals["words"] += result["words"]
            totals["characters"] += result["characters"]
        except PermissionError:
            msg = f"Permission denied: {filepath}"
            logger.error(msg)
            errors.append(msg)
            print(f"Error: {msg}", file=sys.stderr)
        except UnicodeDecodeError:
            msg = f"Cannot decode file as UTF-8: {filepath}"
            logger.error(msg)
            errors.append(msg)
            print(f"Error: {msg}", file=sys.stderr)
        except OSError as exc:
            msg = f"Could not read {filepath}: {exc}"
            logger.error(msg)
            errors.append(msg)
            print(f"Error: {msg}", file=sys.stderr)

    if not results and errors:
        # All files failed
        print("Error: No files could be read.", file=sys.stderr)
        sys.exit(1)

    if not results and not errors:
        # Should not happen with nargs='+', but just in case
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Format and print output
    multiple = len(results) > 1

    if args.json_output:
        output = format_json(results, totals)
    else:
        output = format_plain(results, totals, args.verbose, multiple)

    print(output)

    # Exit with error code if some files failed
    if errors:
        logger.warning("%d file(s) could not be read.", len(errors))
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
