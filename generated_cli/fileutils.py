#!/usr/bin/env python3
"""fileutils - A file utility CLI tool for counting lines, searching patterns, and displaying file metadata."""

import argparse
import os
import re
import sys
import time

__version__ = '1.0.0'


def handle_count(args):
    """Count the number of lines in a file."""
    path = args.path
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1
    with open(path, 'r') as f:
        lines = f.readlines()
    count = len(lines)
    if args.verbose:
        non_empty = sum(1 for line in lines if line.strip())
        print(f"File: {path}")
        print(f"Total lines: {count}")
        print(f"Non-empty lines: {non_empty}")
        print(f"Empty lines: {count - non_empty}")
    else:
        print(f"{count}")
    return 0


def handle_search(args):
    """Search for a pattern in files within a directory."""
    directory = args.directory
    pattern = args.pattern
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return 1
    compiled = re.compile(pattern)
    total_matches = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled.search(line):
                            total_matches += 1
                            print(f"{filepath}:{line_num}: {line.rstrip()}")
            except (IOError, OSError) as e:
                if args.verbose:
                    print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    if args.verbose:
        print(f"\nTotal matches found: {total_matches}")
    return 0


def handle_info(args):
    """Show file metadata and information."""
    path = args.path
    if not os.path.exists(path):
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1
    stat = os.stat(path)
    size = stat.st_size
    mtime = time.ctime(stat.st_mtime)
    atime = time.ctime(stat.st_atime)
    ctime = time.ctime(stat.st_ctime)
    print(f"Path: {path}")
    print(f"Size: {size} bytes")
    print(f"Modified: {mtime}")
    if args.verbose:
        print(f"Accessed: {atime}")
        print(f"Created: {ctime}")
        print(f"Is file: {os.path.isfile(path)}")
        print(f"Is directory: {os.path.isdir(path)}")
        print(f"Is symlink: {os.path.islink(path)}")
        print(f"Absolute path: {os.path.abspath(path)}")
        mode = oct(stat.st_mode)
        print(f"Mode: {mode}")
    return 0


def main():
    """Main entry point for the fileutils CLI tool."""
    parser = argparse.ArgumentParser(
        description='A file utility CLI tool for counting lines, searching patterns, and displaying file metadata.'
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # count subcommand
    count_parser = subparsers.add_parser('count', help='Count the number of lines in a file')
    count_parser.add_argument('--path', type=str, required=True, help='Path to the file to count lines in')

    # search subcommand
    search_parser = subparsers.add_parser('search', help='Search for a pattern in files within a directory')
    search_parser.add_argument('--pattern', type=str, required=True, help='The regex pattern to search for')
    search_parser.add_argument('--directory', type=str, required=True, help='The directory to search in')

    # info subcommand
    info_parser = subparsers.add_parser('info', help='Show file metadata and information')
    info_parser.add_argument('--path', type=str, required=True, help='Path to the file to inspect')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        'count': handle_count,
        'search': handle_search,
        'info': handle_info,
    }

    try:
        return handlers[args.command](args)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
