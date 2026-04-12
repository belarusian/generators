"""
Codebase index for code mode.

Provides the Planner with visibility into the codebase structure:
- File tree with sizes
- Function and class definitions
- Import graph (what imports what)

Configuration via .compass/index.json:
- exclude: paths to skip during indexing
- priority_classes: classes shown first in context
- priority_functions: functions shown first in context
- max_context_chars: context size limit
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FunctionDef:
    """A function definition."""
    name: str
    file: str
    line: int
    args: List[str]
    docstring: Optional[str] = None


@dataclass
class MethodDef:
    """A method definition within a class."""
    name: str
    class_name: str
    file: str
    line: int
    args: List[str]
    docstring: Optional[str] = None


@dataclass
class ExportedSymbol:
    """A public symbol re-exported by a package."""
    name: str
    source: str


@dataclass
class ClassDef:
    """A class definition."""
    name: str
    file: str
    line: int
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class FileSummary:
    """Summary of a single file."""
    path: str
    lines: int
    imports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)


@dataclass
class IndexConfig:
    """Configuration for codebase indexing."""
    exclude: List[str] = field(default_factory=lambda: ["tests/", "scripts/"])
    include_classes: List[str] = field(default_factory=list)  # Force include even if in excluded path
    exclude_classes: List[str] = field(default_factory=list)  # Never show these
    include_functions: List[str] = field(default_factory=list)  # Force include even if in excluded path
    exclude_functions: List[str] = field(default_factory=list)  # Never show these
    priority_classes: List[str] = field(default_factory=list)
    priority_functions: List[str] = field(default_factory=list)
    max_context_chars: int = 12000

    @classmethod
    def load(cls, project_path: str) -> "IndexConfig":
        """Load config from .compass/index.json or return defaults."""
        config_path = Path(project_path) / ".compass" / "index.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                return cls(
                    exclude=data.get("exclude", ["tests/", "scripts/"]),
                    include_classes=data.get("include_classes", []),
                    exclude_classes=data.get("exclude_classes", []),
                    include_functions=data.get("include_functions", []),
                    exclude_functions=data.get("exclude_functions", []),
                    priority_classes=data.get("priority_classes", []),
                    priority_functions=data.get("priority_functions", []),
                    max_context_chars=data.get("max_context_chars", 12000),
                )
            except Exception:
                pass
        return cls()

    def save(self, project_path: str) -> None:
        """Save config to .compass/index.json."""
        config_dir = Path(project_path) / ".compass"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "index.json"

        data = {
            "_comment": "Codebase index configuration - edit to customize what the Oracle sees",
            "exclude": self.exclude,
            "include_classes": self.include_classes,
            "exclude_classes": self.exclude_classes,
            "include_functions": self.include_functions,
            "exclude_functions": self.exclude_functions,
            "priority_classes": self.priority_classes,
            "priority_functions": self.priority_functions,
            "max_context_chars": self.max_context_chars,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded."""
        for pattern in self.exclude:
            if path.startswith(pattern):
                return True
        return False


@dataclass
class CodebaseIndex:
    """Index of the entire codebase."""
    root: str
    files: Dict[str, FileSummary] = field(default_factory=dict)
    functions: Dict[str, FunctionDef] = field(default_factory=dict)
    classes: Dict[str, ClassDef] = field(default_factory=dict)
    methods: Dict[str, MethodDef] = field(default_factory=dict)
    public_api: Dict[str, List[ExportedSymbol]] = field(default_factory=dict)
    config: IndexConfig = field(default_factory=IndexConfig)

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "file_count": len(self.files),
            "function_count": len(self.functions),
            "class_count": len(self.classes),
            "method_count": len(self.methods),
        }

    def dump(self, path: str = None) -> str:
        """
        Dump full index to a file for inspection.

        Creates .compass/index_dump.txt with all indexed items.
        Use this to see what's indexed, then edit index.json to correct.
        """
        lines = [
            "# CODEBASE INDEX DUMP",
            "# This file shows everything that was indexed.",
            "# To control what the Oracle sees, edit index.json:",
            "#   - exclude_classes: [...] to hide specific classes",
            "#   - exclude_functions: [...] to hide specific functions",
            "#   - include_classes: [...] to force-show classes from excluded paths",
            "#   - priority_classes: [...] to show classes first",
            "",
            f"# Generated from: {self.root}",
            f"# Files: {len(self.files)}, Classes: {len(self.classes)}, Functions: {len(self.functions)}, Methods: {len(self.methods)}",
            "",
            "=" * 60,
            "CLASSES",
            "=" * 60,
        ]

        for name, cls in sorted(self.classes.items()):
            lines.append(f"{cls.name}")
            lines.append(f"  file: {cls.file}:{cls.line}")
            if cls.docstring:
                lines.append(f"  doc: {cls.docstring[:80]}")
            if cls.methods:
                lines.append(f"  methods: {', '.join(cls.methods)}")
            lines.append("")

        lines.extend([
            "=" * 60,
            "FUNCTIONS",
            "=" * 60,
        ])

        for name, fn in sorted(self.functions.items()):
            args = ", ".join(fn.args[:5])
            if len(fn.args) > 5:
                args += ", ..."
            lines.append(f"{fn.name}({args})")
            lines.append(f"  file: {fn.file}:{fn.line}")
            if fn.docstring:
                lines.append(f"  doc: {fn.docstring[:80]}")
            lines.append("")

        lines.extend([
            "=" * 60,
            "METHODS",
            "=" * 60,
        ])

        for name, method in sorted(self.methods.items()):
            args = ", ".join(method.args[:5])
            if len(method.args) > 5:
                args += ", ..."
            lines.append(f"{method.class_name}.{method.name}({args})")
            lines.append(f"  file: {method.file}:{method.line}")
            lines.append("")

        content = "\n".join(lines)

        # Write to file
        if path is None:
            path = Path(self.root) / ".compass" / "index_dump.txt"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        return str(path)

    def get_file_symbols(self, file_path: str, max_symbols: int = 12) -> List[Dict[str, int]]:
        """Get symbol locations for a specific file (from AST index)."""
        symbols: List[Dict[str, int]] = []

        for fn in self.functions.values():
            if fn.file == file_path:
                symbols.append({"name": fn.name, "line": fn.line, "kind": "function"})

        for cls in self.classes.values():
            if cls.file == file_path:
                symbols.append({"name": cls.name, "line": cls.line, "kind": "class"})

        for method in self.methods.values():
            if method.file == file_path:
                symbols.append({
                    "name": f"{method.class_name}.{method.name}",
                    "line": method.line,
                    "kind": "method",
                })

        symbols.sort(key=lambda s: s["line"])
        if max_symbols and len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
        return symbols

    # External packages that indicate API integrations
    EXTERNAL_PACKAGES = {
        'requests', 'httpx', 'aiohttp',  # HTTP clients
        'googlemaps', 'google.cloud', 'google.maps',  # Google APIs
        'openai', 'anthropic',  # LLM APIs
        'boto3', 'botocore',  # AWS
        'stripe', 'twilio', 'sendgrid',  # Payment/comms
        'redis', 'pymongo', 'psycopg2', 'sqlalchemy',  # Databases
    }

    def get_context(self, max_chars: int = None) -> str:
        """
        Get a context string for the Planner.

        Includes:
        - External integrations (API clients detected from imports)
        - Public API (package re-exports)
        - File tree (all files with line counts)
        - Classes with docstrings and methods (priority first, excludes tests)
        - Functions with docstrings (priority first, excludes tests)
        - Truncated to max_chars (from config if not specified)
        """
        if max_chars is None:
            max_chars = self.config.max_context_chars

        lines = []

        # Detect external integrations from imports
        external_imports = set()
        for file_info in self.files.values():
            for imp in file_info.imports:
                pkg = imp.split('.')[0]
                if pkg in self.EXTERNAL_PACKAGES:
                    external_imports.add(pkg)

        if external_imports:
            lines.append("EXTERNAL INTEGRATIONS:")
            for pkg in sorted(external_imports):
                desc = {
                    'requests': 'HTTP client (API calls)',
                    'httpx': 'HTTP client (async API calls)',
                    'googlemaps': 'Google Maps/Places API',
                    'openai': 'OpenAI API',
                    'anthropic': 'Anthropic Claude API',
                    'boto3': 'AWS services',
                    'redis': 'Redis cache/queue',
                }.get(pkg, 'external service')
                lines.append(f"  {pkg} - {desc}")
            lines.append("")

        if self.public_api:
            lines.append("PUBLIC API:")
            for package in sorted(self.public_api.keys()):
                symbols = self.public_api[package]
                lines.append(f"  {package}:")
                for sym in symbols[:25]:
                    lines.append(f"    {sym.name} (from {sym.source})")
                if len(symbols) > 25:
                    lines.append(f"    ... +{len(symbols) - 25} more")
            lines.append("")

        lines.append("CODEBASE STRUCTURE:")
        lines.append("")

        # Group files by directory (respecting exclude config)
        dirs: Dict[str, List[str]] = {}
        for path in sorted(self.files.keys()):
            # Skip excluded paths
            if self.config.should_exclude(path):
                continue
            rel_path = path
            parent = str(Path(rel_path).parent)
            if parent == ".":
                parent = "(root)"
            if parent not in dirs:
                dirs[parent] = []
            filename = Path(rel_path).name
            file_info = self.files[path]
            dirs[parent].append(f"{filename} ({file_info.lines}L)")

        # File tree
        for dir_name in sorted(dirs.keys()):
            lines.append(f"{dir_name}/")
            for f in dirs[dir_name][:15]:  # Max 15 files per dir
                lines.append(f"  {f}")
            if len(dirs[dir_name]) > 15:
                lines.append(f"  ... +{len(dirs[dir_name]) - 15} more")

        lines.append("")
        lines.append("KEY DEFINITIONS:")
        lines.append("")

        # Helper to check if class should be included in context
        def should_show_class(name: str, cls: ClassDef) -> bool:
            # Check explicit exclude by class name
            if any(excl in cls.name for excl in self.config.exclude_classes):
                return False
            # Check explicit include by class name (overrides path exclusion)
            if any(incl in cls.name for incl in self.config.include_classes):
                return True
            # Check path-based exclusion
            if self.config.should_exclude(cls.file):
                return False
            return True

        # Helper to check if function should be included in context
        def should_show_function(name: str, fn: FunctionDef) -> bool:
            # Check explicit exclude by function name
            if any(excl in fn.name for excl in self.config.exclude_functions):
                return False
            # Check explicit include by function name (overrides path exclusion)
            if any(incl in fn.name for incl in self.config.include_functions):
                return True
            # Check path-based exclusion
            if self.config.should_exclude(fn.file):
                return False
            return True

        # Classes with docstrings and methods (priority first, skip excluded)
        # No hard limit here - max_context_chars handles truncation at the end
        if self.classes:
            lines.append("Classes:")
            shown_names = set()

            # Priority classes first
            for priority_name in self.config.priority_classes:
                for name, cls in self.classes.items():
                    if priority_name in name and should_show_class(name, cls):
                        self._append_class(lines, name, cls)
                        shown_names.add(name)
                        break

            # Then non-excluded classes
            for name, cls in self.classes.items():
                if name in shown_names:
                    continue
                if not should_show_class(name, cls):
                    continue
                self._append_class(lines, name, cls)

            lines.append("")

        # Top-level functions (priority first, skip excluded)
        # No hard limit - max_context_chars handles truncation
        if self.functions:
            lines.append("Functions:")
            shown_names = set()

            # Priority functions first
            for priority_name in self.config.priority_functions:
                for name, fn in self.functions.items():
                    if priority_name in name and should_show_function(name, fn):
                        self._append_function(lines, name, fn)
                        shown_names.add(name)
                        break

            # Then non-excluded functions
            for name, fn in self.functions.items():
                if name in shown_names:
                    continue
                if not should_show_function(name, fn):
                    continue
                self._append_function(lines, name, fn)

        result = "\n".join(lines)

        # Truncate if too long
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (index truncated)"

        return result

    def _append_class(self, lines: List[str], name: str, cls: ClassDef) -> None:
        """Append a class definition to the context lines."""
        method_str = ", ".join(cls.methods[:5])
        if len(cls.methods) > 5:
            method_str += f", +{len(cls.methods) - 5} more"
        lines.append(f"  {name} ({cls.file}:{cls.line})")
        if cls.docstring:
            doc_line = cls.docstring.split('\n')[0].strip()
            lines.append(f"    \"{doc_line}\"")
        if method_str:
            lines.append(f"    methods: {method_str}")

    def _append_function(self, lines: List[str], name: str, fn: FunctionDef) -> None:
        """Append a function definition to the context lines."""
        args_str = ", ".join(fn.args[:4])
        if len(fn.args) > 4:
            args_str += ", ..."
        line_entry = f"  {name}({args_str}) - {fn.file}:{fn.line}"
        if fn.docstring:
            doc_line = fn.docstring.split('\n')[0].strip()[:60]
            line_entry += f" \"{doc_line}\""
        lines.append(line_entry)

    def search(self, query: str, search_type: str = "content", max_results: int = 20) -> Dict:
        """
        Search through the codebase.

        Args:
            query: Search query string
            search_type: "content", "function", "class", or "file"
            max_results: Maximum results to return

        Returns:
            Dict with matches
        """
        results = {
            "query": query,
            "type": search_type,
            "matches": [],
        }

        query_lower = query.lower()

        if search_type == "file":
            for file_path in self.files.keys():
                if query_lower in file_path.lower():
                    info = self.files[file_path]
                    results["matches"].append({
                        "file": file_path,
                        "lines": info.lines,
                    })

        elif search_type == "function":
            seen_locations = set()  # (file, line) to dedupe

            # AST-based search on function names (structured info)
            for key, fn in self.functions.items():
                if query_lower in fn.name.lower():
                    seen_locations.add((fn.file, fn.line))
                    results["matches"].append({
                        "name": fn.name,
                        "file": fn.file,
                        "line": fn.line,
                        "args": fn.args,
                    })
            for key, method in self.methods.items():
                if query_lower in method.name.lower():
                    seen_locations.add((method.file, method.line))
                    results["matches"].append({
                        "name": f"{method.class_name}.{method.name}",
                        "file": method.file,
                        "line": method.line,
                        "args": method.args,
                        "class": method.class_name,
                    })

            # Text search (catches "def funcname" style queries)
            for file_path in self.files.keys():
                if len(results["matches"]) >= max_results:
                    break
                try:
                    full_path = Path(self.root) / file_path
                    if full_path.exists():
                        for i, line in enumerate(full_path.read_text().split('\n')):
                            if query_lower in line.lower() and any(
                                kw in line for kw in (
                                    'def ', 'func ', 'fn ', 'function ',
                                    'fun ', 'void ', 'int ', 'pub ',
                                )):
                                loc = (file_path, i + 1)
                                if loc not in seen_locations:
                                    seen_locations.add(loc)
                                    results["matches"].append({
                                        "file": file_path,
                                        "line": i + 1,
                                        "content": line.strip()[:100],
                                    })
                                    if len(results["matches"]) >= max_results:
                                        break
                except Exception:
                    continue

        elif search_type == "class":
            seen_locations = set()  # (file, line) to dedupe

            # AST-based search on class names
            for key, cls in self.classes.items():
                if query_lower in cls.name.lower():
                    seen_locations.add((cls.file, cls.line))
                    results["matches"].append({
                        "name": cls.name,
                        "file": cls.file,
                        "line": cls.line,
                        "methods": cls.methods,
                    })

            # Text search (catches "class ClassName" style queries)
            for file_path in self.files.keys():
                if len(results["matches"]) >= max_results:
                    break
                try:
                    full_path = Path(self.root) / file_path
                    if full_path.exists():
                        for i, line in enumerate(full_path.read_text().split('\n')):
                            if query_lower in line.lower() and any(
                                kw in line for kw in (
                                    'class ', 'struct ', 'interface ',
                                    'trait ', 'type ', 'enum ',
                                    'module ',
                                )):
                                loc = (file_path, i + 1)
                                if loc not in seen_locations:
                                    seen_locations.add(loc)
                                    results["matches"].append({
                                        "file": file_path,
                                        "line": i + 1,
                                        "content": line.strip()[:100],
                                    })
                                    if len(results["matches"]) >= max_results:
                                        break
                except Exception:
                    continue

        elif search_type == "content":
            for file_path in self.files.keys():
                try:
                    full_path = Path(self.root) / file_path
                    if full_path.exists():
                        content = full_path.read_text()
                        if query_lower in content.lower():
                            file_lines = content.split('\n')
                            for i, line in enumerate(file_lines):
                                if query_lower in line.lower():
                                    results["matches"].append({
                                        "file": file_path,
                                        "line": i + 1,
                                        "content": line.strip()[:150],
                                    })
                                    if len(results["matches"]) >= max_results:
                                        break
                        if len(results["matches"]) >= max_results:
                            break
                except Exception:
                    continue

        # Limit results
        if len(results["matches"]) > max_results:
            results["matches"] = results["matches"][:max_results]
            results["truncated"] = True

        return results


_SKIP_DIRS = frozenset({
    '__pycache__', 'node_modules', 'venv', '.venv', 'build', 'dist',
    '.git', '.hg', '.svn', 'vendor', 'target', '.tox', '.mypy_cache',
})


def _auto_detect_extensions(project_path: str) -> List[str]:
    """Detect which file extensions exist and have a tree-sitter grammar."""
    try:
        from compass.agents.neo.treesitter import supported_extensions
        supported = set(supported_extensions())
    except Exception:
        supported = set()

    # Always include .py (handled by ast fallback)
    supported.add(".py")

    # Walk project to find what extensions actually exist
    root = Path(project_path)
    found: set = set()
    for item in root.rglob("*"):
        if any(part.startswith('.') or part in _SKIP_DIRS for part in item.parts):
            continue
        if item.is_file() and item.suffix in supported:
            found.add(item.suffix)
            if found == supported:
                break  # no point scanning further

    return sorted(found) if found else [".py"]


def index_codebase(project_path: str, extensions: List[str] = None) -> CodebaseIndex:
    """
    Index a codebase.

    Args:
        project_path: Root directory to index
        extensions: File extensions to include (auto-detected if None)

    Returns:
        CodebaseIndex with file, function, and class information
    """
    if extensions is None:
        extensions = _auto_detect_extensions(project_path)

    root = Path(project_path)

    # Load or create config
    config_path = root / ".compass" / "index.json"
    config = IndexConfig.load(str(root))

    # Auto-generate config on first run
    if not config_path.exists():
        config.save(str(root))

    index = CodebaseIndex(root=str(root), config=config)

    # Find all files
    for ext in extensions:
        for file_path in root.rglob(f"*{ext}"):
            # Skip hidden directories and common excludes
            if any(part.startswith('.') or part in _SKIP_DIRS
                   for part in file_path.parts):
                continue

            try:
                rel_path = str(file_path.relative_to(root))
                _index_file(file_path, rel_path, index)
            except Exception:
                # Skip files that can't be parsed
                pass

    return index


def _try_treesitter(content: str, ext: str, rel_path: str,
                    file_summary: FileSummary, index: CodebaseIndex) -> bool:
    """Try to extract definitions via tree-sitter. Returns True on success."""
    try:
        from compass.agents.neo.treesitter import extract_definitions
        defs = extract_definitions(content.encode("utf-8"), ext, rel_path)
    except Exception:
        return False
    if not defs:
        return False

    for d in defs:
        if d.kind == "function":
            key = f"{rel_path}:{d.name}"
            index.functions[key] = FunctionDef(
                name=d.name, file=rel_path, line=d.line,
                args=d.args, docstring=d.docstring,
            )
            file_summary.functions.append(d.name)
        elif d.kind == "class":
            key = f"{rel_path}:{d.name}"
            index.classes[key] = ClassDef(
                name=d.name, file=rel_path, line=d.line,
                methods=d.methods, docstring=d.docstring,
            )
            file_summary.classes.append(d.name)
        elif d.kind == "method":
            key = f"{rel_path}:{d.class_name}.{d.name}"
            index.methods[key] = MethodDef(
                name=d.name, class_name=d.class_name or "?",
                file=rel_path, line=d.line,
                args=d.args, docstring=d.docstring,
            )
    return True


def _index_file(file_path: Path, rel_path: str, index: CodebaseIndex):
    """Index a single file."""
    try:
        content = file_path.read_text()
        lines = content.count('\n') + 1
    except Exception:
        return

    file_summary = FileSummary(path=rel_path, lines=lines)
    ext = file_path.suffix

    if ext == ".py":
        ts_ok = _try_treesitter(content, ext, rel_path, file_summary, index)
        try:
            tree = ast.parse(content)
            if not ts_ok:
                _extract_definitions(tree, rel_path, file_summary, index)
            else:
                _extract_imports(tree, file_summary)
            _extract_public_api(tree, rel_path, index)
        except SyntaxError:
            pass
    else:
        _try_treesitter(content, ext, rel_path, file_summary, index)

    index.files[rel_path] = file_summary


def _extract_imports(tree: ast.AST, file_summary: FileSummary):
    """Extract import statements from Python AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                file_summary.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                file_summary.imports.append(node.module)


def _extract_definitions(tree: ast.AST, rel_path: str, file_summary: FileSummary, index: CodebaseIndex):
    """Extract function and class definitions from AST."""
    for node in ast.walk(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                file_summary.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                file_summary.imports.append(node.module)

        # Top-level functions
        elif isinstance(node, ast.FunctionDef):
            # Only index top-level functions (not methods)
            if hasattr(node, 'col_offset') and node.col_offset == 0:
                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node)
                fn = FunctionDef(
                    name=node.name,
                    file=rel_path,
                    line=node.lineno,
                    args=args,
                    docstring=docstring[:100] if docstring else None
                )
                # Use qualified name to avoid collisions
                key = f"{rel_path}:{node.name}"
                index.functions[key] = fn
                file_summary.functions.append(node.name)

        # Classes
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
                    method_args = [arg.arg for arg in item.args.args]
                    method_doc = ast.get_docstring(item)
                    method_fn = MethodDef(
                        name=item.name,
                        class_name=node.name,
                        file=rel_path,
                        line=item.lineno,
                        args=method_args,
                        docstring=method_doc[:100] if method_doc else None,
                    )
                    method_key = f"{rel_path}:{node.name}.{item.name}"
                    index.methods[method_key] = method_fn

            docstring = ast.get_docstring(node)
            cls = ClassDef(
                name=node.name,
                file=rel_path,
                line=node.lineno,
                methods=methods,
                docstring=docstring[:100] if docstring else None
            )
            key = f"{rel_path}:{node.name}"
            index.classes[key] = cls
            file_summary.classes.append(node.name)


def _resolve_import_module(package: str, module: Optional[str], level: int) -> Optional[str]:
    """Resolve relative import modules to a fully qualified path."""
    if level == 0:
        return module

    if not package:
        return module

    base_parts = package.split(".")
    if level > len(base_parts):
        return module

    base = ".".join(base_parts[:len(base_parts) - level + 1])
    if module:
        return f"{base}.{module}"
    return base


def _parse_all_list(node: ast.AST) -> Optional[List[str]]:
    """Parse __all__ list/tuple literals into string names."""
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
        return values or None
    return None


def _extract_public_api(tree: ast.AST, rel_path: str, index: CodebaseIndex) -> None:
    """Extract public API re-exports from package __init__.py files."""
    if not rel_path.endswith("__init__.py"):
        return

    package = ".".join(Path(rel_path).parts[:-1])
    if not package:
        return

    exported_names: Optional[List[str]] = None
    reexports: List[ExportedSymbol] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exported_names = _parse_all_list(node.value)
        elif isinstance(node, ast.ImportFrom):
            full_module = _resolve_import_module(package, node.module, node.level)
            if not full_module:
                continue
            for alias in node.names:
                name = alias.asname or alias.name
                reexports.append(ExportedSymbol(name=name, source=full_module))

    if exported_names is None:
        selected = reexports
    else:
        reexport_map = {sym.name: sym for sym in reexports}
        selected = []
        for name in exported_names:
            selected.append(reexport_map.get(name, ExportedSymbol(name=name, source=package)))

    if selected:
        index.public_api[package] = selected
