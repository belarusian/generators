"""
Multi-language code indexing via tree-sitter.

Supports: Python, JavaScript, TypeScript, Go, Rust, Java, Ruby, C, C++.
Gracefully degrades when tree-sitter or a grammar is not installed.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

_ts_available: Optional[bool] = None


def is_available() -> bool:
    """Check if tree-sitter core is installed."""
    global _ts_available
    if _ts_available is None:
        try:
            import tree_sitter  # noqa: F401
            _ts_available = True
        except ImportError:
            _ts_available = False
    return _ts_available


# ---------------------------------------------------------------------------
# Extracted definition (language-neutral)
# ---------------------------------------------------------------------------

@dataclass
class Definition:
    """A single extracted symbol definition."""
    name: str
    kind: str          # 'function', 'class', 'method'
    file: str          # relative path
    line: int          # 1-based
    end_line: int      # 1-based
    args: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    class_name: Optional[str] = None   # for methods
    methods: List[str] = field(default_factory=list)  # for classes


# ---------------------------------------------------------------------------
# Language configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LangConfig:
    """How to extract definitions from a language's tree-sitter AST."""
    module: str
    lang_fn: str = "language"
    # Node types that represent function/method definitions
    func_types: FrozenSet[str] = frozenset()
    # Node types that represent class-like definitions
    class_types: FrozenSet[str] = frozenset()
    # Node types that provide class context to children (Rust impl, etc.)
    # without being class definitions themselves
    method_container_types: FrozenSet[str] = frozenset()
    # Standalone receiver-method types (Go method_declaration)
    receiver_method_types: FrozenSet[str] = frozenset()
    # Variable-assigned function types (JS arrow_function, etc.)
    func_value_types: FrozenSet[str] = frozenset()


# Extension -> language name
EXT_MAP: Dict[str, str] = {}

# Language name -> config
CONFIGS: Dict[str, LangConfig] = {}


def _reg(name: str, config: LangConfig, extensions: List[str]):
    CONFIGS[name] = config
    for ext in extensions:
        EXT_MAP[ext] = name


_reg("python", LangConfig(
    module="tree_sitter_python",
    func_types=frozenset({"function_definition"}),
    class_types=frozenset({"class_definition"}),
), [".py"])

_reg("javascript", LangConfig(
    module="tree_sitter_javascript",
    func_types=frozenset({
        "function_declaration", "generator_function_declaration",
        "method_definition",
    }),
    class_types=frozenset({"class_declaration"}),
    func_value_types=frozenset({"arrow_function", "function_expression", "function"}),
), [".js", ".mjs", ".jsx"])

_reg("typescript", LangConfig(
    module="tree_sitter_typescript",
    lang_fn="language_typescript",
    func_types=frozenset({
        "function_declaration", "generator_function_declaration",
        "method_definition", "method_signature", "abstract_method_signature",
    }),
    class_types=frozenset({
        "class_declaration", "abstract_class_declaration",
        "interface_declaration",
    }),
    func_value_types=frozenset({"arrow_function", "function_expression", "function"}),
), [".ts"])

_reg("tsx", LangConfig(
    module="tree_sitter_typescript",
    lang_fn="language_tsx",
    func_types=frozenset({
        "function_declaration", "generator_function_declaration",
        "method_definition", "method_signature", "abstract_method_signature",
    }),
    class_types=frozenset({
        "class_declaration", "abstract_class_declaration",
        "interface_declaration",
    }),
    func_value_types=frozenset({"arrow_function", "function_expression", "function"}),
), [".tsx"])

_reg("go", LangConfig(
    module="tree_sitter_go",
    func_types=frozenset({"function_declaration"}),
    class_types=frozenset({"type_spec"}),
    receiver_method_types=frozenset({"method_declaration"}),
), [".go"])

_reg("rust", LangConfig(
    module="tree_sitter_rust",
    func_types=frozenset({"function_item", "function_signature_item"}),
    class_types=frozenset({"struct_item", "enum_item", "trait_item"}),
    method_container_types=frozenset({"impl_item", "trait_item"}),
), [".rs"])

_reg("java", LangConfig(
    module="tree_sitter_java",
    func_types=frozenset({"method_declaration", "constructor_declaration"}),
    class_types=frozenset({
        "class_declaration", "interface_declaration",
        "enum_declaration",
    }),
), [".java"])

_reg("ruby", LangConfig(
    module="tree_sitter_ruby",
    func_types=frozenset({"method", "singleton_method"}),
    class_types=frozenset({"class", "module"}),
), [".rb"])

_reg("c", LangConfig(
    module="tree_sitter_c",
    func_types=frozenset({"function_definition"}),
    class_types=frozenset({"struct_specifier"}),
), [".c", ".h"])

_reg("cpp", LangConfig(
    module="tree_sitter_cpp",
    func_types=frozenset({"function_definition"}),
    class_types=frozenset({"class_specifier", "struct_specifier"}),
), [".cpp", ".cc", ".cxx", ".hpp"])


# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------

_parsers: Dict[str, object] = {}   # ext -> Parser
_languages: Dict[str, object] = {}  # ext -> Language


def get_parser(ext: str):
    """Get a cached parser for a file extension. Returns None if unavailable."""
    if ext in _parsers:
        return _parsers[ext]

    if not is_available():
        return None

    lang_name = EXT_MAP.get(ext)
    if not lang_name:
        return None

    config = CONFIGS[lang_name]
    try:
        from tree_sitter import Language, Parser

        mod = __import__(config.module)
        lang_fn = getattr(mod, config.lang_fn)
        language = Language(lang_fn())

        parser = Parser(language)
        _parsers[ext] = parser
        _languages[ext] = language
        return parser
    except Exception:
        _parsers[ext] = None
        return None


def get_config(ext: str) -> Optional[LangConfig]:
    """Get the LangConfig for a file extension."""
    lang_name = EXT_MAP.get(ext)
    return CONFIGS.get(lang_name) if lang_name else None


# ---------------------------------------------------------------------------
# Name / arg / docstring extraction helpers
# ---------------------------------------------------------------------------

def _get_def_name(node) -> Optional[str]:
    """Extract the definition name from a tree-sitter node."""
    # Direct name field (most languages)
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")

    # C/C++: name is buried in a declarator chain
    current = node.child_by_field_name("declarator")
    for _ in range(5):
        if current is None:
            break
        if current.type == "identifier":
            return current.text.decode("utf-8")
        if current.type == "qualified_identifier":
            inner = current.child_by_field_name("name")
            return inner.text.decode("utf-8") if inner else current.text.decode("utf-8")
        current = current.child_by_field_name("declarator")

    return None


_PARAM_FIELDS = ("parameters", "formal_parameters", "method_parameters")


def _extract_args(node) -> List[str]:
    """Best-effort parameter name extraction."""
    # Try direct parameter fields
    for field_name in _PARAM_FIELDS:
        params = node.child_by_field_name(field_name)
        if params:
            return _param_names(params)

    # C/C++: parameters inside the declarator chain
    current = node.child_by_field_name("declarator")
    for _ in range(5):
        if current is None:
            break
        for field_name in _PARAM_FIELDS:
            params = current.child_by_field_name(field_name)
            if params:
                return _param_names(params)
        current = current.child_by_field_name("declarator")

    return []


def _param_names(params_node) -> List[str]:
    """Extract parameter names from a parameter list node."""
    names = []
    for child in params_node.named_children:
        # Direct name field
        name = child.child_by_field_name("name")
        if name:
            names.append(name.text.decode("utf-8"))
            continue
        # C/C++ parameter_declaration uses "declarator" for name
        decl = child.child_by_field_name("declarator")
        if decl:
            # Walk through pointer_declarator chain to find identifier
            cur = decl
            for _ in range(5):
                if cur.type == "identifier":
                    names.append(cur.text.decode("utf-8"))
                    break
                cur = cur.child_by_field_name("declarator") or cur
                if cur.type == "identifier":
                    names.append(cur.text.decode("utf-8"))
                    break
            continue
        # Rust pattern field
        pattern = child.child_by_field_name("pattern")
        if pattern and pattern.type == "identifier":
            names.append(pattern.text.decode("utf-8"))
            continue
        # Bare identifier (Ruby, simple params)
        if child.type == "identifier":
            names.append(child.text.decode("utf-8"))
            continue
    return names


def _extract_docstring(node) -> Optional[str]:
    """Best-effort docstring/comment extraction."""
    # Python-style: first statement in body is a string literal
    body = node.child_by_field_name("body")
    if body and body.named_child_count > 0:
        first = body.named_children[0]
        if first.type == "expression_statement":
            for child in first.named_children:
                if child.type in ("string", "string_literal", "concatenated_string"):
                    text = child.text.decode("utf-8")
                    for q in ('"""', "'''", '"', "'"):
                        if text.startswith(q) and text.endswith(q):
                            text = text[len(q):-len(q)]
                            break
                    return text.strip()[:200]

    # Comment-style: preceding sibling comment (JSDoc, Go, Rust ///)
    prev = node.prev_named_sibling
    if prev and prev.type in ("comment", "line_comment", "block_comment"):
        text = prev.text.decode("utf-8")
        text = text.lstrip("/*#! ").rstrip("*/ ")
        return text[:200] if text else None

    return None


def _get_go_receiver(node) -> Optional[str]:
    """Extract receiver type name from a Go method_declaration."""
    receiver = node.child_by_field_name("receiver")
    if not receiver:
        return None
    for param in receiver.named_children:
        type_node = param.child_by_field_name("type")
        if not type_node:
            continue
        # Pointer receiver: *Foo
        if type_node.type == "pointer_type":
            for child in type_node.named_children:
                if child.type == "type_identifier":
                    return child.text.decode("utf-8")
        elif type_node.type == "type_identifier":
            return type_node.text.decode("utf-8")
    return None


def _get_container_name(node) -> Optional[str]:
    """Get the type name from a method container (Rust impl_item, etc.)."""
    type_node = node.child_by_field_name("type")
    if type_node:
        if type_node.type == "type_identifier":
            return type_node.text.decode("utf-8")
        # Generic type: impl<T> Foo<T>
        for child in type_node.named_children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    return None


def _validate_class_node(node, lang_name: str) -> bool:
    """Extra validation for class-like nodes that need filtering."""
    # Go type_spec: only structs and interfaces, not type aliases
    if lang_name == "go" and node.type == "type_spec":
        type_child = node.child_by_field_name("type")
        return type_child is not None and type_child.type in (
            "struct_type", "interface_type",
        )
    # C/C++ struct_specifier: must have a body (not just forward decl)
    if node.type == "struct_specifier":
        return node.child_by_field_name("body") is not None
    return True


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

def _walk_tree(root_node, config: LangConfig, lang_name: str, rel_path: str):
    """Walk tree and extract definitions. Returns list of Definition."""
    functions: List[Definition] = []
    classes: List[Definition] = []
    methods: List[Definition] = []

    def _make_func(node, name, class_ctx=None):
        args = _extract_args(node)
        doc = _extract_docstring(node)
        line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        if class_ctx:
            methods.append(Definition(
                name=name, kind="method", file=rel_path,
                line=line, end_line=end_line, args=args,
                docstring=doc[:100] if doc else None,
                class_name=class_ctx,
            ))
        else:
            functions.append(Definition(
                name=name, kind="function", file=rel_path,
                line=line, end_line=end_line, args=args,
                docstring=doc[:100] if doc else None,
            ))

    def visit(node, class_ctx=None):
        ntype = node.type

        # --- Class-like definition ---
        if ntype in config.class_types:
            name = _get_def_name(node)
            if name and _validate_class_node(node, lang_name):
                pre = len(methods)
                # Visit children with class context
                for child in node.children:
                    visit(child, class_ctx=name)
                cls_method_names = [m.name for m in methods[pre:]]
                doc = _extract_docstring(node)
                classes.append(Definition(
                    name=name, kind="class", file=rel_path,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=doc[:100] if doc else None,
                    methods=cls_method_names,
                ))
                return

        # --- Method container (Rust impl_item, not itself a class) ---
        if ntype in config.method_container_types and ntype not in config.class_types:
            container_name = _get_container_name(node)
            for child in node.children:
                visit(child, class_ctx=container_name or class_ctx)
            return

        # --- Go receiver methods ---
        if ntype in config.receiver_method_types:
            name = _get_def_name(node)
            receiver = _get_go_receiver(node)
            if name:
                _make_func(node, name, class_ctx=receiver)
            return

        # --- Function-like definition ---
        if ntype in config.func_types:
            name = _get_def_name(node)
            if name:
                _make_func(node, name, class_ctx=class_ctx)
            return  # don't descend into function bodies

        # --- Variable-assigned functions (JS/TS const foo = () => ...) ---
        if ntype == "variable_declarator" and config.func_value_types:
            value = node.child_by_field_name("value")
            if value and value.type in config.func_value_types:
                name = _get_def_name(node)
                if name:
                    # Args are on the value node (the arrow/function)
                    args = _extract_args(value)
                    doc = _extract_docstring(node)
                    line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    kind = "method" if class_ctx else "function"
                    dest = methods if class_ctx else functions
                    dest.append(Definition(
                        name=name, kind=kind, file=rel_path,
                        line=line, end_line=end_line, args=args,
                        docstring=doc[:100] if doc else None,
                        class_name=class_ctx,
                    ))
                return

        # --- Recurse ---
        for child in node.children:
            visit(child, class_ctx)

    visit(root_node)
    return functions + classes + methods


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def supported_extensions() -> List[str]:
    """Return file extensions for which a grammar is installed."""
    if not is_available():
        return []
    available = []
    seen_modules: Dict[str, bool] = {}
    for ext, lang_name in EXT_MAP.items():
        mod_name = CONFIGS[lang_name].module
        if mod_name not in seen_modules:
            try:
                __import__(mod_name)
                seen_modules[mod_name] = True
            except ImportError:
                seen_modules[mod_name] = False
        if seen_modules[mod_name]:
            available.append(ext)
    return available


def extract_definitions(source: bytes, ext: str, rel_path: str) -> List[Definition]:
    """Parse source bytes and extract definitions.

    Returns a list of Definition objects, or [] if the language is
    unsupported or tree-sitter is unavailable.
    """
    parser = get_parser(ext)
    if parser is None:
        return []

    config = get_config(ext)
    if config is None:
        return []

    lang_name = EXT_MAP.get(ext, "")

    try:
        tree = parser.parse(source)
        return _walk_tree(tree.root_node, config, lang_name, rel_path)
    except Exception:
        return []
