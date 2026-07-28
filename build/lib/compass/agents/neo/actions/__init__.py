"""
Actions package - singledispatch handlers for action types.

Types are pure dataclasses in types.py.
Handlers register with singledispatch in each module.
Importing this package registers all handlers.
"""

# Import types from types.py (these are what we export)
from compass.agents.neo.types import (
    ReadFileAction,
    WriteFileAction,
    DeleteFileAction,
    CreateDirAction,
    RunCommandAction,
    ExecAction,
    SearchAction,
    IndexAction,
    GrepAction,
    EditFileAction,
    ShellCommandAction,
    AskOracleAction,
    ProgramAction,
)

# Import handler modules to register singledispatch handlers
# (side effect: registers @display.register, @validate.register, etc.)
from compass.agents.neo.actions import (
    read_file,
    write_file,
    delete_file,
    create_dir,
    run_command,
    exec as exec_module,
    search,
    index,
    grep,
    edit_file,
    shell_command,
    ask_oracle,
    program,
)

__all__ = [
    "ReadFileAction",
    "WriteFileAction",
    "DeleteFileAction",
    "CreateDirAction",
    "RunCommandAction",
    "ExecAction",
    "SearchAction",
    "IndexAction",
    "GrepAction",
    "EditFileAction",
    "ShellCommandAction",
    "AskOracleAction",
    "ProgramAction",
]
