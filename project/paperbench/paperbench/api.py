"""
PaperBench API - High-level utilities for working with PaperBench programmatically.

This module is designed to be lightweight and importable without heavy dependencies.
It provides easy-to-use functions for:
- Getting initial prompts for rollouts
- Running setup on containers for specific papers
- Accessing paper metadata and files

Example usage:
    from paperbench.api import (
        get_initial_prompt,
        get_paper_info,
        list_paper_ids,
        run_paper_setup,
    )

    # Get the initial prompt for any paper
    prompt = get_initial_prompt()

    # Get info about a specific paper
    paper = get_paper_info("rice")
    print(paper.title)

    # List all available papers
    papers = list_paper_ids()
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypedDict

import yaml

# Avoid heavy imports - use inline imports where needed
# This allows api.py to be imported without pulling in all dependencies

if TYPE_CHECKING:
    from datasets import Dataset
    from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface


# Type for chat messages (avoid openai dependency at import time)
class ChatMessage(TypedDict):
    role: str
    content: str


# Constants
WORKSPACE_BASE = "/home"
SUBMISSION_DIR = WORKSPACE_BASE + "/submission"


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _get_paperbench_module_root() -> Path:
    """Returns an absolute path to the root of the PaperBench module."""
    path = Path(__file__).parent.resolve()
    assert path.name == "paperbench", (
        f"Expected the module directory to be `paperbench`, but got `{path.name}`."
    )
    return path


def _get_paperbench_data_dir() -> Path:
    """Returns an absolute path to the PaperBench data directory."""
    override = os.environ.get("PAPERBENCH_DATA_DIR")
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path

    default_path = _get_paperbench_module_root().parent / "data"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Unable to locate the PaperBench data directory. "
        f"Checked: {default_path}. "
        "Set the PAPERBENCH_DATA_DIR environment variable to point to the data directory."
    )


def _load_yaml_dict(fpath: Path) -> dict[str, Any]:
    """Loads a YAML file and returns its contents as a dictionary."""
    with open(fpath, "r") as file:
        contents = yaml.safe_load(file)
    return contents


# ---------------------------------------------------------------------------
# Paper dataclass (self-contained, no external dependencies)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperInfo:
    """
    Information about a paper in the PaperBench dataset.

    Attributes:
        id: Unique identifier for the paper
        title: Full title of the paper
        paper_pdf: Path to the PDF file
        paper_md: Path to the markdown version
        addendum: Path to the addendum file with additional context
        judge_addendum: Path to the judge-specific addendum
        assets: Path to the assets directory
        blacklist: Path to the blacklist file (forbidden resources)
        rubric: Path to the rubric JSON file
    """

    id: str
    title: str
    paper_pdf: Path
    paper_md: Path
    addendum: Path
    judge_addendum: Path
    assets: Path
    blacklist: Path
    rubric: Path


def _get_paper_info(paper_id: str) -> PaperInfo:
    """Internal function to load paper info from the registry."""
    papers_dir = _get_paperbench_data_dir() / "papers"
    config_path = papers_dir / paper_id / "config.yaml"

    if not config_path.exists():
        available = list_paper_ids()
        raise ValueError(
            f"Paper '{paper_id}' not found. Available papers: {available}"
        )

    config = _load_yaml_dict(config_path)

    return PaperInfo(
        id=config["id"],
        title=config["title"],
        paper_pdf=papers_dir / paper_id / "paper.pdf",
        paper_md=papers_dir / paper_id / "paper.md",
        addendum=papers_dir / paper_id / "addendum.md",
        judge_addendum=papers_dir / paper_id / "judge.addendum.md",
        assets=papers_dir / paper_id / "assets",
        rubric=papers_dir / paper_id / "rubric.json",
        blacklist=papers_dir / paper_id / "blacklist.txt",
    )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def get_instructions_path(code_only: bool = False) -> Path:
    """
    Get the path to the instructions file.

    Args:
        code_only: If True, return path to code-only instructions

    Returns:
        Path to the instructions file
    """
    root = _get_paperbench_module_root()
    if code_only:
        return root / "instructions" / "code_only_instructions.txt"
    return root / "instructions" / "instructions.txt"


def get_initial_prompt(code_only: bool = False) -> list[ChatMessage]:
    """
    Returns the initial prompt message for a paperbench rollout.

    This is the first message given to the LLM at the start of a rollout.
    The prompt is paper-agnostic - paper-specific content (PDF, rubric, etc.)
    is uploaded to the container separately during setup.

    Args:
        code_only: If True, use the simplified code-only instructions

    Returns:
        A list containing a single user message with the instructions

    Example:
        >>> prompt = get_initial_prompt()
        >>> print(prompt[0]["role"])
        'user'
        >>> print(prompt[0]["content"][:50])
        'You are tasked with reproducing a research paper.'
    """
    instructions_path = get_instructions_path(code_only)
    instructions_text = instructions_path.read_text()
    return [{"role": "user", "content": instructions_text}]


def get_instructions_text(code_only: bool = False) -> str:
    """
    Returns the raw instructions text for a paperbench rollout.

    Args:
        code_only: If True, return code-only instructions

    Returns:
        The instructions text as a string
    """
    instructions_path = get_instructions_path(code_only)
    return instructions_path.read_text()


def get_paper_info(paper_id: str) -> PaperInfo:
    """
    Get metadata and file paths for a specific paper.

    Args:
        paper_id: The paper ID (e.g., "rice", "lca-on-the-line")

    Returns:
        PaperInfo object with id, title, and paths to paper files

    Example:
        >>> paper = get_paper_info("rice")
        >>> print(paper.title)
        'RICE: Breaking Through the Training Bottlenecks...'
        >>> print(paper.paper_pdf.exists())
        True
    """
    return _get_paper_info(paper_id)


def list_paper_ids() -> list[str]:
    """
    List all available paper IDs in the registry.

    Returns:
        Sorted list of paper IDs

    Example:
        >>> papers = list_paper_ids()
        >>> 'rice' in papers
        True
    """
    papers_dir = _get_paperbench_data_dir() / "papers"
    paper_configs = papers_dir.rglob("config.yaml")
    paper_ids = [f.parent.stem for f in sorted(paper_configs)]
    return paper_ids


def get_papers_directory() -> Path:
    """
    Get the path to the papers data directory.

    Returns:
        Path to the papers directory containing all paper data
    """
    return _get_paperbench_data_dir() / "papers"


async def run_paper_setup(
    paper_id: str,
    computer: "ComputerInterface",
    prompt_content: str | None = None,
    code_only: bool = False,
    agent_env_path: Path | None = None,
) -> None:
    """
    Run the paperbench setup for a single paper on an already-started container.

    This uploads all necessary files to the container to prepare it for an agent
    to attempt reproducing the paper. Files uploaded include:
    - instructions.txt (the prompt)
    - paper.pdf, paper.md (the paper itself)
    - addendum.md (additional context)
    - blacklist.txt (forbidden resources)
    - assets/ (any supporting files)

    Args:
        paper_id: The paper ID (e.g., "rice", "lca-on-the-line")
        computer: An already-started ComputerInterface (from a ComputerRuntime)
        prompt_content: Optional custom prompt text. If None, uses default instructions.
        code_only: If True and prompt_content is None, use code-only instructions
        agent_env_path: Optional path to agent.env file with additional env vars

    Example:
        >>> from alcatraz import LocalConfig
        >>> from nanoeval_alcatraz import AlcatrazComputerRuntime
        >>> from nanoeval.solvers.computer_tasks import ComputerConfiguration, NetworkMode
        >>>
        >>> config = ComputerConfiguration(
        ...     cwd="/home",
        ...     docker_image="pb-env:latest",
        ...     network_mode=NetworkMode.UNPROXIED,
        ... )
        >>> runtime = AlcatrazComputerRuntime(env=LocalConfig())
        >>>
        >>> async with runtime._start_computer(config) as computer:
        ...     await run_paper_setup("rice", computer)
        ...     # Container is now ready for agent execution
    """
    # Get paper metadata and file paths
    paper = _get_paper_info(paper_id)

    # Get prompt content if not provided
    if prompt_content is None:
        prompt_content = get_instructions_text(code_only)

    # Create workspace directory structure
    await computer.check_shell_command(f"mkdir -p {WORKSPACE_BASE}/paper/assets")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write instructions to temp file then upload
        instructions = Path(tmp_dir) / "instructions.txt"
        instructions.write_text(prompt_content)

        # Upload all paper files
        file_mappings = [
            (instructions, f"{WORKSPACE_BASE}/instructions.txt"),
            (paper.paper_pdf, f"{WORKSPACE_BASE}/paper/paper.pdf"),
            (paper.paper_md, f"{WORKSPACE_BASE}/paper/paper.md"),
            (paper.addendum, f"{WORKSPACE_BASE}/paper/addendum.md"),
            (paper.blacklist, f"{WORKSPACE_BASE}/paper/blacklist.txt"),
        ]

        for src, dst in file_mappings:
            if src.exists():
                with open(src, "rb") as f:
                    await computer.upload(f.read(), dst)

    # Upload paper assets
    if paper.assets.exists():
        for asset in paper.assets.glob("*"):
            with open(asset, "rb") as f:
                await computer.upload(f.read(), f"{WORKSPACE_BASE}/paper/assets/{asset.name}")

    # Upload optional agent env file
    if agent_env_path and agent_env_path.exists():
        with open(agent_env_path, "rb") as f:
            await computer.upload(f.read(), f"{WORKSPACE_BASE}/agent.env")

    # Create submission directory
    await computer.check_shell_command(f"mkdir -p {SUBMISSION_DIR}")


def get_paper_files(paper_id: str) -> dict[str, Path]:
    """
    Get a dictionary of all file paths for a paper.

    Args:
        paper_id: The paper ID

    Returns:
        Dictionary mapping file type to Path

    Example:
        >>> files = get_paper_files("rice")
        >>> files.keys()
        dict_keys(['paper_pdf', 'paper_md', 'addendum', 'blacklist', 'rubric', 'assets'])
    """
    paper = _get_paper_info(paper_id)
    return {
        "paper_pdf": paper.paper_pdf,
        "paper_md": paper.paper_md,
        "addendum": paper.addendum,
        "blacklist": paper.blacklist,
        "rubric": paper.rubric,
        "assets": paper.assets,
        "judge_addendum": paper.judge_addendum,
    }


def read_paper_markdown(paper_id: str) -> str:
    """
    Read the markdown content of a paper.

    Args:
        paper_id: The paper ID

    Returns:
        The paper content as markdown text
    """
    paper = _get_paper_info(paper_id)
    return paper.paper_md.read_text()


def read_paper_addendum(paper_id: str) -> str:
    """
    Read the addendum content for a paper.

    The addendum provides additional context needed to reproduce the paper,
    as well as clarifications about what is not in scope.

    Args:
        paper_id: The paper ID

    Returns:
        The addendum content as text
    """
    paper = _get_paper_info(paper_id)
    return paper.addendum.read_text()


def read_paper_rubric(paper_id: str) -> str:
    """
    Read the rubric JSON content for a paper.

    The rubric defines the hierarchical grading criteria for the paper.

    Args:
        paper_id: The paper ID

    Returns:
        The rubric content as JSON string
    """
    paper = _get_paper_info(paper_id)
    return paper.rubric.read_text()


# ---------------------------------------------------------------------------
# Tool definitions for external use
# ---------------------------------------------------------------------------


class FunctionDefinition(TypedDict):
    """Function definition inside a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ChatCompletionToolParam(TypedDict):
    """Tool definition in OpenAI ChatCompletionToolParam format."""

    type: str
    function: FunctionDefinition


# Keep ToolDefinition for backwards compatibility (flat format)
class ToolDefinition(TypedDict):
    """Tool definition in flat format (legacy)."""

    type: str
    name: str
    description: str
    parameters: dict[str, Any]


# Internal tool definitions (flat format for internal use)
_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": "Use this function to execute bash commands.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
    {
        "name": "python-tool",
        "description": (
            "Use the python function to execute Python code.\n\n"
            "The Python tool executes single-run Python scripts. Important notes:\n"
            "1. Each execution is independent - no state is preserved between runs\n"
            "2. You must explicitly use print() statements to see any output\n"
            "3. Simply writing expressions (like in notebooks) will not display results\n"
            "4. The script cannot accept interactive input during execution\n"
            "5. Return statements alone won't produce visible output\n"
            "6. All variables and imports are cleared between executions\n"
            "7. Standard output (via print()) is the only way to see results"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file_chunk",
        "description": "Read a chunk of lines from a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed).",
                    "default": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 50, max: 50).",
                    "default": 50,
                },
            },
            "required": ["file"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_file",
        "description": (
            "Search for a keyword or phrase in a file and return matching lines with context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for (case-insensitive).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of lines of context to show before and after each match (default: 2).",
                    "default": 2,
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of matches to return per page (default: 5).",
                    "default": 5,
                },
                "page": {
                    "type": "integer",
                    "description": "Which page of results to return (1-indexed, default: 1).",
                    "default": 1,
                },
            },
            "required": ["file", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit",
        "description": "Signal that you are completely finished.",
        "parameters": {
            "type": "object",
            "properties": {
                "end_message": {
                    "type": "string",
                    "description": "Final message to signal that you are finished.",
                },
            },
            "required": ["end_message"],
            "additionalProperties": False,
        },
    },
]


def get_oai_tools() -> list[ChatCompletionToolParam]:
    """
    Get paperbench tools in OpenAI ChatCompletionToolParam format.

    This is the format expected by OpenAI's chat completions API
    and frameworks that use it (like your verifiers Environment).

    Returns:
        List of tools in ChatCompletionToolParam format with nested 'function' key

    Example:
        >>> oai_tools = get_oai_tools()
        >>> oai_tools[0]["type"]
        'function'
        >>> oai_tools[0]["function"]["name"]
        'bash'
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in _TOOL_DEFINITIONS
    ]


def get_tool_definitions() -> list[ToolDefinition]:
    """
    Get paperbench tool definitions in flat format (legacy).

    For OpenAI-compatible format, use get_oai_tools() instead.

    Returns:
        List of tool definitions in flat format
    """
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in _TOOL_DEFINITIONS
    ]


def get_tool_names() -> list[str]:
    """Get list of available tool names."""
    return [tool["name"] for tool in _TOOL_DEFINITIONS]


def is_submit_tool_call(tool_name: str) -> bool:
    """
    Check if a tool call is the submit tool (signals completion).

    Use this to detect when the agent has finished its work.

    Args:
        tool_name: The name of the tool being called

    Returns:
        True if this is the submit tool

    Example:
        >>> is_submit_tool_call("submit")
        True
        >>> is_submit_tool_call("bash")
        False
    """
    return tool_name == "submit"


# ---------------------------------------------------------------------------
# Rubric utilities for building reward functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricNode:
    """
    A single node in the paperbench rubric tree.

    The rubric is a hierarchical structure of requirements, where leaf nodes
    have task categories that define how they should be evaluated:
    - "Code Development": Check if submission contains correct implementation
    - "Code Execution": Check if reproduce.sh runs successfully
    - "Result Analysis": Check if outputs match expected results

    Attributes:
        id: Unique identifier for this task
        requirements: Human-readable description of what's required
        weight: Relative importance (for weighted scoring)
        task_category: Only for leaf nodes - how to evaluate
        finegrained_task_category: More specific category (optional)
        sub_tasks: Child nodes (empty for leaf nodes)
    """

    id: str
    requirements: str
    weight: int
    task_category: str | None
    finegrained_task_category: str | None
    sub_tasks: list["RubricNode"]

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (has no sub-tasks)."""
        return len(self.sub_tasks) == 0

    def get_leaf_nodes(self) -> list["RubricNode"]:
        """Get all leaf nodes in this subtree."""
        if self.is_leaf():
            return [self]
        leaves = []
        for sub in self.sub_tasks:
            leaves.extend(sub.get_leaf_nodes())
        return leaves

    def count_leaves(self) -> int:
        """Count the number of leaf nodes."""
        return len(self.get_leaf_nodes())

    def total_weight(self) -> int:
        """Sum of all weights in the tree."""
        if self.is_leaf():
            return self.weight
        return sum(sub.total_weight() for sub in self.sub_tasks)


def _dict_to_rubric_node(data: dict[str, Any]) -> RubricNode:
    """Convert a rubric dict to RubricNode."""
    sub_tasks = [_dict_to_rubric_node(sub) for sub in data.get("sub_tasks", [])]
    return RubricNode(
        id=data["id"],
        requirements=data["requirements"],
        weight=data["weight"],
        task_category=data.get("task_category"),
        finegrained_task_category=data.get("finegrained_task_category"),
        sub_tasks=sub_tasks,
    )


def get_rubric(paper_id: str) -> RubricNode:
    """
    Get the parsed rubric tree for a paper.

    The rubric defines hierarchical grading criteria. Use this to:
    - Understand what requirements need to be met
    - Build custom reward/grading functions
    - Analyze task complexity (number of leaves, weights, etc.)

    Args:
        paper_id: The paper ID

    Returns:
        RubricNode tree structure

    Example:
        >>> rubric = get_rubric("rice")
        >>> rubric.id
        'rice'
        >>> len(rubric.get_leaf_nodes())
        42  # number of gradable criteria
    """
    import json

    paper = _get_paper_info(paper_id)
    with open(paper.rubric) as f:
        data = json.load(f)
    return _dict_to_rubric_node(data)


def get_rubric_dict(paper_id: str) -> dict[str, Any]:
    """
    Get the raw rubric as a dictionary.

    Args:
        paper_id: The paper ID

    Returns:
        The rubric data as a dict
    """
    import json

    paper = _get_paper_info(paper_id)
    with open(paper.rubric) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset format utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperBenchExample:
    """
    A single paperbench example formatted for use in training/evaluation frameworks.

    Attributes:
        paper_id: Unique identifier for the paper
        title: Paper title
        prompt: The initial instructions given to the agent
        num_criteria: Number of leaf-level grading criteria
        total_weight: Sum of all criterion weights
        categories: Set of task categories present in the rubric
    """

    paper_id: str
    title: str
    prompt: str
    num_criteria: int
    total_weight: int
    categories: set[str]


def get_paperbench_dataset(code_only: bool = False) -> list[PaperBenchExample]:
    """
    Get all paperbench papers as a dataset of examples.

    This formats papers for use in external frameworks that expect
    (prompt, task_info) pairs.

    Args:
        code_only: If True, use code-only instructions and filter rubric

    Returns:
        List of PaperBenchExample objects

    Example:
        >>> dataset = get_paperbench_dataset()
        >>> len(dataset)
        20  # number of papers
        >>> dataset[0].paper_id
        'rice'
    """
    prompt_text = get_instructions_text(code_only)
    examples = []

    for paper_id in list_paper_ids():
        paper = _get_paper_info(paper_id)
        rubric = get_rubric(paper_id)

        # Get categories from leaf nodes
        leaves = rubric.get_leaf_nodes()
        categories = {leaf.task_category for leaf in leaves if leaf.task_category}

        # If code_only, filter to only Code Development
        if code_only:
            leaves = [l for l in leaves if l.task_category == "Code Development"]
            categories = {"Code Development"} if leaves else set()

        examples.append(
            PaperBenchExample(
                paper_id=paper_id,
                title=paper.title,
                prompt=prompt_text,
                num_criteria=len(leaves),
                total_weight=sum(l.weight for l in leaves),
                categories=categories,
            )
        )

    return examples


def get_paper_as_dict(paper_id: str, code_only: bool = False) -> dict[str, Any]:
    """
    Get a single paper formatted as a dictionary.

    Useful for creating custom dataset formats.

    Args:
        paper_id: The paper ID
        code_only: If True, use code-only mode

    Returns:
        Dictionary with paper info, prompt, and rubric summary
    """
    paper = _get_paper_info(paper_id)
    rubric = get_rubric(paper_id)
    leaves = rubric.get_leaf_nodes()

    if code_only:
        leaves = [l for l in leaves if l.task_category == "Code Development"]

    return {
        "paper_id": paper_id,
        "title": paper.title,
        "prompt": get_instructions_text(code_only),
        "paper_pdf_path": str(paper.paper_pdf),
        "paper_md_path": str(paper.paper_md),
        "addendum_path": str(paper.addendum),
        "rubric_path": str(paper.rubric),
        "num_criteria": len(leaves),
        "total_weight": sum(l.weight for l in leaves),
        "task_categories": list({l.task_category for l in leaves if l.task_category}),
    }


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


async def execute_bash(computer: "ComputerInterface", cmd: str) -> str:
    """
    Execute a bash command on the container.

    Args:
        computer: The ComputerInterface to execute on
        cmd: The bash command to run

    Returns:
        Command output as a string
    """
    result = await computer.send_shell_command(cmd=cmd)
    return result.output.decode("utf-8").strip()


async def execute_python(computer: "ComputerInterface", code: str) -> str:
    """
    Execute Python code on the container.

    The code is written to a temp file and executed with python3.

    Args:
        computer: The ComputerInterface to execute on
        code: The Python code to run

    Returns:
        Script output as a string
    """
    result = await computer.send_shell_command("mktemp -d")
    tmp_dir = result.output.decode("utf-8").strip()
    await computer.upload(code.encode("utf-8"), f"{tmp_dir}/code.py")
    result = await computer.send_shell_command(f"python3 {tmp_dir}/code.py")
    return result.output.decode("utf-8").strip()


async def read_file_chunk(
    computer: "ComputerInterface",
    file: str,
    start_line: int = 1,
    max_lines: int = 50,
) -> str:
    """
    Read a chunk of a file with line numbers.

    Args:
        computer: The ComputerInterface to execute on
        file: Path to the file
        start_line: Line to start reading from (1-indexed)
        max_lines: Maximum lines to read (max 50)

    Returns:
        File content with line numbers
    """
    if start_line < 1:
        return "ERROR: start_line must be >= 1"
    if max_lines < 1:
        return "ERROR: max_lines must be >= 1"
    if max_lines > 50:
        return "ERROR: max_lines cannot exceed 50"

    result = await computer.send_shell_command(f"cat {file}")
    content = result.output.decode("utf-8").strip()
    lines = content.splitlines()

    if start_line > len(lines):
        return f"ERROR: start_line ({start_line}) is beyond total lines ({len(lines)})"

    end_line = min(start_line + max_lines - 1, len(lines))
    chunk = lines[start_line - 1 : end_line]
    numbered = [f"{i + start_line}: {line}" for i, line in enumerate(chunk)]

    summary = f"File has {len(lines)} total lines. Showing lines {start_line} to {end_line}.\n\n"
    return summary + "\n".join(numbered)


async def search_file(
    computer: "ComputerInterface",
    file: str,
    query: str,
    context_lines: int = 2,
    max_matches: int = 5,
    page: int = 1,
) -> str:
    """
    Search for text in a file and return matches with context.

    Args:
        computer: The ComputerInterface to execute on
        file: Path to the file to search
        query: Text to search for (case-insensitive)
        context_lines: Lines of context before/after each match
        max_matches: Maximum matches per page
        page: Which page of results (1-indexed)

    Returns:
        Matching lines with context and line numbers
    """
    result = await computer.send_shell_command(f"cat {file}")
    content = result.output.decode("utf-8").strip()
    lines = content.splitlines()

    # Find matching line indices (case-insensitive)
    query_lower = query.lower()
    match_indices = [i for i, line in enumerate(lines) if query_lower in line.lower()]

    if not match_indices:
        return f"No matches found for '{query}' in {file}"

    total_matches = len(match_indices)
    total_pages = (total_matches + max_matches - 1) // max_matches

    if page < 1 or page > total_pages:
        return f"Invalid page {page}. Total pages: {total_pages}"

    # Get matches for this page
    start_idx = (page - 1) * max_matches
    end_idx = min(start_idx + max_matches, total_matches)
    page_matches = match_indices[start_idx:end_idx]

    output_parts = [f"Found {total_matches} matches. Showing page {page}/{total_pages}.\n"]

    for match_idx in page_matches:
        # Get context window
        ctx_start = max(0, match_idx - context_lines)
        ctx_end = min(len(lines), match_idx + context_lines + 1)

        output_parts.append(f"\n--- Match at line {match_idx + 1} ---")
        for i in range(ctx_start, ctx_end):
            marker = ">>>" if i == match_idx else "   "
            output_parts.append(f"{marker} {i + 1}: {lines[i]}")

    return "\n".join(output_parts)


async def execute_submit(end_message: str) -> str:
    """
    Handle the submit tool call (signals agent completion).

    This is a no-op that just returns confirmation.
    The actual completion detection is done via is_submit_tool_call().

    Args:
        end_message: The agent's final message

    Returns:
        Confirmation string
    """
    return f"Submission received: {end_message}"


async def execute_tool(
    computer: "ComputerInterface",
    tool_name: str,
    tool_args: dict[str, Any],
) -> str:
    """
    Execute a paperbench tool by name.

    This is a convenience function that dispatches to the appropriate
    tool execution function based on the tool name.

    Args:
        computer: The ComputerInterface to execute on
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool

    Returns:
        Tool execution result as a string

    Example:
        >>> result = await execute_tool(computer, "bash", {"cmd": "ls -la"})
    """
    if tool_name == "bash":
        return await execute_bash(computer, tool_args["cmd"])
    elif tool_name == "python-tool":
        return await execute_python(computer, tool_args["code"])
    elif tool_name == "read_file_chunk":
        return await read_file_chunk(
            computer,
            tool_args["file"],
            tool_args.get("start_line", 1),
            tool_args.get("max_lines", 50),
        )
    elif tool_name == "search_file":
        return await search_file(
            computer,
            tool_args["file"],
            tool_args["query"],
            tool_args.get("context_lines", 2),
            tool_args.get("max_matches", 5),
            tool_args.get("page", 1),
        )
    elif tool_name == "submit":
        return await execute_submit(tool_args["end_message"])
    else:
        return f"ERROR: Unknown tool '{tool_name}'"


# Type for tool functions
ToolFunction = Callable[..., Awaitable[str]]


def get_tool_map(computer: "ComputerInterface") -> dict[str, Callable[..., Awaitable[str]]]:
    """
    Get a mapping from tool names to callable functions.

    This returns a dict that maps tool names to async callables.
    Each callable takes the tool arguments as keyword arguments.

    Args:
        computer: The ComputerInterface to bind tools to

    Returns:
        Dict mapping tool name -> async callable

    Example:
        >>> tool_map = get_tool_map(computer)
        >>> result = await tool_map["bash"](cmd="ls -la")
    """
    return {
        "bash": lambda **kwargs: execute_bash(computer, kwargs["cmd"]),
        "python-tool": lambda **kwargs: execute_python(computer, kwargs["code"]),
        "read_file_chunk": lambda **kwargs: read_file_chunk(
            computer,
            kwargs["file"],
            kwargs.get("start_line", 1),
            kwargs.get("max_lines", 50),
        ),
        "search_file": lambda **kwargs: search_file(
            computer,
            kwargs["file"],
            kwargs["query"],
            kwargs.get("context_lines", 2),
            kwargs.get("max_matches", 5),
            kwargs.get("page", 1),
        ),
        "submit": lambda **kwargs: execute_submit(kwargs["end_message"]),
    }


# ---------------------------------------------------------------------------
# HuggingFace Dataset utilities
# ---------------------------------------------------------------------------


def get_hf_dataset(code_only: bool = False) -> "Dataset":
    """
    Get paperbench as a HuggingFace Dataset.

    Returns a Dataset with columns compatible with the verifiers framework:
    - example_id: int - unique identifier
    - prompt: list[dict] - initial messages for the rollout
    - answer: str - empty (paperbench uses judge-based scoring)
    - task: str - paper_id
    - info: dict - paper metadata and rubric info

    Args:
        code_only: If True, use code-only mode

    Returns:
        HuggingFace Dataset object

    Example:
        >>> dataset = get_hf_dataset()
        >>> len(dataset)
        20
        >>> dataset[0]["task"]
        'rice'
    """
    from datasets import Dataset as HFDataset

    prompt_messages = get_initial_prompt(code_only)
    rows = []

    for idx, paper_id in enumerate(list_paper_ids()):
        paper = _get_paper_info(paper_id)
        rubric = get_rubric(paper_id)
        leaves = rubric.get_leaf_nodes()

        if code_only:
            leaves = [l for l in leaves if l.task_category == "Code Development"]

        rows.append({
            "example_id": idx,
            "prompt": prompt_messages,
            "answer": "",  # Paperbench uses judge-based scoring, not exact match
            "task": paper_id,
            "info": {
                "paper_id": paper_id,
                "title": paper.title,
                "paper_pdf_path": str(paper.paper_pdf),
                "paper_md_path": str(paper.paper_md),
                "addendum_path": str(paper.addendum),
                "rubric_path": str(paper.rubric),
                "num_criteria": len(leaves),
                "total_weight": sum(l.weight for l in leaves),
                "task_categories": list({l.task_category for l in leaves if l.task_category}),
                "oai_tools": get_oai_tools(),
            },
        })

    return HFDataset.from_list(rows)


def get_paper_hf_row(paper_id: str, code_only: bool = False) -> dict[str, Any]:
    """
    Get a single paper formatted as a HuggingFace Dataset row.

    This returns a dict with the same columns as get_hf_dataset(),
    useful for creating single-paper datasets.

    Args:
        paper_id: The paper ID
        code_only: If True, use code-only mode

    Returns:
        Dict with example_id, prompt, answer, task, info columns
    """
    paper = _get_paper_info(paper_id)
    rubric = get_rubric(paper_id)
    leaves = rubric.get_leaf_nodes()

    if code_only:
        leaves = [l for l in leaves if l.task_category == "Code Development"]

    return {
        "example_id": 0,
        "prompt": get_initial_prompt(code_only),
        "answer": "",
        "task": paper_id,
        "info": {
            "paper_id": paper_id,
            "title": paper.title,
            "paper_pdf_path": str(paper.paper_pdf),
            "paper_md_path": str(paper.paper_md),
            "addendum_path": str(paper.addendum),
            "rubric_path": str(paper.rubric),
            "num_criteria": len(leaves),
            "total_weight": sum(l.weight for l in leaves),
            "task_categories": list({l.task_category for l in leaves if l.task_category}),
            "oai_tools": get_oai_tools(),
        },
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Types
    "ChatMessage",
    "PaperInfo",
    "ChatCompletionToolParam",
    "FunctionDefinition",
    "ToolDefinition",
    "RubricNode",
    "PaperBenchExample",
    # Constants
    "WORKSPACE_BASE",
    "SUBMISSION_DIR",
    # Prompt functions
    "get_instructions_path",
    "get_initial_prompt",
    "get_instructions_text",
    # Paper functions
    "get_paper_info",
    "list_paper_ids",
    "get_papers_directory",
    "get_paper_files",
    "read_paper_markdown",
    "read_paper_addendum",
    "read_paper_rubric",
    # Setup function
    "run_paper_setup",
    # Tool utilities - definitions
    "get_oai_tools",
    "get_tool_definitions",
    "get_tool_names",
    "is_submit_tool_call",
    # Tool utilities - execution
    "execute_bash",
    "execute_python",
    "read_file_chunk",
    "search_file",
    "execute_submit",
    "execute_tool",
    "get_tool_map",
    # Rubric utilities
    "get_rubric",
    "get_rubric_dict",
    # Dataset utilities - legacy
    "get_paperbench_dataset",
    "get_paper_as_dict",
    # Dataset utilities - HuggingFace
    "get_hf_dataset",
    "get_paper_hf_row",
]
