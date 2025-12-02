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
from typing import TYPE_CHECKING, Any, TypedDict

import yaml

# Avoid heavy imports - use inline imports where needed
# This allows api.py to be imported without pulling in all dependencies

if TYPE_CHECKING:
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
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Types
    "ChatMessage",
    "PaperInfo",
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
]
