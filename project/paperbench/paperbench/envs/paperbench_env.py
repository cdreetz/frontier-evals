"""
PaperBench Environment for the verifiers framework.

This module provides a complete Environment implementation for running
PaperBench evaluations with the verifiers framework.

Example usage:
    from paperbench.envs.paperbench_env import PaperBenchEnvironment
    from openai import AsyncOpenAI

    env = PaperBenchEnvironment(paper_ids=["rice"])
    client = AsyncOpenAI()

    results = await env.evaluate(
        client=client,
        model="gpt-4o",
        num_examples=1,
        rollouts_per_example=1,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict
from typing import Any, AsyncIterator

from datasets import Dataset
from openai import AsyncOpenAI

# Paperbench imports
from paperbench.api import (
    get_hf_dataset,
    get_initial_prompt,
    get_oai_tools,
    get_rubric,
    is_submit_tool_call,
    list_paper_ids,
    run_paper_setup,
)

# Container/sandbox imports
from alcatraz import LocalConfig
from nanoeval.solvers.computer_tasks import ComputerConfiguration, NetworkMode
from nanoeval_alcatraz import AlcatrazComputerRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool execution (inline to avoid circular imports)
# ---------------------------------------------------------------------------


async def _execute_bash(computer, cmd: str) -> str:
    """Execute bash command on container."""
    result = await computer.send_shell_command(cmd=cmd)
    return result.output.decode("utf-8").strip()


async def _execute_python(computer, code: str) -> str:
    """Execute Python code on container."""
    result = await computer.send_shell_command("mktemp -d")
    tmp_dir = result.output.decode("utf-8").strip()
    await computer.upload(code.encode("utf-8"), f"{tmp_dir}/code.py")
    result = await computer.send_shell_command(f"python3 {tmp_dir}/code.py")
    return result.output.decode("utf-8").strip()


async def _read_file_chunk(computer, file: str, start_line: int = 1, max_lines: int = 50) -> str:
    """Read file chunk with line numbers."""
    result = await computer.send_shell_command(f"cat {file} 2>&1")
    content = result.output.decode("utf-8").strip()

    if "No such file" in content or "cannot access" in content.lower():
        return f"ERROR: File not found: {file}"

    lines = content.splitlines()
    if not lines:
        return f"File {file} is empty."

    start_line = max(1, start_line)
    max_lines = min(50, max(1, max_lines))

    if start_line > len(lines):
        return f"ERROR: start_line ({start_line}) beyond file length ({len(lines)} lines)"

    end_line = min(start_line + max_lines - 1, len(lines))
    chunk = lines[start_line - 1 : end_line]
    numbered = [f"{i + start_line}: {line}" for i, line in enumerate(chunk)]

    return f"File has {len(lines)} lines. Showing {start_line}-{end_line}.\n\n" + "\n".join(numbered)


async def _search_file(computer, file: str, query: str, context_lines: int = 2, max_matches: int = 5, page: int = 1) -> str:
    """Search file for query with context."""
    result = await computer.send_shell_command(f"cat {file} 2>&1")
    content = result.output.decode("utf-8").strip()

    if "No such file" in content:
        return f"ERROR: File not found: {file}"

    lines = content.splitlines()
    query_lower = query.lower()
    matches = [i for i, line in enumerate(lines) if query_lower in line.lower()]

    if not matches:
        return f"No matches for '{query}' in {file}"

    total = len(matches)
    pages = (total + max_matches - 1) // max_matches
    page = max(1, min(page, pages))

    start = (page - 1) * max_matches
    end = min(start + max_matches, total)

    parts = [f"Found {total} matches. Page {page}/{pages}.\n"]
    for idx in matches[start:end]:
        ctx_start = max(0, idx - context_lines)
        ctx_end = min(len(lines), idx + context_lines + 1)
        parts.append(f"\n--- Line {idx + 1} ---")
        for i in range(ctx_start, ctx_end):
            marker = ">>>" if i == idx else "   "
            parts.append(f"{marker} {i + 1}: {lines[i]}")

    return "\n".join(parts)


async def execute_tool(computer, tool_name: str, tool_args: dict) -> str:
    """Execute a paperbench tool."""
    try:
        if tool_name == "bash":
            return await _execute_bash(computer, tool_args.get("cmd", ""))
        elif tool_name == "python-tool":
            return await _execute_python(computer, tool_args.get("code", ""))
        elif tool_name == "read_file_chunk":
            return await _read_file_chunk(
                computer,
                tool_args.get("file", ""),
                tool_args.get("start_line", 1),
                tool_args.get("max_lines", 50),
            )
        elif tool_name == "search_file":
            return await _search_file(
                computer,
                tool_args.get("file", ""),
                tool_args.get("query", ""),
                tool_args.get("context_lines", 2),
                tool_args.get("max_matches", 5),
                tool_args.get("page", 1),
            )
        elif tool_name == "submit":
            return f"Submission received: {tool_args.get('end_message', '')}"
        else:
            return f"ERROR: Unknown tool '{tool_name}'"
    except Exception as e:
        return f"ERROR executing {tool_name}: {str(e)}"


# ---------------------------------------------------------------------------
# Sandbox/Container Management
# ---------------------------------------------------------------------------


@asynccontextmanager
async def create_sandbox(
    docker_image: str = "python:3.11-slim",
    network_mode: NetworkMode = NetworkMode.UNPROXIED,
) -> AsyncIterator[Any]:
    """
    Create and manage a sandbox container for paper reproduction.

    Args:
        docker_image: Docker image to use
        network_mode: Network access mode

    Yields:
        ComputerInterface for executing commands
    """
    runtime = AlcatrazComputerRuntime(env=LocalConfig())
    config = ComputerConfiguration(
        cwd="/home",
        docker_image=docker_image,
        network_mode=network_mode,
    )

    async with runtime._start_computer(config) as computer:
        yield computer


# ---------------------------------------------------------------------------
# Simple Grading (without full judge - uses rubric structure)
# ---------------------------------------------------------------------------


async def simple_grade(computer, paper_id: str, code_only: bool = False) -> dict:
    """
    Perform simple grading based on submission state.

    This is a simplified grader that checks:
    - Whether reproduce.sh exists
    - Whether it runs without errors
    - Basic output validation

    For full grading, use the paperbench judge system.

    Returns:
        Dict with score and details
    """
    rubric = get_rubric(paper_id)
    total_leaves = len(rubric.get_leaf_nodes())

    # Check if submission directory exists
    result = await computer.send_shell_command("ls -la /home/submission 2>&1")
    submission_content = result.output.decode("utf-8").strip()

    if "No such file" in submission_content:
        return {
            "score": 0.0,
            "submission_exists": False,
            "reproduce_exists": False,
            "reproduce_ran": False,
            "details": "No submission directory found",
            "total_criteria": total_leaves,
        }

    # Check for reproduce.sh
    result = await computer.send_shell_command("test -f /home/submission/reproduce.sh && echo 'exists'")
    reproduce_exists = "exists" in result.output.decode("utf-8")

    if not reproduce_exists:
        return {
            "score": 0.1,  # Some credit for having a submission
            "submission_exists": True,
            "reproduce_exists": False,
            "reproduce_ran": False,
            "details": "Submission exists but no reproduce.sh found",
            "total_criteria": total_leaves,
        }

    # Try running reproduce.sh (with timeout)
    result = await computer.send_shell_command(
        "cd /home/submission && timeout 60 bash reproduce.sh > reproduce.log 2>&1; echo $?"
    )
    exit_code = result.output.decode("utf-8").strip()

    reproduce_ran = exit_code == "0"

    # Read reproduce.log for details
    result = await computer.send_shell_command("cat /home/submission/reproduce.log 2>&1 | tail -50")
    log_tail = result.output.decode("utf-8").strip()

    if reproduce_ran:
        score = 0.5  # Base score for successful reproduction
    else:
        score = 0.2  # Partial credit for having reproduce.sh

    return {
        "score": score,
        "submission_exists": True,
        "reproduce_exists": True,
        "reproduce_ran": reproduce_ran,
        "exit_code": exit_code,
        "log_tail": log_tail[:1000],  # Truncate log
        "details": "reproduce.sh ran successfully" if reproduce_ran else f"reproduce.sh failed with exit code {exit_code}",
        "total_criteria": total_leaves,
    }


# ---------------------------------------------------------------------------
# PaperBench Environment
# ---------------------------------------------------------------------------


class PaperBenchEnvironment:
    """
    PaperBench Environment for running paper reproduction evaluations.

    This environment:
    1. Creates a Docker container for each rollout
    2. Sets up the paper files (PDF, instructions, etc.)
    3. Runs an agent loop with tool calls
    4. Grades the submission when complete

    Args:
        paper_ids: List of paper IDs to evaluate (default: all papers)
        code_only: Whether to use code-only mode
        docker_image: Docker image for containers
        max_turns: Maximum agent turns before stopping
        system_prompt: Optional system prompt override
    """

    def __init__(
        self,
        paper_ids: list[str] | None = None,
        code_only: bool = False,
        docker_image: str = "python:3.11-slim",
        max_turns: int = 50,
        system_prompt: str | None = None,
    ):
        self.paper_ids = paper_ids or list_paper_ids()
        self.code_only = code_only
        self.docker_image = docker_image
        self.max_turns = max_turns
        self.system_prompt = system_prompt

        # Build dataset
        self.dataset = self._build_dataset()
        self.oai_tools = get_oai_tools()

        self.logger = logging.getLogger(f"{__name__}.PaperBenchEnvironment")

    def _build_dataset(self) -> Dataset:
        """Build HuggingFace dataset for selected papers."""
        prompt_messages = get_initial_prompt(self.code_only)

        if self.system_prompt:
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                *prompt_messages,
            ]

        rows = []
        for idx, paper_id in enumerate(self.paper_ids):
            rubric = get_rubric(paper_id)
            leaves = rubric.get_leaf_nodes()

            if self.code_only:
                leaves = [l for l in leaves if l.task_category == "Code Development"]

            rows.append({
                "example_id": idx,
                "prompt": prompt_messages,
                "answer": "",
                "task": paper_id,
                "paper_id": paper_id,
                "num_criteria": len(leaves),
            })

        return Dataset.from_list(rows)

    async def setup_state(self, state: dict, computer) -> dict:
        """
        Set up rollout-level state including sandbox.

        Args:
            state: Current state dict
            computer: The ComputerInterface

        Returns:
            Updated state with sandbox and tools bound
        """
        paper_id = state["paper_id"]

        # Run paper setup (upload files to container)
        await run_paper_setup(paper_id, computer, code_only=self.code_only)

        # Store computer reference
        state["computer"] = computer
        state["paper_id"] = paper_id
        state["messages"] = deepcopy(state["prompt"])
        state["turn"] = 0
        state["is_completed"] = False
        state["submit_called"] = False
        state["tool_calls"] = []
        state["grade"] = None

        self.logger.info(f"Setup complete for paper: {paper_id}")
        return state

    async def step(self, state: dict, client: AsyncOpenAI, model: str) -> dict:
        """
        Execute one step of the agent loop.

        Args:
            state: Current state
            client: OpenAI client
            model: Model to use

        Returns:
            Updated state
        """
        state["turn"] += 1

        # Get model response
        response = await client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=self.oai_tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # Add assistant message to history
        state["messages"].append(assistant_message.model_dump())

        # Check for tool calls
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                self.logger.debug(f"Tool call: {tool_name}({tool_args})")

                # Record tool call
                state["tool_calls"].append({
                    "name": tool_name,
                    "args": tool_args,
                    "turn": state["turn"],
                })

                # Check for submit
                if is_submit_tool_call(tool_name):
                    state["submit_called"] = True
                    state["is_completed"] = True
                    tool_result = f"Submission received: {tool_args.get('end_message', '')}"
                else:
                    # Execute tool
                    tool_result = await execute_tool(
                        state["computer"],
                        tool_name,
                        tool_args,
                    )

                # Add tool result to messages
                state["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
        else:
            # No tool calls - check finish reason
            if choice.finish_reason == "stop":
                # Model stopped without tool call - might be done
                pass

        # Check max turns
        if state["turn"] >= self.max_turns:
            state["is_completed"] = True
            self.logger.info(f"Max turns ({self.max_turns}) reached")

        return state

    async def grade_rollout(self, state: dict) -> dict:
        """
        Grade the completed rollout.

        Args:
            state: Final state after rollout

        Returns:
            State with grade added
        """
        paper_id = state["paper_id"]
        computer = state["computer"]

        self.logger.info(f"Grading submission for {paper_id}")

        # Run grading
        grade = await simple_grade(computer, paper_id, self.code_only)
        state["grade"] = grade
        state["reward"] = grade["score"]

        self.logger.info(f"Grade for {paper_id}: {grade['score']:.2f}")

        return state

    async def rollout(
        self,
        example: dict,
        client: AsyncOpenAI,
        model: str,
    ) -> dict:
        """
        Run a complete rollout for one example.

        Args:
            example: Dataset row with paper info
            client: OpenAI client
            model: Model to use

        Returns:
            Final state with grade
        """
        paper_id = example["paper_id"]
        self.logger.info(f"Starting rollout for paper: {paper_id}")

        # Initialize state from example
        state = {
            "example_id": example["example_id"],
            "prompt": example["prompt"],
            "paper_id": paper_id,
            "task": example["task"],
        }

        # Create sandbox and run
        async with create_sandbox(docker_image=self.docker_image) as computer:
            # Setup
            state = await self.setup_state(state, computer)

            # Agent loop
            while not state["is_completed"]:
                state = await self.step(state, client, model)

            # Grade
            state = await self.grade_rollout(state)

        # Clean up computer reference (not serializable)
        state.pop("computer", None)

        # Build completion from messages
        state["completion"] = state["messages"][len(state["prompt"]):]

        self.logger.info(f"Rollout complete for {paper_id}, score: {state['reward']:.2f}")

        return state

    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = 1,
    ) -> list[dict]:
        """
        Run evaluation on the dataset.

        Args:
            client: OpenAI client
            model: Model to use
            num_examples: Number of examples (-1 for all)
            rollouts_per_example: Rollouts per paper
            max_concurrent: Max concurrent rollouts

        Returns:
            List of final states with grades
        """
        # Select examples
        if num_examples > 0:
            examples = self.dataset.select(range(min(num_examples, len(self.dataset))))
        else:
            examples = self.dataset

        # Expand for rollouts_per_example
        all_examples = []
        for example in examples:
            for r in range(rollouts_per_example):
                ex = dict(example)
                ex["rollout_idx"] = r
                all_examples.append(ex)

        self.logger.info(f"Running {len(all_examples)} rollouts ({len(examples)} papers x {rollouts_per_example} rollouts)")

        # Run rollouts (with concurrency limit)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_sem(example):
            async with semaphore:
                return await self.rollout(example, client, model)

        tasks = [run_with_sem(ex) for ex in all_examples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for ex, result in zip(all_examples, results):
            if isinstance(result, Exception):
                self.logger.error(f"Rollout failed for {ex['paper_id']}: {result}")
                final_results.append({
                    "paper_id": ex["paper_id"],
                    "example_id": ex["example_id"],
                    "error": str(result),
                    "reward": 0.0,
                })
            else:
                final_results.append(result)

        # Summary
        rewards = [r.get("reward", 0.0) for r in final_results]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        self.logger.info(f"Evaluation complete. Avg reward: {avg_reward:.2f}")

        return final_results


# ---------------------------------------------------------------------------
# Convenience function to run a single paper
# ---------------------------------------------------------------------------


async def run_single_paper(
    paper_id: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    max_turns: int = 50,
    docker_image: str = "python:3.11-slim",
    code_only: bool = False,
) -> dict:
    """
    Run a single paper evaluation end-to-end.

    Args:
        paper_id: Paper ID to evaluate
        model: Model to use
        api_key: OpenAI API key (uses env var if not provided)
        base_url: Optional base URL for API
        max_turns: Max agent turns
        docker_image: Docker image for sandbox
        code_only: Whether to use code-only mode

    Returns:
        Final state with grade

    Example:
        >>> result = await run_single_paper("rice", model="gpt-4o")
        >>> print(f"Score: {result['reward']}")
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    env = PaperBenchEnvironment(
        paper_ids=[paper_id],
        code_only=code_only,
        docker_image=docker_image,
        max_turns=max_turns,
    )

    results = await env.evaluate(
        client=client,
        model=model,
        num_examples=1,
        rollouts_per_example=1,
    )

    return results[0]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaperBench evaluation")
    parser.add_argument("--paper", type=str, required=True, help="Paper ID to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--max-turns", type=int, default=50, help="Max agent turns")
    parser.add_argument("--docker-image", type=str, default="python:3.11-slim", help="Docker image")
    parser.add_argument("--code-only", action="store_true", help="Use code-only mode")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = asyncio.run(run_single_paper(
        paper_id=args.paper,
        model=args.model,
        max_turns=args.max_turns,
        docker_image=args.docker_image,
        code_only=args.code_only,
    ))

    print("\n" + "=" * 60)
    print(f"Paper: {result.get('paper_id')}")
    print(f"Score: {result.get('reward', 0):.2f}")
    print(f"Turns: {result.get('turn', 0)}")
    print(f"Submit called: {result.get('submit_called', False)}")
    if result.get("grade"):
        print(f"Grade details: {result['grade'].get('details', 'N/A')}")
    print("=" * 60)
