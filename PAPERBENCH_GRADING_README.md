# PaperBench Grading Integration Guide

This document explains how the PaperBench grading system works and how to integrate it with the verifiers framework.

## Overview

PaperBench uses a **hierarchical rubric-based grading system** where:

1. **Rubric Structure**: Each paper has a rubric.json file defining a tree of requirements
2. **Leaf Nodes**: The actual criteria to evaluate (scored 0 or 1)
3. **Internal Nodes**: Aggregate scores from children using weighted averaging
4. **Root Score**: The final weighted score (0.0 to 1.0)

## Grading Process

```
                    Root (score = 0.75)
                   /                   \
          Section A (0.8)          Section B (0.5)
         /     |     \                  |
      Leaf1  Leaf2  Leaf3             Leaf4
      (1.0)  (1.0)  (0.0)             (1.0)

Score calculation:
- Leaf nodes: Binary 0 or 1 from LLM evaluation
- Internal nodes: weighted_sum(child.score * child.weight) / total_weight
```

## Key Components

### 1. Task Categories (Leaf Node Types)

Each leaf node has a `task_category` that determines how it's evaluated:

| Category | Question Asked | Evaluates |
|----------|---------------|-----------|
| `Code Development` | Does the code contain a correct implementation? | Source code analysis |
| `Code Execution` | Does reproduce.sh run successfully? | Execution logs |
| `Result Analysis` | Do outputs match expected results? | Output files, logs |

### 2. Judge Types

```python
# For production (LLM-based evaluation)
judge_type="simple"  # Uses GPT-4o by default

# For testing the pipeline
judge_type="random"  # Random 0/1 scores

# For always-pass testing
judge_type="dummy"   # Always returns 1.0
```

### 3. SimpleJudge LLM Evaluation

For each leaf criterion, the SimpleJudge:

1. **Ranks files** by relevance to the criterion
2. **Constructs a prompt** with:
   - The paper (markdown)
   - Relevant submission files
   - reproduce.sh and reproduce.log
   - The specific criterion
3. **Evaluates** using structured reasoning:
   - **Expectations**: What correct implementation looks like
   - **Reality**: What the submission actually contains
   - **Score**: Binary 0 or 1 with explanation
4. **Parses** the response into structured output

## Integration with Verifiers

### Key Changes from Your Original Implementation

```python
# OLD: Simple heuristic grading
async def _simple_grade(self, computer, paper_id):
    # Only checked if reproduce.sh exists/runs
    return {"score": 0.5 if success else 0.2, ...}

# NEW: Full rubric-based grading
async def _run_rubric_grading(self, computer, paper_id):
    graded_tree = await run_judge(
        submission_path=Path("/home/submission"),
        paper_id=paper_id,
        judge_type=self.judge_type,
        code_only=self.code_only,
        completer_config=completer_config,
        computer=computer,
    )
    return graded_tree  # Contains per-criterion scores
```

### Required Imports

```python
# Grading infrastructure
from paperbench.grade import run_judge, JudgeOutput
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.rubric.tasks import TaskNode

# For SimpleJudge
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
```

### Score Extraction

```python
class PaperBenchRubric(Rubric):
    async def score_rollout(self, state: State) -> float:
        grade = state.get("grade")
        if isinstance(grade, dict) and "graded_task_tree" in grade:
            tree = grade["graded_task_tree"]
            return tree.score if isinstance(tree, GradedTaskNode) else tree.get("score", 0.0)
        return 0.0
```

## Usage Example

```python
# Load environment with LLM-based grading
env = load_environment(
    max_steps=50,
    code_only=False,
    paper_ids=["rice"],  # Optional filter
    judge_type="simple",
    judge_model="gpt-4o-2024-08-06",
)

# For testing (faster, no API calls)
env = load_environment(
    judge_type="random",  # or "dummy"
)
```

## Grade Output Structure

After grading, `state["grade"]` contains:

```python
{
    "score": 0.75,  # Root score (0.0-1.0)
    "graded_task_tree": GradedTaskNode,  # Full tree with per-criterion scores
    "num_leaf_nodes": 15,  # Total criteria
    "num_passed": 11,  # Criteria with score > 0.5
    "judge_type": "simple",
}
```

### Extracting Detailed Grades

```python
from paperbench_grading_integration import extract_leaf_grades, get_grade_summary

# Get per-criterion breakdown
leaf_grades = extract_leaf_grades(state["grade"]["graded_task_tree"])
# [{"id": "...", "requirements": "...", "score": 1, "explanation": "..."}, ...]

# Get summary statistics
summary = get_grade_summary(state["grade"]["graded_task_tree"])
# {"overall_score": 0.75, "passed": 11, "failed": 4, "by_category": {...}}
```

## Notes

1. **API Keys**: SimpleJudge requires OpenAI API key for LLM calls
2. **Cost**: Each leaf criterion requires one LLM call (~$0.01-0.05 per criterion)
3. **Time**: Full grading takes 1-5 minutes depending on rubric size
4. **Fallback**: If full grading fails, simple heuristic grading is used

## File Reference

| File | Purpose |
|------|---------|
| `paperbench/grade.py` | `run_judge()` function, `JudgeOutput` class |
| `paperbench/judge/base.py` | Base `Judge` class with recursive grading |
| `paperbench/judge/simple.py` | `SimpleJudge` - LLM-based evaluation |
| `paperbench/judge/dummyrandom.py` | `DummyJudge`, `RandomJudge` for testing |
| `paperbench/judge/graded_task_node.py` | `GradedTaskNode`, score calculations |
| `paperbench/judge/constants.py` | Prompts used for LLM evaluation |
