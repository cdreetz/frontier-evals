from pathlib import Path
from typing import Literal
from openai.types.chat import ChatCompletionMessageParam

def get_initial_prompt_for_paper(
    paper_id: str,
    code_only: bool = False,
) -> list[ChatCompletionMessageParam]:
    paperbench_root = Path(__file__).parent.resolve()

    if code_only:
        instruction_path = paperbench_root / "paperbench" / "instructions" / "code_only_instructions.txt"
    else:
        instruction_path = paperbench_root / "paperbench" / "instructions" / "instructions.txt"

    instructions_text = instruction_path.read_text()

    return [{"role": "user", "content": instructions_text}]


if __name__ == "__main__":
    prompt = get_initial_prompt_for_paper("rice")

    print(prompt[0]["content"][:500])


