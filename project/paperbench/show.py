class MyEnv(ToolEnv):
  def __init__(self):
    self.sandbox_client = SandboxClient()
    self.tools: list[Callable] = tools or []
    self.oai_tools: list[ChatCompletionFunctionToolParam] = [
        convert_func_to_oai_tool(tool) for tool in self.tools
    ]
    self.tool_map: dict[str, Callable] = {
        getattr(tool, "__name__", tool.__class__.__name__): tool
        for tool in self.tools
    }

  def setup_state(self, state):
    sandbox = self.sandbox_client.create()
    state["sandbox_id"] = sandbox.id
    return state

  async def is_completed(self, completions, state):
    completed = super().is_completed(self)
    if completed:
      await grade_submission(state["sandbox_id"])
      grade_result = download_result(state["sandbox_id"])
      state["grade_result"] = grade_result
    return completed, state

class MyRubric(Rubric):
    pass

def correctness_reward(prompt, answer, completion, state):
    return 1.0 if correct else 0.0

def grade_reward(prompt, answer, completion, state):
    grade_result = state["grade_result"]
    grade = grade_result["score"]
    return int(grade)


def format_dataset():
    dataset = Dataset.from_dict({
        "prompt": [],
        "answer": []
    })
    return dataset

def load_environment(**kwargs):
    dataset = format_dataset()
    rubric = MyRubric(
        funcs=[correctness_reward,grade_reward],
        weights=[0.5, 0.5]
    )
    env = MyEnv(
        dataset=dataset,
        rubric=rubric
    )
    return env

