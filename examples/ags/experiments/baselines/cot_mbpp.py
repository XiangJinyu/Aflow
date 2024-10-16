from examples.ags.scripts.operator import Operator
from examples.ags.scripts.graph import SolveGraph
from examples.ags.benchmark.mbpp import mbpp_evaluation
from examples.ags.scripts.operator_an import GenerateOp
from metagpt.actions.action_node import ActionNode 
from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from pydantic import BaseModel, Field
from typing import Tuple

MBPP_PROMPT_COT = """
{question}\nPlease provide a step-by-step explanation in text, followed by your Python function, ensure the output code is self-contained, meaning it should have the correct function name and return statement, without any additional text."""


class GenerateOp(BaseModel):
    solution: str = Field(default="", description="Python Solution For This Question.")

class CoTGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(name, llm)

    async def __call__(self, problem, function_name, mode: str = None):
        prompt = MBPP_PROMPT_COT.format(question=problem)
        fill_kwargs = {"context": prompt, "llm": self.llm, "function_name": function_name}
        if mode:
            fill_kwargs["mode"] = mode
        node = await ActionNode.from_pydantic(GenerateOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()
        return response

class CoTSolveGraph(SolveGraph):
    def __init__(self, name: str, llm_config, dataset: str):
        super().__init__(name, llm_config, dataset)
        self.cot_generate = CoTGenerate(self.llm)

    async def __call__(self, question: str, entry_point) -> Tuple[str, str]:
        solution = await self.cot_generate(question, entry_point, mode="code_fill")
        return solution["solution"], self.llm.cost_manager.total_cost

if __name__ == "__main__":
    async def main():
        llm_config = ModelsConfig.default().get("gpt-4o-mini")
        # llm_config = ModelsConfig.default().get("gpt-35-turbo-1106")
        graph = CoTSolveGraph(name="CoT", llm_config=llm_config, dataset="MBPP")
        file_path = "examples/ags/data/mbpp-new-new.jsonl"
        samples = 86
        path = "examples/ags/data/baselines/general/mbpp"
        score = await mbpp_evaluation(graph, file_path, samples, path, test=True)
        return score

    import asyncio 
    asyncio.run(main())