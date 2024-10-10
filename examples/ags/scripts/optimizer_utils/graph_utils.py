import os
import re
import json
from typing import List
import traceback
import time

from examples.ags.scripts.prompts.optimize_prompt import (
    GRAPH_CUSTOM_USE,
    GRAPH_INPUT,
    GRAPH_OPTIMIZE_PROMPT,
    GRAPH_TEMPLATE
)


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, graphs_path: str):
        graphs_path = graphs_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{graphs_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "SolveGraph")
            return graph_class
        except ImportError as e:
            print(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, graphs_path: str):
        prompt_file_path = os.path.join(graphs_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(graphs_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            print(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            print(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class SolveGraph:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/graphs/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(self, experience: str, score: float, graph: str, prompt: str,
                                     operator_description: str, type: str, log_data: str) -> str:
        graph_input = GRAPH_INPUT.format(
            experience=experience, score=score, graph=graph, prompt=prompt, operator_description=operator_description,
            type=type, log=log_data
        )
        graph_system = GRAPH_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + GRAPH_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                print(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    print("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph = GRAPH_TEMPLATE.format(graph=response["graph"], round=round_number, dataset=dataset)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
