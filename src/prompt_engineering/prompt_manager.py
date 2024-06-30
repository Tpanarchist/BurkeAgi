# src\prompt_engineering\prompt_manager.py

import json
import os
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()
        # This should be replaced with actual KB service integration
        self.kb_service = DummyKBService()

    def load_prompts(self) -> Dict[str, Any]:
        if os.path.exists(self.prompts_file):
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Prompts file {self.prompts_file} not found. Creating a new one.")
            return {}

    def save_prompts(self):
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)

    def get_prompt(self, prompt_name: str, agent: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            prompt_template = self.prompts[agent][prompt_name]['current']
            if context:
                return prompt_template.format(**context)
            return prompt_template
        except KeyError:
            logger.error(f"Prompt '{prompt_name}' for agent '{agent}' not found.")
            return ""

    def add_prompt(self, prompt_name: str, agent: str, prompt_template: str):
        if agent not in self.prompts:
            self.prompts[agent] = {}
        if prompt_name not in self.prompts[agent]:
            self.prompts[agent][prompt_name] = {'versions': [], 'current': ''}
        self.version_prompt(prompt_name, agent)
        self.prompts[agent][prompt_name]['current'] = prompt_template
        self.save_prompts()
        logger.info(f"Prompt '{prompt_name}' for agent '{agent}' added/updated.")

    def generate_prompt(self, task: str, agent: str, technique: str = "standard", **kwargs) -> str:
        base_prompt = f"As an AI assistant functioning as the {agent} layer, your task is to {task}."
        
        if technique == "few_shot":
            examples = kwargs.get("examples", [])
            base_prompt += "\n\nHere are some examples:"
            for example in examples:
                base_prompt += f"\nInput: {example['input']}\nOutput: {example['output']}"
            base_prompt += "\n\nNow, please provide the output for the following input:\nInput: {input}"
        
        elif technique == "cot":
            base_prompt += "\n\nLet's approach this step-by-step:\n1)"
        
        elif technique == "tree_of_thought":
            depth = kwargs.get("depth", 3)
            base_prompt += "\n\nLet's explore multiple paths to solve this problem:"
            for i in range(1, depth + 1):
                base_prompt += f"\n\nDepth {i}:"
                for j in range(1, 4):  # Assuming 3 branches at each depth
                    base_prompt += f"\nBranch {j}:"
        
        elif technique == "zero_shot_cot":
            base_prompt += "\n\nLet's solve this problem step by step. First,"

        return base_prompt

    def enhance_prompt_with_kb(self, prompt: str, kb_query: str) -> str:
        kb_content = self.kb_service.search_articles(kb_query)
        return f"Based on the following information:\n\n{kb_content}\n\n{prompt}"

    def save_prompt_as_article(self, prompt_name: str, agent: str):
        prompt = self.get_prompt(prompt_name, agent)
        self.kb_service.create_article(f"Prompt: {prompt_name}\nAgent: {agent}\n\n{prompt}")

    def generate_prompt_from_article(self, article_title: str) -> str:
        article_content = self.kb_service.search_articles(article_title)
        return f"Based on the following information:\n\n{article_content}\n\nPlease provide a response for: {{query}}"

    def create_spr_prompt(self, concept: str) -> str:
        return f"""Create a Sparse Priming Representation (SPR) for the following concept:
        Concept: {concept}
        An SPR is a concise list of statements that capture the essence of the concept.
        Each statement should be clear, concise, and self-contained.
        Aim for 5-10 statements that collectively represent the key aspects of the concept."""

    def create_hmcs_prompt(self, log_entries: List[str], summary_length: int = 5) -> str:
        logs = "\n".join(log_entries)
        return f"""Given the following log entries:
        {logs}
        Please provide a hierarchical summary of these logs, condensing the information into {summary_length} key points or less. 
        Focus on the most important and relevant information, organizing it in a way that captures the essence of the logs efficiently."""

    def get_or_generate_prompt(self, prompt_name: str, agent: str, task: str, technique: str = "standard", kb_query: Optional[str] = None, **kwargs) -> str:
        if prompt_name in self.prompts.get(agent, {}):
            prompt = self.get_prompt(prompt_name, agent, kwargs)
        else:
            prompt = self.generate_prompt(task, agent, technique, **kwargs)
            self.add_prompt(prompt_name, agent, prompt)
        
        if kb_query:
            prompt = self.enhance_prompt_with_kb(prompt, kb_query)
        
        return prompt

    def version_prompt(self, prompt_name: str, agent: str):
        if agent in self.prompts and prompt_name in self.prompts[agent]:
            current_prompt = self.prompts[agent][prompt_name]['current']
            timestamp = datetime.now().isoformat()
            self.prompts[agent][prompt_name]['versions'].append({
                'timestamp': timestamp,
                'content': current_prompt
            })
            logger.info(f"Created new version of prompt '{prompt_name}' for agent '{agent}'")

# Dummy KB service for demonstration purposes
class DummyKBService:
    def search_articles(self, query: str) -> str:
        return f"This is a dummy KB article for query: {query}"

    def create_article(self, content: str):
        logger.info(f"Created dummy KB article with content: {content}")

# Example usage
if __name__ == "__main__":
    prompt_manager = PromptManager()

    # Adding a sample prompt
    prompt_manager.add_prompt(
        "greeting",
        "general",
        "Hello, {name}! Welcome to {project_name}."
    )

    # Retrieving and using the prompt
    greeting = prompt_manager.get_prompt(
        "greeting",
        "general",
        {"name": "Alice", "project_name": "BurkeAGI"}
    )
    print("Basic prompt:", greeting)

    # Generating a CoT prompt
    cot_prompt = prompt_manager.get_or_generate_prompt(
        "problem_solving",
        "cognitive_control",
        "solve a complex problem",
        technique="cot",
        input="How can we reduce carbon emissions in urban areas?"
    )
    print("\nCoT prompt:", cot_prompt)

    # Creating an SPR prompt
    spr_prompt = prompt_manager.create_spr_prompt("Artificial General Intelligence")
    print("\nSPR prompt:", spr_prompt)

    # Creating an HMCS prompt
    log_entries = [
        "2023-06-15 10:00:00 - System initialized",
        "2023-06-15 10:05:00 - Received user query about climate change",
        "2023-06-15 10:10:00 - Accessed knowledge base for relevant information",
        "2023-06-15 10:15:00 - Generated response using ethical decision-making framework",
        "2023-06-15 10:20:00 - User feedback received: response rated 4/5 stars"
    ]
    hmcs_prompt = prompt_manager.create_hmcs_prompt(log_entries)
    print("\nHMCS prompt:", hmcs_prompt)

    # Demonstrating KB integration (with dummy service)
    kb_enhanced_prompt = prompt_manager.get_or_generate_prompt(
        "ethical_decision",
        "aspirational",
        "make an ethical decision",
        kb_query="AI ethics guidelines"
    )
    print("\nKB-enhanced prompt:", kb_enhanced_prompt)

    # Demonstrating prompt versioning
    prompt_manager.add_prompt(
        "greeting",
        "general",
        "Greetings, {name}! Welcome to the {project_name} project."
    )
    print("\nPrompt versions:", json.dumps(prompt_manager.prompts['general']['greeting'], indent=2))