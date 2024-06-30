# src/agents/base_agent.py

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..prompt_engineering.prompt_manager import PromptManager
from ..knowledge_base.kb_service import KBService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, agent_name: str, layer_name: str):
        self.agent_name = agent_name
        self.layer_name = layer_name
        self.client = OpenAI()
        self.prompt_manager = PromptManager()
        self.kb_service = KBService()

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 150, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return ""

    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1) -> List[str]:
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=n
            )
            return [image.url for image in response.data]
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            return []

    def transcribe_audio(self, audio_file_path: str, prompt: Optional[str] = None) -> str:
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    prompt=prompt
                )
            return response.text
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return ""

    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            return response.content
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            return b""

    def create_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in creating embedding: {str(e)}")
            return []

    def moderate_content(self, text: str) -> Dict[str, Any]:
        try:
            response = self.client.moderations.create(input=text)
            return response.results[0].model_dump()
        except Exception as e:
            logger.error(f"Error in content moderation: {str(e)}")
            return {}

    def get_prompt(self, prompt_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        return self.prompt_manager.get_prompt(prompt_name, self.layer_name, context)

    def add_prompt(self, prompt_name: str, prompt_template: str):
        self.prompt_manager.add_prompt(prompt_name, self.layer_name, prompt_template)

    def enhance_prompt_with_kb(self, prompt: str, kb_query: str) -> str:
        return self.prompt_manager.enhance_prompt_with_kb(prompt, kb_query)

    def create_kb_article(self, title: str, content: str, content_type: str = "text") -> Dict[str, Any]:
        return self.kb_service.create_article(title, content, content_type)

    def search_kb_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.kb_service.search_articles(query, top_k)

    def update_kb_article(self, title: str, content: str, content_type: str = "text") -> Dict[str, Any]:
        return self.kb_service.update_article(title, content, content_type)

    def get_kb_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        return self.kb_service.get_article_content(title)

    def generate_kb_summary(self, title: str, max_tokens: int = 150) -> str:
        return self.kb_service.generate_summary(title, max_tokens)

    def process_input(self, input_data: Any) -> Any:
        """
        Process input data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_input method")

    def generate_output(self, processed_data: Any) -> Any:
        """
        Generate output based on processed data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_output method")

    def run(self, input_data: Any) -> Any:
        """
        Main execution method for the agent.
        """
        processed_data = self.process_input(input_data)
        return self.generate_output(processed_data)

# Example usage
if __name__ == "__main__":
    agent = BaseAgent("example_agent", "example_layer")
    
    # Chat completion example
    chat_response = agent.chat_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ])
    print("Chat response:", chat_response)

    # Image generation example
    image_urls = agent.generate_image("A futuristic city with flying cars")
    print("Generated image URLs:", image_urls)

    # Prompt management example
    agent.add_prompt("greeting", "Hello, {name}! Welcome to {project}.")
    greeting = agent.get_prompt("greeting", {"name": "Alice", "project": "ChatDev"})
    print("Greeting:", greeting)

    # KB integration example
    kb_article = agent.create_kb_article("AI Ethics", "AI ethics is the study of ethical issues related to artificial intelligence.")
    print("Created KB article:", kb_article)

    kb_search_results = agent.search_kb_articles("ethics in AI")
    print("KB search results:", kb_search_results)