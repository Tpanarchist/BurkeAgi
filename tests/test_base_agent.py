# tests/test_base_agent.py

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.base_agent import BaseAgent

class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self.agent = BaseAgent("test_agent", "test_layer")

    @patch('agents.base_agent.OpenAI')
    def test_chat_completion(self, mock_openai):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        response = self.agent.chat_completion([{"role": "user", "content": "Hello"}])
        self.assertEqual(response, "Test response")

    @patch('agents.base_agent.OpenAI')
    def test_generate_image(self, mock_openai):
        mock_image = MagicMock()
        mock_image.url = "https://test-image-url.com"
        mock_openai.return_value.images.generate.return_value.data = [mock_image]

        urls = self.agent.generate_image("Test prompt")
        self.assertEqual(urls, ["https://test-image-url.com"])

    @patch('agents.base_agent.OpenAI')
    def test_transcribe_audio(self, mock_openai):
        mock_transcription = MagicMock()
        mock_transcription.text = "Test transcription"
        mock_openai.return_value.audio.transcriptions.create.return_value = mock_transcription

        with patch('builtins.open', MagicMock()):
            transcription = self.agent.transcribe_audio("test.mp3")
        self.assertEqual(transcription, "Test transcription")

    @patch('agents.base_agent.OpenAI')
    def test_text_to_speech(self, mock_openai):
        mock_speech = MagicMock()
        mock_speech.content = b"Test audio content"
        mock_openai.return_value.audio.speech.create.return_value = mock_speech

        audio = self.agent.text_to_speech("Test text")
        self.assertEqual(audio, b"Test audio content")

    @patch('agents.base_agent.OpenAI')
    def test_create_embedding(self, mock_openai):
        mock_embedding = MagicMock()
        mock_embedding.data[0].embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = mock_embedding

        embedding = self.agent.create_embedding("Test text")
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch('agents.base_agent.OpenAI')
    def test_moderate_content(self, mock_openai):
        mock_moderation = MagicMock()
        mock_moderation.results[0].model_dump.return_value = {"flagged": False}
        mock_openai.return_value.moderations.create.return_value = mock_moderation

        result = self.agent.moderate_content("Test content")
        self.assertEqual(result, {"flagged": False})

    @patch('agents.base_agent.PromptManager')
    def test_get_prompt(self, mock_prompt_manager):
        mock_prompt_manager.return_value.get_prompt.return_value = "Test prompt"
        
        prompt = self.agent.get_prompt("test_prompt", {"var": "value"})
        self.assertEqual(prompt, "Test prompt")
        mock_prompt_manager.return_value.get_prompt.assert_called_with("test_prompt", "test_layer", {"var": "value"})

    @patch('agents.base_agent.PromptManager')
    def test_add_prompt(self, mock_prompt_manager):
        self.agent.add_prompt("test_prompt", "Test prompt template")
        mock_prompt_manager.return_value.add_prompt.assert_called_with("test_prompt", "test_layer", "Test prompt template")

    @patch('agents.base_agent.PromptManager')
    def test_enhance_prompt_with_kb(self, mock_prompt_manager):
        mock_prompt_manager.return_value.enhance_prompt_with_kb.return_value = "Enhanced prompt"
        
        enhanced_prompt = self.agent.enhance_prompt_with_kb("Base prompt", "KB query")
        self.assertEqual(enhanced_prompt, "Enhanced prompt")
        mock_prompt_manager.return_value.enhance_prompt_with_kb.assert_called_with("Base prompt", "KB query")

    @patch('agents.base_agent.KBService')
    def test_create_kb_article(self, mock_kb_service):
        mock_kb_service.return_value.create_article.return_value = {"title": "Test", "content": "Test content"}
        
        article = self.agent.create_kb_article("Test", "Test content")
        self.assertEqual(article, {"title": "Test", "content": "Test content"})
        mock_kb_service.return_value.create_article.assert_called_with("Test", "Test content", "text")

    @patch('agents.base_agent.KBService')
    def test_search_kb_articles(self, mock_kb_service):
        mock_kb_service.return_value.search_articles.return_value = [{"title": "Test", "content": "Test content"}]
        
        results = self.agent.search_kb_articles("Test query")
        self.assertEqual(results, [{"title": "Test", "content": "Test content"}])
        mock_kb_service.return_value.search_articles.assert_called_with("Test query", 5)

    def test_process_input(self):
        with self.assertRaises(NotImplementedError):
            self.agent.process_input("Test input")

    def test_generate_output(self):
        with self.assertRaises(NotImplementedError):
            self.agent.generate_output("Test processed data")

    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.agent.run("Test input")

if __name__ == '__main__':
    unittest.main()