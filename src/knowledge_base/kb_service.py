# src\knowledge_base\kb_service.py

import json
import os
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KBService:
    def __init__(self, storage_dir: str = "kb_storage"):
        self.storage_dir = storage_dir
        self.client = OpenAI()
        self.ensure_storage_dir()
        self.embedding_cache = {}

    def ensure_storage_dir(self):
        os.makedirs(self.storage_dir, exist_ok=True)

    def create_article(self, title: str, content: str, content_type: str = "text") -> Dict[str, Any]:
        embedding = self.get_embedding(content)
        article = {
            "title": title,
            "content": content,
            "content_type": content_type,
            "embedding": embedding,
            "versions": [{
                "content": content,
                "timestamp": datetime.now().isoformat()
            }]
        }
        self._save_article(article)
        logger.info(f"Created new article: {title}")
        return article

    def search_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.get_embedding(query)
        articles = self.load_all_articles()
        
        similarities = [
            self.cosine_similarity(query_embedding, article['embedding'])
            for article in articles
        ]
        
        sorted_articles = [
            article for _, article in sorted(
                zip(similarities, articles),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        
        return sorted_articles[:top_k]

    def update_article(self, title: str, content: str, content_type: str = "text") -> Dict[str, Any]:
        article = self.get_article_content(title)
        if not article:
            logger.error(f"Article not found: {title}")
            return {}
        
        embedding = self.get_embedding(content)
        article["content"] = content
        article["content_type"] = content_type
        article["embedding"] = embedding
        article["versions"].append({
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_article(article)
        logger.info(f"Updated article: {title}")
        return article

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        file_path = os.path.join(self.storage_dir, f"{title}.json")
        if not os.path.exists(file_path):
            logger.error(f"Article not found: {title}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                article = json.load(f)
            return article
        except json.JSONDecodeError:
            logger.error(f"Error decoding article: {title}")
            return None

    def get_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        response = self.client.embeddings.create(input=text, model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        self.embedding_cache[text] = embedding
        return embedding

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def load_all_articles(self) -> List[Dict[str, Any]]:
        articles = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.storage_dir, filename), 'r') as f:
                        articles.append(json.load(f))
                except json.JSONDecodeError:
                    logger.error(f"Error decoding article: {filename}")
        return articles

    def generate_summary(self, title: str, max_tokens: int = 150) -> str:
        article = self.get_article_content(title)
        if not article:
            return ""
        
        prompt = f"Please summarize the following article:\n\nTitle: {article['title']}\n\nContent: {article['content']}\n\nSummary:"
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def bulk_import_articles(self, articles: List[Dict[str, str]]) -> None:
        for article in articles:
            self.create_article(article['title'], article['content'], article.get('content_type', 'text'))
        logger.info(f"Bulk imported {len(articles)} articles")

    def delete_article(self, title: str) -> bool:
        file_path = os.path.join(self.storage_dir, f"{title}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted article: {title}")
            return True
        else:
            logger.error(f"Article not found: {title}")
            return False

    def list_articles(self) -> List[str]:
        return [filename[:-5] for filename in os.listdir(self.storage_dir) if filename.endswith('.json')]

    def _save_article(self, article: Dict[str, Any]) -> None:
        file_path = os.path.join(self.storage_dir, f"{article['title']}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(article, f)
        except IOError:
            logger.error(f"Error saving article: {article['title']}")

# Example usage and testing
if __name__ == "__main__":
    kb_service = KBService()

    # Test creating an article
    new_article = kb_service.create_article(
        "AI Ethics",
        "AI ethics is the set of moral principles and techniques to guide the development and use of artificial intelligence technologies."
    )
    print("New article created:", new_article)

    # Test searching for articles
    search_results = kb_service.search_articles("ethical considerations in AI")
    print("Search results:", search_results)

    # Test updating an article
    updated_article = kb_service.update_article(
        "AI Ethics", 
        "AI ethics encompasses the moral principles and guidelines that govern the development, deployment, and use of artificial intelligence systems to ensure they benefit humanity and do not cause harm."
    )
    print("Updated article:", updated_article)

    # Test getting article content
    article_content = kb_service.get_article_content("AI Ethics")
    print("Article content:", article_content)

    # Test generating a summary
    summary = kb_service.generate_summary("AI Ethics")
    print("Article summary:", summary)

    # Test bulk importing articles
    kb_service.bulk_import_articles([
        {"title": "Machine Learning", "content": "Machine learning is a subset of AI focused on creating systems that learn from data."},
        {"title": "Natural Language Processing", "content": "NLP is a field of AI that focuses on the interaction between computers and humans using natural language."}
    ])

    # Test listing articles
    print("All articles:", kb_service.list_articles())

    # Test deleting an article
    kb_service.delete_article("Machine Learning")
    print("Articles after deletion:", kb_service.list_articles())