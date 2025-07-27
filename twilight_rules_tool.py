"""
Twilight Imperium Rules Search Tool
Step 4: LangChain-compatible tool for searching game rules

This module creates a search tool that can be used by LangChain agents
to find relevant rule chunks from the Twilight Imperium PDFs.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.schema import Document

# Load environment variables
load_dotenv()


class TwilightRulesSearcher:
    """
    A class to handle searching Twilight Imperium rules using FAISS vector store
    """
    
    def __init__(self, vector_store_path: str = "processed_rules/vector_store"):
        """
        Initialize the rules searcher
        
        Args:
            vector_store_path: Path to the saved FAISS vector store
        """
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model = None
        self.vector_store = None
        self.config = None
        
        # Load configuration
        self._load_config()
        
        # Initialize embedding model
        self._initialize_embeddings()
        
        # Load vector store
        self._load_vector_store()
    
    def _load_config(self):
        """Load the embedding configuration from Step 3"""
        config_path = Path("processed_rules/embedding_config.json")
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "Please run the embedding_generator.ipynb notebook first!"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"✅ Loaded config: {self.config['total_vectors']} vectors, "
              f"{self.config['embedding_dimension']}D embeddings")
    
    def _initialize_embeddings(self):
        """Initialize the OpenAI embeddings model"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or create a .env file with your key."
            )
        
        self.embedding_model = OpenAIEmbeddings(
            model=self.config['model_name'],
            openai_api_key=api_key
        )
        
        print(f"✅ Initialized embedding model: {self.config['model_name']}")
    
    def _load_vector_store(self):
        """Load the FAISS vector store from disk"""
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Please run the embedding_generator.ipynb notebook first!"
            )
        
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Loaded vector store with {self.vector_store.index.ntotal} vectors")
    
    def search_rules(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant rule chunks based on a query
        
        Args:
            query: The search query (e.g., "How do I move ships?")
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing rule chunks and metadata
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not loaded. Please initialize first.")
        
        # Perform similarity search
        similar_docs = self.vector_store.similarity_search(query=query, k=k)
        
        # Format results
        results = []
        for i, doc in enumerate(similar_docs):
            result = {
                'rank': i + 1,
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'doc_type': doc.metadata.get('doc_type', 'unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'section': doc.metadata.get('section', None),
                'char_count': doc.metadata.get('char_count', len(doc.page_content)),
                'word_count': doc.metadata.get('word_count', len(doc.page_content.split()))
            }
            results.append(result)
        
        return results
    
    def search_rules_formatted(self, query: str, k: int = 3) -> str:
        """
        Search for rules and return formatted text for LLM consumption
        
        Args:
            query: The search query
            k: Number of results to include
            
        Returns:
            Formatted string with search results
        """
        results = self.search_rules(query, k)
        
        if not results:
            return f"No relevant rules found for query: '{query}'"
        
        formatted_output = f"🔍 **Search Results for: '{query}'**\n\n"
        
        for result in results:
            formatted_output += f"**Result {result['rank']}** "
            formatted_output += f"({result['doc_type']} - {result['source']})\n"
            
            if result['section']:
                formatted_output += f"📍 Section: {result['section']}\n"
            
            formatted_output += f"📝 Content:\n{result['content']}\n"
            formatted_output += f"📊 {result['word_count']} words | Chunk: {result['chunk_id']}\n\n"
            formatted_output += "---\n\n"
        
        return formatted_output


def create_twilight_rules_tool() -> Tool:
    """
    Create a LangChain Tool for searching Twilight Imperium rules
    
    Returns:
        LangChain Tool object that can be used by agents
    """
    # Initialize the searcher
    searcher = TwilightRulesSearcher()
    
    def search_twilight_rules(query: str) -> str:
        """
        Search function that will be called by the LangChain agent
        
        Args:
            query: User's question about Twilight Imperium rules
            
        Returns:
            Formatted search results
        """
        try:
            return searcher.search_rules_formatted(query, k=3)
        except Exception as e:
            return f"Error searching rules: {str(e)}"
    
    # Create the LangChain Tool
    tool = Tool(
        name="SearchTwilightRules",
        description=(
            "Search the official Twilight Imperium Fourth Edition rules. "
            "Use this tool when users ask about game rules, mechanics, combat, "
            "movement, strategy cards, victory conditions, or any gameplay questions. "
            "Input should be a clear question about the game rules."
        ),
        func=search_twilight_rules
    )
    
    return tool


def test_tool():
    """Test the rules search tool with sample queries"""
    print("🧪 Testing Twilight Imperium Rules Search Tool")
    print("=" * 60)
    
    try:
        # Create the tool
        tool = create_twilight_rules_tool()
        
        # Test queries
        test_queries = [
            "Can I place diplomacy first abilty on a system that other players own?",
            "What are strategy cards?",
            "How do I win the game?",
            "What happens during ground combat?",
            "How do I activate a system?"
        ]
        
        for i, query in enumerate(test_queries[:2], 1):  # Test first 2
            print(f"\n🔸 Test {i}: '{query}'")
            print("-" * 40)
            
            result = tool.func(query)
            print(result)
            
        print("✅ Tool testing complete!")
        
    except Exception as e:
        print(f"❌ Error testing tool: {e}")
        return False
    
    return True


if __name__ == "__main__":
    """
    Run this script directly to test the tool
    """
    print("🚀 Twilight Imperium Rules Search Tool - Step 4")
    print("=" * 60)
    
    # Test the tool
    success = test_tool()
    
    if success:
        print("\n🎉 Tool is ready for Step 5: LangChain Agent Integration!")
        print("\n📋 Usage in LangChain:")
        print("```python")
        print("from twilight_rules_tool import create_twilight_rules_tool")
        print("")
        print("# Create the tool")
        print("rules_tool = create_twilight_rules_tool()")
        print("")
        print("# Use in an agent")
        print("tools = [rules_tool]")
        print("# ... agent setup ...")
        print("```")
    else:
        print("\n❌ Tool setup failed. Please check your vector store and API key.") 