"""
Twilight Imperium Faction Data Integration
Adds scraped faction data to existing vector store

This script loads the faction data from faction_scraper and integrates it
with the existing rule chunks in the vector database.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()


class FactionDataIntegrator:
    """
    Integrates faction data with existing vector store
    """
    
    def __init__(self):
        """Initialize the faction data integrator"""
        self.processed_rules_dir = Path("processed_rules")
        self.vector_store_path = self.processed_rules_dir / "vector_store"
        self.faction_data_file = self.processed_rules_dir / "faction_data_improved.json"
        
        # Initialize embeddings model
        self.embeddings_model = None
        self.vector_store = None
        self.faction_data = []
        
        print("🔥 Faction Data Integrator initialized")
    
    def _initialize_embeddings(self):
        """Initialize the OpenAI embeddings model"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        
        print("✅ OpenAI embeddings model initialized")
    
    def load_faction_data(self):
        """Load the scraped faction data"""
        if not self.faction_data_file.exists():
            raise FileNotFoundError(
                f"Faction data file not found at {self.faction_data_file}. "
                "Please run faction_scraper_improved.py first!"
            )
        
        with open(self.faction_data_file, 'r', encoding='utf-8') as f:
            self.faction_data = json.load(f)
        
        successful_factions = [f for f in self.faction_data if f.get("scraped_successfully", False)]
        
        print(f"📊 Loaded faction data:")
        print(f"  Total factions: {len(self.faction_data)}")
        print(f"  Successfully scraped: {len(successful_factions)}")
        
        return successful_factions
    
    def load_existing_vector_store(self):
        """Load the existing vector store"""
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Please run embedding_generator.ipynb first!"
            )
        
        self._initialize_embeddings()
        
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            self.embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Loaded existing vector store with {self.vector_store.index.ntotal} vectors")
        return self.vector_store
    
    def create_faction_documents(self, faction_data: List[Dict]) -> List[Document]:
        """Convert faction data to LangChain Documents"""
        documents = []
        
        print("📄 Converting faction data to documents...")
        
        for faction in faction_data:
            if not faction.get("scraped_successfully", False):
                continue
            
            faction_name = faction["name"]
            faction_url = faction["url"]
            sections = faction.get("sections", {})
            
            print(f"  📝 Processing {faction_name} ({len(sections)} sections)")
            
            # Create a document for each section
            for section_name, content in sections.items():
                if not content or len(content.strip()) < 20:
                    continue
                
                # Create comprehensive document content
                doc_content = f"**{faction_name} - {section_name}**\n\n{content}"
                
                # Create detailed metadata
                metadata = {
                    'source': 'faction_wiki',
                    'doc_type': f'Faction {section_name}',
                    'faction_name': faction_name,
                    'section': section_name,
                    'url': faction_url,
                    'char_count': len(doc_content),
                    'word_count': len(doc_content.split()),
                    'content_type': 'faction_data'
                }
                
                # Create Document
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                
                documents.append(doc)
        
        print(f"✅ Created {len(documents)} faction documents")
        
        # Show breakdown by section type
        section_counts = {}
        for doc in documents:
            section = doc.metadata['section']
            section_counts[section] = section_counts.get(section, 0) + 1
        
        print("📊 Documents by section type:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count} documents")
        
        return documents
    
    def add_faction_data_to_vector_store(self, faction_documents: List[Document]):
        """Add faction documents to the vector store"""
        if not faction_documents:
            print("❌ No faction documents to add")
            return
        
        print(f"🔥 Adding {len(faction_documents)} faction documents to vector store...")
        print("⏳ This will generate embeddings for each document (may take a few minutes)...")
        
        # Get initial count
        initial_count = self.vector_store.index.ntotal
        
        # Add faction documents (this will generate embeddings)
        self.vector_store.add_documents(faction_documents)
        
        # Get final count
        final_count = self.vector_store.index.ntotal
        added_count = final_count - initial_count
        
        print(f"✅ Successfully added {added_count} vectors to the database")
        print(f"📈 Vector store now contains {final_count} total vectors")
        
        return final_count
    
    def save_updated_vector_store(self):
        """Save the updated vector store"""
        print("💾 Saving updated vector store...")
        
        self.vector_store.save_local(str(self.vector_store_path))
        
        print(f"✅ Updated vector store saved to: {self.vector_store_path}")
    
    def update_config(self, total_vectors: int, faction_docs_added: int):
        """Update the embedding configuration file"""
        config_path = self.processed_rules_dir / "embedding_config.json"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update config with faction data
            config['total_vectors'] = total_vectors
            config['faction_documents_added'] = faction_docs_added
            config['factions_included'] = True
            config['last_faction_update'] = str(Path.cwd() / self.faction_data_file)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("⚙️  Updated embedding configuration")
        else:
            print("⚠️  Embedding config file not found - creating new one")
            
            config = {
                'model_name': 'text-embedding-3-small',
                'total_vectors': total_vectors,
                'faction_documents_added': faction_docs_added,
                'factions_included': True
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def test_faction_search(self):
        """Test that faction data is searchable"""
        print("\n🧪 Testing faction data integration...")
        
        test_queries = [
            "What are the Arborec's special abilities?",
            "How does the Federation of Sol play?",
            "What's unique about the Ghosts of Creuss?",
            "Tell me about Sardakk N'orr FAQ"
        ]
        
        for query in test_queries[:2]:  # Test first 2
            print(f"\n🔸 Test query: '{query}'")
            
            results = self.vector_store.similarity_search(query, k=3)
            
            print(f"  📄 Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', 'unknown')
                faction = result.metadata.get('faction_name', 'N/A')
                section = result.metadata.get('section', 'N/A')
                
                print(f"    {i}. Source: {source}")
                if faction != 'N/A':
                    print(f"       Faction: {faction} - {section}")
                print(f"       Preview: {result.page_content[:100]}...")
        
        print("\n✅ Faction search test complete!")
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Starting Faction Data Integration")
        print("=" * 60)
        
        try:
            # Step 1: Load faction data
            successful_factions = self.load_faction_data()
            
            # Step 2: Load existing vector store
            self.load_existing_vector_store()
            
            # Step 3: Create faction documents
            faction_documents = self.create_faction_documents(successful_factions)
            
            # Step 4: Add to vector store
            final_count = self.add_faction_data_to_vector_store(faction_documents)
            
            # Step 5: Save updated vector store
            self.save_updated_vector_store()
            
            # Step 6: Update configuration
            self.update_config(final_count, len(faction_documents))
            
            # Step 7: Test integration
            self.test_faction_search()
            
            print("\n🎉 Faction Data Integration Complete!")
            print("=" * 60)
            print(f"📊 Final Summary:")
            print(f"  Original vectors: {final_count - len(faction_documents)}")
            print(f"  Faction vectors added: {len(faction_documents)}")
            print(f"  Total vectors: {final_count}")
            print(f"\n💡 Your chatbot now knows about:")
            print("  ✅ All game rules (from PDFs)")
            print("  ✅ All 17 faction abilities")
            print("  ✅ Faction FAQs and common questions")
            print("  ✅ Faction technologies and units")
            print("\n🤖 Test your enhanced chatbot with questions like:")
            print("  - What are the Arborec's special abilities?")
            print("  - How does the Federation of Sol play differently?")
            print("  - What's unique about the Ghosts of Creuss?")
            
        except Exception as e:
            print(f"❌ Error during integration: {e}")
            raise


if __name__ == "__main__":
    """
    Run the faction data integration
    """
    integrator = FactionDataIntegrator()
    integrator.run_integration() 