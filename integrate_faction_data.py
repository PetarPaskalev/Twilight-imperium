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
    Supports organized folder structure with multiple faction sets
    """
    
    def __init__(self, faction_set: str = "all"):
        """
        Initialize the faction data integrator
        
        Args:
            faction_set: Which faction data to integrate ("base", "pok", "codex", "all")
        """
        self.faction_set = faction_set
        self.processed_rules_dir = Path("processed_rules")
        self.vector_store_path = self.processed_rules_dir / "vector_store"
        
        # Set up faction data paths
        self.faction_data_dir = self.processed_rules_dir / "faction_data"
        self.faction_data_files = self._get_faction_data_files()
        
        # Initialize embeddings model
        self.embeddings_model = None
        self.vector_store = None
        self.faction_data = []
        
        print("🔥 Enhanced Faction Data Integrator initialized")
        print(f"📦 Target faction set: {faction_set}")
        print(f"📁 Data source: {self.faction_data_dir}")
    
    def _get_faction_data_files(self) -> List[Path]:
        """Get the faction data files to process based on faction_set"""
        files = []
        
        if self.faction_set == "base":
            base_file = self.faction_data_dir / "base_game" / "base_game_factions.json"
            if base_file.exists():
                files.append(base_file)
        
        elif self.faction_set == "pok":
            pok_file = self.faction_data_dir / "prophecy_of_kings" / "prophecy_of_kings_factions.json"
            if pok_file.exists():
                files.append(pok_file)
        
        elif self.faction_set == "codex":
            codex_file = self.faction_data_dir / "codex" / "codex_factions.json"
            if codex_file.exists():
                files.append(codex_file)
        
        elif self.faction_set == "all":
            # Try organized files first
            combined_file = self.faction_data_dir / "combined" / "all_factions_combined.json"
            if combined_file.exists():
                files.append(combined_file)
            else:
                # Fall back to individual files
                for sub_set in ["base", "pok", "codex"]:
                    sub_files = self._get_faction_data_files_for_set(sub_set)
                    files.extend(sub_files)
            
            # Also check for legacy file
            legacy_file = self.processed_rules_dir / "faction_data_improved.json"
            if legacy_file.exists() and not files:
                files.append(legacy_file)
        
        else:
            raise ValueError(f"Invalid faction_set: {self.faction_set}. Use 'base', 'pok', 'codex', or 'all'")
        
        return files
    
    def _get_faction_data_files_for_set(self, faction_set: str) -> List[Path]:
        """Helper to get files for a specific faction set"""
        file_map = {
            "base": self.faction_data_dir / "base_game" / "base_game_factions.json",
            "pok": self.faction_data_dir / "prophecy_of_kings" / "prophecy_of_kings_factions.json",
            "codex": self.faction_data_dir / "codex" / "codex_factions.json"
        }
        
        file_path = file_map.get(faction_set)
        if file_path and file_path.exists():
            return [file_path]
        return []
    
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
        """Load the scraped faction data from organized files"""
        if not self.faction_data_files:
            print(f"❌ No faction data files found for '{self.faction_set}' faction set")
            print(f"📁 Expected locations:")
            
            if self.faction_set == "base":
                print(f"  {self.faction_data_dir / 'base_game' / 'base_game_factions.json'}")
            elif self.faction_set == "pok":
                print(f"  {self.faction_data_dir / 'prophecy_of_kings' / 'prophecy_of_kings_factions.json'}")
            elif self.faction_set == "codex":
                print(f"  {self.faction_data_dir / 'codex' / 'codex_factions.json'}")
            elif self.faction_set == "all":
                print(f"  {self.faction_data_dir / 'combined' / 'all_factions_combined.json'}")
                print(f"  OR individual files from base_game/, prophecy_of_kings/, codex/")
            
            raise FileNotFoundError("Please run faction_scraper_improved.py first!")
        
        print(f"📂 Loading faction data from {len(self.faction_data_files)} file(s):")
        
        self.faction_data = []
        for file_path in self.faction_data_files:
            print(f"  📄 Loading: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.faction_data.extend(data)
                print(f"    ✅ Loaded {len(data)} factions")
        
        successful_factions = [f for f in self.faction_data if f.get("scraped_successfully", False)]
        
        print(f"\n📊 Faction Data Summary:")
        print(f"  Total factions loaded: {len(self.faction_data)}")
        print(f"  Successfully scraped: {len(successful_factions)}")
        print(f"  Failed scrapes: {len(self.faction_data) - len(successful_factions)}")
        
        # Show breakdown by faction set
        faction_sets = {}
        for faction in self.faction_data:
            faction_name = faction.get('name', 'Unknown')
            if faction_name in ["The Arborec", "The Barony of Letnev", "The Clan of Saar", "The Embers of Muaat", "The Emirates of Hacan", "The Federation of Sol", "The Ghosts of Creuss", "The L1Z1X Mindnet", "The Mentak Coalition", "The Naalu Collective", "The Nekro Virus", "Sardakk N'orr", "The Universities of Jol-Nar", "The Winnu", "The Xxcha Kingdom", "The Yin Brotherhood", "The Yssaril Tribes"]:
                faction_sets['Base Game'] = faction_sets.get('Base Game', 0) + 1
            elif faction_name in ["The Argent Flight", "The Empyrean", "The Mahact Gene-Sorcerers", "The Naaz-Rokha Alliance", "The Nomad", "The Titans of Ul", "The Vuil'Raith Cabal"]:
                faction_sets['Prophecy of Kings'] = faction_sets.get('Prophecy of Kings', 0) + 1
            elif faction_name in ["The Council Keleres"]:
                faction_sets['Codex'] = faction_sets.get('Codex', 0) + 1
        
        print(f"\n  📦 Factions by set:")
        for set_name, count in faction_sets.items():
            print(f"    {set_name}: {count} factions")
        
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
            config['last_faction_update'] = str(Path.cwd() / self.faction_data_files[0]) # Use the first file as a proxy for the set
            
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
    Run the faction data integration with support for different faction sets
    """
    import sys
    
    print("🚀 Twilight Imperium Faction Data Integration")
    print("=" * 60)
    
    # Determine faction set from command line argument
    faction_set = "all"  # default
    if len(sys.argv) > 1:
        faction_set = sys.argv[1].lower()
        if faction_set not in ["base", "pok", "codex", "all"]:
            print(f"❌ Invalid faction set: {faction_set}")
            print("Valid options: base, pok, codex, all")
            print("\nUsage examples:")
            print("  python integrate_faction_data.py all     # Integrate all faction data")
            print("  python integrate_faction_data.py base    # Integrate only base game factions")
            print("  python integrate_faction_data.py pok     # Integrate only PoK factions")
            print("  python integrate_faction_data.py codex   # Integrate only Codex factions")
            sys.exit(1)
    
    print(f"🎯 Integrating {faction_set} faction data into vector store...")
    
    try:
        integrator = FactionDataIntegrator(faction_set=faction_set)
        integrator.run_integration()
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        print("\n💡 Make sure you have:")
        print("  1. Run faction_scraper_improved.py first")
        print("  2. Set your OPENAI_API_KEY environment variable")
        print("  3. Created the initial vector store (embedding_generator.ipynb)")
        sys.exit(1) 