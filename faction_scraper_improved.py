"""
Improved Twilight Imperium Faction Data Scraper
Enhanced version with support for all faction sets and organized folder structure

This version supports:
- Base game factions (17)
- Prophecy of Kings expansion factions (7) 
- Codex factions (1)
- All factions combined (25)

Features better debugging and more robust section extraction
for the complex Fandom wiki structure.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()


class ImprovedTwilightFactionScraper:
    """
    Improved scraper for Twilight Imperium faction data from Fandom wiki
    Supports all faction sets: base game, Prophecy of Kings, and Codex
    """
    
    def __init__(self, debug_mode: bool = True, faction_set: str = "all"):
        """
        Initialize the improved faction scraper
        
        Args:
            debug_mode: Enable debug output and limit to 2 factions for testing
            faction_set: Which factions to scrape ("base", "pok", "codex", "all")
        """
        self.debug_mode = debug_mode
        self.faction_set = faction_set
        self.base_url = "https://twilight-imperium.fandom.com/wiki/"
        
        # Base game factions (original 17)
        self.base_factions = [
            "The Arborec",
            "The Barony of Letnev", 
            "The Clan of Saar",
            "The Embers of Muaat",
            "The Emirates of Hacan",
            "The Federation of Sol",
            "The Ghosts of Creuss",
            "The L1Z1X Mindnet",
            "The Mentak Coalition",
            "The Naalu Collective",
            "The Nekro Virus",
            "Sardakk N'orr",
            "The Universities of Jol-Nar",
            "The Winnu",
            "The Xxcha Kingdom",
            "The Yin Brotherhood",
            "The Yssaril Tribes"
        ]
        
        # Prophecy of Kings expansion factions (7)
        self.pok_factions = [
            "The Argent Flight",
            "The Empyrean",
            "The Mahact Gene-Sorcerers",
            "The Naaz-Rokha Alliance",
            "The Nomad",
            "The Titans of Ul",
            "The Vuil'Raith Cabal"
        ]
        
        # Codex factions (1)
        self.codex_factions = [
            "The Council Keleres"
        ]
        
        # Select which factions to scrape
        if faction_set == "base":
            self.factions = self.base_factions
            self.set_description = "Base Game"
        elif faction_set == "pok":
            self.factions = self.pok_factions
            self.set_description = "Prophecy of Kings"
        elif faction_set == "codex":
            self.factions = self.codex_factions
            self.set_description = "Codex"
        elif faction_set == "all":
            self.factions = self.base_factions + self.pok_factions + self.codex_factions
            self.set_description = "All Factions"
        else:
            raise ValueError(f"Invalid faction_set: {faction_set}. Use 'base', 'pok', 'codex', or 'all'")
        
        # More flexible section patterns to catch variations
        self.section_patterns = {
            "Faction Abilities": ["Faction Abilities", "Abilities", "Special Abilities"],
            "FAQ": ["FAQ", "Frequently Asked Questions", "Q&A"],
            "Faction Technologies": ["Faction Technologies", "Technologies", "Tech", "Faction Technology"],
            "Faction Specific Units": ["Faction Specific Units", "Unique Units", "Special Units"],
            "Leaders": ["Leaders", "Faction Leaders"],
            "Flagship": ["Flagship", "Faction Flagship"],
            "Mech": ["Mech", "Faction Mech"],
            "Faction Promissory Note": ["Faction Promissory Note", "Promissory Note", "Promissory"]
        }
        
        # Create organized folder structure
        self.processed_rules_dir = Path("processed_rules")
        self.faction_data_dir = self.processed_rules_dir / "faction_data"
        self.faction_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for different faction sets
        self.base_dir = self.faction_data_dir / "base_game"
        self.pok_dir = self.faction_data_dir / "prophecy_of_kings"
        self.codex_dir = self.faction_data_dir / "codex"
        self.combined_dir = self.faction_data_dir / "combined"
        
        for dir_path in [self.base_dir, self.pok_dir, self.codex_dir, self.combined_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.faction_data = []
        
        # Better headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        print(f"✅ Improved faction scraper initialized")
        print(f"📦 Faction Set: {self.set_description} ({len(self.factions)} factions)")
        print(f"📁 Data will be saved to: {self.faction_data_dir}")
        if debug_mode:
            print("🐛 Debug mode enabled - will show detailed extraction info")
    
    def _get_output_filename(self) -> str:
        """Get the appropriate output filename based on faction set"""
        filename_map = {
            "base": "base_game_factions.json",
            "pok": "prophecy_of_kings_factions.json", 
            "codex": "codex_factions.json",
            "all": "all_factions_combined.json"
        }
        return filename_map.get(self.faction_set, "faction_data.json")
    
    def _get_output_directory(self) -> Path:
        """Get the appropriate output directory based on faction set"""
        dir_map = {
            "base": self.base_dir,
            "pok": self.pok_dir,
            "codex": self.codex_dir,
            "all": self.combined_dir
        }
        return dir_map.get(self.faction_set, self.faction_data_dir)

    def _generate_faction_url(self, faction_name: str) -> str:
        """Generate the correct Fandom wiki URL for a faction"""
        url_name = faction_name.replace(" ", "_")
        return f"{self.base_url}{quote(url_name)}"
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text more thoroughly"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common wiki markup and navigation elements
        unwanted_phrases = [
            "[edit]", "Edit", "[]", "Category:", "File:", "Main Page",
            "Community portal", "Random page", "Recent changes",
            "What links here", "Related changes", "Special pages"
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")
        
        return text.strip()
    
    def _find_all_headers(self, soup: BeautifulSoup) -> List[tuple]:
        """Find all headers and their text for debugging"""
        headers = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for header in soup.find_all(tag):
                header_text = header.get_text().strip()
                if header_text:
                    headers.append((tag, header_text))
        return headers
    
    def _extract_section_content_improved(self, soup: BeautifulSoup, section_key: str) -> tuple:
        """Improved section extraction with multiple patterns and better parsing"""
        patterns = self.section_patterns.get(section_key, [section_key])
        
        for pattern in patterns:
            if self.debug_mode:
                print(f"    🔍 Looking for pattern: '{pattern}'")
            
            # Try different approaches to find the section
            section_content = self._try_multiple_extraction_methods(soup, pattern)
            
            if section_content:
                if self.debug_mode:
                    print(f"    ✅ Found content using pattern: '{pattern}'")
                return section_content, pattern
        
        return "", None
    
    def _try_multiple_extraction_methods(self, soup: BeautifulSoup, pattern: str) -> str:
        """Try multiple methods to extract section content"""
        
        # Method 1: Look for exact header match
        content = self._extract_by_header_match(soup, pattern)
        if content:
            return content
        
        # Method 2: Look for partial header match (case insensitive)
        content = self._extract_by_partial_match(soup, pattern)
        if content:
            return content
        
        # Method 3: Look in specific div containers (Fandom specific)
        content = self._extract_from_content_divs(soup, pattern)
        if content:
            return content
        
        return ""
    
    def _extract_by_header_match(self, soup: BeautifulSoup, pattern: str) -> str:
        """Extract content by looking for exact header matches"""
        for header_tag in ['h2', 'h3', 'h4', 'h5']:
            headers = soup.find_all(header_tag)
            for header in headers:
                header_text = header.get_text().strip()
                if header_text == pattern:
                    return self._get_content_after_header(header)
        return ""
    
    def _extract_by_partial_match(self, soup: BeautifulSoup, pattern: str) -> str:
        """Extract content by looking for partial matches"""
        pattern_lower = pattern.lower()
        
        for header_tag in ['h2', 'h3', 'h4', 'h5']:
            headers = soup.find_all(header_tag)
            for header in headers:
                header_text = header.get_text().strip().lower()
                if pattern_lower in header_text or header_text in pattern_lower:
                    return self._get_content_after_header(header)
        return ""
    
    def _extract_from_content_divs(self, soup: BeautifulSoup, pattern: str) -> str:
        """Extract content from Fandom-specific div structures"""
        # Look for content in common Fandom div classes
        content_divs = soup.find_all('div', class_=['mw-content-text', 'WikiaArticle', 'page-content'])
        
        for div in content_divs:
            # Look for the pattern in text content
            if pattern.lower() in div.get_text().lower():
                # Try to extract relevant paragraph or list content
                paragraphs = div.find_all(['p', 'ul', 'ol', 'div'])
                relevant_content = []
                
                for p in paragraphs:
                    text = p.get_text().strip()
                    if pattern.lower() in text.lower() or len(text) > 50:  # Include substantial content
                        relevant_content.append(text)
                
                if relevant_content:
                    return "\n".join(relevant_content)
        
        return ""
    
    def _get_content_after_header(self, header) -> str:
        """Extract content that comes after a header"""
        content_parts = []
        current = header.next_sibling
        
        while current:
            # Stop if we hit another header of the same or higher level
            if current.name in ['h1', 'h2', 'h3', 'h4'] and current != header:
                break
            
            if current.name:
                # Handle different content types
                if current.name in ['p', 'div']:
                    text = current.get_text().strip()
                    if text and len(text) > 10:  # Only include substantial content
                        content_parts.append(text)
                
                elif current.name in ['ul', 'ol']:
                    # Handle lists with bullet points
                    for li in current.find_all('li'):
                        li_text = li.get_text().strip()
                        if li_text:
                            content_parts.append(f"• {li_text}")
                
                elif current.name == 'table':
                    # Extract table content if relevant
                    table_text = current.get_text().strip()
                    if table_text and len(table_text) > 20:
                        content_parts.append(table_text)
            
            current = current.next_sibling
        
        return "\n".join(content_parts)
    
    def scrape_faction(self, faction_name: str) -> Dict[str, Any]:
        """Scrape data for a single faction with improved extraction"""
        print(f"🔍 Scraping {faction_name}...")
        
        url = self._generate_faction_url(faction_name)
        faction_data = {
            "name": faction_name,
            "url": url,
            "sections": {},
            "scraped_successfully": False,
            "debug_info": {}
        }
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Show all headers found
            if self.debug_mode:
                all_headers = self._find_all_headers(soup)
                print(f"  🐛 Found {len(all_headers)} headers on page:")
                for tag, text in all_headers[:10]:  # Show first 10
                    print(f"    {tag}: {text}")
                if len(all_headers) > 10:
                    print(f"    ... and {len(all_headers) - 10} more")
            
            # Extract each section with improved method
            sections_found = 0
            for section_key in self.section_patterns.keys():
                print(f"  🔍 Looking for: {section_key}")
                
                content, found_pattern = self._extract_section_content_improved(soup, section_key)
                
                if content:
                    cleaned_content = self._clean_text(content)
                    if len(cleaned_content) > 20:  # Only save substantial content
                        faction_data["sections"][section_key] = cleaned_content
                        sections_found += 1
                        print(f"  ✅ Found: {section_key} (using pattern: {found_pattern})")
                        print(f"      Preview: {cleaned_content[:100]}...")
                    else:
                        print(f"  ⚠️  Found {section_key} but content too short")
                else:
                    print(f"  ❌ Missing: {section_key}")
            
            if sections_found > 0:
                faction_data["scraped_successfully"] = True
                print(f"  📊 Successfully extracted {sections_found}/{len(self.section_patterns)} sections")
            else:
                print(f"  ❌ No sections found for {faction_name}")
                
                # If debugging, save the page content for manual inspection
                if self.debug_mode:
                    debug_file = self.processed_rules_dir / f"debug_{faction_name.replace(' ', '_')}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(str(soup.prettify()))
                    print(f"  🐛 Saved debug HTML to: {debug_file}")
            
            # Respectful delay
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ Error scraping {faction_name}: {e}")
        
        return faction_data
    
    def scrape_all_factions(self) -> List[Dict[str, Any]]:
        """Scrape data for all factions"""
        print(f"🚀 Starting to scrape {len(self.factions)} factions...")
        print("=" * 60)
        
        all_faction_data = []
        successful_scrapes = 0
        
        # Test with just first faction if in debug mode
        test_factions = self.factions[:2] if self.debug_mode else self.factions
        
        for i, faction in enumerate(test_factions, 1):
            print(f"\n[{i}/{len(test_factions)}] ", end="")
            faction_data = self.scrape_faction(faction)
            all_faction_data.append(faction_data)
            
            if faction_data["scraped_successfully"]:
                successful_scrapes += 1
        
        print(f"\n" + "=" * 60)
        print(f"📊 Scraping Summary:")
        print(f"  Total attempted: {len(test_factions)}")
        print(f"  Successfully scraped: {successful_scrapes}")
        print(f"  Failed: {len(test_factions) - successful_scrapes}")
        
        if self.debug_mode:
            print("\n🐛 Debug mode was enabled - only tested first 2 factions")
            print("   Run with debug_mode=False to scrape all factions")
        
        self.faction_data = all_faction_data
        return all_faction_data
    
    def save_faction_data(self, custom_filename: str = None):
        """Save scraped faction data to organized folders"""
        if not self.faction_data:
            print("❌ No faction data to save")
            return
        
        output_dir = self._get_output_directory()
        output_filename = custom_filename or self._get_output_filename()
        
        output_file = output_dir / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.faction_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 {self.set_description} faction data saved to: {output_file}")
        
        # Also save a copy to the main processed_rules directory for backward compatibility
        if self.faction_set == "all":
            legacy_file = self.processed_rules_dir / "faction_data_improved.json"
            with open(legacy_file, 'w', encoding='utf-8') as f:
                json.dump(self.faction_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Legacy copy saved to: {legacy_file}")
    
    def create_summary_report(self):
        """Create a summary report of all scraped factions"""
        if not self.faction_data:
            return
        
        successful = [f for f in self.faction_data if f.get("scraped_successfully", False)]
        failed = [f for f in self.faction_data if not f.get("scraped_successfully", False)]
        
        # Count sections by type
        section_stats = {}
        for faction in successful:
            for section_name in faction.get("sections", {}).keys():
                section_stats[section_name] = section_stats.get(section_name, 0) + 1
        
        report = {
            "faction_set": self.faction_set,
            "set_description": self.set_description,
            "total_factions": len(self.faction_data),
            "successful_scrapes": len(successful),
            "failed_scrapes": len(failed),
            "success_rate": f"{(len(successful)/len(self.faction_data)*100):.1f}%" if self.faction_data else "0%",
            "section_statistics": section_stats,
            "successful_factions": [f["name"] for f in successful],
            "failed_factions": [f["name"] for f in failed]
        }
        
        # Save report
        output_dir = self._get_output_directory()
        report_file = output_dir / f"{self.faction_set}_scraping_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Scraping report saved to: {report_file}")
        return report


def test_single_faction(faction_name: str = "The Arborec", faction_set: str = "base"):
    """Test scraping a single faction for debugging"""
    print(f"🧪 Testing scraping for: {faction_name}")
    print("=" * 50)
    
    scraper = ImprovedTwilightFactionScraper(debug_mode=True, faction_set=faction_set)
    faction_data = scraper.scrape_faction(faction_name)
    
    print(f"\n📋 Results for {faction_name}:")
    print(f"  Success: {faction_data['scraped_successfully']}")
    print(f"  Sections found: {len(faction_data['sections'])}")
    
    for section, content in faction_data['sections'].items():
        print(f"\n  📄 {section}:")
        print(f"    Length: {len(content)} characters")
        print(f"    Preview: {content[:200]}...")

def test_faction_set(faction_set: str):
    """Test scraping for a specific faction set"""
    print(f"🧪 Testing {faction_set} faction set")
    print("=" * 60)
    
    scraper = ImprovedTwilightFactionScraper(debug_mode=True, faction_set=faction_set)
    
    # Test first faction from the set
    if scraper.factions:
        test_faction = scraper.factions[0]
        print(f"Testing with: {test_faction}")
        faction_data = scraper.scrape_faction(test_faction)
        
        print(f"\n📋 Test Results:")
        print(f"  Faction: {test_faction}")
        print(f"  Success: {faction_data['scraped_successfully']}")
        print(f"  Sections found: {len(faction_data['sections'])}")
        print(f"  URL tested: {faction_data['url']}")
        
        if faction_data['sections']:
            print(f"\n  📄 Sections extracted:")
            for section in faction_data['sections'].keys():
                print(f"    ✅ {section}")
        else:
            print("  ❌ No sections extracted")
    
    return scraper


if __name__ == "__main__":
    import sys
    
    print("🚀 Twilight Imperium Enhanced Faction Scraper")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python faction_scraper_improved.py base          # Scrape base game factions")
        print("  python faction_scraper_improved.py pok           # Scrape Prophecy of Kings factions")
        print("  python faction_scraper_improved.py codex         # Scrape Codex factions")
        print("  python faction_scraper_improved.py all           # Scrape all 25 factions")
        print("  python faction_scraper_improved.py test-base     # Test base game faction")
        print("  python faction_scraper_improved.py test-pok      # Test PoK faction")
        print("  python faction_scraper_improved.py test-codex    # Test Codex faction")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Test modes
    if command == "test-base":
        test_faction_set("base")
    elif command == "test-pok":
        test_faction_set("pok")
    elif command == "test-codex":
        test_faction_set("codex")
    elif command == "test":
        # Legacy test mode
        test_single_faction()
    
    # Production scraping modes
    elif command in ["base", "pok", "codex", "all"]:
        print(f"🎯 Starting {command} faction scraping...")
        
        # Choose debug mode based on faction set
        debug_mode = (command in ["codex"]) or (len(sys.argv) > 2 and sys.argv[2] == "debug")
        
        scraper = ImprovedTwilightFactionScraper(debug_mode=debug_mode, faction_set=command)
        
        # Run the scraping
        scraper.scrape_all_factions()
        scraper.save_faction_data()
        scraper.create_summary_report()
        
        print(f"\n🎉 {scraper.set_description} scraping complete!")
        print(f"📁 Check the {scraper._get_output_directory()} folder for results")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: base, pok, codex, all, test-base, test-pok, test-codex")
        sys.exit(1) 