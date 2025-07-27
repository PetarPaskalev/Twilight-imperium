"""
Improved Twilight Imperium Faction Data Scraper
Enhanced version with better HTML parsing and section detection

This version includes better debugging and more robust section extraction
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
    """
    
    def __init__(self, debug_mode: bool = True):
        """Initialize the improved faction scraper"""
        self.debug_mode = debug_mode
        self.base_url = "https://twilight-imperium.fandom.com/wiki/"
        self.factions = [
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
        
        self.processed_rules_dir = Path("processed_rules")
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
        
        print(f"✅ Improved faction scraper initialized for {len(self.factions)} factions")
        if debug_mode:
            print("🐛 Debug mode enabled - will show detailed extraction info")
    
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
    
    def save_faction_data(self, filename: str = "faction_data_improved.json"):
        """Save scraped faction data to file"""
        if not self.faction_data:
            print("❌ No faction data to save")
            return
        
        self.processed_rules_dir.mkdir(exist_ok=True)
        faction_file = self.processed_rules_dir / filename
        
        with open(faction_file, 'w', encoding='utf-8') as f:
            json.dump(self.faction_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Faction data saved to: {faction_file}")


def test_single_faction(faction_name: str = "The Arborec"):
    """Test scraping a single faction for debugging"""
    print(f"🧪 Testing scraping for: {faction_name}")
    print("=" * 50)
    
    scraper = ImprovedTwilightFactionScraper(debug_mode=True)
    faction_data = scraper.scrape_faction(faction_name)
    
    print(f"\n📋 Results for {faction_name}:")
    print(f"  Success: {faction_data['scraped_successfully']}")
    print(f"  Sections found: {len(faction_data['sections'])}")
    
    for section, content in faction_data['sections'].items():
        print(f"\n  📄 {section}:")
        print(f"    Length: {len(content)} characters")
        print(f"    Preview: {content[:200]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - single faction
        test_single_faction()
    else:
        # Full scraping
        scraper = ImprovedTwilightFactionScraper(debug_mode=False)
        scraper.scrape_all_factions()
        scraper.save_faction_data() 