"""
Expanded Twilight Imperium Fandom Data Scraper
Scrapes various game elements from the Fandom wiki beyond just factions

Supported data types:
- Factions (using existing scraper)
- Relics
- Action Cards
- Agenda Cards
- Technologies
- Planets
- Objectives
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TwilightFandomDataScraper:
    """
    Scraper for various Twilight Imperium game elements from Fandom wiki
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the Fandom data scraper
        
        Args:
            debug_mode: Enable debug output and limit scraping for testing
        """
        self.debug_mode = debug_mode
        self.base_url = "https://twilight-imperium.fandom.com/wiki/"
        
        # Configure what to scrape
        self.data_types = {
            "relics": {
                "page": "Relic",
                "list_selector": None,  # Will scrape main page
                "description": "Powerful artifacts found on frontier planets"
            },
            "action_cards": {
                "page": "Action_Card",
                "list_selector": None,
                "description": "Cards players can play during tactical or strategic actions"
            },
            "agenda_cards": {
                "page": "Agenda_Card",
                "list_selector": None,
                "description": "Political cards revealed during the Agenda Phase"
            },
            "technologies": {
                "page": "Technology",
                "list_selector": None,
                "description": "Upgrades and advancements for factions"
            },
            "planets": {
                "page": "Planet",
                "list_selector": None,
                "description": "Worlds that can be controlled for resources and influence"
            },
            "objectives": {
                "page": "Objective",
                "list_selector": None,
                "description": "Victory point goals for players"
            }
        }
        
        # Create output directory
        self.processed_rules_dir = Path("processed_rules")
        self.fandom_data_dir = self.processed_rules_dir / "fandom_data"
        self.fandom_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Better headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        print("✅ Fandom data scraper initialized")
        print(f"📁 Data will be saved to: {self.fandom_data_dir}")
        if debug_mode:
            print("🐛 Debug mode enabled")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common wiki markup
        unwanted_phrases = [
            "[edit]", "Edit", "[]", "Category:", "File:",
            "Main Page", "Community portal", "Random page"
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")
        
        return text.strip()
    
    def _extract_page_content(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract structured content from a Fandom page"""
        content = {
            "main_text": "",
            "tables": [],
            "lists": [],
            "sections": {}
        }
        
        # Find main content div
        main_content = soup.find('div', class_=['mw-parser-output', 'mw-content-text'])
        
        if not main_content:
            return content
        
        # Extract main text from paragraphs
        paragraphs = main_content.find_all('p')
        content["main_text"] = "\n\n".join([
            self._clean_text(p.get_text()) 
            for p in paragraphs 
            if self._clean_text(p.get_text())
        ])
        
        # Extract tables (useful for card lists, stats, etc.)
        tables = main_content.find_all('table')
        for table in tables:
            rows = []
            for row in table.find_all('tr'):
                cells = [self._clean_text(cell.get_text()) for cell in row.find_all(['td', 'th'])]
                if any(cells):  # Only add if row has content
                    rows.append(cells)
            if rows:
                content["tables"].append(rows)
        
        # Extract lists
        lists = main_content.find_all(['ul', 'ol'])
        for lst in lists:
            items = [self._clean_text(li.get_text()) for li in lst.find_all('li')]
            if items:
                content["lists"].append(items)
        
        # Extract sections with headers
        for header_tag in ['h2', 'h3', 'h4']:
            headers = main_content.find_all(header_tag)
            for header in headers:
                section_title = self._clean_text(header.get_text())
                if section_title and len(section_title) < 100:
                    # Get content after header until next header
                    section_content = []
                    current = header.next_sibling
                    
                    while current:
                        if current.name in ['h1', 'h2', 'h3', 'h4']:
                            break
                        if current.name in ['p', 'ul', 'ol']:
                            text = self._clean_text(current.get_text())
                            if text:
                                section_content.append(text)
                        current = current.next_sibling
                    
                    if section_content:
                        content["sections"][section_title] = "\n".join(section_content)
        
        return content
    
    def scrape_data_type(self, data_type: str) -> Dict[str, Any]:
        """
        Scrape data for a specific type (relics, action cards, etc.)
        
        Args:
            data_type: Type of data to scrape (key from self.data_types)
            
        Returns:
            Dictionary with scraped data
        """
        if data_type not in self.data_types:
            print(f"❌ Unknown data type: {data_type}")
            return {}
        
        config = self.data_types[data_type]
        print(f"\n🔍 Scraping {data_type}...")
        print(f"📄 Description: {config['description']}")
        
        url = f"{self.base_url}{config['page']}"
        
        result = {
            "data_type": data_type,
            "description": config["description"],
            "source_url": url,
            "content": {},
            "scraped_successfully": False
        }
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            content = self._extract_page_content(soup)
            
            # Check if we got meaningful content
            if content["main_text"] or content["sections"]:
                result["content"] = content
                result["scraped_successfully"] = True
                print(f"  ✅ Successfully scraped {data_type}")
                print(f"  📊 Main text: {len(content['main_text'])} characters")
                print(f"  📊 Sections: {len(content['sections'])}")
                print(f"  📊 Tables: {len(content['tables'])}")
                print(f"  📊 Lists: {len(content['lists'])}")
            else:
                print(f"  ⚠️  No substantial content found for {data_type}")
            
            # Respectful delay
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ Error scraping {data_type}: {e}")
        
        return result
    
    def scrape_all_data_types(self, types_to_scrape: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scrape all configured data types
        
        Args:
            types_to_scrape: Optional list of specific types to scrape
                           If None, scrapes all configured types
        """
        if types_to_scrape is None:
            types_to_scrape = list(self.data_types.keys())
        
        print(f"🚀 Starting to scrape {len(types_to_scrape)} data types...")
        print("=" * 60)
        
        all_data = {
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_types": {}
        }
        
        successful = 0
        failed = 0
        
        for data_type in types_to_scrape:
            result = self.scrape_data_type(data_type)
            all_data["data_types"][data_type] = result
            
            if result.get("scraped_successfully"):
                successful += 1
            else:
                failed += 1
        
        print(f"\n" + "=" * 60)
        print(f"📊 Scraping Summary:")
        print(f"  Total attempted: {len(types_to_scrape)}")
        print(f"  Successfully scraped: {successful}")
        print(f"  Failed: {failed}")
        
        return all_data
    
    def save_data(self, data: Dict[str, Any], filename: str = "fandom_data.json"):
        """Save scraped data to file"""
        output_file = self.fandom_data_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Fandom data saved to: {output_file}")
        
        # Also save individual files for each data type
        for data_type, result in data.get("data_types", {}).items():
            if result.get("scraped_successfully"):
                type_file = self.fandom_data_dir / f"{data_type}.json"
                with open(type_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"  📄 Saved {data_type} to: {type_file}")
    
    def convert_to_text_for_embedding(self, data: Dict[str, Any]) -> str:
        """
        Convert scraped data to plain text suitable for embedding
        
        Args:
            data: Scraped data from a single data type
            
        Returns:
            Formatted text string
        """
        if not data.get("scraped_successfully"):
            return ""
        
        text_parts = []
        content = data.get("content", {})
        data_type = data.get("data_type", "unknown")
        
        # Add header
        text_parts.append(f"=== {data_type.upper().replace('_', ' ')} ===\n")
        
        # Add main text
        if content.get("main_text"):
            text_parts.append(content["main_text"])
            text_parts.append("\n")
        
        # Add sections
        for section_name, section_text in content.get("sections", {}).items():
            text_parts.append(f"\n## {section_name}\n")
            text_parts.append(section_text)
            text_parts.append("\n")
        
        # Add lists
        for lst in content.get("lists", []):
            for item in lst:
                text_parts.append(f"• {item}")
            text_parts.append("\n")
        
        return "\n".join(text_parts)


def main():
    """Main function to run the scraper"""
    import sys
    
    print("🚀 Twilight Imperium Fandom Data Scraper")
    print("=" * 60)
    
    # Parse command line arguments
    debug_mode = "--debug" in sys.argv
    
    scraper = TwilightFandomDataScraper(debug_mode=debug_mode)
    
    # Determine what to scrape
    if len(sys.argv) > 1 and sys.argv[1] != "--debug":
        # Scrape specific type
        data_type = sys.argv[1]
        if data_type in scraper.data_types:
            result = scraper.scrape_data_type(data_type)
            data = {
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_types": {data_type: result}
            }
            scraper.save_data(data, f"fandom_{data_type}.json")
        else:
            print(f"❌ Unknown data type: {data_type}")
            print(f"Available types: {', '.join(scraper.data_types.keys())}")
    else:
        # Scrape all types
        data = scraper.scrape_all_data_types()
        scraper.save_data(data)
    
    print("\n🎉 Scraping complete!")


if __name__ == "__main__":
    main()

