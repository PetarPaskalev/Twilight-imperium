"""
Improved Fandom API Integration using fandom-py library
Much more reliable than web scraping!
"""

import fandom
import json
from pathlib import Path
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

# Set the wiki
fandom.set_wiki("twilight-imperium")

class FandomAPIIntegrator:
    """Use official Fandom API instead of web scraping"""
    
    def __init__(self):
        self.fandom_data_dir = Path("processed_rules/fandom_data")
        self.fandom_data_dir.mkdir(parents=True, exist_ok=True)
        self.mw_api = "https://twilight-imperium.fandom.com/api.php"
    
    def search_and_get_pages(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for pages and retrieve their content
        """
        print(f"Searching for: {query}")
        
        # Search for pages (results may be strings or tuples of (title, pageid))
        search_results = fandom.search(query, results=limit)
        
        pages_data = []
        for result in search_results:
            # Normalize result to a page title string
            page_title = result[0] if isinstance(result, (list, tuple)) and result else result
            try:
                # Get the full page
                page = fandom.page(page_title)
                
                pages_data.append({
                    "title": page.title,
                    "summary": page.summary,
                    "content": page.content,
                    "url": page.url
                })
                
                print(f"  Retrieved: {page.title}")
                
            except Exception as e:
                print(f"  Could not retrieve {page_title}: {e}")
        
        return pages_data
    
    def get_category_members(self, category: str) -> List[Dict]:
        """
        Get all pages in a specific category
        This is PERFECT for getting all relics, all action cards, etc.
        """
        print(f"Getting category members for: {category}")
        members: List[Dict[str, Any]] = []
        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": "500",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp = requests.get(self.mw_api, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cms = data.get("query", {}).get("categorymembers", [])
            for cm in cms:
                members.append({
                    "pageid": cm.get("pageid"),
                    "title": cm.get("title"),
                    "ns": cm.get("ns"),
                })
            cont = data.get("continue")
            if cont and cont.get("cmcontinue"):
                cmcontinue = cont["cmcontinue"]
            else:
                break
        print(f"  Found {len(members)} members in {category}")
        return members

    def get_page_extracts(self, titles: List[str]) -> List[Dict]:
        """Batch fetch plain-text extracts for given page titles via MediaWiki API."""
        print(f"Fetching extracts for {len(titles)} pages...")
        results: List[Dict[str, Any]] = []
        # Batch titles to avoid URL length issues (50 per batch is safe)
        for i in range(0, len(titles), 40):
            batch = titles[i:i+40]
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "redirects": "1",
                "formatversion": "2",
                "titles": "|".join(batch),
            }
            resp = requests.get(self.mw_api, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                results.append({
                    "title": page.get("title"),
                    "extract": page.get("extract", ""),
                    "pageid": page.get("pageid")
                })
        print(f"  Retrieved {len(results)} extracts")
        return results

    def get_codex_sections(self, target_substrings: List[str]) -> List[Dict]:
        """
        Pull specific sections from the Codex I - IV page by matching section lines
        against provided substrings (e.g., ["Relics", "Action Cards", "Faction Technologies"]).
        """
        codex_title = "Codex I - IV (Fourth Edition Expansion)"
        print(f"Parsing sections from: {codex_title}")
        # 1) List sections
        params = {
            "action": "parse",
            "format": "json",
            "page": codex_title,
            "prop": "sections",
        }
        resp = requests.get(self.mw_api, params=params, timeout=30)
        resp.raise_for_status()
        sections = resp.json().get("parse", {}).get("sections", [])
        if not sections:
            print("  No sections found")
            return []
        # 2) Filter sections by target substrings (case-insensitive)
        to_fetch = []
        lc_targets = [s.lower() for s in target_substrings]
        for s in sections:
            line = s.get("line", "")
            index = s.get("index")
            if any(t in line.lower() for t in lc_targets):
                to_fetch.append({"line": line, "index": index})
        print(f"  Matched {len(to_fetch)} sections: {[s['line'] for s in to_fetch]}")
        # 3) Fetch HTML for each section, convert to plain text
        results: List[Dict[str, Any]] = []
        for s in to_fetch:
            params = {
                "action": "parse",
                "format": "json",
                "page": codex_title,
                "prop": "text",
                "section": s["index"],
            }
            r2 = requests.get(self.mw_api, params=params, timeout=30)
            r2.raise_for_status()
            html = r2.json().get("parse", {}).get("text", {}).get("*", "")
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            # Remove unwanted elements
            for el in soup.select(".reference, .mw-editsection"):
                el.decompose()
            text = soup.get_text("\n").strip()
            results.append({
                "page": codex_title,
                "section_line": s["line"],
                "section_index": s["index"],
                "text": text
            })
        print(f"  Extracted {len(results)} Codex sections")
        return results
    
    def get_specific_pages(self, page_titles: List[str]) -> List[Dict]:
        """
        Get multiple specific pages
        """
        pages_data = []
        
        for title in page_titles:
            try:
                page = fandom.page(title)
                pages_data.append({
                    "title": page.title,
                    "summary": page.summary,
                    "content": page.content,
                    "url": page.url
                })
                print(f"  Retrieved: {title}")
            except:
                print(f"  Page not found: {title}")
        
        return pages_data


# Example usage:
if __name__ == "__main__":
    integrator = FandomAPIIntegrator()
    
    # Search for relics
    relics = integrator.search_and_get_pages("relic", limit=5)
    
    # Search for action cards
    action_cards = integrator.search_and_get_pages("action card", limit=10)
    
    # Get specific pages
    tech_pages = integrator.get_specific_pages([
        "Technology",
        "Unit Upgrade Technology",
        "Faction Technology"
    ])

    # Category-based pulls (adjust category names if needed)
    # Likely categories to try: "Category:Relics", "Category:Action Cards"
    try:
        relic_members = integrator.get_category_members("Category:Relics")
        action_members = integrator.get_category_members("Category:Action Cards")
        # Fetch extracts for first few as sample
        relic_extracts = integrator.get_page_extracts([m["title"] for m in relic_members[:10]])
        action_extracts = integrator.get_page_extracts([m["title"] for m in action_members[:10]])
        # Save samples
        out_dir = integrator.fandom_data_dir
        (out_dir / "category_samples.json").write_text(json.dumps({
            "relics_sample": relic_extracts,
            "action_cards_sample": action_extracts
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Saved category_samples.json")
    except Exception as e:
        print(f"Category fetch failed: {e}")

    # Codex sections for Relics / Action Cards / Faction Technologies
    try:
        codex_sections = integrator.get_codex_sections([
            "Relics",
            "Action Cards",
            "Faction Technologies"
        ])
        (integrator.fandom_data_dir / "codex_sections.json").write_text(
            json.dumps(codex_sections, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Saved codex_sections.json")
    except Exception as e:
        print(f"Codex section fetch failed: {e}")
