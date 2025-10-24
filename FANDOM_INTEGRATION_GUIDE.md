# 🌐 Fandom API Integration Guide

## 📊 Your Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
├─────────────────────────────────────────────────────────────┤
│  1. PDFs (pdf_processor.ipynb)                              │
│     • ti-k0289_learn_to_playcompressed.pdf                  │
│     • ti10_rulebook_web-good.pdf                            │
│                                                              │
│  2. DOCX Files (docx_processor.ipynb)                       │
│     • twilight_4_pravila.docx                               │
│     • Copy of Official Dane FAQ.docx                        │
│                                                              │
│  3. Fandom Wiki (fandom_data_processor.ipynb) ⭐ NEW!       │
│     • Relics                                                 │
│     • Action Cards                                           │
│     • Agenda Cards                                           │
│     • Technologies                                           │
│     • Planets                                                │
│     • Objectives                                             │
│     • Factions (faction_scraper_improved.py)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING                                │
├─────────────────────────────────────────────────────────────┤
│  1. Extract Text                                             │
│  2. Chunk into 800 chars (100 overlap)                      │
│  3. Generate Embeddings (OpenAI text-embedding-3-small)     │
│  4. Add to FAISS Vector Store                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              FAISS VECTOR DATABASE                           │
│           processed_rules/vector_store/                      │
│                                                              │
│  Current: 511 vectors                                        │
│  After DOCX: ~511 + new chunks                              │
│  After Fandom: ~511 + DOCX + Fandom chunks                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  CHATBOT & API                               │
├─────────────────────────────────────────────────────────────┤
│  • twilight_rules_tool.py (loads vector store)              │
│  • twilight_chatbot_langgraph.py (LangGraph bot)           │
│  • twilight_api.py (FastAPI server)                         │
│  • frontend/ (Next.js UI)                                    │
│                                                              │
│  ✅ Automatically uses ALL data in vector store!            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Use Fandom Integration

### Step 1: Run the Fandom Data Processor Notebook

```bash
# Activate your conda environment
conda activate twilight-imperium

# Open Jupyter
jupyter notebook fandom_data_processor.ipynb
```

Then **run all cells**. This will:
1. ✅ Scrape data from Fandom wiki (relics, action cards, etc.)
2. ✅ Chunk the data
3. ✅ Add to your existing vector database
4. ✅ Save the updated database

### Step 2: That's It! 🎉

Your chatbot **automatically** uses the updated vector database. No code changes needed!

---

## 📝 What Gets Scraped from Fandom

| Data Type      | Description                                  | Example Questions            |
|----------------|----------------------------------------------|------------------------------|
| **Relics**     | Powerful artifacts on frontier planets       | "What are relics?"           |
| **Action Cards** | Cards for tactical/strategic actions       | "How do action cards work?"  |
| **Agenda Cards** | Political cards in Agenda Phase            | "What are agenda cards?"     |
| **Technologies** | Faction upgrades and advancements          | "Tell me about tech"         |
| **Planets**    | Worlds with resources and influence          | "What planets exist?"        |
| **Objectives** | Victory point goals                          | "How do objectives work?"    |
| **Factions**   | All 25 factions (already implemented)        | "Tell me about The Arborec"  |

---

## 🔧 Customization Options

### Scrape Specific Data Types

Edit cell 3 in `fandom_data_processor.ipynb`:

```python
# Only scrape what you want
data_types_to_scrape = [
    "relics",           # ✅ Keep
    "action_cards",     # ✅ Keep
    # "agenda_cards",   # ❌ Skip
    # "technologies",   # ❌ Skip
]
```

### Use the Standalone Scraper

Run from command line:

```bash
# Scrape all types
python fandom_data_scraper.py

# Scrape specific type
python fandom_data_scraper.py relics
python fandom_data_scraper.py action_cards

# Debug mode (test first)
python fandom_data_scraper.py --debug
```

Data saved to: `processed_rules/fandom_data/`

---

## 🗂️ File Structure

```
D:\Twilight-imperium\
│
├── dataset/                          # Original files
│   ├── ti-k0289_learn_to_playcompressed.pdf
│   ├── ti10_rulebook_web-good.pdf
│   ├── twilight_4_pravila.docx
│   └── Copy of Official Dane FAQ.docx
│
├── processed_rules/                  # Processed data
│   ├── vector_store/                 # FAISS database ⭐
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── embedding_config.json         # Vector store config
│   ├── fandom_data/                  # Fandom scraped data
│   └── faction_data/                 # Faction data
│
├── Notebooks (Processing Pipelines):
│   ├── pdf_processor.ipynb           # Step 1: PDFs
│   ├── text_chunker.ipynb            # Step 2: Chunking
│   ├── embedding_generator.ipynb     # Step 3: Embeddings
│   ├── docx_processor.ipynb          # DOCX files ⭐ NEW
│   └── fandom_data_processor.ipynb   # Fandom data ⭐ NEW
│
├── Scrapers:
│   ├── fandom_data_scraper.py        # ⭐ NEW: Scrape game data
│   └── faction_scraper_improved.py   # Scrape faction info
│
├── Chatbot:
│   ├── twilight_rules_tool.py        # Loads vector store
│   ├── twilight_chatbot_langgraph.py # LangGraph bot
│   └── twilight_api.py               # FastAPI server
│
└── frontend/                         # Next.js UI
```

---

## ⚡ Quick Start Commands

### 1. Process Word Documents
```bash
jupyter notebook docx_processor.ipynb
# Run all cells
```

### 2. Add Fandom Data
```bash
jupyter notebook fandom_data_processor.ipynb
# Run all cells
```

### 3. Test Locally
```bash
# Start API server
python twilight_api.py

# In another terminal, start frontend
cd frontend
npm run dev
```

### 4. Deploy to Production
```bash
git add .
git commit -m "Added Word docs and Fandom data to vector database"
git push origin main
```

Your hosting service (Render/Railway/etc.) will automatically redeploy with the updated vector database! 🚀

---

## 🎯 Key Points

1. **All data goes into ONE vector database**
   - `processed_rules/vector_store/`
   - Your chatbot loads this automatically

2. **No code changes needed**
   - Just run notebooks to add data
   - `twilight_rules_tool.py` loads whatever is in the vector store

3. **Data sources are additive**
   - PDFs → DOCX → Fandom
   - Each notebook ADDS to the existing database
   - Never deletes previous data

4. **Easy to extend**
   - Want more data types? Edit `fandom_data_scraper.py`
   - Add new sections to `data_types` dict
   - Run the notebook again

---

## 📊 Expected Results

After running all notebooks:

| Stage           | Vectors | Sources                          |
|-----------------|---------|----------------------------------|
| Initial         | 286     | PDFs only                        |
| + Factions      | 511     | PDFs + Faction data              |
| + DOCX          | ~600+   | PDFs + Factions + Word docs      |
| + Fandom        | ~800+   | PDFs + Factions + DOCX + Fandom  |

Your chatbot will be **significantly more knowledgeable** about:
- ✅ Game rules (PDFs)
- ✅ FAQs (DOCX)
- ✅ All factions (Fandom)
- ✅ Relics, action cards, technologies, etc. (Fandom)

---

## 🐛 Troubleshooting

### "Vector store not found"
→ Run `embedding_generator.ipynb` first to create the initial vector store

### "OpenAI API key not found"
→ Make sure your `.env` file has: `OPENAI_API_KEY=sk-...`

### Fandom scraping fails
→ Check internet connection
→ Fandom wiki might be down temporarily
→ Try with debug mode: `python fandom_data_scraper.py --debug`

### Vector database too large
→ Each chunk costs ~$0.0001 to embed
→ ~1000 chunks ≈ $0.10 total
→ This is one-time cost (embeddings are cached)

---

## 🎉 Next Steps

1. ✅ **Run `docx_processor.ipynb`** to add Word documents
2. ✅ **Run `fandom_data_processor.ipynb`** to add Fandom data
3. ✅ **Test your chatbot** with new types of questions
4. ✅ **Deploy to production** and enjoy your enhanced chatbot!

Need to add more data types? Just edit `fandom_data_scraper.py` and add to the `data_types` dictionary! 🚀

