# Current Chunking Strategy - Complete Explanation

## 📊 Current Setup Overview

### What You Have Now

**Vector Store:**
- **Total Vectors:** 862 embeddings
- **Original Chunks:** 286 chunks
- **Embedding Model:** `text-embedding-3-small` (1536 dimensions)
- **Storage:** FAISS vector database

**Data Sources:**
- Learn to Play Guide: 182 chunks
- Official Rulebook: 104 chunks
- Twilight Pravila (DOCX): 84 chunks
- Official FAQ: 84 chunks
- Fandom Technologies: 21 chunks
- Faction Data: 51 documents
- Total DOCX documents: 168

---

## 🔧 Current Chunking Configuration

### Parameters

```python
chunk_size = 800        # Maximum characters per chunk
chunk_overlap = 100     # Characters overlapping between chunks
```

### Separator Priority (in order)

1. `\n\n` - Double newlines (paragraph breaks)
2. `\n` - Single newlines (line breaks)
3. `. ` - Sentence endings
4. `, ` - Clause separators
5. ` ` - Word boundaries
6. `""` - Character level (last resort)

### How It Works

1. **Text Splitting Process:**
   - Starts with full document text
   - Tries to split at paragraph breaks first (`\n\n`)
   - If chunk would exceed 800 chars, tries next separator
   - Continues down the priority list until chunk fits
   - Creates overlap of 100 chars between adjacent chunks

2. **Example Flow:**
   ```
   Original text: 2000 characters
   
   Chunk 1: chars 0-800 (ends at paragraph break)
   Chunk 2: chars 700-1500 (100 char overlap, starts at char 700)
   Chunk 3: chars 1400-2000 (100 char overlap, starts at char 1400)
   ```

3. **Metadata Added:**
   - `source`: Document identifier (e.g., "learn_to_play")
   - `doc_type`: Human-readable name (e.g., "Learn to Play Guide")
   - `chunk_id`: Unique identifier (e.g., "learn_to_play_chunk_042")
   - `chunk_index`: Position in document (0-based)
   - `total_chunks`: Total chunks in this document
   - `char_count`: Character count of this chunk
   - `word_count`: Word count of this chunk
   - `section`: Detected section header (if found)

---

## 🔍 Current Search Behavior

### How Search Works

1. **User Query:** "What does the Leadership strategy card do?"

2. **Vector Search:**
   - Query is embedded using same model (`text-embedding-3-small`)
   - FAISS finds top `k=5` most similar chunks
   - Returns chunks ranked by cosine similarity

3. **Results Format:**
   ```python
   {
     'rank': 1,
     'content': '...chunk text...',
     'source': 'rulebook',
     'doc_type': 'Official Rulebook',
     'chunk_id': 'rulebook_chunk_042',
     'section': 'Strategy Cards',
     'char_count': 750,
     'word_count': 120
   }
   ```

4. **Current Tool Usage:**
   - `search_rules_formatted()` returns top `k=3` results
   - Results are formatted with metadata
   - Passed to LLM as context

---

## 📈 Current Statistics

### Chunk Size Distribution
- **Average:** 583 characters
- **Minimum:** 14 characters (very short chunks)
- **Maximum:** 800 characters (hitting the limit)
- **Most chunks:** Between 500-700 characters

### Overlap Analysis
- **Overlap:** 100 characters (~12.5% of chunk size)
- **Purpose:** Maintains context across chunk boundaries
- **Example:** If a rule spans two chunks, overlap ensures both chunks contain part of the rule

### Section Detection
- **Chunks with sections:** Only 4 out of 286 (1.4%)
- **Current detection:** Looks for short lines (<100 chars, <8 words) in first 3 lines
- **Limitation:** Most chunks don't have detected section headers

---

## ⚡ Current Performance Characteristics

### Strengths

1. **Consistent Size:** Most chunks are 500-700 chars (good for embeddings)
2. **Context Preservation:** 100-char overlap maintains continuity
3. **Structure Aware:** Tries to split at natural boundaries (paragraphs, sentences)
4. **Rich Metadata:** Each chunk has source tracking and statistics

### Potential Issues

1. **Fixed Size:** 800 chars might be too small for complex rules
2. **Overlap Might Be Insufficient:** 100 chars (~12.5%) may not capture full context
3. **Section Detection Weak:** Only 1.4% of chunks have detected sections
4. **No Semantic Chunking:** Splits by characters, not by meaning/topic
5. **Small Chunks:** Some chunks are only 14 characters (likely headers/footers)

---

## 🎯 How Confidence Scoring Works With Chunking

### Current Flow

1. **User asks question** → Query embedded
2. **Vector search** → Finds top 3-5 relevant chunks
3. **Chunks sent to LLM** → As context for answering
4. **LLM generates response** → With logprobs enabled
5. **Confidence calculated** → From token-level probabilities
6. **Confidence logged** → Backend only (0.0-1.0 scale)

### Chunking Impact on Confidence

- **Good chunks** → Better context → Higher confidence
- **Irrelevant chunks** → Confusing context → Lower confidence
- **Incomplete chunks** → Missing information → Lower confidence
- **Overlapping chunks** → Redundant context → May affect confidence

---

## 🤔 Questions to Consider for Improvement

1. **Chunk Size:**
   - Is 800 chars optimal? Should it be larger (1000-1500) for complex rules?
   - Should different document types have different chunk sizes?

2. **Overlap:**
   - Is 100 chars enough? Should it be 20-25% of chunk size?
   - Should overlap be adaptive based on content?

3. **Semantic Chunking:**
   - Should we chunk by topic/rule rather than fixed size?
   - Can we detect rule boundaries (e.g., "Strategy Cards" section)?

4. **Multi-level Chunking:**
   - Should we have hierarchical chunks (sections → rules → details)?
   - Can we chunk at different granularities for different query types?

5. **Metadata Enhancement:**
   - Can we better detect and tag sections?
   - Should we add rule type tags (combat, movement, strategy cards, etc.)?

6. **Search Strategy:**
   - Should we use different `k` values based on query complexity?
   - Should we re-rank results using LLM before sending to response?

---

## 📝 Next Steps

**Tell me what you have in mind!** 

Some ideas you might be considering:
- Adaptive chunk sizes based on content type
- Semantic chunking (by topic/rule rather than fixed size)
- Better section detection and tagging
- Multi-level chunking (hierarchical)
- Query-aware chunking (different strategies for different questions)
- Hybrid search (combining vector search with keyword search)
- Re-ranking chunks before sending to LLM

What's your vision for improving the chunking strategy?

