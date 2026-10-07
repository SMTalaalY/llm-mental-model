# RAG Flow: Q&A Notes

| # | Question | Answer |
|---|----------|--------|
| 1 | What do the three functions do? | `load_document` reads .txt files from a folder into Document objects. `split_documents` cuts them into smaller chunks. `create_vector_store` embeds each chunk (text to vector via OpenAI) and stores the vectors in ChromaDB. |
| 2 | What is a LangChain Document? | One object with two parts: `page_content` (the text) and `metadata` (a dict, e.g. `{'source': 'docs/file.txt'}`). Loaders like DirectoryLoader/TextLoader produce Documents; they are not part of them. |
| 3 | What does `load_dotenv()` do? | Reads the `.env` file and loads variables like `OPENAI_API_KEY` into the environment. Without it, OpenAIEmbeddings fails with a missing API key error. |
| 4 | Does the code read PDF/DOCX/PPT? | No, only .txt (via `glob="*.txt"` and TextLoader). Other formats need other loaders, e.g. `PyPDFLoader`, `Docx2txtLoader`. |
| 5 | Why chunk documents? | (1) Embedding models have an input limit. (2) Precision: one vector for a whole document blurs all topics together, while small chunks have sharp, searchable meaning. (3) Only relevant pieces get sent to the LLM. |
| 6 | What unit is `chunk_size=800`? | Characters (measured with `len()`). Tokens only if you use `from_tiktoken_encoder`. |
| 7 | What is `chunk_overlap`? | Number of characters repeated between neighboring chunks. With 0, ideas cut at chunk boundaries lose context. Typical: 10-20% of chunk size (100-150 for 800). |
| 8 | Default separator of CharacterTextSplitter? What if a paragraph is over 800 chars? | Default is `"\n\n"` (blank line). Oversized paragraphs are kept as one oversized chunk with a warning, not dropped or cut. Fix: use `RecursiveCharacterTextSplitter`. |
| 9 | With 20 chunks, what does the print logic show? | Only 5 chunks (`chunks[:5]`). The "more chunks" message needs over 50, so 15 chunks are hidden silently. Bug: the condition should be `> 5`. |
| 10 | What is an embedding? | A list of numbers representing the meaning of a whole chunk. Similar meanings get similar numbers, like GPS coordinates on a map of meaning. It is NOT an ID; IDs carry no meaning. |
| 11 | Output of `text-embedding-3-small`? | 1536 numbers per chunk. 100 chunks gives 100 x 1536. |
| 12 | What does `"hnsw:space": "cosine"` mean? | HNSW = fast search index (avoids comparing against every vector). Cosine = distance measure comparing the angle between vectors (direction of meaning, ignoring length). |
| 13 | What does `persist_directory` do? | Saves vectors to disk. Without it, Chroma keeps them in memory only and they vanish when the script ends. |
| 14 | What happens if you run the script 3 times? | Duplicates: every chunk is stored 3 times. Retrieval then returns copies of the same chunk, wasting top-k slots. |
| 15 | Does every run cost money? How to avoid it? | Yes, every chunk is re-sent to OpenAI and billed per token. Fix: skip embedding if `db/chroma_db` already exists. |
| 16 | Which half of RAG is this script? | The indexing (ingestion) half. Missing half: retrieval + generation (embed question, search Chroma, send top chunks + question to the LLM). |
| 17 | How do you use the stored vectorstore? | Load it: `Chroma(persist_directory=..., embedding_function=...)`, then `similarity_search(query, k=3)`. Must use the SAME embedding model as indexing. |

## Key Rules to Remember
- Chunk size is in characters by default, not tokens.
- Always use some overlap (10-20%).
- Prefer `RecursiveCharacterTextSplitter`.
- Don't re-run indexing blindly: it causes duplicates and costs money.
- Same embedding model for indexing and querying, always.


# Tokens vs Chunks vs Vectors (RAG Notes)

## The Example Text (one document)

> My mother makes the best chicken biryani in Karachi every Sunday afternoon. She soaks the basmati rice for thirty minutes, fries the onions until they turn golden brown, and layers the rice with spicy masala. The whole house smells amazing, and my cousins always show up uninvited when they hear biryani is ready.

---

## Step 1: Chunking (YOU do this with a text splitter)

Say `chunk_size=150` characters and `chunk_overlap=30`. The splitter cuts the document into pieces:

| Chunk | Text |
|---|---|
| Chunk 1 | My mother makes the best chicken biryani in Karachi every Sunday afternoon. She soaks the basmati rice for thirty minutes, |
| Chunk 2 | ...rice for thirty minutes, fries the onions until they turn golden brown, and layers the rice with spicy masala. |
| Chunk 3 | ...with spicy masala. The whole house smells amazing, and my cousins always show up uninvited when they hear biryani is ready. |

Notice the **overlap**: the end of one chunk repeats at the start of the next ("rice for thirty minutes", "with spicy masala"). This keeps ideas from being cut in half.

*(Chunk boundaries above are approximate, for illustration.)*

---

## Step 2: Tokenization (the MODEL does this internally)

Take **Chunk 1** only. The tokenizer splits it into small pieces:

```
"My" | " mother" | " makes" | " the" | " best" | " chicken" | " bir" | "yani" |
" in" | " Kar" | "achi" | " every" | " Sunday" | " afternoon" | "." |
" She" | " soaks" | " the" | " bas" | "mati" | " rice" | " for" |
" thirty" | " minutes" | ","
```

Key points:
- Tokens are pieces of the **exact** text. Nothing is changed or capitalized differently.
- Common words ("the", "rice") are usually 1 token.
- Rare words ("biryani", "Karachi", "basmati") get split into pieces.
- The space is often attached to the front of the word (`" mother"`).
- Exact splits differ per model. The splits above are **illustrative**.

Each token is then mapped to an **ID number**:

```
"My" → 5159,  " mother" → 6691,  " bir" → 15296, ...
```

These IDs are just lookup numbers. **They carry no meaning.** ID 6691 and ID 6692 are not related.

*(ID numbers above are made up, for illustration.)*

---

## Step 3: Embedding (the MODEL turns the whole chunk into ONE vector)

The embedding model reads **all tokens of Chunk 1 together** and outputs **one vector**:

```
Chunk 1  →  [0.021, -0.443, 0.118, 0.307, ..., -0.052]   (1536 numbers)
Chunk 2  →  [0.015, -0.398, 0.201, 0.144, ..., 0.077]    (1536 numbers)
Chunk 3  →  [-0.110, 0.231, 0.090, -0.276, ..., 0.163]   (1536 numbers)
```

3 chunks = 3 vectors. Not one vector per token.

These 3 vectors are what gets stored in **ChromaDB**.

---

## Step 4: Why This Matters at Search Time

User asks: **"How long should I soak basmati rice?"**

1. The question is embedded into its own vector (same model, 1536 numbers).
2. ChromaDB compares it with the 3 stored vectors (cosine similarity).
3. **Chunk 1** is the closest match because it talks about soaking basmati rice for thirty minutes.
4. Chunk 1 is sent to the LLM along with the question, which then answers: "30 minutes."

---

## The Hierarchy

| Level | Who makes it | Size | Example |
|---|---|---|---|
| Document | You (the file) | Whole file | `biryani_story.txt` |
| Chunk | You (text splitter) | ~150-1000 characters | "My mother makes the best chicken biryani..." |
| Token | Model (tokenizer) | ~4 characters | `" bir"`, `"yani"` |
| Vector | Model (embedding) | 1536 numbers | One per chunk |

---

## Rules to Remember

- **Token ≠ Chunk.** A chunk contains many tokens.
- **You make chunks. The model makes tokens.**
- **One chunk = one vector.** Tokens are used internally, but you store one vector per chunk.
- **Token IDs are not embeddings.** IDs are meaningless lookup numbers. Embeddings carry meaning.
- **Rough math:** 1 token ≈ 4 English characters, so an 800-character chunk ≈ 200 tokens.
- **Same embedding model** for storing chunks and embedding questions, always.

# Part 2: How an LLM Generates Text (vs. an Embedding Model)
 
## The LLM Pipeline (e.g. GPT writing an answer)
 
```
HUMAN TEXT
    ↓
TOKENIZATION          text → tokens  (" bir", "yani")
    ↓
TOKEN IDs             tokens → lookup numbers (no meaning)
    ↓
TOKEN EMBEDDINGS      each ID → a vector (meaning starts here)
    ↓
TRANSFORMER LAYERS    repeated many times:
   ├─ Attention       each token looks at other tokens for context
   └─ Feed-forward    processes each token further
   (ALL of this runs on weights/parameters)
    ↓
LOGITS                raw score for every possible next token
    ↓
SOFTMAX               scores → probabilities (add up to 100%)
    ↓
SAMPLING              pick one token (temperature controls randomness)
    ↓
NEXT TOKEN
    ↓
REPEAT 🔄             add new token to input, run loop again
    ↓                 until a stop token is produced
AI RESPONSE
```
 
## The Embedding Model Pipeline (what RAG uses to store chunks)
 
```
CHUNK TEXT
    ↓
TOKENIZATION → TOKEN IDs → TOKEN EMBEDDINGS → TRANSFORMER LAYERS
    ↓
POOLING               combine all token vectors into ONE
    ↓
ONE VECTOR PER CHUNK  (e.g. 1536 numbers) → stored in ChromaDB
```
 
**No logits, no softmax, no next token, no repeat.** An embedding model only *understands* text. It never *writes* text.
 
## Side by Side
 
| | Embedding Model | LLM (Generator) |
|---|---|---|
| Job | Turn text into meaning-vector | Write text, one token at a time |
| Output | One vector per chunk | Next token, repeated until done |
| Example | `text-embedding-3-small` | GPT, Claude |
| Used in RAG for | Indexing chunks + embedding the question | Writing the final answer |
 
## Glossary (Corrected)
 
| Term | Meaning |
|---|---|
| Token | A small piece of text |
| Token ID | A lookup number for a token. Carries no meaning |
| Embedding | A vector that represents meaning (per token inside a model, or per chunk in RAG) |
| Vector | A list of numbers |
| Vector space | The "map" where all vectors live; close = similar meaning. Not a step |
| Parameters / Weights | Same idea: the learned numbers inside the model (billions of them) |
| Transformer | The model architecture: attention + feed-forward layers, stacked |
| Attention | Lets each token decide which other tokens matter for its meaning |
| Logits | Raw scores for every possible next token |
| Softmax | Converts logits into probabilities |
| Sampling | Picks the actual next token from those probabilities |
| Temperature | Low = safe, predictable picks. High = more random, creative picks |
| Training | Adjusting parameters so predictions improve |
 
## Rules to Remember (Part 2)
 
- Weights and parameters are the same idea. Don't treat them as two things.
- Weights aren't one box. They're everywhere in the model.
- The LLM loop: predict → sample → append → repeat.
- RAG uses TWO models: an embedding model (find chunks) and an LLM (write the answer).


# Cosine Similarity

Cosine Similarity measures the angle between vectors, not their magnitudes.
Cosine Similarity is the reason we are able to fetch matching chunks based on user query

**Formula**

`cosine_similarity = A.B/(||A||.||B||)`
where:
*A* = Vector Embedding A (same for B)
*A.B* = Dot Product between A and B
*||A||* = Magnitude (Length) of Vector A
*||B||* = Magnitude (Length) of Vector B

The Similarity Score in Cosine Similarity is between 0 and 1 with 1 being closest and 0 being farthest. 
