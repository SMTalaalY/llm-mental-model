# LLM Mental Model

## Pre-Training: Step 1 - Collecting the Internet

**Pre-training** = training a neural network to predict the next token, using a huge amount of internet text.
Before training, we need that text. Step 1 is collecting it and cleaning it.

> Analogy: before a student can read a library, someone must collect the books
> and throw out the junk, duplicates, and torn pages.

### Finding the Raw Data

Two ways to get internet text:
1. Crawl the web yourself
2. Use **Common Crawl**, a non-profit that has crawled the web since 2007
   (April 2024: ~2.7 billion pages, ~386 TiB of raw HTML)

Raw crawled data is messy, so it goes through a cleaning pipeline.
Example: the **FineWeb** dataset pipeline:

| Step | What it does | Plain-English version |
|---|---|---|
| 1. URL filtering | Blocks bad domains (spam, malware, adult, marketing, hate sites) | Don't even open books from bad shops |
| 2. Text extraction | Strips HTML, menus, ads, code; keeps the main text | Tear off the cover and adverts, keep the story |
| 3. Language filtering | Detects the language; FineWeb keeps pages that are >65% English | Keep mostly English books |
| 4. Gopher filtering | Quality rules (too short, too repetitive, too many symbols) | Throw out low-quality pages |
| 5. Deduplication (MinHash) | Removes near-duplicate pages | Keep one copy, not 50 photocopies |
| 6. C4 filters | More cleanup rules (boilerplate, broken text) | Remove filler and junk lines |
| 7. Custom filters | The dataset team's own extra rules | House rules |
| 8. PII removal | Removes personal info (emails, addresses, phone numbers) | Black out private details |

**Result:** FineWeb is about 44 TB of clean text, roughly 15 trillion tokens.

**Why it matters:** these choices shape the model.
Example: filtering for English is one reason models can be weaker in languages like Urdu.

## Tokenization: Turning Text into Numbers

Neural networks can't read text. They need a **1-D sequence of symbols** from a fixed set (a vocabulary).

1. Text → **UTF-8 encoding** → raw bits (0s and 1s)

![alt text](image-1.png)

2. Bits are too long a sequence with too few symbols (only 0 and 1).
   **Trade-off:** we want *shorter sequences* with *more symbols*.

![alt text](image.png)

3. Group 8 bits into a **byte** → 256 possible symbols (0–255)

![alt text](image-2.png)

4. **Byte Pair Encoding (BPE)** shrinks it further:
   - Find the most common pair of neighbouring symbols (e.g. 116 followed by 32)
   - Replace that pair with a **new symbol** (256)
   - Repeat many times → 257, 258, …
   - GPT-4 ends up with a vocabulary of ~100,277 tokens

![alt text](image-3.png)

5. **Tokenization** = converting raw text into these tokens (numbers).

> Analogy: instead of spelling every word letter by letter,
> you invent shortcuts for common letter-pairs, then for common words.
> Messages get shorter, but you need a bigger dictionary.

### Tools

- **tiktoken**: OpenAI's tokenizer library (Python)
- **Tiktokenizer**: a website that *visualizes* how text splits into tokens

![alt text](image-4.png)
![alt text](image-5.png)

**Try this:** "hello world" vs "Hello World" vs "helloworld" give different tokens.
Spaces and capital letters matter.


## Inference

Inference = using the trained model to generate text. No learning happens here; the weights are fixed.

1. Give the model a prefix (the prompt), as tokens
2. It outputs a probability for every token in its vocabulary
3. One token is **sampled** (randomly, weighted by probability)
4. That token is appended to the sequence → repeat from step 2

This is why the same prompt can give different answers: sampling is random.

## Post-Training

Pre-training produces a **base model**: an internet document simulator.
It continues text; it doesn't answer like an assistant.

Post-training turns it into an assistant:
- Train on datasets of example **conversations** written by human labelers
  following company guidelines
- We "program" the assistant implicitly, through examples, not code
- The assistant is a statistical imitation of those labelers
- Then reinforcement learning improves it further
- Much cheaper than pre-training (hours/days vs months)

## Context Window

The context window = the model's **working memory**: all tokens it can see right now
(prompt + conversation + any documents we paste in).

| Knowledge in the **weights** | Knowledge in the **context window** |
|---|---|
| Vague recollection, like something you read a month ago | Like having the page open in front of you |
| Can be wrong or outdated | Directly visible, much more reliable |

The window has a size limit. When it's full, older tokens fall out.

## Hallucination

Hallucination = the model confidently makes things up.

**Why it happens:**
1. Training conversations always show confident answers, so the model
   imitates a confident *style* even when it doesn't know
2. It predicts statistically likely tokens, not verified facts
3. Training and evaluation can reward guessing over saying "I don't know"

**How to reduce it:**
1. Train the model with examples where the correct answer is "I don't know"
2. Give it tools (like web search) to look things up
3. **Put the facts into the context window** → this is what RAG does
