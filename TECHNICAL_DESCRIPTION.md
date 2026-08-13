# Technical Description — myRAG.py

## Overview

`myRAG.py` is a single-script **Retrieval-Augmented Generation (RAG)** pipeline built on top of [LlamaIndex](https://www.llamaindex.ai/) with **Google Gemini** serving as both the language model and the embedding model.

### What is RAG? (A Short Introduction)

A plain large language model (LLM) can only answer from what it learned during training. It does not know about your private documents (for example, the files in `./data`), and it may "hallucinate" — make up plausible-looking but wrong information.

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM access to your own documents at query time. Instead of answering from memory alone, the system first finds the most relevant passages in your documents and then asks the LLM to answer *based on those passages*. The answer is therefore **grounded** in real source text, and the script can point to the exact documents it used.

The name breaks down into three parts:

| Term | Meaning | Where it happens in this script |
| --- | --- | --- |
| **Retrieval** | Finding the most relevant text chunks from your documents | Semantic similarity search over the vector index |
| **Augmented** | Adding those retrieved chunks to the LLM's input | The chunks are passed to the LLM together with the question |
| **Generation** | The LLM writing a natural-language answer | `query_engine.query(...)` returns the final response |

### How the script works, step by step

1. **Load configuration** — The script loads the `GOOGLE_API_KEY` from a `.env` file and checks that it exists, so the Google models can be called.

2. **Configure the models and chunking** — It defines which LLM will write answers (`gemini-3.1-pro-preview`), which embedding model will convert text to vectors (`gemini-embedding-2`), and how documents are split into chunks (512 tokens with 50 tokens of overlap).

3. **Read the documents** — It reads every file in `./data` (PDFs are parsed with a dedicated `PDFReader`) and reports which files were loaded. If nothing is loaded, it stops with an error.

4. **Split the documents into chunks** — Long documents are cut into smaller, overlapping pieces (chunks) so that relevant details are easier to find later, and so each piece fits within the embedding model's input limit.

5. **Embed the chunks (turn text into numbers)** — Each chunk is passed through the embedding model, which converts it into a high-dimensional vector (a list of numbers). Texts with similar meaning get vectors that are close to each other in this numeric space. These vectors are stored in a vector index.

6. **Ask the question** — The question is a hard-coded string in the code (`"Balázs melyik évben állt munkába?"`); there is no user interface prompt in this demo.

7. **Embed the question** — The same embedding model converts the question into a vector using the same numeric space as the document chunks.

8. **Retrieve the most relevant chunks** — The system compares the question vector against every chunk vector and selects the top 2 most semantically similar chunks (`similarity_top_k=2`).

9. **Generate the answer** — The retrieved chunks are passed to the LLM together with the original question. The LLM writes an answer that is grounded in those chunks instead of relying only on its training memory.

10. **Present the answer with citations** — The script prints the generated answer, then lists each source chunk with its file name, relevance score, and a quoted snippet, so the user can trace where the information came from.

The full pipeline — ingestion, chunking, embedding, indexing, retrieval, and generation — executes in-process on every run. It is designed as a minimal, self-contained demonstration of a RAG workflow rather than a production service.

## Dependencies

| Package / Module | Purpose |
| --- | --- |
| `os`, `pathlib.Path` | Operating-system access and path handling |
| `dotenv.load_dotenv` | Loads environment variables (notably `GOOGLE_API_KEY`) from a `.env` file |
| `llama_index.core.SimpleDirectoryReader` | Reads documents from a directory |
| `llama_index.core.VectorStoreIndex` | Builds and queries the in-memory vector index |
| `llama_index.core.Settings` | Global configuration object for LLM, embedding model, and chunking |
| `llama_index.llms.google_genai.GoogleGenAI` | Gemini language model integration (response generation) |
| `llama_index.embeddings.google_genai.GoogleGenAIEmbedding` | Gemini embedding model integration (vectorization) |
| `llama_index.readers.file.PDFReader` | PDF extraction reader (built on `pypdf`) |

## Component Breakdown

### 1. Environment Configuration & API Key Validation

- `load_dotenv()` reads key/value pairs from a `.env` file into the process environment.
- A fallback manual assignment (`os.environ["GOOGLE_API_KEY"]`) is provided but commented out.
- The script validates that `GOOGLE_API_KEY` is present. If missing, it raises a `ValueError` with a descriptive message, preventing execution against an unconfigured backend.

### 2. Global `Settings` Configuration

LlamaIndex uses a global `Settings` singleton to configure default components:

- **LLM:** `GoogleGenAI(model="gemini-3.1-pro-preview")` — the model responsible for generating the answer.
- **Embedding model:** `GoogleGenAIEmbedding(model_name="gemini-embedding-2")` — the model that turns text chunks into embedding vectors.
- **Chunking parameters:**
  - `Settings.chunk_size = 512` — target token count per chunk.
  - `Settings.chunk_overlap = 50` — overlapping tokens between adjacent chunks to preserve cross-boundary context.

### 3. Document Ingestion

Inside `main()`:

- A `file_extractor` mapping assigns `PDFReader` to the `.pdf` extension.
- `SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor).load_data()` reads all documents from `./data`.
- A set comprehension deduplicates and reports the loaded filenames, ensuring no PDF is silently dropped.
- If zero documents are loaded, the script raises a `RuntimeError` rather than building an empty index.

### 4. Vector Index Construction

- `VectorStoreIndex.from_documents(documents)` builds an in-memory vector index.
- Internally, this splits documents into chunks (guided by `Settings`), generates embeddings via the configured embedding model, and stores the resulting vectors.
- The index is **not persisted** — it is rebuilt from scratch on every execution.

### 5. Query Engine

- `index.as_query_engine(similarity_top_k=2)` creates a query engine that retrieves the top 2 most semantically similar chunks for a given question.
- The query itself is a hard-coded string (`"Balázs melyik évben állt munkába?"`). A second commented-out example question exists earlier in the file.

### 6. Response & Citation Reporting

- `query_engine.query(kerdes)` executes the full retrieval + generation flow.
- The generated answer is printed via `response.response`.
- For each source node, the script prints:
  - originating file name (`node.metadata["file_name"]`),
  - semantic relevance score (`node.score`, formatted to 4 decimals),
  - a truncated quoted snippet of the retrieved content (first ~180 characters, newlines collapsed).

## Pipeline Flow

```
+---------------------+
| Load .env / API key |
+---------------------+
           |
           v
+----------------------------+
| Configure Settings (LLM,   |
| embedding, chunking)       |
+----------------------------+
           |
           v
+----------------------------+
| Read documents from ./data |
| (PDFReader for .pdf)       |
+----------------------------+
           |
           v
+--------------------------------+
| VectorStoreIndex.from_documents|
|  - chunk (512 / overlap 50)   |
|  - embed (gemini-embedding-2) |
+--------------------------------+
           |
           v
+----------------------------+
| Query Engine (top_k = 2)   |
|  - retrieve relevant chunks|
|  - generate answer         |
|  - attach source nodes     |
+----------------------------+
           |
           v
+----------------------------+
| Print answer + citations   |
+----------------------------+
```

## Configuration Parameters

| Parameter | Value | Location | Description |
| --- | --- | --- | --- |
| LLM model | `gemini-3.1-pro-preview` | `Settings.llm` | Language model for answer generation |
| Embedding model | `gemini-embedding-2` | `Settings.embed_model` | Embedding model for vectorization |
| Chunk size | `512` | `Settings.chunk_size` | Tokens per text chunk |
| Chunk overlap | `50` | `Settings.chunk_overlap` | Overlap between chunks |
| Input directory | `./data` | `SimpleDirectoryReader(input_dir=...)` | Source document location |
| PDF extractor | `PDFReader()` | `file_extractor` | Handles `.pdf` files |
| Retrieval count | `2` | `similarity_top_k=2` | Number of source chunks retrieved |
| API key variable | `GOOGLE_API_KEY` | `load_dotenv()` / validation | Google API authentication |

## Code Structure Map

| Lines | Element | Responsibility |
| --- | --- | --- |
| 1–11 | Imports | Module imports for LlamaIndex, Gemini integrations, PDF reader, dotenv |
| 14 | `load_dotenv()` | Load `.env` environment variables |
| 16–23 | API key check | Validate presence of `GOOGLE_API_KEY`, raise `ValueError` if absent |
| 29 | `Settings.llm = GoogleGenAI(...)` | Configure Gemini LLM |
| 32 | `Settings.embed_model = GoogleGenAIEmbedding(...)` | Configure Gemini embedding model |
| 35–36 | `Settings.chunk_size` / `chunk_overlap` | Configure chunking |
| 39 | `def main():` | Entry point for the RAG pipeline |
| 42–45 | `SimpleDirectoryReader(...).load_data()` | Document ingestion |
| 49–50 | Filename set + log | Deduplicate and report loaded files |
| 52–56 | Empty check | `RuntimeError` if no documents loaded |
| 60 | `VectorStoreIndex.from_documents(documents)` | Build vector index |
| 63 | `index.as_query_engine(similarity_top_k=2)` | Create retrieval query engine |
| 67 | `kerdes = "Balázs ..."` | Hard-coded query |
| 71 | `query_engine.query(kerdes)` | Execute retrieval + generation |
| 75 | `print(response.response)` | Print generated answer |
| 81–87 | Source loop | Print source filename, score, and snippet |
| 90–91 | `if __name__ == "__main__": main()` | Script entry guard |

## Technical Observations & Limitations

- **Hard-coded query:** The question is embedded directly in the source code; there is no CLI argument, interactive input, or API surface for dynamic queries. The commented-out example question suggests this is intentional for demonstration.
- **No persistence:** The vector index lives only in memory. Each run re-reads, re-chunks, and re-embeds all documents, which is inefficient for large or frequently queried corpora. LlamaIndex supports `StorageContext` persistence, which is not used here.
- **Chunking redundancy:** `chunk_size` / `chunk_overlap` are set on `Settings`, but `VectorStoreIndex.from_documents` performs its own chunking. The `Settings` values serve as defaults; an explicit transformation would provide finer control.
- **Model-name verification:** The configured models (`gemini-3.1-pro-preview` and `gemini-embedding-2`) should be verified against Google's currently available Gemini model identifiers. Mismatched or unavailable model names will cause runtime failures.
- **Citation format:** The source-reporting block mimics notebook-style citation logic by surfacing filenames, relevance scores, and quoted snippets, but the score is the raw node score and is not normalized for cross-run comparison.
- **Single-threaded, synchronous execution:** The entire pipeline is synchronous and runs sequentially, appropriate for a minimal demonstration but not for high-throughput workloads.
- **Limited file-type coverage:** Only `.pdf` files are explicitly mapped to a custom reader; other supported formats (if any) rely on `SimpleDirectoryReader` defaults.