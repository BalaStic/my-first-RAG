import os
from pathlib import Path
from dotenv import load_dotenv

# Import LlamaIndex modules
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Reader required for PDF loading (based on pypdf)
from llama_index.readers.file import PDFReader

# 1. Load environment variables (from the .env file)
load_dotenv()

# If you are not using a .env file, enter your key here:
# os.environ["GOOGLE_API_KEY"] = "AIzaSy_YOUR_API_KEY"

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError(
        "❌ GOOGLE_API_KEY is not set! "
        "Set it in the .env file or assign os.environ['GOOGLE_API_KEY'] near the beginning of this script."
    )

# 2. Configure Gemini LLM and embeddings
print("🤖 Configuring Gemini models...")

# LLM for generating responses (Gemini 1.5 Pro or Flash)
Settings.llm = GoogleGenAI(model="gemini-3.1-pro-preview")

# Embedding model for vector search (Google's official embedding model)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")

# Chunking settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 3. Run the RAG pipeline
def main():
    # Load documents from the ./data directory
    print("\n📚 Loading documents from the ./data directory...")
    file_extractor = {".pdf": PDFReader()}
    documents = SimpleDirectoryReader(
        input_dir="./data", file_extractor=file_extractor
    ).load_data()

    # Verify that every file was actually loaded
    # (so a PDF is not silently skipped if something goes wrong)
    loaded_files = {d.metadata.get("file_name") for d in documents}
    print(f"   Loaded documents ({len(documents)}): {loaded_files}")

    if len(documents) == 0:
        raise RuntimeError(
            "❌ No documents were loaded from the ./data directory! "
            "Check that the directory exists and is not empty."
        )

    # Build the vector index (chunking and Gemini embedding run in the background)
    print("⚡ Building the vector index...")
    index = VectorStoreIndex.from_documents(documents)

    # Instantiate the query engine (retrieve the top 2 most relevant results)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # Ask a question
    # question = "Which standard does the braking system comply with, and what is the project budget?"
    question = "In which year did Balázs start working?"
    print(f"\n❓ Question: {question}\n")

    # Generate a response
    response = query_engine.query(question)

    # 5. Print results
    print("💡 Answer:")
    print(response.response)

    print("\n" + "=" * 60)
    print("📌 SOURCES AND CITATIONS (NotebookLM citation logic):")
    print("=" * 60)

    for i, node in enumerate(response.source_nodes, 1):
        file_name = node.metadata.get("file_name", "Unknown file")
        score = f"{node.score:.4f}" if node.score is not None else "N/A"
        content_snippet = node.node.get_content().strip().replace("\n", " ")

        print(f"\n[{i}] File: {file_name} (Semantic relevance: {score})")
        print(f"    Quoted excerpt: \"{content_snippet[:180]}...\"")


if __name__ == "__main__":
    main()