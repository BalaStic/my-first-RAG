import os
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex modulok importálása
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# PDF beolvasáshoz szükséges reader (pypdf alapú)
from llama_index.readers.file import PDFReader

# 1. Környezeti változók betöltése (.env fájlból)
load_dotenv()

# HA nem használsz .env fájlt, ide írd be a kulcsodat:
# os.environ["GOOGLE_API_KEY"] = "AIzaSy_A_TE_API_KULCSOD"

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError(
        "❌ A GOOGLE_API_KEY nincs beállítva! "
        "Állítsd be a .env fájlban vagy a kód elején az os.environ['GOOGLE_API_KEY'] értékét."
    )

# 2. Gemini LLM és Embedding beállítása
print("🤖 Gemini modellek konfigurálása...")

# LLM a válaszadáshoz (Gemini 1.5 Pro vagy Flash)
Settings.llm = GoogleGenAI(model="gemini-3.1-pro-preview")

# Embedding modell a vektoros kereséshez (Google hivatalos embedding modellje)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")

# Chunking (darabolási) beállítások
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 3. A RAG Pipeline futtatása
def main():
    # Dokumentumok beolvasása a ./data mappából
    print("\n📚 Dokumentumok beolvasása a ./data mappából...")
    file_extractor = {".pdf": PDFReader()}
    documents = SimpleDirectoryReader(
        input_dir="./data", file_extractor=file_extractor
    ).load_data()

    # Ellenőrzés, hogy minden fájl valóban be legyen olvasva
    # (így nem vesz el csendben egy PDF-et, ha valami rosszul menne)
    beolvasott_fajlok = {d.metadata.get("file_name") for d in documents}
    print(f"   Beolvasott dokumentumok ({len(documents)} db): {beolvasott_fajlok}")

    if len(documents) == 0:
        raise RuntimeError(
            "❌ Egyetlen dokumentum sem lett beolvasva a ./data mappából! "
            "Ellenőrizd, hogy a mappa létezik és nem üres."
        )

    # Vektorindex építése (a háttérben lefut a chunking + Gemini embedding)
    print("⚡ Vektorindex építése...")
    index = VectorStoreIndex.from_documents(documents)

    # Query Engine példányosítása (a top 2 legrelevánsabb találatot kérjük le)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # Kérdés feltevése
    # kerdes = "Milyen szabványnak felel meg a fékrendszer és mekkora a projekt büdzséje?"
    kerdes = "Balázs melyik évben állt munkába?"
    print(f"\n❓ Kérdés: {kerdes}\n")

    # Válasz generálása
    response = query_engine.query(kerdes)

    # 5. Eredmények kiíratása
    print("💡 Válasz:")
    print(response.response)

    print("\n" + "=" * 60)
    print("📌 FORRÁSOK ÉS IDÉZETEK (NotebookLM hivatkozási logika):")
    print("=" * 60)

    for i, node in enumerate(response.source_nodes, 1):
        file_name = node.metadata.get("file_name", "Ismeretlen fájl")
        score = f"{node.score:.4f}" if node.score is not None else "N/A"
        content_snippet = node.node.get_content().strip().replace("\n", " ")

        print(f"\n[{i}] Fájl: {file_name} (Szemantikai relevancia: {score})")
        print(f"    Beidézett részlet: \"{content_snippet[:180]}...\"")


if __name__ == "__main__":
    main()