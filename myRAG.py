import os
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex modulok importálása
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

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


# 3. Minta adatok előkészítése (hogy a script tesztelhető legyen)
def setup_sample_data():
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    sample_file = data_dir / "projekt_specifikacio.txt"
    if not sample_file.exists():
        sample_file.write_text(
            """
=== Autóipari Szoftverprojekt Specifikáció 2026 ===
Dokumentumazonosító: SPEC-2026-AUT-01
Szerző: Kovács Péter (Vezető Szoftverfejlesztő)

1. Biztonsági és Minőségi Szabványok:
Az elektronikus fékrendszer (EBS) vezérlőszoftvere szigorúan az ISO 26262 ASIL-D előírásoknak megfelelően készül.
A diagnosztikai teszteléshez Vector CANoe restbus szimulációt használunk.

2. Architekturális Döntések:
A mikrokontroller kommunikációja CAN-FD buszon történik.
A belső fejlesztési felületen Python-alapú automatizált tesztscriptek futnak.

3. Projekt MÉRLEGEk ÉS HATÁRIDŐK:
A fázis 1 lezárása 2026 harmadik negyedévében (Q3) esedékes.
A projekt teljes költségvetési kerete 65 millió HUF.
""",
            encoding="utf-8",
        )
        print(f"📄 Minta fájl létrehozva: {sample_file}")


# 4. A RAG Pipeline futtatása
def main():
    setup_sample_data()

    # Dokumentumok beolvasása a ./data mappából
    print("\n📚 Dokumentumok beolvasása a ./data mappából...")
    documents = SimpleDirectoryReader(input_dir="./data").load_data()

    # Vektorindex építése (a háttérben lefut a chunking + Gemini embedding)
    print("⚡ Vektorindex építése...")
    index = VectorStoreIndex.from_documents(documents)

    # Query Engine példányosítása (a top 2 legrelevánsabb találatot kérjük le)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # Kérdés feltevése
    kerdes = "Milyen szabványnak felel meg a fékrendszer és mekkora a projekt büdzséje?"
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