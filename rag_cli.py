import sys
import logging
from datetime import datetime
from vector_store import VectorStoreManager
from rag_system import OllamaClient, RAGSystem

# Configura logging di base
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Domande di test predefinite (opzionali)
TEST_QUESTIONS = [
    "Qual è la popolazione di Rimini?",
    "Chi era Sigismondo Pandolfo Malatesta?",
    "Quali sono le principali attrazioni turistiche di Rimini?",
    "Quando è stato costruito il Ponte di Tiberio?",
    "Quali film ha girato Federico Fellini su Rimini?"
]

def main():
    # Inizializza Vector Store
    vector_store = VectorStoreManager(
        embedding_model="sentence-transformers/paraphrase-mpnet-base-v2",
        persist_directory="./chroma_db",
        collection_name="rimini_knowledge_base",
    )
    stats = vector_store.get_statistics()
    if stats['total_documents'] == 0:
        logger.error("Nessun documento indicizzato. Esegui prima l'indicizzazione dei documenti.")
        sys.exit(1)

    # Inizializza Ollama client con modello fisso
    llm_client = OllamaClient(
        base_url="http://localhost:11434",
        model="mistral:7b",
        temperature=0.7
    )

    # Costruisci RAGSystem
    rag = RAGSystem(
        vector_store=vector_store,
        llm_client=llm_client,
        retrieval_k=10,
        rerank_k=3
    )

    # Modalità interattiva
    print("Sistema RAG attivo. Digita 'exit' per uscire.")
    while True:
        try:
            q = input("❓ Domanda: ").strip()
            if q.lower() in ['exit', 'quit']:
                print("👋 Arrivederci!")
                break
            if not q:
                continue
            start = datetime.now()
            resp = rag.answer_question(q, include_sources=True)
            elapsed = (datetime.now() - start).total_seconds()
            print(f"\n💡 Risposta (Confidence: {resp.confidence:.2%}, Tempo: {elapsed:.2f}s):")
            print(resp.answer)
            print("\n" + "-"*60)
        except KeyboardInterrupt:
            print("\n👋 Interrotto dall'utente")
            break
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione: {e}")

if __name__ == '__main__':
    main()
