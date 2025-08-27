"""
Script di Setup e Verifica del Sistema RAG
Automatizza l'intero processo di setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_requirements():
    """Verifica che tutte le dipendenze siano installate"""
    logger.info("\n🔍 Verifica dipendenze...")
    
    required_modules = [
        'pypdf',
        'chardet',
        'sentence_transformers',
        'chromadb',
        'numpy',
        'tqdm',
        'requests'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"  ✅ {module}")
        except ImportError:
            missing.append(module)
            logger.error(f"  ❌ {module}")
    
    if missing:
        logger.error(f"\n⚠️ Moduli mancanti: {', '.join(missing)}")
        logger.info("Installa con: pip install -r requirements.txt")
        return False
    
    return True


def check_documents():
    """Verifica presenza documenti"""
    logger.info("\n📁 Verifica documenti...")
    
    docs_dir = Path("documents")
    if not docs_dir.exists():
        logger.error("  ❌ Directory 'documents' non trovata")
        logger.info("  Crea la directory e inserisci il PDF di Rimini")
        return False
    
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("  ❌ Nessun PDF trovato in 'documents'")
        return False
    
    logger.info(f"  ✅ Trovati {len(pdf_files)} file PDF:")
    for pdf in pdf_files:
        logger.info(f"     - {pdf.name}")
    
    return True


def check_ollama():
    """Verifica che Ollama sia installato e in esecuzione"""
    logger.info("\n🤖 Verifica Ollama...")
    
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"  ✅ Ollama in esecuzione")
            
            if models:
                logger.info(f"  📦 Modelli disponibili:")
                for model in models:
                    logger.info(f"     - {model['name']}")
                
                # Cerca DeepSeek
                if any('mistral:7b' in m['name'].lower() for m in models):
                    logger.info(f"  ✅ DeepSeek trovato")
                else:
                    logger.warning(f"  ⚠️ DeepSeek non trovato")
                    logger.info("  Installa con: ollama pull deepseek")
            else:
                logger.warning("  ⚠️ Nessun modello installato")
                logger.info("  Installa DeepSeek con: ollama pull deepseek")
            
            return True
        else:
            logger.error(f"  ❌ Ollama risponde ma con errore: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("  ❌ Ollama non raggiungibile")
        logger.info("  Avvia Ollama con: ollama serve")
        return False
    except Exception as e:
        logger.error(f"  ❌ Errore: {e}")
        return False


def run_indexing():
    """Esegue l'indicizzazione dei documenti"""
    logger.info("\n📚 Indicizzazione documenti...")
    
    # Verifica se già indicizzato
    indexed_data = Path("indexed_data")
    if indexed_data.exists() and (indexed_data / "chunks.pkl").exists():
        logger.info("  ℹ️ Documenti già indicizzati")
        
        # Carica statistiche
        metadata_file = indexed_data / "indexing_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                stats = metadata.get('statistics', {})
                logger.info(f"  📊 Chunk totali: {stats.get('total_chunks', 'N/A')}")
        
        response = input("\n  Vuoi re-indicizzare? (s/N): ").strip().lower()
        if response != 's':
            return True
    
    # Esegui indicizzazione
    logger.info("\n  ▶️ Avvio indicizzazione...")
    try:
        from indexing_pipeline import IndexingPipeline
        
        pipeline = IndexingPipeline(
            documents_dir="./PreProcessing_scripts/processed",
            output_dir="./indexed_data",
            embedding_model="sentence-transformers/paraphrase-mpnet-base-v2"
        )
        
        results = pipeline.run_pipeline(
            chunking_strategy='hierarchical',
            process_limit=None
        )
        
        logger.info(f"\n  ✅ Indicizzazione completata!")
        logger.info(f"     - Documenti: {results['total_documents']}")
        logger.info(f"     - Chunk: {results['total_chunks']}")
        logger.info(f"     - Token medi: {results['chunk_statistics']['avg_tokens']:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Errore nell'indicizzazione: {e}")
        return False


def create_vector_store():
    """Crea/aggiorna il vector store"""
    logger.info("\n🗃️ Creazione Vector Store...")
    
    # Verifica se esiste
    chroma_db = Path("chroma_db")
    if chroma_db.exists():
        logger.info("  ℹ️ Vector store già esistente")
        response = input("\n  Vuoi ricreare il database? (s/N): ").strip().lower()
        if response != 's':
            # Verifica solo il contenuto
            try:
                from vector_store import VectorStoreManager
                vs = VectorStoreManager(persist_directory="./chroma_db")
                stats = vs.get_statistics()
                logger.info(f"  📊 Documenti nel database: {stats['total_documents']}")
                return True
            except Exception as e:
                logger.error(f"  ⚠️ Errore nel verificare il database: {e}")
    
    # Crea vector store
    logger.info("\n  ▶️ Creazione vector store...")
    try:
        from vector_store import VectorStoreManager, ChunkIndexer
        
        # Inizializza vector store
        vector_store = VectorStoreManager(
            embedding_model="sentence-transformers/paraphrase-mpnet-base-v2",
            persist_directory="./chroma_db",
            collection_name="rimini_knowledge_base"
        )
        
        # Indicizza chunk
        indexer = ChunkIndexer(vector_store)
        num_indexed = indexer.index_from_pipeline_output(
            chunks_file="indexed_data/chunks.pkl",
            reset_store=True
        )
        
        logger.info(f"\n  ✅ Vector store creato!")
        logger.info(f"     - Documenti indicizzati: {num_indexed}")
        
        # Test di ricerca
        logger.info("\n  🔍 Test di ricerca...")
        results = vector_store.search("popolazione di Rimini", n_results=1)
        if results['results']:
            logger.info("  ✅ Ricerca funzionante")
        else:
            logger.warning("  ⚠️ Nessun risultato nel test")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Errore nella creazione del vector store: {e}")
        return False


def test_rag_system():
    """Test del sistema RAG completo"""
    logger.info("\n🧪 Test Sistema RAG...")
    
    try:
        from vector_store import VectorStoreManager
        from rag_system import RAGSystem, OllamaClient
        
        # Inizializza componenti
        vector_store = VectorStoreManager(
            persist_directory="./chroma_db",
            collection_name="rimini_knowledge_base"
        )
        
        llm_client = OllamaClient(
            model="mistral:7b",
            temperature=0.7
        )
        
        rag_system = RAGSystem(
            vector_store=vector_store,
            llm_client=llm_client
        )
        
        # Test con domanda semplice
        test_question = "Qual è la popolazione di Rimini?"
        logger.info(f"\n  ❓ Domanda test: {test_question}")
        
        response = rag_system.answer_question(
            test_question,
            include_sources=False,
            verbose=False
        )
        
        if response.answer:
            logger.info(f"  💡 Risposta: {response.answer[:200]}...")
            logger.info(f"  📊 Confidence: {response.confidence:.2%}")
            logger.info("\n  ✅ Sistema RAG funzionante!")
            return True
        else:
            logger.error("  ❌ Nessuna risposta generata")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ Errore nel test RAG: {e}")
        return False


def main():
    """Funzione principale di setup"""
    print("\n" + "="*60)
    print("🚀 SETUP SISTEMA RAG")
    print("="*60)
    
    # Step 1: Verifica dipendenze
    if not check_requirements():
        logger.error("\n❌ Setup interrotto: dipendenze mancanti")
        return
    
    # Step 2: Verifica documenti
    if not check_documents():
        logger.error("\n❌ Setup interrotto: documenti mancanti")
        return
    
    # Step 3: Verifica Ollama
    ollama_ok = check_ollama()
    if not ollama_ok:
        logger.warning("\n⚠️ Ollama non disponibile - il sistema funzionerà solo per indicizzazione")
        response = input("\nVuoi continuare comunque? (s/N): ").strip().lower()
        if response != 's':
            return
    
    # Step 4: Indicizzazione
    if not run_indexing():
        logger.error("\n❌ Setup interrotto: errore nell'indicizzazione")
        return
    
    # Step 5: Vector Store
    if not create_vector_store():
        logger.error("\n❌ Setup interrotto: errore nel vector store")
        return
    
    # Step 6: Test completo (solo se Ollama disponibile)
    if ollama_ok:
        if test_rag_system():
            print("\n" + "="*60)
            print("✅ SETUP COMPLETATO CON SUCCESSO!")
            print("="*60)
            print("\n📝 Prossimi passi:")
            print("1. Avvia il sistema RAG: python rag_system.py")
            print("2. Fai le tue domande sui documenti di Rimini!")
            print("\n💡 Suggerimenti:")
            print("- Aggiungi altri documenti nella cartella 'documents'")
            print("- Modifica i parametri in rag_system.py per ottimizzare")
            print("- Prova diversi modelli Ollama (llama2, mistral, etc.)")
        else:
            logger.warning("\n⚠️ Setup completato ma test RAG fallito")
            logger.info("Verifica che Ollama sia configurato correttamente")
    else:
        print("\n" + "="*60)
        print("✅ SETUP PARZIALE COMPLETATO")
        print("="*60)
        print("\n📝 Sistema pronto per indicizzazione e ricerca")
        print("Per abilitare la generazione di risposte:")
        print("1. Installa Ollama: https://ollama.ai")
        print("2. Avvia Ollama: ollama serve")
        print("3. Installa DeepSeek: ollama pull mistral:7b")
        print("4. Rilancia questo script")


if __name__ == "__main__":
    main()