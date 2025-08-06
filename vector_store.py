"""
Vector Store Manager
Gestisce la generazione di embeddings e il salvataggio/recupero da ChromaDB
"""

import os
import json
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# ChromaDB e embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Gestisce il vector store con ChromaDB e gli embeddings
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "rimini_docs"):
        
        self.embedding_model_name = embedding_model
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Crea directory se non esiste
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Inizializza modello di embedding
        logger.info(f"Caricamento modello di embedding: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Dimensione embeddings: {self.embedding_dimension}")
        
        # Inizializza ChromaDB con persistenza
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Crea o carica collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Configura la collection in ChromaDB"""
        try:
            # Prova a ottenere collection esistente
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Collection '{self.collection_name}' caricata")
            logger.info(f"Documenti esistenti: {self.collection.count()}")
        except:
            # Crea nuova collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Usa cosine similarity
            )
            logger.info(f"Nuova collection '{self.collection_name}' creata")
    
    def reset_collection(self):
        """Resetta la collection (elimina tutti i dati)"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' eliminata")
        except:
            pass
        
        # Ricrea collection
        self._setup_collection()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings per una lista di testi
        
        Args:
            texts: lista di testi da embedare
            batch_size: dimensione batch per processing
            
        Returns:
            Array numpy con embeddings
        """
        logger.info(f"Generazione embeddings per {len(texts)} testi...")
        
        embeddings = []
        
        # Processa in batch con progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        # Concatena tutti gli embeddings
        all_embeddings = np.vstack(embeddings)
        
        logger.info(f"Embeddings generati: shape {all_embeddings.shape}")
        return all_embeddings
    
    def add_documents(self, 
                     texts: List[str],
                     metadatas: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None,
                     batch_size: int = 32) -> int:
        """
        Aggiunge documenti al vector store
        
        Args:
            texts: lista di testi
            metadatas: lista di metadata per ogni testo
            ids: lista di ID unici (opzionale)
            batch_size: dimensione batch per embeddings
            
        Returns:
            Numero di documenti aggiunti
        """
        if not texts:
            logger.warning("Nessun testo da aggiungere")
            return 0
        
        # Genera IDs se non forniti
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Genera embeddings
        embeddings = self.generate_embeddings(texts, batch_size)
        
        # Prepara metadata (ChromaDB richiede valori serializzabili)
        clean_metadatas = []
        for metadata in metadatas:
            clean_meta = {}
            for key, value in metadata.items():
                # Converti valori non serializzabili in stringhe
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            clean_metadatas.append(clean_meta)
        
        # Aggiungi a ChromaDB in batch
        logger.info(f"Aggiunta di {len(texts)} documenti a ChromaDB...")
        
        # ChromaDB ha un limite di batch, dividi se necessario
        chroma_batch_size = 100
        
        for i in tqdm(range(0, len(texts), chroma_batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + chroma_batch_size, len(texts))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=clean_metadatas[i:end_idx]
            )
        
        logger.info(f"✅ Aggiunti {len(texts)} documenti al vector store")
        logger.info(f"Totale documenti nel database: {self.collection.count()}")
        
        return len(texts)
    
    def search(self, 
              query: str,
              n_results: int = 5,
              filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cerca documenti simili alla query
        
        Args:
            query: testo della query
            n_results: numero di risultati da restituire
            filter_dict: filtri opzionali sui metadata
            
        Returns:
            Dizionario con risultati e metadata
        """
        # Genera embedding della query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cerca in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict if filter_dict else None
        )
        
        # Formatta risultati
        formatted_results = {
            'query': query,
            'results': []
        }
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                formatted_results['results'].append(result)
        
        return formatted_results
    
    def search_with_score(self, 
                         query: str,
                         n_results: int = 5,
                         score_threshold: float = 0.3) -> List[Tuple[str, Dict, float]]:
        """
        Cerca documenti con score di similarità
        
        Args:
            query: testo della query
            n_results: numero massimo di risultati
            score_threshold: soglia minima di similarità (0-1)
            
        Returns:
            Lista di tuple (testo, metadata, score)
        """
        results = self.search(query, n_results)
        
        filtered_results = []
        for result in results['results']:
            # ChromaDB restituisce distanza, convertiamo in similarità
            # Per cosine distance: similarity = 1 - distance
            if result['distance'] is not None:
                similarity = 1 - result['distance']
                
                if similarity >= score_threshold:
                    filtered_results.append((
                        result['text'],
                        result['metadata'],
                        similarity
                    ))
        
        return filtered_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Ottiene statistiche sul vector store"""
        count = self.collection.count()
        
        # Ottieni un campione per statistiche
        sample = self.collection.get(limit=min(100, count))
        
        stats = {
            'total_documents': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dimension,
            'persist_directory': str(self.persist_directory)
        }
        
        # Analizza metadata se disponibili
        if sample['metadatas']:
            metadata_keys = set()
            for meta in sample['metadatas']:
                metadata_keys.update(meta.keys())
            stats['metadata_fields'] = list(metadata_keys)
        
        return stats
    
    def export_to_file(self, output_path: str = "vector_store_export.json"):
        """Esporta tutto il contenuto del vector store in un file"""
        all_data = self.collection.get()
        
        export_data = {
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
            'timestamp': datetime.now().isoformat(),
            'total_documents': self.collection.count(),
            'documents': []
        }
        
        for i in range(len(all_data['ids'])):
            doc = {
                'id': all_data['ids'][i],
                'text': all_data['documents'][i] if all_data['documents'] else None,
                'metadata': all_data['metadatas'][i] if all_data['metadatas'] else {}
            }
            export_data['documents'].append(doc)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vector store esportato in: {output_path}")
        return output_path


class ChunkIndexer:
    """
    Classe helper per indicizzare i chunk creati dalla pipeline
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
    
    def index_from_pipeline_output(self, 
                                  chunks_file: str = "indexed_data/chunks.pkl",
                                  reset_store: bool = False) -> int:
        """
        Indicizza i chunk salvati dalla pipeline
        
        Args:
            chunks_file: percorso al file pickle con i chunk
            reset_store: se True, resetta il database prima di indicizzare
            
        Returns:
            Numero di chunk indicizzati
        """
        # Carica chunk
        logger.info(f"Caricamento chunk da: {chunks_file}")
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        
        logger.info(f"Caricati {len(chunks)} chunk")
        
        # Reset se richiesto
        if reset_store:
            logger.info("Reset del vector store...")
            self.vector_store.reset_collection()
        
        # Prepara dati per indicizzazione
        texts = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            texts.append(chunk.content)
            
            # Prepara metadata
            metadata = chunk.metadata.copy()
            metadata['char_count'] = chunk.char_count
            metadata['token_count'] = chunk.token_count
            metadatas.append(metadata)
            
            # Crea ID unico con UUID per garantire unicità
            chunk_id = f"{metadata.get('file_hash', 'unknown')[:8]}_{chunk.chunk_index}_{str(uuid.uuid4())[:8]}"
            ids.append(chunk_id)
        
        # Indicizza
        count = self.vector_store.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=32
        )
        
        return count


# Script di esempio
if __name__ == "__main__":
    # Inizializza vector store
    vector_store = VectorStoreManager(
        embedding_model="sentence-transformers/paraphrase-mpnet-base-v2",
        persist_directory="./chroma_db",
        collection_name="rimini_knowledge_base"
    )
    
    # Indicizza i chunk dalla pipeline
    indexer = ChunkIndexer(vector_store)
    
    # Indicizza (reset=True per pulire database precedente)
    num_indexed = indexer.index_from_pipeline_output(
        chunks_file="indexed_data/chunks.pkl",
        reset_store=True  # Pulisce database precedente
    )
    
    print(f"\n✅ Indicizzati {num_indexed} chunk nel vector database")
    
    # Mostra statistiche
    stats = vector_store.get_statistics()
    print("\n📊 Statistiche Vector Store:")
    print(json.dumps(stats, indent=2))
    
    # Test di ricerca
    print("\n🔍 Test di ricerca:")
    test_queries = [
        "Qual è la popolazione di Rimini?",
        "Storia dei Malatesta",
        "Spiagge e turismo balneare",
        "Federico Fellini"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = vector_store.search_with_score(query, n_results=3, score_threshold=0.2)
        
        for i, (text, metadata, score) in enumerate(results, 1):
            print(f"\n  Risultato {i} (score: {score:.3f}):")
            print(f"  Chunk: {metadata.get('chunk_index', 'N/A')}")
            print(f"  Testo: {text[:150]}...")