"""
Pipeline completa di indicizzazione documenti
Orchestrazione del processo di scanning, loading, processing e chunking
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from document_loader import DocumentScanner, DocumentLoader, Document
from text_processor import TextPreprocessor, TextChunker, TextChunk

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Pipeline completa per l'indicizzazione di documenti"""
    
    def __init__(self, 
                 documents_dir: str,
                 output_dir: str = "./indexed_data",
                 embedding_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"):
        
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        
        # Crea directory output se non esiste
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza componenti
        self.scanner = DocumentScanner(documents_dir)
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor(preserve_structure=True)
        self.chunker = TextChunker(
            model_name=embedding_model,
            target_chunk_size=350,  # Ottimale per paraphrase-mpnet-base-v2
            max_chunk_size=450,
            min_chunk_size=100,
            overlap_tokens=50
        )
        
        # Storage per risultati
        self.manifest = []
        self.documents = []
        self.all_chunks = []
        self.index_metadata = {}
    
    def run_pipeline(self, 
                    chunking_strategy: str = 'semantic',
                    process_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Esegue la pipeline completa di indicizzazione
        
        Args:
            chunking_strategy: strategia di chunking ('semantic', 'hierarchical', 'sliding')
            process_limit: numero massimo di documenti da processare (None = tutti)
        
        Returns:
            Dizionario con statistiche e risultati
        """
        logger.info("=" * 60)
        logger.info("INIZIO PIPELINE DI INDICIZZAZIONE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Scanning
        logger.info("\n📁 STEP 1: Scanning della directory...")
        self.manifest = self.scanner.scan_directory(recursive=True)
        logger.info(f"   Trovati {len(self.manifest)} documenti")
        
        # Salva manifest
        self._save_manifest()
        
        # Limita documenti se richiesto
        docs_to_process = self.manifest[:process_limit] if process_limit else self.manifest
        
        # Step 2: Loading e Processing
        logger.info(f"\n📄 STEP 2: Caricamento e processing di {len(docs_to_process)} documenti...")
        
        for i, file_info in enumerate(docs_to_process, 1):
            logger.info(f"\n   [{i}/{len(docs_to_process)}] Processing: {file_info['name']}")
            
            try:
                # Carica documento
                document = self.loader.load_document(file_info)
                self.documents.append(document)
                
                # Preprocessa testo
                cleaned_text = self.preprocessor.preprocess(
                    document.content,
                    doc_type=document.metadata.file_type
                )
                
                # Crea metadata per i chunk
                chunk_metadata = {
                    'source_file': document.metadata.file_path,
                    'file_name': document.metadata.file_name,
                    'file_type': document.metadata.file_type,
                    'file_hash': document.metadata.file_hash,
                    'priority': document.metadata.priority
                }
                
                # Step 3: Chunking
                logger.info(f"   ✂️  Chunking con strategia: {chunking_strategy}")
                chunks = self.chunker.chunk_document(
                    cleaned_text,
                    chunk_metadata,
                    strategy=chunking_strategy
                )
                
                # Aggiungi riferimento al documento originale
                for chunk in chunks:
                    chunk.metadata['doc_index'] = i - 1
                
                self.all_chunks.extend(chunks)
                
                # Statistiche documento
                logger.info(f"   ✅ Creati {len(chunks)} chunk")
                logger.info(f"      Token medi: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
                
            except Exception as e:
                logger.error(f"   ❌ Errore nel processare {file_info['name']}: {e}")
                continue
        
        # Step 4: Validazione e statistiche
        logger.info("\n📊 STEP 4: Validazione e statistiche finali...")
        stats = self._calculate_statistics()
        
        # Step 5: Salvataggio risultati
        logger.info("\n💾 STEP 5: Salvataggio risultati...")
        self._save_results()
        
        # Tempo totale
        elapsed_time = (datetime.now() - start_time).total_seconds()
        stats['processing_time_seconds'] = elapsed_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETATA")
        logger.info(f"Tempo totale: {elapsed_time:.2f} secondi")
        logger.info(f"Documenti processati: {len(self.documents)}")
        logger.info(f"Chunk totali creati: {len(self.all_chunks)}")
        logger.info("=" * 60)
        
        return stats
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche sull'indicizzazione"""
        if not self.all_chunks:
            return {}
        
        import numpy as np
        
        token_counts = [c.token_count for c in self.all_chunks]
        char_counts = [c.char_count for c in self.all_chunks]
        
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': len(self.all_chunks),
            'total_characters': sum(char_counts),
            'total_tokens_estimated': sum(token_counts),
            'avg_chunks_per_doc': len(self.all_chunks) / len(self.documents) if self.documents else 0,
            'chunk_statistics': {
                'avg_tokens': np.mean(token_counts),
                'std_tokens': np.std(token_counts),
                'min_tokens': min(token_counts),
                'max_tokens': max(token_counts),
                'median_tokens': np.median(token_counts),
                'avg_chars': np.mean(char_counts),
                'chunks_over_512': sum(1 for t in token_counts if t > 512),
                'chunks_under_100': sum(1 for t in token_counts if t < 100)
            },
            'document_types': self._count_document_types(),
            'priority_distribution': self._count_priorities()
        }
        
        # Avvisi se ci sono problemi
        if stats['chunk_statistics']['chunks_over_512'] > 0:
            logger.warning(f"⚠️  {stats['chunk_statistics']['chunks_over_512']} chunk superano il limite di 512 token!")
        
        return stats
    
    def _count_document_types(self) -> Dict[str, int]:
        """Conta documenti per tipo"""
        types = {}
        for doc in self.documents:
            doc_type = doc.metadata.file_type
            types[doc_type] = types.get(doc_type, 0) + 1
        return types
    
    def _count_priorities(self) -> Dict[str, int]:
        """Conta documenti per priorità"""
        priorities = {}
        for doc in self.documents:
            priority = doc.metadata.priority
            priorities[priority] = priorities.get(priority, 0) + 1
        return priorities
    
    def _save_manifest(self):
        """Salva il manifest dei documenti"""
        manifest_path = self.output_dir / "document_manifest.json"
        
        # Converti datetime per JSON
        manifest_serializable = []
        for item in self.manifest:
            item_copy = item.copy()
            if isinstance(item_copy.get('created'), datetime):
                item_copy['created'] = item_copy['created'].isoformat()
            if isinstance(item_copy.get('modified'), datetime):
                item_copy['modified'] = item_copy['modified'].isoformat()
            manifest_serializable.append(item_copy)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Manifest salvato: {manifest_path}")
    
    def _save_results(self):
        """Salva tutti i risultati dell'indicizzazione"""
        # 1. Salva chunks come JSON (per debugging/ispezione)
        chunks_json_path = self.output_dir / "chunks.json"
        chunks_data = []
        
        for chunk in self.all_chunks:
            chunks_data.append({
                'chunk_index': chunk.chunk_index,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'char_count': chunk.char_count,
                'token_count': chunk.token_count
            })
        
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Chunks salvati (JSON): {chunks_json_path}")
        
        # 2. Salva chunks come pickle (per uso programmatico)
        chunks_pickle_path = self.output_dir / "chunks.pkl"
        with open(chunks_pickle_path, 'wb') as f:
            pickle.dump(self.all_chunks, f)
        
        logger.info(f"   Chunks salvati (pickle): {chunks_pickle_path}")
        
        # 3. Salva metadata dell'indicizzazione
        metadata_path = self.output_dir / "indexing_metadata.json"
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'embedding_model': self.embedding_model,
            'documents_dir': str(self.documents_dir),
            'statistics': self._calculate_statistics(),
            'chunker_config': {
                'target_chunk_size': self.chunker.target_chunk_size,
                'max_chunk_size': self.chunker.max_chunk_size,
                'min_chunk_size': self.chunker.min_chunk_size,
                'overlap_tokens': self.chunker.overlap_tokens
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Metadata salvati: {metadata_path}")
        
        # 4. Crea file di testo leggibile con esempi
        examples_path = self.output_dir / "chunk_examples.txt"
        with open(examples_path, 'w', encoding='utf-8') as f:
            f.write("ESEMPI DI CHUNK CREATI\n")
            f.write("=" * 80 + "\n\n")
            
            # Mostra primi 5 e ultimi 5 chunk
            example_chunks = self.all_chunks[:5] + self.all_chunks[-5:] if len(self.all_chunks) > 10 else self.all_chunks
            
            for chunk in example_chunks:
                f.write(f"Chunk #{chunk.chunk_index}\n")
                f.write(f"File: {chunk.metadata.get('file_name', 'Unknown')}\n")
                f.write(f"Token: {chunk.token_count} | Caratteri: {chunk.char_count}\n")
                f.write("-" * 40 + "\n")
                f.write(chunk.content[:500] + "...\n" if len(chunk.content) > 500 else chunk.content + "\n")
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"   Esempi salvati: {examples_path}")
    
    def get_chunks_for_embedding(self) -> List[str]:
        """
        Ritorna lista di testi pronti per l'embedding
        """
        return [chunk.content for chunk in self.all_chunks]
    
    def get_chunks_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Ritorna chunks con metadata per il vector database
        """
        return [
            {
                'id': f"{chunk.metadata['file_hash'][:8]}_{chunk.chunk_index}",
                'text': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in self.all_chunks
        ]


# Script di esempio
if __name__ == "__main__":
    # Configurazione
    DOCUMENTS_DIR = "./documents"  # Directory con il PDF di Rimini
    OUTPUT_DIR = "./indexed_data"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
    
    # Crea e esegui pipeline
    pipeline = IndexingPipeline(
        documents_dir=DOCUMENTS_DIR,
        output_dir=OUTPUT_DIR,
        embedding_model=EMBEDDING_MODEL
    )
    
    # Esegui indicizzazione
    results = pipeline.run_pipeline(
        chunking_strategy='hierarchical',  # Ottimo per documenti Wikipedia
        process_limit=None  # Processa tutti i documenti
    )
    
    # Mostra risultati
    print("\n📈 RISULTATI FINALI:")
    print(json.dumps(results, indent=2, default=str))
    
    # Prepara per embedding (prossimo step)
    print(f"\n✨ Pronti {len(pipeline.all_chunks)} chunk per embedding con {EMBEDDING_MODEL}")
    
    # Esempio: ottieni primi 3 chunk per test
    sample_chunks = pipeline.get_chunks_with_metadata()[:3]
    for chunk_data in sample_chunks:
        print(f"\nChunk ID: {chunk_data['id']}")
        print(f"Testo (primi 100 char): {chunk_data['text'][:100]}...")
        print(f"Metadata: {chunk_data['metadata']}")