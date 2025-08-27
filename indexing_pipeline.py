"""
Pipeline completa di indicizzazione documenti con supporto Multimodale
Orchestrazione del processo di scanning, loading, processing e chunking
con gestione di immagini, tabelle e altri elementi multimodali
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from dataclasses import asdict

# Import dei moduli custom
from document_loader import DocumentScanner, MultimodalDocumentLoader, Document, MultimodalElement
from text_processor import TextPreprocessor, MultimodalTextChunker, TextChunk

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalIndexingPipeline:
    """Pipeline completa per l'indicizzazione di documenti con supporto multimodale"""
    
    def __init__(self, 
                 documents_dir: str,
                 output_dir: str = "./indexed_data",
                 embedding_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
                 preprocessed_file: Optional[str] = None,
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 save_images: bool = True,
                 images_dir: str = "./extracted_images"):
        
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        self.preprocessed_file = preprocessed_file and Path(preprocessed_file)
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.save_images = save_images
        self.images_dir = Path(images_dir)
        
        # Crea directory output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_images and extract_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza componenti
        self.scanner = DocumentScanner(documents_dir)
        self.loader = MultimodalDocumentLoader(
            extract_images=extract_images,
            extract_tables=extract_tables,
            save_images=save_images,
            images_dir=str(images_dir)
        )
        self.preprocessor = TextPreprocessor(
            preserve_structure=True,
            preserve_multimodal=True
        )
        self.chunker = MultimodalTextChunker(
            model_name=embedding_model,
            target_chunk_size=350,
            max_chunk_size=450,
            min_chunk_size=100,
            overlap_tokens=50,
            preserve_multimodal=True,
            split_tables=False
        )
        
        # Storage per risultati
        self.manifest = []
        self.documents = []
        self.all_chunks = []
        self.all_images = []
        self.all_tables = []
        self.all_elements = []
        self.index_metadata = {}
    
    def run_pipeline(self, 
                    chunking_strategy: str = 'multimodal_aware',
                    process_limit: Optional[int] = None,
                    save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Esegue la pipeline completa di indicizzazione con supporto multimodale
        
        Args:
            chunking_strategy: strategia di chunking ('semantic', 'hierarchical', 'multimodal_aware')
            process_limit: numero massimo di documenti da processare (None = tutti)
            save_intermediate: salva risultati intermedi
        
        Returns:
            Dizionario con statistiche e risultati
        """
        logger.info("=" * 60)
        logger.info("INIZIO PIPELINE DI INDICIZZAZIONE MULTIMODALE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Check if using preprocessed documents
        if self.preprocessed_file:
            logger.info("\n📄 Usando documenti preprocessati...")
            success = self._load_preprocessed_documents()
            if not success:
                logger.warning("⚠️  Caricamento preprocessati fallito, uso normale")
                self.preprocessed_file = None
        
        # Step 1: Scanning
        if not self.preprocessed_file:
            logger.info("\n📁 STEP 1: Scanning della directory...")
            self.manifest = self.scanner.scan_directory(recursive=True)
            logger.info(f"   Trovati {len(self.manifest)} documenti")
            
            # Salva manifest
            if save_intermediate:
                self._save_manifest()
            
            # Limita documenti se richiesto
            docs_to_process = self.manifest[:process_limit] if process_limit else self.manifest
            
            # Step 2: Loading e Processing con supporto multimodale
            logger.info(f"\n📄 STEP 2: Caricamento e processing di {len(docs_to_process)} documenti...")
            
            for i, file_info in enumerate(docs_to_process, 1):
                logger.info(f"\n   [{i}/{len(docs_to_process)}] Processing: {file_info['name']}")
                
                try:
                    # Carica documento con elementi multimodali
                    document = self.loader.load_document(file_info)
                    self.documents.append(document)
                    
                    # Raccogli elementi multimodali
                    if document.images:
                        self.all_images.extend(document.images)
                        logger.info(f"   🖼️  Trovate {len(document.images)} immagini")
                    
                    if document.tables:
                        self.all_tables.extend(document.tables)
                        logger.info(f"   📋 Trovate {len(document.tables)} tabelle")
                    
                    if document.elements:
                        self.all_elements.extend(document.elements)
                    
                    # Preprocessa testo preservando riferimenti multimodali
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
                        'priority': document.metadata.priority,
                        'has_images': document.metadata.has_images,
                        'has_tables': document.metadata.has_tables,
                        'image_count': document.metadata.image_count,
                        'table_count': document.metadata.table_count
                    }
                    
                    # Step 3: Chunking con strategia multimodale
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
                    multimodal_chunks = sum(1 for c in chunks if c.has_multimodal)
                    logger.info(f"   ✅ Creati {len(chunks)} chunk (di cui {multimodal_chunks} con multimodale)")
                    logger.info(f"      Token medi: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Errore nel processare {file_info['name']}: {e}")
                    continue
        
        # Step 3.5: Processamento elementi multimodali
        if self.all_images or self.all_tables:
            logger.info("\n🎨 STEP 3.5: Processamento elementi multimodali...")
            self._process_multimodal_elements()
        
        # Step 4: Validazione e statistiche
        logger.info("\n📊 STEP 4: Validazione e statistiche finali...")
        stats = self._calculate_statistics()
        
        # Step 5: Salvataggio risultati
        logger.info("\n💾 STEP 5: Salvataggio risultati...")
        self._save_results(save_intermediate)
        
        # Step 6: Creazione indice per retrieval
        logger.info("\n🔍 STEP 6: Creazione indice per retrieval...")
        self._create_retrieval_index()
        
        # Tempo totale
        elapsed_time = (datetime.now() - start_time).total_seconds()
        stats['processing_time_seconds'] = elapsed_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETATA CON SUCCESSO")
        logger.info(f"Tempo totale: {elapsed_time:.2f} secondi")
        logger.info(f"Documenti processati: {len(self.documents)}")
        logger.info(f"Chunk totali creati: {len(self.all_chunks)}")
        logger.info(f"Immagini estratte: {len(self.all_images)}")
        logger.info(f"Tabelle estratte: {len(self.all_tables)}")
        logger.info("=" * 60)
        
        return stats
    
    def _load_preprocessed_documents(self) -> bool:
        """Carica documenti dal file preprocessato"""
        if not self.preprocessed_file or not self.preprocessed_file.exists():
            return False
        
        try:
            logger.info(f"Caricamento da: {self.preprocessed_file}")
            
            # Crea un documento dal file preprocessato
            from document_loader import Document, DocumentMetadata
            
            with open(self.preprocessed_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cerca metadata JSON associato
            metadata_file = self.preprocessed_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
            else:
                metadata_dict = {}
            
            # Crea documento
            doc_metadata = DocumentMetadata(
                file_path=str(self.preprocessed_file),
                file_name=self.preprocessed_file.name,
                file_type='preprocessed',
                file_size=self.preprocessed_file.stat().st_size,
                created_date=datetime.fromtimestamp(self.preprocessed_file.stat().st_ctime),
                modified_date=datetime.fromtimestamp(self.preprocessed_file.stat().st_mtime),
                file_hash=self._calculate_file_hash(str(self.preprocessed_file)),
                priority='high',
                has_images='total_images' in metadata_dict and metadata_dict['total_images'] > 0,
                has_tables='total_tables' in metadata_dict and metadata_dict['total_tables'] > 0,
                image_count=metadata_dict.get('total_images', 0),
                table_count=metadata_dict.get('total_tables', 0)
            )
            
            document = Document(
                content=content,
                metadata=doc_metadata
            )
            
            self.documents = [document]
            
            # Se c'è un manifest multimodale, caricalo
            multimodal_manifest = self.output_dir / "multimodal_manifest.json"
            if multimodal_manifest.exists():
                with open(multimodal_manifest, 'r') as f:
                    mm_data = json.load(f)
                    self.all_images = mm_data.get('images', [])
                    self.all_tables = mm_data.get('tables', [])
            
            logger.info(f"✓ Documento preprocessato caricato con successo")
            return True
            
        except Exception as e:
            logger.error(f"Errore nel caricare documenti preprocessati: {e}")
            return False
    
    def _process_multimodal_elements(self):
        """Processa e indicizza elementi multimodali"""
        # Aggiungi metadata agli elementi
        for img in self.all_images:
            if 'source_doc' not in img:
                # Trova documento sorgente
                for doc in self.documents:
                    if any(doc_img['image_id'] == img['image_id'] for doc_img in doc.images):
                        img['source_doc'] = doc.metadata.file_name
                        break
        
        for table in self.all_tables:
            if 'source_doc' not in table:
                for doc in self.documents:
                    if any(doc_table['table_id'] == table['table_id'] for doc_table in doc.tables):
                        table['source_doc'] = doc.metadata.file_name
                        break
        
        # Collega elementi ai chunk
        for chunk in self.all_chunks:
            if chunk.has_multimodal:
                chunk.metadata['linked_elements'] = {
                    'images': [],
                    'tables': []
                }
                
                for ref in chunk.multimodal_refs:
                    if 'IMMAGINE_' in ref:
                        img_id = ref.replace('[IMMAGINE_', '').replace(']', '')
                        matching_img = next((img for img in self.all_images if img['image_id'] == img_id), None)
                        if matching_img:
                            chunk.metadata['linked_elements']['images'].append(img_id)
                    
                    elif 'TABELLA_' in ref:
                        table_id = ref.replace('[TABELLA_', '').replace(']', '')
                        matching_table = next((table for table in self.all_tables if table['table_id'] == table_id), None)
                        if matching_table:
                            chunk.metadata['linked_elements']['tables'].append(table_id)
        
        logger.info(f"  Processati {len(self.all_images)} immagini e {len(self.all_tables)} tabelle")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche complete sull'indicizzazione"""
        if not self.all_chunks:
            return {}
        
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
            'priority_distribution': self._count_priorities(),
            'multimodal_statistics': {
                'total_images': len(self.all_images),
                'total_tables': len(self.all_tables),
                'total_elements': len(self.all_elements),
                'chunks_with_multimodal': sum(1 for c in self.all_chunks if c.has_multimodal),
                'table_chunks': sum(1 for c in self.all_chunks if c.is_table_chunk),
                'docs_with_images': sum(1 for d in self.documents if d.metadata.has_images),
                'docs_with_tables': sum(1 for d in self.documents if d.metadata.has_tables)
            }
        }
        
        # Calcola distribuzione riferimenti multimodali
        multimodal_refs_per_chunk = [len(c.multimodal_refs) for c in self.all_chunks if c.has_multimodal]
        if multimodal_refs_per_chunk:
            stats['multimodal_statistics']['avg_refs_per_chunk'] = np.mean(multimodal_refs_per_chunk)
            stats['multimodal_statistics']['max_refs_per_chunk'] = max(multimodal_refs_per_chunk)
        
        # Avvisi
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
    
    def _save_results(self, save_intermediate: bool = True):
        """Salva tutti i risultati dell'indicizzazione con supporto multimodale"""
        
        # 1. Salva chunks come JSON (per debugging/ispezione)
        chunks_json_path = self.output_dir / "chunks.json"
        chunks_data = []
        
        for chunk in self.all_chunks:
            chunk_dict = {
                'chunk_index': chunk.chunk_index,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'char_count': chunk.char_count,
                'token_count': chunk.token_count,
                'has_multimodal': chunk.has_multimodal,
                'multimodal_refs': chunk.multimodal_refs,
                'is_table_chunk': chunk.is_table_chunk
            }
            chunks_data.append(chunk_dict)
        
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Chunks salvati (JSON): {chunks_json_path}")
        
        # 2. Salva chunks come pickle (per uso programmatico)
        chunks_pickle_path = self.output_dir / "chunks.pkl"
        with open(chunks_pickle_path, 'wb') as f:
            pickle.dump(self.all_chunks, f)
        
        logger.info(f"   Chunks salvati (pickle): {chunks_pickle_path}")
        
        # 3. Salva manifest multimodale
        if self.all_images or self.all_tables:
            multimodal_manifest_path = self.output_dir / "multimodal_manifest.json"
            multimodal_manifest = {
                'creation_date': datetime.now().isoformat(),
                'images': self.all_images,
                'tables': self.all_tables,
                'chunks_with_multimodal': [
                    {
                        'chunk_index': c.chunk_index,
                        'multimodal_refs': c.multimodal_refs,
                        'linked_elements': c.metadata.get('linked_elements', {})
                    }
                    for c in self.all_chunks if c.has_multimodal
                ]
            }
            
            with open(multimodal_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(multimodal_manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   Manifest multimodale salvato: {multimodal_manifest_path}")
        
        # 4. Salva metadata dell'indicizzazione
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
                'overlap_tokens': self.chunker.overlap_tokens,
                'preserve_multimodal': self.chunker.preserve_multimodal
            },
            'multimodal_config': {
                'extract_images': self.extract_images,
                'extract_tables': self.extract_tables,
                'save_images': self.save_images,
                'images_dir': str(self.images_dir)
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Metadata salvati: {metadata_path}")
        
        # 5. Crea file di testo leggibile con esempi
        if save_intermediate:
            examples_path = self.output_dir / "chunk_examples.txt"
            with open(examples_path, 'w', encoding='utf-8') as f:
                f.write("ESEMPI DI CHUNK CREATI\n")
                f.write("=" * 80 + "\n\n")
                
                # Esempi di chunk normali
                normal_chunks = [c for c in self.all_chunks if not c.has_multimodal][:3]
                if normal_chunks:
                    f.write("CHUNK TESTUALI:\n")
                    f.write("-" * 40 + "\n")
                    for chunk in normal_chunks:
                        f.write(f"Chunk #{chunk.chunk_index}\n")
                        f.write(f"File: {chunk.metadata.get('file_name', 'Unknown')}\n")
                        f.write(f"Token: {chunk.token_count} | Caratteri: {chunk.char_count}\n")
                        f.write(chunk.content[:300] + "...\n" if len(chunk.content) > 300 else chunk.content + "\n")
                        f.write("\n")
                
                # Esempi di chunk con multimodale
                multimodal_chunks = [c for c in self.all_chunks if c.has_multimodal][:3]
                if multimodal_chunks:
                    f.write("\nCHUNK CON ELEMENTI MULTIMODALI:\n")
                    f.write("-" * 40 + "\n")
                    for chunk in multimodal_chunks:
                        f.write(f"Chunk #{chunk.chunk_index}\n")
                        f.write(f"File: {chunk.metadata.get('file_name', 'Unknown')}\n")
                        f.write(f"Riferimenti: {chunk.multimodal_refs}\n")
                        f.write(f"Token: {chunk.token_count} | Caratteri: {chunk.char_count}\n")
                        f.write(chunk.content[:300] + "...\n" if len(chunk.content) > 300 else chunk.content + "\n")
                        f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"   Esempi salvati: {examples_path}")
    
    def _create_retrieval_index(self):
        """Crea indice ottimizzato per il retrieval con supporto multimodale"""
        retrieval_index = {
            'version': '2.0',
            'created': datetime.now().isoformat(),
            'total_chunks': len(self.all_chunks),
            'chunks': [],
            'multimodal_lookup': {
                'images': {},
                'tables': {}
            }
        }
        
        # Crea entry per ogni chunk
        for chunk in self.all_chunks:
            chunk_entry = {
                'id': f"{chunk.metadata.get('file_hash', 'unknown')[:8]}_{chunk.chunk_index}",
                'index': chunk.chunk_index,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'token_count': chunk.token_count,
                'has_multimodal': chunk.has_multimodal
            }
            
            if chunk.has_multimodal:
                chunk_entry['multimodal_refs'] = chunk.multimodal_refs
                chunk_entry['linked_elements'] = chunk.metadata.get('linked_elements', {})
                
                # Aggiungi al lookup
                for img_id in chunk.metadata.get('linked_elements', {}).get('images', []):
                    if img_id not in retrieval_index['multimodal_lookup']['images']:
                        retrieval_index['multimodal_lookup']['images'][img_id] = []
                    retrieval_index['multimodal_lookup']['images'][img_id].append(chunk.chunk_index)
                
                for table_id in chunk.metadata.get('linked_elements', {}).get('tables', []):
                    if table_id not in retrieval_index['multimodal_lookup']['tables']:
                        retrieval_index['multimodal_lookup']['tables'][table_id] = []
                    retrieval_index['multimodal_lookup']['tables'][table_id].append(chunk.chunk_index)
            
            retrieval_index['chunks'].append(chunk_entry)
        
        # Salva indice
        index_path = self.output_dir / "retrieval_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_index, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Indice di retrieval creato: {index_path}")
        
        # Crea anche versione pickle per performance
        index_pickle_path = self.output_dir / "retrieval_index.pkl"
        with open(index_pickle_path, 'wb') as f:
            pickle.dump(retrieval_index, f)
        
        logger.info(f"   Indice pickle creato: {index_pickle_path}")
    
    def get_chunks_for_embedding(self) -> List[str]:
        """Ritorna lista di testi pronti per l'embedding"""
        return [chunk.content for chunk in self.all_chunks]
    
    def get_chunks_with_metadata(self) -> List[Dict[str, Any]]:
        """Ritorna chunks con metadata completi per il vector database"""
        chunks_data = []
        
        for chunk in self.all_chunks:
            chunk_data = {
                'id': f"{chunk.metadata.get('file_hash', 'unknown')[:8]}_{chunk.chunk_index}",
                'text': chunk.content,
                'metadata': chunk.metadata,
                'has_multimodal': chunk.has_multimodal
            }
            
            if chunk.has_multimodal:
                # Estrai ID degli elementi referenziati
                image_refs = [ref.replace('[IMMAGINE_', '').replace(']', '') 
                             for ref in chunk.multimodal_refs if 'IMMAGINE_' in ref]
                table_refs = [ref.replace('[TABELLA_', '').replace(']', '') 
                             for ref in chunk.multimodal_refs if 'TABELLA_' in ref]
                
                chunk_data['multimodal_refs'] = {
                    'images': image_refs,
                    'tables': table_refs,
                    'all_refs': chunk.multimodal_refs
                }
            
            chunks_data.append(chunk_data)
        
        return chunks_data
    
    def get_multimodal_element(self, element_id: str, element_type: str = 'auto') -> Optional[Dict]:
        """Recupera un elemento multimodale specifico"""
        if element_type == 'auto':
            # Determina tipo dall'ID
            if 'img' in element_id.lower():
                element_type = 'image'
            elif 't' in element_id.lower():
                element_type = 'table'
        
        if element_type == 'image':
            for img in self.all_images:
                if img['image_id'] == element_id:
                    return img
        elif element_type == 'table':
            for table in self.all_tables:
                if table['table_id'] == element_id:
                    return table
        
        return None
    
    def get_chunks_for_element(self, element_id: str) -> List[TextChunk]:
        """Ritorna tutti i chunk che riferiscono un elemento multimodale"""
        related_chunks = []
        
        for chunk in self.all_chunks:
            if chunk.has_multimodal:
                # Cerca l'ID nei riferimenti
                for ref in chunk.multimodal_refs:
                    if element_id in ref:
                        related_chunks.append(chunk)
                        break
        
        return related_chunks
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcola hash SHA256 del file"""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# Script di esempio
if __name__ == "__main__":
    # Configurazione
    DOCUMENTS_DIR = "./documents"  # Directory con documenti
    PREPROCESSED_FILE = "./processed/documents_multimodal.txt"  # File preprocessato (opzionale)
    OUTPUT_DIR = "./indexed_data"
    IMAGES_DIR = "./extracted_images"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
    
    # Crea e esegui pipeline multimodale
    pipeline = MultimodalIndexingPipeline(
        documents_dir=DOCUMENTS_DIR,
        output_dir=OUTPUT_DIR,
        embedding_model=EMBEDDING_MODEL,
        preprocessed_file=PREPROCESSED_FILE if Path(PREPROCESSED_FILE).exists() else None,
        extract_images=True,
        extract_tables=True,
        save_images=True,
        images_dir=IMAGES_DIR
    )
    
    # Esegui indicizzazione con supporto multimodale
    results = pipeline.run_pipeline(
        chunking_strategy='multimodal_aware',  # Usa strategia ottimizzata per multimodale
        process_limit=None,  # Processa tutti i documenti
        save_intermediate=True
    )
    
    # Mostra risultati
    print("\n📈 RISULTATI FINALI:")
    print(json.dumps(results, indent=2, default=str))
    
    # Prepara per embedding
    print(f"\n✨ Pronti {len(pipeline.all_chunks)} chunk per embedding con {EMBEDDING_MODEL}")
    print(f"   - Chunk con elementi multimodali: {results['multimodal_statistics']['chunks_with_multimodal']}")
    print(f"   - Immagini totali: {results['multimodal_statistics']['total_images']}")
    print(f"   - Tabelle totali: {results['multimodal_statistics']['total_tables']}")
    
    # Esempio: ottieni chunk per una specifica immagine
    if pipeline.all_images:
        first_image = pipeline.all_images[0]
        print(f"\n🖼️ Esempio immagine: {first_image['image_id']}")
        related_chunks = pipeline.get_chunks_for_element(first_image['image_id'])
        print(f"   Chunk correlati: {len(related_chunks)}")
        
    # Esempio: ottieni dati per vector database
    chunks_for_db = pipeline.get_chunks_with_metadata()[:3]
    for chunk_data in chunks_for_db:
        print(f"\n📦 Chunk ID: {chunk_data['id']}")
        print(f"   Multimodale: {chunk_data['has_multimodal']}")
        if chunk_data['has_multimodal']:
            print(f"   Riferimenti: {chunk_data.get('multimodal_refs', {})}")
        print(f"   Testo (primi 100 char): {chunk_data['text'][:100]}...")