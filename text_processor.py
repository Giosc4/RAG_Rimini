"""
Text Processor e Chunking System
Gestisce il processing del testo e la divisione in chunk ottimizzati per paraphrase-mpnet-base-v2
"""

import re
import unicodedata
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Rappresenta un chunk di testo processato"""
    content: str
    metadata: Dict[str, Any]
    char_count: int
    token_count: int
    chunk_index: int
    overlap_with_previous: int = 0
    overlap_with_next: int = 0


class TextPreprocessor:
    """Preprocessa e pulisce il testo prima del chunking"""
    
    def __init__(self, preserve_structure: bool = True):
        self.preserve_structure = preserve_structure
        
    def preprocess(self, text: str, doc_type: str = 'generic') -> str:
        """
        Preprocessa il testo in base al tipo di documento
        """
        logger.info(f"Preprocessing testo ({len(text)} caratteri) - tipo: {doc_type}")
        
        # Pipeline di preprocessing
        text = self._normalize_unicode(text)
        text = self._clean_whitespace(text)
        
        if doc_type == 'pdf':
            text = self._clean_pdf_artifacts(text)
        
        text = self._fix_hyphenation(text)
        text = self._normalize_punctuation(text)
        
        # Rimuovi header/footer ripetitivi (specifico per PDF)
        if doc_type == 'pdf':
            text = self._remove_repetitive_headers(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalizza caratteri Unicode"""
        return unicodedata.normalize('NFKC', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Pulisce spazi multipli mantenendo struttura paragrafi"""
        # Sostituisci spazi multipli con singolo spazio
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalizza newline multiple (max 2 per preservare paragrafi)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Rimuovi spazi a inizio/fine riga
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        
        return '\n'.join(lines)
    
    def _clean_pdf_artifacts(self, text: str) -> str:
        """Rimuove artefatti comuni da PDF"""
        # Rimuovi numeri di pagina isolati
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Rimuovi linee contenenti solo caratteri speciali
        text = re.sub(r'^[_\-=*]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """Corregge parole spezzate con trattino a fine riga"""
        # Pattern per parole spezzate (minuscola-\n minuscola)
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalizza punteggiatura"""
        # Assicura spazio dopo punteggiatura
        text = re.sub(r'([.!?,:;])([A-Z])', r'\1 \2', text)
        return text
    
    def _remove_repetitive_headers(self, text: str, threshold: int = 3) -> str:
        """Rimuove header/footer che si ripetono nel documento"""
        lines = text.split('\n')
        line_counts = {}
        
        # Conta occorrenze di ogni linea
        for line in lines:
            if len(line) > 10 and len(line) < 100:  # Potenziali header/footer
                line_counts[line] = line_counts.get(line, 0) + 1
        
        # Identifica linee ripetitive
        repetitive_lines = {line for line, count in line_counts.items() 
                          if count >= threshold}
        
        # Rimuovi linee ripetitive
        if repetitive_lines:
            logger.info(f"Rimosse {len(repetitive_lines)} linee ripetitive")
            lines = [line for line in lines if line not in repetitive_lines]
        
        return '\n'.join(lines)
    
    def extract_structure(self, text: str) -> Dict[str, List[str]]:
        """Estrae struttura del documento (sezioni, paragrafi, ecc.)"""
        structure = {
            'sections': [],
            'paragraphs': [],
            'lists': []
        }
        
        # Identifica sezioni (linee che sembrano titoli)
        section_pattern = r'^[A-Z][^.!?]*$'  # Linea che inizia con maiuscola senza punteggiatura finale
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 3 and len(line) < 100:
                if re.match(section_pattern, line.strip()):
                    structure['sections'].append({
                        'title': line.strip(),
                        'position': i
                    })
        
        # Identifica paragrafi
        paragraphs = text.split('\n\n')
        structure['paragraphs'] = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        return structure


class TextChunker:
    """
    Sistema di chunking ottimizzato per paraphrase-mpnet-base-v2
    Max tokens: 512, Target: 300-400 tokens
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
                 target_chunk_size: int = 350,
                 max_chunk_size: int = 450,
                 min_chunk_size: int = 100,
                 overlap_tokens: int = 50):
        
        self.model_name = model_name
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        
        # Carica tokenizer del modello per conteggio preciso
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Modello {model_name} caricato per tokenizzazione")
        except:
            logger.warning("Modello non disponibile, uso stima approssimativa tokens")
            self.model = None
    
    def chunk_document(self, 
                      text: str, 
                      metadata: Dict[str, Any],
                      strategy: str = 'semantic') -> List[TextChunk]:
        """
        Divide il documento in chunk usando la strategia specificata
        
        Strategie:
        - 'semantic': divide per unità semantiche (paragrafi, frasi)
        - 'sliding': finestra scorrevole con overlap
        - 'hierarchical': rispetta la struttura del documento
        """
        logger.info(f"Chunking con strategia: {strategy}")
        
        if strategy == 'semantic':
            return self._semantic_chunking(text, metadata)
        elif strategy == 'sliding':
            return self._sliding_window_chunking(text, metadata)
        elif strategy == 'hierarchical':
            return self._hierarchical_chunking(text, metadata)
        else:
            return self._semantic_chunking(text, metadata)
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking basato su unità semantiche"""
        chunks = []
        
        # Dividi prima per paragrafi
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self._estimate_tokens(para)
            
            # Se il paragrafo è troppo grande, dividilo per frasi
            if para_tokens > self.max_chunk_size:
                # Processa il chunk corrente prima
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, metadata, chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Dividi il paragrafo grande
                para_chunks = self._split_large_paragraph(para, metadata, chunk_index)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                
            # Se aggiungere questo paragrafo supera il target
            elif current_tokens + para_tokens > self.target_chunk_size:
                # Salva il chunk corrente
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, metadata, chunk_index
                    ))
                    chunk_index += 1
                
                # Inizia nuovo chunk
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                # Aggiungi al chunk corrente
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Salva l'ultimo chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, metadata, chunk_index
            ))
        
        logger.info(f"Creati {len(chunks)} chunk semantici")
        return chunks
    
    def _sliding_window_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking con finestra scorrevole e overlap"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            current_chunk = []
            current_tokens = 0
            
            # Costruisci chunk fino al target size
            j = i
            while j < len(sentences) and current_tokens < self.target_chunk_size:
                sent_tokens = self._estimate_tokens(sentences[j])
                if current_tokens + sent_tokens <= self.max_chunk_size:
                    current_chunk.append(sentences[j])
                    current_tokens += sent_tokens
                    j += 1
                else:
                    break
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                
                # Calcola overlap
                if chunk_index > 0 and self.overlap_tokens > 0:
                    chunk.overlap_with_previous = self._calculate_overlap(
                        chunks[-1].content, chunk_text
                    )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Avanza con overlap
            sentences_to_skip = max(1, len(current_chunk) - 2)  # Mantieni 2 frasi per overlap
            i += sentences_to_skip
        
        logger.info(f"Creati {len(chunks)} chunk con sliding window")
        return chunks
    
    def _hierarchical_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking che rispetta la struttura gerarchica del documento"""
        # Per documenti Wikipedia, identifica sezioni
        chunks = []
        chunk_index = 0
        
        # Pattern per identificare sezioni (titoli in Wikipedia)
        section_pattern = r'^#+\s+(.+)$|^([A-Z][^.!?]{3,50})$'
        
        lines = text.split('\n')
        current_section = []
        current_section_title = "Introduzione"
        
        for line in lines:
            # Controlla se è un titolo di sezione
            if re.match(section_pattern, line.strip()) and len(line) < 100:
                # Processa la sezione precedente
                if current_section:
                    section_text = '\n'.join(current_section)
                    section_chunks = self._process_section(
                        section_text, 
                        current_section_title,
                        metadata,
                        chunk_index
                    )
                    chunks.extend(section_chunks)
                    chunk_index += len(section_chunks)
                
                # Inizia nuova sezione
                current_section_title = line.strip()
                current_section = []
            else:
                current_section.append(line)
        
        # Processa l'ultima sezione
        if current_section:
            section_text = '\n'.join(current_section)
            section_chunks = self._process_section(
                section_text, 
                current_section_title,
                metadata,
                chunk_index
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Creati {len(chunks)} chunk gerarchici")
        return chunks
    
    def _process_section(self, 
                        section_text: str, 
                        section_title: str,
                        metadata: Dict[str, Any],
                        start_index: int) -> List[TextChunk]:
        """Processa una sezione del documento"""
        section_metadata = metadata.copy()
        section_metadata['section'] = section_title
        
        # Se la sezione è piccola, ritorna come chunk singolo
        tokens = self._estimate_tokens(section_text)
        if tokens <= self.target_chunk_size:
            return [self._create_chunk(section_text, section_metadata, start_index)]
        
        # Altrimenti usa semantic chunking sulla sezione
        return self._semantic_chunking(section_text, section_metadata)
    
    def _split_large_paragraph(self, 
                               paragraph: str, 
                               metadata: Dict[str, Any],
                               start_index: int) -> List[TextChunk]:
        """Divide un paragrafo troppo grande in chunk più piccoli"""
        chunks = []
        sentences = self._split_into_sentences(paragraph)
        
        current_chunk = []
        current_tokens = 0
        chunk_index = start_index
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sent_tokens > self.target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                chunk_index += 1
                current_chunk = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide il testo in frasi"""
        # Pattern per split su frasi (punto seguito da spazio e maiuscola)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """Stima il numero di token nel testo"""
        if self.model:
            # Usa il tokenizer del modello reale
            try:
                tokens = self.model.tokenizer.tokenize(text)
                return len(tokens)
            except:
                pass
        
        # Stima approssimativa: ~1 token ogni 4 caratteri
        return len(text) // 4
    
    def _create_chunk(self, 
                     text: str, 
                     metadata: Dict[str, Any],
                     chunk_index: int) -> TextChunk:
        """Crea un oggetto TextChunk"""
        return TextChunk(
            content=text,
            metadata={
                **metadata,
                'chunk_index': chunk_index,
                'chunking_strategy': 'semantic'
            },
            char_count=len(text),
            token_count=self._estimate_tokens(text),
            chunk_index=chunk_index
        )
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """Calcola token di overlap tra due testi"""
        words1 = set(text1.split()[-20:])  # Ultime 20 parole
        words2 = set(text2.split()[:20])   # Prime 20 parole
        return len(words1.intersection(words2))
    
    def validate_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Valida e fornisce statistiche sui chunk creati"""
        stats = {
            'total_chunks': len(chunks),
            'avg_char_count': np.mean([c.char_count for c in chunks]),
            'avg_token_count': np.mean([c.token_count for c in chunks]),
            'min_tokens': min([c.token_count for c in chunks]),
            'max_tokens': max([c.token_count for c in chunks]),
            'chunks_over_limit': sum(1 for c in chunks if c.token_count > 512),
            'chunks_under_minimum': sum(1 for c in chunks if c.token_count < self.min_chunk_size)
        }
        
        logger.info(f"Validazione chunk:")
        logger.info(f"  - Totale: {stats['total_chunks']}")
        logger.info(f"  - Token medi: {stats['avg_token_count']:.1f}")
        logger.info(f"  - Range token: {stats['min_tokens']}-{stats['max_tokens']}")
        
        if stats['chunks_over_limit'] > 0:
            logger.warning(f"  ⚠️  {stats['chunks_over_limit']} chunk superano i 512 token!")
        
        return stats


# Esempio di utilizzo
if __name__ == "__main__":
    # Importa il document loader
    from document_loader import DocumentLoader, DocumentScanner
    
    # Carica un documento
    DOCUMENTS_DIR = "./documents"
    
    scanner = DocumentScanner(DOCUMENTS_DIR)
    manifest = scanner.scan_directory()
    
    if manifest:
        # Carica il primo documento (PDF di Rimini)
        loader = DocumentLoader()
        document = loader.load_document(manifest[0])
        
        # Preprocessa il testo
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess(
            document.content, 
            doc_type=document.metadata.file_type
        )
        
        # Crea chunk
        chunker = TextChunker()
        
        # Prova diverse strategie
        for strategy in ['semantic', 'hierarchical']:
            print(f"\n{'='*60}")
            print(f"Strategia: {strategy}")
            print('='*60)
            
            chunks = chunker.chunk_document(
                cleaned_text,
                {'source': document.metadata.file_path, 'type': document.metadata.file_type},
                strategy=strategy
            )
            
            # Valida e mostra statistiche
            stats = chunker.validate_chunks(chunks)
            
            # Mostra esempi di chunk
            print(f"\nEsempio primo chunk:")
            print(f"Tokens: {chunks[0].token_count}")
            print(f"Testo: {chunks[0].content[:200]}...")
            
            if len(chunks) > 1:
                print(f"\nEsempio chunk centrale:")
                mid_idx = len(chunks) // 2
                print(f"Tokens: {chunks[mid_idx].token_count}")
                print(f"Testo: {chunks[mid_idx].content[:200]}...")