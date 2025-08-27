"""
Text Processor e Chunking System con supporto Multimodale
Gestisce il processing del testo e la divisione in chunk ottimizzati per paraphrase-mpnet-base-v2
preservando riferimenti a elementi multimodali (immagini, tabelle, grafici)
"""

import re
import unicodedata
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Rappresenta un chunk di testo processato con supporto multimodale"""
    content: str
    metadata: Dict[str, Any]
    char_count: int
    token_count: int
    chunk_index: int
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    has_multimodal: bool = False
    multimodal_refs: List[str] = field(default_factory=list)
    is_table_chunk: bool = False


class TextPreprocessor:
    """Preprocessa e pulisce il testo preservando riferimenti multimodali"""
    
    def __init__(self, preserve_structure: bool = True, preserve_multimodal: bool = True):
        self.preserve_structure = preserve_structure
        self.preserve_multimodal = preserve_multimodal
        
    def preprocess(self, text: str, doc_type: str = 'generic') -> str:
        """
        Preprocessa il testo preservando riferimenti multimodali
        """
        logger.info(f"Preprocessing testo ({len(text)} caratteri) - tipo: {doc_type}")
        
        # Proteggi riferimenti multimodali
        protected_refs = []
        if self.preserve_multimodal:
            text, protected_refs = self._protect_multimodal_refs(text)
        
        # Pipeline di preprocessing
        text = self._normalize_unicode(text)
        text = self._clean_whitespace(text)
        
        if doc_type == 'pdf':
            text = self._clean_pdf_artifacts(text)
        
        text = self._fix_hyphenation(text)
        text = self._normalize_punctuation(text)
        
        if doc_type == 'pdf':
            text = self._remove_repetitive_headers(text)
        
        # Ripristina riferimenti multimodali
        if protected_refs:
            text = self._restore_multimodal_refs(text, protected_refs)
        
        return text
    
    def _protect_multimodal_refs(self, text: str) -> Tuple[str, List[str]]:
        """Protegge i riferimenti multimodali durante la pulizia"""
        protected = []
        
        # Pattern per riferimenti multimodali
        patterns = [
            r'\[IMMAGINE_[^\]]+\]',
            r'\[TABELLA_[^\]]+\]',
            r'\[GRAFICO_[^\]]+\]',
            r'\[FIGURA_[^\]]+\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                placeholder = f"<<<PROTECTED_{len(protected)}>>>"
                text = text.replace(match, placeholder)
                protected.append(match)
        
        return text, protected
    
    def _restore_multimodal_refs(self, text: str, protected_refs: List[str]) -> str:
        """Ripristina i riferimenti multimodali protetti"""
        for i, ref in enumerate(protected_refs):
            placeholder = f"<<<PROTECTED_{i}>>>"
            text = text.replace(placeholder, ref)
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalizza caratteri Unicode"""
        return unicodedata.normalize('NFKC', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Pulisce spazi multipli mantenendo struttura paragrafi"""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        return '\n'.join(lines)
    
    def _clean_pdf_artifacts(self, text: str) -> str:
        """Rimuove artefatti comuni da PDF"""
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[_\-=*]{3,}\s*$', '', text, flags=re.MULTILINE)
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """Corregge parole spezzate con trattino a fine riga"""
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalizza punteggiatura"""
        text = re.sub(r'([.!?,:;])([A-Z])', r'\1 \2', text)
        return text
    
    def _remove_repetitive_headers(self, text: str, threshold: int = 3) -> str:
        """Rimuove header/footer che si ripetono nel documento"""
        lines = text.split('\n')
        line_counts = {}
        
        for line in lines:
            if len(line) > 10 and len(line) < 100:
                line_counts[line] = line_counts.get(line, 0) + 1
        
        repetitive_lines = {line for line, count in line_counts.items() 
                          if count >= threshold}
        
        if repetitive_lines:
            logger.info(f"Rimosse {len(repetitive_lines)} linee ripetitive")
            lines = [line for line in lines if line not in repetitive_lines]
        
        return '\n'.join(lines)
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """Estrae struttura del documento inclusi elementi multimodali"""
        structure = {
            'sections': [],
            'paragraphs': [],
            'lists': [],
            'multimodal_elements': []
        }
        
        # Identifica sezioni
        section_pattern = r'^[A-Z][^.!?]*$'
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
        
        # Identifica elementi multimodali
        multimodal_patterns = {
            'image': r'\[IMMAGINE_([^\]]+)\]',
            'table': r'\[TABELLA_([^\]]+)\]',
            'chart': r'\[GRAFICO_([^\]]+)\]'
        }
        
        for elem_type, pattern in multimodal_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                structure['multimodal_elements'].append({
                    'type': elem_type,
                    'id': match.group(1),
                    'position': match.start(),
                    'full_ref': match.group(0)
                })
        
        return structure


class MultimodalTextChunker:
    """
    Sistema di chunking ottimizzato per paraphrase-mpnet-base-v2 con supporto multimodale
    Max tokens: 512, Target: 300-400 tokens
    Preserva integrità di riferimenti multimodali
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
                 target_chunk_size: int = 350,
                 max_chunk_size: int = 450,
                 min_chunk_size: int = 100,
                 overlap_tokens: int = 50,
                 preserve_multimodal: bool = True,
                 split_tables: bool = False):
        
        self.model_name = model_name
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        self.preserve_multimodal = preserve_multimodal
        self.split_tables = split_tables
        
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
        preservando elementi multimodali
        
        Strategie:
        - 'semantic': divide per unità semantiche (paragrafi, frasi)
        - 'sliding': finestra scorrevole con overlap
        - 'hierarchical': rispetta la struttura del documento
        - 'multimodal_aware': ottimizzato per documenti con immagini/tabelle
        """
        logger.info(f"Chunking con strategia: {strategy}")
        
        # Aggiungi informazioni multimodali ai metadata
        if self.preserve_multimodal:
            multimodal_refs = self._extract_multimodal_refs(text)
            metadata['has_multimodal'] = len(multimodal_refs) > 0
            metadata['multimodal_count'] = len(multimodal_refs)
        
        if strategy == 'semantic':
            return self._semantic_chunking(text, metadata)
        elif strategy == 'sliding':
            return self._sliding_window_chunking(text, metadata)
        elif strategy == 'hierarchical':
            return self._hierarchical_chunking(text, metadata)
        elif strategy == 'multimodal_aware':
            return self._multimodal_aware_chunking(text, metadata)
        else:
            return self._semantic_chunking(text, metadata)
    
    def _multimodal_aware_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking ottimizzato per documenti con elementi multimodali"""
        chunks = []
        
        # Estrai riferimenti multimodali
        multimodal_refs = self._extract_multimodal_refs(text)
        
        # Dividi il testo preservando i riferimenti
        segments = self._split_preserving_multimodal(text, multimodal_refs)
        
        chunk_index = 0
        for segment in segments:
            segment_text = segment['text']
            segment_type = segment.get('type', 'text')
            
            # Se è una tabella e non vogliamo dividerle
            if segment_type == 'table' and not self.split_tables:
                chunk = self._create_chunk(segment_text, metadata, chunk_index)
                chunk.is_table_chunk = True
                chunk.has_multimodal = True
                chunk.multimodal_refs = [segment.get('ref_id', '')]
                chunks.append(chunk)
                chunk_index += 1
                
            # Se il segmento è piccolo, crea un singolo chunk
            elif self._estimate_tokens(segment_text) <= self.target_chunk_size:
                chunk = self._create_chunk(segment_text, metadata, chunk_index)
                
                # Aggiungi info multimodale se presente
                if segment.get('multimodal_refs'):
                    chunk.has_multimodal = True
                    chunk.multimodal_refs = segment['multimodal_refs']
                
                chunks.append(chunk)
                chunk_index += 1
                
            # Altrimenti dividi il segmento
            else:
                segment_chunks = self._split_large_segment(segment_text, metadata, chunk_index)
                
                # Propaga info multimodale ai chunk del segmento
                if segment.get('multimodal_refs'):
                    for sc in segment_chunks:
                        sc.has_multimodal = True
                        sc.multimodal_refs = segment['multimodal_refs']
                
                chunks.extend(segment_chunks)
                chunk_index += len(segment_chunks)
        
        logger.info(f"Creati {len(chunks)} chunk multimodal-aware")
        return chunks
    
    def _split_preserving_multimodal(self, text: str, multimodal_refs: List[Dict]) -> List[Dict]:
        """Divide il testo preservando l'integrità dei riferimenti multimodali"""
        segments = []
        last_pos = 0
        
        # Ordina riferimenti per posizione
        sorted_refs = sorted(multimodal_refs, key=lambda x: x['position'])
        
        for ref in sorted_refs:
            ref_start = ref['position']
            ref_end = ref_start + len(ref['full_ref'])
            
            # Aggiungi testo prima del riferimento
            if ref_start > last_pos:
                before_text = text[last_pos:ref_start].strip()
                if before_text:
                    segments.append({
                        'text': before_text,
                        'type': 'text',
                        'multimodal_refs': []
                    })
            
            # Determina contesto del riferimento
            context_start = max(0, ref_start - 200)  # 200 caratteri prima
            context_end = min(len(text), ref_end + 200)  # 200 caratteri dopo
            
            # Estrai contesto con il riferimento
            context_text = text[context_start:context_end]
            
            # Crea segmento per elemento multimodale con contesto
            segment_type = 'table' if ref['type'] == 'table' else 'multimodal'
            segments.append({
                'text': context_text,
                'type': segment_type,
                'ref_id': ref['id'],
                'multimodal_refs': [ref['full_ref']]
            })
            
            last_pos = context_end
        
        # Aggiungi testo rimanente
        if last_pos < len(text):
            remaining_text = text[last_pos:].strip()
            if remaining_text:
                segments.append({
                    'text': remaining_text,
                    'type': 'text',
                    'multimodal_refs': []
                })
        
        return segments
    
    def _extract_multimodal_refs(self, text: str) -> List[Dict]:
        """Estrae tutti i riferimenti multimodali dal testo"""
        refs = []
        
        patterns = {
            'image': r'\[IMMAGINE_([^\]]+)\]',
            'table': r'\[TABELLA_([^\]]+)\]',
            'chart': r'\[GRAFICO_([^\]]+)\]',
            'figure': r'\[FIGURA_([^\]]+)\]'
        }
        
        for ref_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                refs.append({
                    'type': ref_type,
                    'id': match.group(1),
                    'full_ref': match.group(0),
                    'position': match.start()
                })
        
        return refs
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking basato su unità semantiche preservando multimodale"""
        chunks = []
        
        # Dividi prima per paragrafi
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        current_multimodal_refs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self._estimate_tokens(para)
            para_refs = self._extract_multimodal_refs(para) if self.preserve_multimodal else []
            
            # Se il paragrafo contiene riferimenti multimodali e supera il target
            # ma non il max, tienilo intero
            if para_refs and para_tokens <= self.max_chunk_size:
                # Salva chunk corrente se esiste
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                    if current_multimodal_refs:
                        chunk.has_multimodal = True
                        chunk.multimodal_refs = [ref['full_ref'] for ref in current_multimodal_refs]
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                    current_multimodal_refs = []
                
                # Crea chunk dedicato per paragrafo con multimodale
                chunk = self._create_chunk(para, metadata, chunk_index)
                chunk.has_multimodal = True
                chunk.multimodal_refs = [ref['full_ref'] for ref in para_refs]
                chunks.append(chunk)
                chunk_index += 1
                
            # Se il paragrafo è troppo grande
            elif para_tokens > self.max_chunk_size:
                # Processa chunk corrente
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                    if current_multimodal_refs:
                        chunk.has_multimodal = True
                        chunk.multimodal_refs = [ref['full_ref'] for ref in current_multimodal_refs]
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                    current_multimodal_refs = []
                
                # Dividi il paragrafo grande
                para_chunks = self._split_large_paragraph(para, metadata, chunk_index)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                
            # Se aggiungere questo paragrafo supera il target
            elif current_tokens + para_tokens > self.target_chunk_size:
                # Salva il chunk corrente
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                    if current_multimodal_refs:
                        chunk.has_multimodal = True
                        chunk.multimodal_refs = [ref['full_ref'] for ref in current_multimodal_refs]
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Inizia nuovo chunk
                current_chunk = [para]
                current_tokens = para_tokens
                current_multimodal_refs = para_refs
            else:
                # Aggiungi al chunk corrente
                current_chunk.append(para)
                current_tokens += para_tokens
                current_multimodal_refs.extend(para_refs)
        
        # Salva l'ultimo chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = self._create_chunk(chunk_text, metadata, chunk_index)
            if current_multimodal_refs:
                chunk.has_multimodal = True
                chunk.multimodal_refs = [ref['full_ref'] for ref in current_multimodal_refs]
            chunks.append(chunk)
        
        logger.info(f"Creati {len(chunks)} chunk semantici")
        return chunks
    
    def _sliding_window_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking con finestra scorrevole preservando multimodale"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            current_chunk = []
            current_tokens = 0
            current_refs = []
            
            j = i
            while j < len(sentences) and current_tokens < self.target_chunk_size:
                sent = sentences[j]
                sent_tokens = self._estimate_tokens(sent)
                sent_refs = self._extract_multimodal_refs(sent)
                
                # Non spezzare riferimenti multimodali
                if sent_refs and current_tokens + sent_tokens > self.max_chunk_size:
                    break
                
                if current_tokens + sent_tokens <= self.max_chunk_size:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
                    current_refs.extend(sent_refs)
                    j += 1
                else:
                    break
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                
                if current_refs:
                    chunk.has_multimodal = True
                    chunk.multimodal_refs = [ref['full_ref'] for ref in current_refs]
                
                if chunk_index > 0 and self.overlap_tokens > 0:
                    chunk.overlap_with_previous = self._calculate_overlap(
                        chunks[-1].content, chunk_text
                    )
                
                chunks.append(chunk)
                chunk_index += 1
            
            sentences_to_skip = max(1, len(current_chunk) - 2)
            i += sentences_to_skip
        
        logger.info(f"Creati {len(chunks)} chunk con sliding window")
        return chunks
    
    def _hierarchical_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunking che rispetta la struttura gerarchica preservando multimodale"""
        chunks = []
        chunk_index = 0
        
        section_pattern = r'^#+\s+(.+)$|^([A-Z][^.!?]{3,50})$'
        
        lines = text.split('\n')
        current_section = []
        current_section_title = "Introduzione"
        current_section_refs = []
        
        for line in lines:
            line_refs = self._extract_multimodal_refs(line) if self.preserve_multimodal else []
            
            # Check if line is a section title
            if re.match(section_pattern, line.strip()) and len(line) < 100 and not line_refs:
                # Process previous section
                if current_section:
                    section_text = '\n'.join(current_section)
                    section_chunks = self._process_section(
                        section_text, 
                        current_section_title,
                        metadata,
                        chunk_index,
                        current_section_refs
                    )
                    chunks.extend(section_chunks)
                    chunk_index += len(section_chunks)
                
                # Start new section
                current_section_title = line.strip()
                current_section = []
                current_section_refs = []
            else:
                current_section.append(line)
                current_section_refs.extend(line_refs)
        
        # Process last section
        if current_section:
            section_text = '\n'.join(current_section)
            section_chunks = self._process_section(
                section_text, 
                current_section_title,
                metadata,
                chunk_index,
                current_section_refs
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Creati {len(chunks)} chunk gerarchici")
        return chunks
    
    def _process_section(self, 
                        section_text: str, 
                        section_title: str,
                        metadata: Dict[str, Any],
                        start_index: int,
                        multimodal_refs: List[Dict] = None) -> List[TextChunk]:
        """Processa una sezione del documento preservando multimodale"""
        section_metadata = metadata.copy()
        section_metadata['section'] = section_title
        
        tokens = self._estimate_tokens(section_text)
        
        # Se la sezione è piccola, ritorna come chunk singolo
        if tokens <= self.target_chunk_size:
            chunk = self._create_chunk(section_text, section_metadata, start_index)
            if multimodal_refs:
                chunk.has_multimodal = True
                chunk.multimodal_refs = [ref['full_ref'] for ref in multimodal_refs]
            return [chunk]
        
        # Altrimenti usa semantic chunking sulla sezione
        return self._semantic_chunking(section_text, section_metadata)
    
    def _split_large_paragraph(self, 
                               paragraph: str, 
                               metadata: Dict[str, Any],
                               start_index: int) -> List[TextChunk]:
        """Divide un paragrafo grande preservando multimodale"""
        chunks = []
        sentences = self._split_into_sentences(paragraph)
        
        current_chunk = []
        current_tokens = 0
        current_refs = []
        chunk_index = start_index
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            sent_refs = self._extract_multimodal_refs(sentence)
            
            # Non spezzare se contiene riferimenti multimodali
            if sent_refs and current_tokens + sent_tokens > self.max_chunk_size and current_chunk:
                # Salva chunk corrente
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                if current_refs:
                    chunk.has_multimodal = True
                    chunk.multimodal_refs = [ref['full_ref'] for ref in current_refs]
                chunks.append(chunk)
                chunk_index += 1
                
                # Inizia nuovo chunk
                current_chunk = [sentence]
                current_tokens = sent_tokens
                current_refs = sent_refs
                
            elif current_tokens + sent_tokens > self.target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(chunk_text, metadata, chunk_index)
                if current_refs:
                    chunk.has_multimodal = True
                    chunk.multimodal_refs = [ref['full_ref'] for ref in current_refs]
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = [sentence]
                current_tokens = sent_tokens
                current_refs = sent_refs
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
                current_refs.extend(sent_refs)
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_chunk(chunk_text, metadata, chunk_index)
            if current_refs:
                chunk.has_multimodal = True
                chunk.multimodal_refs = [ref['full_ref'] for ref in current_refs]
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_segment(self, text: str, metadata: Dict[str, Any], start_index: int) -> List[TextChunk]:
        """Divide un segmento grande in chunk più piccoli"""
        # Usa split_large_paragraph come base
        return self._split_large_paragraph(text, metadata, start_index)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide il testo in frasi preservando riferimenti multimodali"""
        # Pattern modificato per non dividere nei riferimenti
        multimodal_pattern = r'\[[A-Z]+_[^\]]+\]'
        
        # Proteggi riferimenti multimodali
        protected = []
        matches = re.findall(multimodal_pattern, text)
        for i, match in enumerate(matches):
            placeholder = f"<<<PROTECTED_SENT_{i}>>>"
            text = text.replace(match, placeholder)
            protected.append(match)
        
        # Dividi in frasi
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Ripristina riferimenti
        result = []
        for sent in sentences:
            for i, ref in enumerate(protected):
                placeholder = f"<<<PROTECTED_SENT_{i}>>>"
                sent = sent.replace(placeholder, ref)
            if sent.strip():
                result.append(sent.strip())
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Stima il numero di token nel testo"""
        if self.model:
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
        """Crea un oggetto TextChunk con supporto multimodale"""
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_index'] = chunk_index
        chunk_metadata['chunking_strategy'] = 'multimodal_aware'
        
        return TextChunk(
            content=text,
            metadata=chunk_metadata,
            char_count=len(text),
            token_count=self._estimate_tokens(text),
            chunk_index=chunk_index
        )
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """Calcola token di overlap tra due testi"""
        words1 = set(text1.split()[-20:])
        words2 = set(text2.split()[:20])
        return len(words1.intersection(words2))
    
    def validate_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Valida e fornisce statistiche sui chunk creati incluso multimodale"""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'avg_char_count': np.mean([c.char_count for c in chunks]),
            'avg_token_count': np.mean([c.token_count for c in chunks]),
            'min_tokens': min([c.token_count for c in chunks]),
            'max_tokens': max([c.token_count for c in chunks]),
            'chunks_over_limit': sum(1 for c in chunks if c.token_count > 512),
            'chunks_under_minimum': sum(1 for c in chunks if c.token_count < self.min_chunk_size),
            'multimodal_stats': {
                'chunks_with_multimodal': sum(1 for c in chunks if c.has_multimodal),
                'total_multimodal_refs': sum(len(c.multimodal_refs) for c in chunks),
                'table_chunks': sum(1 for c in chunks if c.is_table_chunk)
            }
        }
        
        logger.info(f"Validazione chunk:")
        logger.info(f"  - Totale: {stats['total_chunks']}")
        logger.info(f"  - Token medi: {stats['avg_token_count']:.1f}")
        logger.info(f"  - Range token: {stats['min_tokens']}-{stats['max_tokens']}")
        logger.info(f"  - Chunk con multimodale: {stats['multimodal_stats']['chunks_with_multimodal']}")
        
        if stats['chunks_over_limit'] > 0:
            logger.warning(f"  ⚠️  {stats['chunks_over_limit']} chunk superano i 512 token!")
        
        return stats


# Esempio di utilizzo
if __name__ == "__main__":
    # Importa il document loader
    from document_loader import MultimodalDocumentLoader, DocumentScanner
    
    # Carica un documento
    DOCUMENTS_DIR = "./documents"
    
    scanner = DocumentScanner(DOCUMENTS_DIR)
    manifest = scanner.scan_directory()
    
    if manifest:
        # Carica il primo documento con supporto multimodale
        loader = MultimodalDocumentLoader(
            extract_images=True,
            extract_tables=True
        )
        document = loader.load_document(manifest[0])
        
        # Preprocessa il testo preservando multimodale
        preprocessor = TextPreprocessor(preserve_multimodal=True)
        cleaned_text = preprocessor.preprocess(
            document.content, 
            doc_type=document.metadata.file_type
        )
        
        # Crea chunk con supporto multimodale
        chunker = MultimodalTextChunker(preserve_multimodal=True)
        
        # Prova diverse strategie
        for strategy in ['semantic', 'multimodal_aware']:
            print(f"\n{'='*60}")
            print(f"Strategia: {strategy}")
            print('='*60)
            
            chunks = chunker.chunk_document(
                cleaned_text,
                {
                    'source': document.metadata.file_path,
                    'type': document.metadata.file_type,
                    'has_images': document.metadata.has_images,
                    'has_tables': document.metadata.has_tables
                },
                strategy=strategy
            )
            
            # Valida e mostra statistiche
            stats = chunker.validate_chunks(chunks)
            
            # Mostra esempi di chunk
            print(f"\nEsempio primo chunk:")
            print(f"Tokens: {chunks[0].token_count}")
            print(f"Multimodale: {chunks[0].has_multimodal}")
            if chunks[0].multimodal_refs:
                print(f"Riferimenti: {chunks[0].multimodal_refs}")
            print(f"Testo: {chunks[0].content[:200]}...")
            
            # Mostra chunk con elementi multimodali
            multimodal_chunks = [c for c in chunks if c.has_multimodal]
            if multimodal_chunks:
                print(f"\nTrovati {len(multimodal_chunks)} chunk con elementi multimodali")
                for mc in multimodal_chunks[:3]:
                    print(f"  - Chunk #{mc.chunk_index}: {mc.multimodal_refs}")