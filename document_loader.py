"""
Document Loader e Scanner
Gestisce lo scanning della directory, il caricamento dei file e l'estrazione dei metadata
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import logging

# Importazioni per gestire diversi tipi di file
import pypdf
from pypdf import PdfReader
import chardet  # per rilevare encoding dei file di testo

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Classe per i metadata del documento"""
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    created_date: datetime
    modified_date: datetime
    file_hash: str
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    source: str = "local"
    priority: str = "medium"
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Classe per rappresentare un documento caricato"""
    content: str
    metadata: DocumentMetadata
    pages: Optional[List[Dict[str, Any]]] = None


class DocumentScanner:
    """Scanner per trovare e catalogare i documenti nella directory"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.md': 'markdown',
        '.docx': 'word',
        '.html': 'html',
        '.json': 'json'
    }
    
    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
        if not self.base_directory.exists():
            raise ValueError(f"Directory {base_directory} non esiste")
        
        self.manifest = []
        
    def scan_directory(self, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Scansiona la directory e crea un manifest dei documenti trovati
        """
        logger.info(f"Scanning directory: {self.base_directory}")
        
        if recursive:
            files = self.base_directory.rglob("*")
        else:
            files = self.base_directory.glob("*")
        
        for file_path in files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_info = self._get_file_info(file_path)
                    self.manifest.append(file_info)
                    logger.info(f"Found: {file_path.name} ({file_info['type']})")
        
        # Ordina per priorità (PDF di Rimini per primo)
        self.manifest = self._prioritize_files(self.manifest)
        
        logger.info(f"Totale documenti trovati: {len(self.manifest)}")
        return self.manifest
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Estrae informazioni base del file"""
        stats = file_path.stat()
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'type': self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown'),
            'extension': file_path.suffix.lower(),
            'size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_ctime),
            'modified': datetime.fromtimestamp(stats.st_mtime),
            'priority': self._determine_priority(file_path)
        }
    
    def _determine_priority(self, file_path: Path) -> str:
        """Determina la priorità del documento"""
        name_lower = file_path.name.lower()
        
        # Alta priorità per documenti principali
        if 'rimini' in name_lower and 'wikipedia' in name_lower:
            return 'high'
        elif 'rimini' in name_lower:
            return 'high'
        elif file_path.suffix.lower() == '.pdf':
            return 'medium'
        else:
            return 'low'
    
    def _prioritize_files(self, files: List[Dict]) -> List[Dict]:
        """Ordina i file per priorità"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(files, key=lambda x: priority_order.get(x['priority'], 3))
    
    def save_manifest(self, output_path: str = "document_manifest.json"):
        """Salva il manifest in un file JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Converti datetime in stringhe per JSON
            manifest_serializable = []
            for item in self.manifest:
                item_copy = item.copy()
                item_copy['created'] = item_copy['created'].isoformat()
                item_copy['modified'] = item_copy['modified'].isoformat()
                manifest_serializable.append(item_copy)
            
            json.dump(manifest_serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest salvato in: {output_path}")


class DocumentLoader:
    """Loader per caricare diversi tipi di documenti"""
    
    def __init__(self):
        self.loaders = {
            'pdf': self._load_pdf,
            'text': self._load_text,
            'markdown': self._load_text,
            'json': self._load_json,
            'html': self._load_html
        }
    
    def load_document(self, file_info: Dict[str, Any]) -> Document:
        """
        Carica un documento in base al suo tipo
        """
        file_type = file_info['type']
        file_path = file_info['path']
        
        logger.info(f"Loading {file_type} document: {file_info['name']}")
        
        # Seleziona il loader appropriato
        loader = self.loaders.get(file_type, self._load_text)
        
        # Carica il contenuto
        content, extra_metadata = loader(file_path)
        
        # Crea metadata
        metadata = self._create_metadata(file_info, extra_metadata)
        
        # Crea e ritorna il documento
        return Document(content=content, metadata=metadata)
    
    def _load_pdf(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file PDF"""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Estrai testo da tutte le pagine
                text_content = []
                pages_data = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text_content.append(page_text)
                    
                    # Salva anche dati per pagina (utile per chunking)
                    pages_data.append({
                        'page_num': i + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                
                # Metadata extra del PDF
                extra_metadata = {
                    'page_count': len(reader.pages),
                    'pages_data': pages_data
                }
                
                # Se ci sono metadata nel PDF
                if reader.metadata:
                    pdf_meta = {}
                    if reader.metadata.title:
                        pdf_meta['title'] = reader.metadata.title
                    if reader.metadata.author:
                        pdf_meta['author'] = reader.metadata.author
                    if reader.metadata.subject:
                        pdf_meta['subject'] = reader.metadata.subject
                    extra_metadata['pdf_metadata'] = pdf_meta
                
                return '\n\n'.join(text_content), extra_metadata
                
        except Exception as e:
            logger.error(f"Errore nel caricamento del PDF {file_path}: {e}")
            raise
    
    def _load_text(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file di testo con rilevamento encoding"""
        # Rileva encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        
        # Leggi con encoding corretto
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        extra_metadata = {
            'encoding': encoding,
            'line_count': content.count('\n') + 1
        }
        
        return content, extra_metadata
    
    def _load_json(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file JSON"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Converti JSON in testo formattato
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        extra_metadata = {
            'json_keys': list(data.keys()) if isinstance(data, dict) else None
        }
        
        return content, extra_metadata
    
    def _load_html(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file HTML (implementazione base)"""
        # Per ora carica come testo, in futuro puoi usare BeautifulSoup
        return self._load_text(file_path)
    
    def _create_metadata(self, file_info: Dict, extra_metadata: Dict) -> DocumentMetadata:
        """Crea oggetto metadata completo"""
        # Calcola hash del file
        file_hash = self._calculate_file_hash(file_info['path'])
        
        metadata = DocumentMetadata(
            file_path=file_info['path'],
            file_name=file_info['name'],
            file_type=file_info['type'],
            file_size=file_info['size'],
            created_date=file_info['created'],
            modified_date=file_info['modified'],
            file_hash=file_hash,
            priority=file_info['priority'],
            custom_metadata=extra_metadata
        )
        
        # Aggiungi metadata specifici
        if 'encoding' in extra_metadata:
            metadata.encoding = extra_metadata['encoding']
        if 'page_count' in extra_metadata:
            metadata.page_count = extra_metadata['page_count']
        
        return metadata
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcola hash SHA256 del file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# Esempio di utilizzo
if __name__ == "__main__":
    # Specifica la directory dei documenti
    DOCUMENTS_DIR = "./documents"  # Modifica con il tuo percorso
    
    # 1. Scanning della directory
    scanner = DocumentScanner(DOCUMENTS_DIR)
    manifest = scanner.scan_directory(recursive=True)
    scanner.save_manifest()
    
    # 2. Caricamento documenti
    loader = DocumentLoader()
    
    for file_info in manifest:
        try:
            document = loader.load_document(file_info)
            
            print(f"\n{'='*60}")
            print(f"Documento: {document.metadata.file_name}")
            print(f"Tipo: {document.metadata.file_type}")
            print(f"Dimensione: {document.metadata.file_size:,} bytes")
            print(f"Hash: {document.metadata.file_hash[:16]}...")
            
            if document.metadata.page_count:
                print(f"Pagine: {document.metadata.page_count}")
            
            print(f"Priorità: {document.metadata.priority}")
            print(f"Caratteri totali: {len(document.content):,}")
            print(f"Prime 200 caratteri: {document.content[:200]}...")
            
        except Exception as e:
            logger.error(f"Errore nel processare {file_info['name']}: {e}")