"""
Document Loader e Scanner con supporto Multimodale
Gestisce lo scanning della directory, il caricamento dei file e l'estrazione dei metadata
inclusi elementi multimodali (immagini, tabelle, grafici)
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import logging
import base64
from io import BytesIO

# Importazioni per gestire diversi tipi di file
try:
    from pypdf import PdfReader
    import fitz  # PyMuPDF per estrazione avanzata
except ImportError:
    print("pypdf/PyMuPDF non installato. Installa con: pip install pypdf PyMuPDF")
    PdfReader = None
    fitz = None

try:
    import chardet
except ImportError:
    print("chardet non installato. Installa con: pip install chardet")
    chardet = None

try:
    from PIL import Image
except ImportError:
    print("PIL non installato. Installa con: pip install Pillow")
    Image = None

try:
    import pandas as pd
except ImportError:
    print("pandas non installato. Installa con: pip install pandas")
    pd = None

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Classe per i metadata del documento con supporto multimodale"""
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
    has_images: bool = False
    has_tables: bool = False
    image_count: int = 0
    table_count: int = 0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalElement:
    """Elemento multimodale estratto dal documento"""
    element_type: str  # 'image', 'table', 'chart'
    element_id: str
    content: Any  # Può essere path, data, o struttura dati
    page_number: Optional[int] = None
    position: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Classe per rappresentare un documento caricato con elementi multimodali"""
    content: str
    metadata: DocumentMetadata
    pages: Optional[List[Dict[str, Any]]] = None
    elements: List[MultimodalElement] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


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
        
        # Ordina per priorità
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
            manifest_serializable = []
            for item in self.manifest:
                item_copy = item.copy()
                item_copy['created'] = item_copy['created'].isoformat()
                item_copy['modified'] = item_copy['modified'].isoformat()
                manifest_serializable.append(item_copy)
            
            json.dump(manifest_serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest salvato in: {output_path}")


class MultimodalDocumentLoader:
    """Loader per caricare diversi tipi di documenti con supporto multimodale"""
    
    def __init__(self, 
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 save_images: bool = False,
                 images_dir: str = "./extracted_images"):
        
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.save_images = save_images
        self.images_dir = Path(images_dir)
        
        if save_images and extract_images:
            self.images_dir.mkdir(exist_ok=True)
        
        self.loaders = {
            'pdf': self._load_pdf_multimodal,
            'text': self._load_text,
            'markdown': self._load_text,
            'json': self._load_json,
            'html': self._load_html,
            'word': self._load_docx
        }
    
    def load_document(self, file_info: Dict[str, Any]) -> Document:
        """
        Carica un documento in base al suo tipo con supporto multimodale
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
        
        # Crea documento
        document = Document(content=content, metadata=metadata)
        
        # Aggiungi elementi multimodali se presenti
        if 'elements' in extra_metadata:
            document.elements = extra_metadata['elements']
        if 'images' in extra_metadata:
            document.images = extra_metadata['images']
            metadata.image_count = len(extra_metadata['images'])
            metadata.has_images = True
        if 'tables' in extra_metadata:
            document.tables = extra_metadata['tables']
            metadata.table_count = len(extra_metadata['tables'])
            metadata.has_tables = True
        if 'pages_data' in extra_metadata:
            document.pages = extra_metadata['pages_data']
        
        return document
    
    def _load_pdf_multimodal(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file PDF con estrazione multimodale"""
        if fitz:
            # Usa PyMuPDF per estrazione avanzata
            return self._load_pdf_with_fitz(file_path)
        elif PdfReader:
            # Fallback a pypdf per estrazione base
            return self._load_pdf_basic(file_path)
        else:
            raise ImportError("Nessuna libreria PDF disponibile")
    
    def _load_pdf_with_fitz(self, file_path: str) -> tuple[str, Dict]:
        """Estrazione avanzata da PDF con PyMuPDF"""
        pdf_document = fitz.open(file_path)
        
        text_content = []
        pages_data = []
        all_images = []
        all_tables = []
        all_elements = []
        
        # Estrai metadata PDF
        pdf_metadata = pdf_document.metadata
        
        for page_num, page in enumerate(pdf_document, 1):
            # Estrai testo
            page_text = page.get_text()
            text_content.append(page_text)
            
            page_data = {
                'page_num': page_num,
                'text': page_text,
                'char_count': len(page_text)
            }
            
            # Estrai immagini se richiesto
            if self.extract_images:
                page_images = self._extract_images_from_page(page, page_num, Path(file_path).stem)
                for img in page_images:
                    all_images.append(img)
                    # Aggiungi placeholder nel testo
                    placeholder_pos = len(page_text)
                    page_text += f"\n[IMMAGINE_{img['image_id']}]\n"
                    
                    # Crea elemento
                    elem = MultimodalElement(
                        element_type='image',
                        element_id=img['image_id'],
                        content=img.get('path', img.get('data')),
                        page_number=page_num,
                        metadata=img
                    )
                    all_elements.append(elem)
                
                page_data['images'] = len(page_images)
            
            # Estrai tabelle se richiesto
            if self.extract_tables:
                page_tables = self._extract_tables_from_page(page, page_num)
                for table in page_tables:
                    all_tables.append(table)
                    # Aggiungi placeholder nel testo
                    page_text += f"\n[TABELLA_{table['table_id']}]\n"
                    
                    # Crea elemento
                    elem = MultimodalElement(
                        element_type='table',
                        element_id=table['table_id'],
                        content=table['content'],
                        page_number=page_num,
                        metadata=table
                    )
                    all_elements.append(elem)
                
                page_data['tables'] = len(page_tables)
            
            pages_data.append(page_data)
        
        pdf_document.close()
        
        # Combina tutto il testo
        full_text = '\n\n'.join(text_content)
        
        # Prepara metadata extra
        extra_metadata = {
            'page_count': len(pages_data),
            'pages_data': pages_data,
            'elements': all_elements,
            'images': all_images,
            'tables': all_tables
        }
        
        if pdf_metadata:
            extra_metadata['pdf_metadata'] = pdf_metadata
        
        return full_text, extra_metadata
    
    def _extract_images_from_page(self, page, page_num: int, doc_name: str) -> List[Dict]:
        """Estrae immagini da una pagina PDF"""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY o RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    
                    img_id = f"p{page_num}_img{img_index+1}"
                    
                    img_info = {
                        'image_id': img_id,
                        'page': page_num,
                        'width': pix.width,
                        'height': pix.height,
                        'format': 'PNG'
                    }
                    
                    if self.save_images and Image:
                        # Salva immagine su disco
                        img_filename = f"{doc_name}_{img_id}.png"
                        img_path = self.images_dir / img_filename
                        
                        img_pil = Image.open(BytesIO(img_data))
                        img_pil.save(img_path)
                        img_info['path'] = str(img_path)
                        logger.debug(f"  Salvata immagine: {img_filename}")
                    else:
                        # Conserva come base64
                        img_info['data'] = base64.b64encode(img_data).decode()
                    
                    images.append(img_info)
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Errore estrazione immagine {img_index}: {e}")
        
        except Exception as e:
            logger.warning(f"Errore estrazione immagini pagina {page_num}: {e}")
        
        return images
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """Estrae tabelle da una pagina PDF usando euristiche"""
        tables = []
        
        try:
            # Estrai testo strutturato
            blocks = page.get_text("blocks")
            
            for block_idx, block in enumerate(blocks):
                if self._looks_like_table(block):
                    table_data = self._parse_table_from_block(block)
                    if table_data and len(table_data) > 1:
                        table_id = f"p{page_num}_t{len(tables)+1}"
                        tables.append({
                            'table_id': table_id,
                            'page': page_num,
                            'content': table_data,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0
                        })
        except Exception as e:
            logger.warning(f"Errore estrazione tabelle: {e}")
        
        return tables
    
    def _looks_like_table(self, block) -> bool:
        """Verifica se un blocco sembra una tabella"""
        if len(block) < 5:
            return False
        
        text = str(block[4]) if len(block) > 4 else ""
        
        # Cerca pattern tipici delle tabelle
        if '\t' in text or '|' in text:
            return True
        
        # Verifica allineamento con spazi multipli
        if '  ' in text:
            lines = text.split('\n')
            if len(lines) > 2:
                return True
        
        return False
    
    def _parse_table_from_block(self, block) -> List[List[str]]:
        """Estrae dati tabellari da un blocco"""
        if len(block) < 5:
            return []
        
        text = str(block[4])
        lines = text.split('\n')
        table_data = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Prova diversi delimitatori
            if '\t' in line:
                cells = line.split('\t')
            elif '|' in line:
                cells = [c.strip() for c in line.split('|')]
            else:
                cells = [c.strip() for c in line.split('  ') if c.strip()]
            
            if cells:
                table_data.append(cells)
        
        return table_data
    
    def _load_pdf_basic(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file PDF con pypdf (fallback senza multimodale)"""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                text_content = []
                pages_data = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text_content.append(page_text)
                    
                    pages_data.append({
                        'page_num': i + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                
                extra_metadata = {
                    'page_count': len(reader.pages),
                    'pages_data': pages_data
                }
                
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
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            if chardet:
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
            else:
                encoding = 'utf-8'
        
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
        
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        extra_metadata = {
            'json_keys': list(data.keys()) if isinstance(data, dict) else None
        }
        
        return content, extra_metadata
    
    def _load_html(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file HTML"""
        return self._load_text(file_path)
    
    def _load_docx(self, file_path: str) -> tuple[str, Dict]:
        """Carica un file Word (implementazione base)"""
        # Per ora usa estrazione testo base
        return self._load_text(file_path)
    
    def _create_metadata(self, file_info: Dict, extra_metadata: Dict) -> DocumentMetadata:
        """Crea oggetto metadata completo"""
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
    DOCUMENTS_DIR = "./documents"
    
    # 1. Scanning della directory
    scanner = DocumentScanner(DOCUMENTS_DIR)
    manifest = scanner.scan_directory(recursive=True)
    scanner.save_manifest()
    
    # 2. Caricamento documenti con supporto multimodale
    loader = MultimodalDocumentLoader(
        extract_images=True,
        extract_tables=True,
        save_images=True,
        images_dir="./extracted_images"
    )
    
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
            
            if document.metadata.has_images:
                print(f"Immagini: {document.metadata.image_count}")
            
            if document.metadata.has_tables:
                print(f"Tabelle: {document.metadata.table_count}")
            
            print(f"Priorità: {document.metadata.priority}")
            print(f"Caratteri totali: {len(document.content):,}")
            print(f"Prime 200 caratteri: {document.content[:200]}...")
            
            # Mostra elementi multimodali
            if document.elements:
                print(f"\nElementi multimodali trovati: {len(document.elements)}")
                for elem in document.elements[:5]:  # Primi 5
                    print(f"  - {elem.element_type}: {elem.element_id} (pagina {elem.page_number})")
            
        except Exception as e:
            logger.error(f"Errore nel processare {file_info['name']}: {e}")