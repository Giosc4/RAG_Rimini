
per prima cosa eseguire:    .\venv\Scripts\Activate.ps1

i files: document_loader.py, text_processor.py vengono eseguiti da indexing_pipeline.py

per fare domande eseguire il file rag_cli.py


Fare il preprocessing dei files che devono essere caricati nell'LLM. 
    questi devono divenetare un file txt





















---------------------

# Struttura del Progetto

## File Principali e loro Funzioni

### 1. document_loader.py
- **Funzione**: Carica e processa i documenti da diverse fonti
- **Componenti**:
  - `DocumentScanner`: Scansiona directory per trovare documenti supportati
  - `DocumentLoader`: Carica e estrae testo da vari formati (PDF, TXT, MD, DOCX, HTML, JSON)
- **Dipendenze**:
  - pypdf
  - chardet
- **Input**: Directory contenente i documenti
- **Output**: Documenti processati e loro metadata

### 2. text_processor.py
- **Funzione**: Preprocessa e chunking del testo
- **Componenti**:
  - `TextPreprocessor`: Pulizia e normalizzazione del testo
  - `TextChunker`: Divide il testo in chunks semanticamente significativi
- **Dipendenze**:
  - sentence-transformers
  - numpy
- **Input**: Testo estratto dai documenti
- **Output**: Chunks di testo processati

### 3. indexing_pipeline.py
- **Funzione**: Orchestrazione del processo di indicizzazione
- **Componenti**:
  - Coordina document_loader e text_processor
  - Gestisce il salvataggio dei chunks
- **Dipendenze**:
  - document_loader.py
  - text_processor.py
- **Input**: Directory dei documenti
- **Output**: File JSON e PKL con i chunks processati

### 4. vector_store.py
- **Funzione**: Gestisce il database vettoriale
- **Componenti**:
  - `VectorStoreManager`: Gestisce ChromaDB e embeddings
- **Dipendenze**:
  - chromadb
  - sentence-transformers
  - numpy
  - tqdm
- **Input**: Chunks di testo processati
- **Output**: Database vettoriale in ChromaDB

### 5. rag_system.py
- **Funzione**: Sistema RAG principale
- **Componenti**:
  - `RAGSystem`: Integra vector store e LLM
  - `OllamaClient`: Interfaccia con il modello linguistico
- **Dipendenze**:
  - vector_store.py
  - requests
- **Input**: Query dell'utente
- **Output**: Risposte generate

### 6. rag_cli.py
- **Funzione**: Interfaccia a linea di comando
- **Dipendenze**:
  - rag_system.py
  - vector_store.py
- **Input**: Comandi utente da terminale
- **Output**: Risposte formattate

## Pipeline di Esecuzione

1. **Setup Iniziale**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preparazione Documenti**:
   - Inserire i documenti nella cartella `documents/`
   - Assicurarsi che i file siano nei formati supportati (PDF, TXT, MD, DOCX, HTML, JSON)

3. **Indicizzazione**:
   ```python
   python indexing_pipeline.py
   ```
   - Questo eseguirà document_loader.py e text_processor.py
   - Creerà i file necessari in indexed_data/

4. **Esecuzione RAG**:
   ```python
   python rag_cli.py
   ```
   - Avvia l'interfaccia per fare domande al sistema
   - Utilizza il database vettoriale in chroma_db/

## Note Importanti
- Il sistema dà priorità ai documenti contenenti "rimini" nel nome
- I documenti vengono convertiti in chunks di testo ottimizzati per il RAG
- Gli embeddings vengono salvati persistentemente in ChromaDB
- Il sistema utilizza Ollama come LLM