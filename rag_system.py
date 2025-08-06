"""
Sistema RAG Completo
Integra retrieval dal vector store con generazione usando Ollama/DeepSeek
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import requests
from dataclasses import dataclass

from vector_store import VectorStoreManager

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Struttura per la risposta del sistema RAG"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float


class OllamaClient:
    """
    Client per interagire con Ollama API
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "mistral:7b",
                 temperature: float = 0.7):
        
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        
        # Verifica connessione
        self._check_connection()
    
    def _check_connection(self):
        """Verifica che Ollama sia raggiungibile"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ Connesso a Ollama. Modelli disponibili: {[m['name'] for m in models.get('models', [])]}")
            else:
                logger.warning(f"⚠️ Ollama raggiungibile ma risposta non valida: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Impossibile connettersi a Ollama: {e}")
            logger.info("Assicurati che Ollama sia in esecuzione (ollama serve)")
    
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                max_tokens: int = 1000) -> str:
        """
        Genera testo usando Ollama
        
        Args:
            prompt: il prompt principale
            system_prompt: prompt di sistema opzionale
            max_tokens: numero massimo di token da generare
            
        Returns:
            Testo generato
        """
        # Costruisci prompt completo
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Chiamata API Ollama
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"Errore Ollama: {response.status_code}")
                return "Errore nella generazione della risposta"
                
        except Exception as e:
            logger.error(f"Errore nella chiamata a Ollama: {e}")
            return f"Errore: {str(e)}"


class RAGSystem:
    """
    Sistema RAG completo che integra retrieval e generazione
    """
    
    def __init__(self,
                 vector_store: VectorStoreManager,
                 llm_client: Optional[OllamaClient] = None,
                 retrieval_k: int = 5,
                 rerank_k: int = 3):
        
        self.vector_store = vector_store
        self.llm_client = llm_client or OllamaClient()
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        
        # Template per prompt
        self.system_prompt = """Sei un assistente esperto che risponde a domande basandosi ESCLUSIVAMENTE sul contesto fornito.
        
REGOLE IMPORTANTI:
1. Usa SOLO le informazioni presenti nel contesto
2. Se l'informazione non è nel contesto, rispondi "Non ho trovato questa informazione nei documenti forniti"
3. Cita sempre da quale parte del contesto proviene l'informazione
4. Sii preciso e conciso
5. Rispondi in italiano"""
        
        self.answer_template = """Contesto disponibile:
{context}

Domanda: {question}

Basandoti ESCLUSIVAMENTE sul contesto fornito sopra, rispondi alla domanda.
Se l'informazione non è presente nel contesto, dichiaralo esplicitamente.

Risposta:"""
    
    def retrieve_context(self, 
                        query: str,
                        k: Optional[int] = None) -> List[Tuple[str, Dict, float]]:
        """
        Recupera il contesto rilevante dal vector store
        
        Args:
            query: la domanda dell'utente
            k: numero di chunk da recuperare
            
        Returns:
            Lista di tuple (testo, metadata, score)
        """
        k = k or self.retrieval_k
        
        # Cerca nel vector store
        results = self.vector_store.search_with_score(
            query=query,
            n_results=k,
            score_threshold=0.2  # Soglia minima di similarità
        )
        
        logger.info(f"📚 Recuperati {len(results)} chunk rilevanti")
        
        return results
    
    def rerank_results(self, 
                      query: str,
                      results: List[Tuple[str, Dict, float]],
                      k: Optional[int] = None) -> List[Tuple[str, Dict, float]]:
        """
        Re-ranking dei risultati (per ora semplice, basato su score)
        
        Args:
            query: la domanda
            results: risultati dal retrieval
            k: numero di risultati da mantenere dopo reranking
            
        Returns:
            Lista riordinata e filtrata
        """
        k = k or self.rerank_k
        
        # Per ora, semplice sorting per score e prendi top-k
        # In futuro si può implementare cross-encoder per reranking
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        return sorted_results[:k]
    
    def format_context(self, 
                      results: List[Tuple[str, Dict, float]]) -> str:
        """
        Formatta i risultati del retrieval come contesto
        
        Args:
            results: chunk recuperati
            
        Returns:
            Contesto formattato come stringa
        """
        context_parts = []
        
        for i, (text, metadata, score) in enumerate(results, 1):
            source = metadata.get('file_name', 'Sconosciuto')
            chunk_idx = metadata.get('chunk_index', 'N/A')
            
            context_part = f"""[Fonte {i} - {source}, Chunk #{chunk_idx} (Rilevanza: {score:.2f})]
{text}
---"""
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, 
                       question: str,
                       include_sources: bool = True,
                       verbose: bool = False) -> RAGResponse:
        """
        Risponde a una domanda usando il sistema RAG completo
        
        Args:
            question: la domanda dell'utente
            include_sources: se includere le fonti nella risposta
            verbose: se mostrare log dettagliati
            
        Returns:
            Oggetto RAGResponse con risposta e metadata
        """
        start_time = datetime.now()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤔 Domanda: {question}")
            print('='*60)
        
        # Step 1: Retrieval
        if verbose:
            print("\n📚 Step 1: Recupero contesto rilevante...")
        
        retrieved_results = self.retrieve_context(question)
        
        if not retrieved_results:
            return RAGResponse(
                query=question,
                answer="Non ho trovato informazioni rilevanti nei documenti per rispondere a questa domanda.",
                sources=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Step 2: Reranking
        if verbose:
            print(f"🎯 Step 2: Reranking ({len(retrieved_results)} → {self.rerank_k} chunk)...")
        
        reranked_results = self.rerank_results(question, retrieved_results)
        
        # Step 3: Format context
        context = self.format_context(reranked_results)
        
        if verbose:
            print(f"\n📄 Contesto selezionato ({len(context)} caratteri)")
            print("-" * 40)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 40)
        
        # Step 4: Generate answer
        if verbose:
            print("\n🤖 Step 3: Generazione risposta con LLM...")
        
        prompt = self.answer_template.format(
            context=context,
            question=question
        )
        
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=500
        )
        
        # Calcola confidence basata sui scores
        avg_score = sum(r[2] for r in reranked_results) / len(reranked_results)
        
        # Prepara sources
        sources = []
        for text, metadata, score in reranked_results:
            sources.append({
                'text_preview': text[:200] + "...",
                'metadata': metadata,
                'relevance_score': score
            })
        
        # Aggiungi fonti alla risposta se richiesto
        if include_sources and sources:
            answer += "\n\n📚 Fonti utilizzate:"
            for i, source in enumerate(sources, 1):
                answer += f"\n{i}. {source['metadata'].get('file_name', 'N/A')} - Chunk #{source['metadata'].get('chunk_index', 'N/A')} (Rilevanza: {source['relevance_score']:.2%})"
        
        response = RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            confidence=avg_score,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        if verbose:
            print(f"\n✅ Risposta generata in {response.processing_time:.2f} secondi")
            print(f"📊 Confidence: {response.confidence:.2%}")
        
        return response
    
    def chat_loop(self):
        """
        Loop interattivo per chat con il sistema RAG
        """
        print("\n" + "="*60)
        print("💬 SISTEMA RAG INTERATTIVO")
        print("="*60)
        print("Digita 'quit' o 'exit' per uscire")
        print("Digita 'help' per aiuto")
        print("="*60)
        
        while True:
            try:
                # Input utente
                question = input("\n❓ La tua domanda: ").strip()
                
                # Comandi speciali
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Arrivederci!")
                    break
                
                if question.lower() == 'help':
                    print("\n📖 AIUTO:")
                    print("- Fai qualsiasi domanda sui documenti indicizzati")
                    print("- Il sistema cercherà le informazioni rilevanti")
                    print("- Le risposte sono basate SOLO sui documenti forniti")
                    print("- Comandi: quit/exit (esci), help (aiuto)")
                    continue
                
                if not question:
                    continue
                
                # Genera risposta
                response = self.answer_question(question, include_sources=True, verbose=True)
                
                # Mostra risposta
                print("\n" + "="*60)
                print("💡 RISPOSTA:")
                print("="*60)
                print(response.answer)
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrotto dall'utente")
                break
            except Exception as e:
                print(f"\n❌ Errore: {e}")
                logger.error(f"Errore nel chat loop: {e}", exc_info=True)


# Script principale
if __name__ == "__main__":
    print("\n🚀 Inizializzazione Sistema RAG...")
    
    # Inizializza vector store
    vector_store = VectorStoreManager(
        embedding_model="sentence-transformers/paraphrase-mpnet-base-v2",
        persist_directory="./chroma_db",
        collection_name="rimini_knowledge_base"
    )
    
    # Verifica che ci siano documenti
    stats = vector_store.get_statistics()
    if stats['total_documents'] == 0:
        print("\n⚠️ Nessun documento nel vector store!")
        print("Esegui prima: python vector_store.py")
        exit(1)
    
    print(f"\n✅ Vector store caricato: {stats['total_documents']} documenti")
    
    # Inizializza Ollama client
    print("\n🤖 Connessione a Ollama/DeepSeek...")
    llm_client = OllamaClient(
        base_url="http://localhost:11434",
        model="mistral:7b",  # o qualsiasi modello tu abbia installato
        temperature=0.7
    )
    
    # Crea sistema RAG
    rag_system = RAGSystem(
        vector_store=vector_store,
        llm_client=llm_client,
        retrieval_k=5,  # Recupera 5 chunk
        rerank_k=3      # Usa i top 3 dopo reranking
    )
    
    print("\n✅ Sistema RAG pronto!")
    
    # Modalità interattiva o test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Modalità test con domande predefinite
        print("\n🧪 MODALITÀ TEST")
        test_questions = [
            "Qual è la popolazione di Rimini?",
            "Chi era Sigismondo Pandolfo Malatesta?",
            "Quali sono le principali attrazioni turistiche di Rimini?",
            "Quando è stato costruito il Ponte di Tiberio?",
            "Quali film ha girato Federico Fellini su Rimini?"
        ]
        
        for q in test_questions:
            response = rag_system.answer_question(q, verbose=False)
            print(f"\n❓ {q}")
            print(f"💡 {response.answer[:300]}...")
            print(f"📊 Confidence: {response.confidence:.2%}")
            print("-" * 60)
    else:
        # Modalità interattiva
        rag_system.chat_loop()