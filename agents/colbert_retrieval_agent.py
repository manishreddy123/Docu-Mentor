from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import torch


class ColBERTRetrievalAgent:
    """
    ColBERT-style multi-vector dense retrieval for improved semantic matching.
    Uses late interaction between query and document token representations.
    """

    def __init__(self):
        try:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.max_doc_length = 512
            self.max_query_length = 64
            print("✅ ColBERT retrieval agent initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize ColBERT agent: {str(e)}")
            self.model = None

    def encode_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Encode documents with token-level representations for ColBERT-style retrieval.
        """
        if not self.model:
            return documents

        try:
            encoded_docs = []

            for doc in documents:
                content = doc.get('content', '')
                if not content.strip():
                    continue

                words = content.split()[:self.max_doc_length]
                truncated_content = ' '.join(words)

                token_embeddings = self.model.encode(
                    [truncated_content],
                    output_value='token_embeddings',
                    convert_to_tensor=True
                )

                doc_embedding = torch.mean(token_embeddings[0], dim=0).cpu().numpy()

                encoded_doc = doc.copy()
                encoded_doc['colbert_tokens'] = token_embeddings[0].cpu().numpy()
                encoded_doc['embedding'] = doc_embedding
                encoded_doc['token_count'] = len(token_embeddings[0])

                encoded_docs.append(encoded_doc)

            print(f"✅ Encoded {len(encoded_docs)} documents with ColBERT representations")
            return encoded_docs

        except Exception as e:
            print(f"⚠️ ColBERT document encoding failed: {str(e)}")
            return documents

    def late_interaction_score(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """
        Compute ColBERT-style late interaction score between query and document tokens.
        """
        try:
            query_norm = query_tokens / (np.linalg.norm(query_tokens, axis=1, keepdims=True) + 1e-8)
            doc_norm = doc_tokens / (np.linalg.norm(doc_tokens, axis=1, keepdims=True) + 1e-8)

            similarity_matrix = np.dot(query_norm, doc_norm.T)

            max_similarities = np.max(similarity_matrix, axis=1)

            score = np.sum(max_similarities)

            return float(score)

        
        except Exception as e:
            print(f"⚠️ Late interaction scoring failed: {str(e)}")
            return 0.0
    
    def retrieve_with_colbert(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using ColBERT-style late interaction.
        """
        if not self.model or not documents:
            return documents[:top_k]
        
        try:
            query_words = query.split()[:self.max_query_length]
            truncated_query = ' '.join(query_words)
            
            query_tokens = self.model.encode(
                [truncated_query],
                output_value='token_embeddings',
                convert_to_tensor=True
            )[0].cpu().numpy()
            
            scored_docs = []
            
            for doc in documents:
                if 'colbert_tokens' not in doc:
                    if 'embedding' in doc:
                        query_emb = np.mean(query_tokens, axis=0)
                        doc_emb = doc['embedding']
                        score = np.dot(query_emb, doc_emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
                        )
                    else:
                        score = 0.0
                else:
                    score = self.late_interaction_score(query_tokens, doc['colbert_tokens'])
                
                doc_with_score = doc.copy()
                doc_with_score['colbert_score'] = score
                scored_docs.append(doc_with_score)
            
            scored_docs.sort(key=lambda x: x.get('colbert_score', 0), reverse=True)
            
            print(f"✅ ColBERT retrieval completed, top score: {scored_docs[0].get('colbert_score', 0):.4f}")
            return scored_docs[:top_k]
            
        except Exception as e:
            print(f"⚠️ ColBERT retrieval failed: {str(e)}")
            return documents[:top_k]
    
    def hybrid_retrieve(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10, 
                       colbert_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Combine ColBERT scores with traditional dense retrieval for hybrid ranking.
        """
        try:
            colbert_results = self.retrieve_with_colbert(query, documents, len(documents))
            
            colbert_scores = [doc.get('colbert_score', 0) for doc in colbert_results]
            if max(colbert_scores) > 0:
                colbert_scores = [s / max(colbert_scores) for s in colbert_scores]
            
            final_results = []
            for i, doc in enumerate(colbert_results):
                traditional_score = doc.get('similarity_score', 0)
                colbert_score = colbert_scores[i]
                
                hybrid_score = (colbert_weight * colbert_score + 
                               (1 - colbert_weight) * traditional_score)
                
                doc_with_hybrid = doc.copy()
                doc_with_hybrid['hybrid_score'] = hybrid_score
                final_results.append(doc_with_hybrid)
            
            final_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            
            print(f"✅ Hybrid retrieval completed with ColBERT weight {colbert_weight}")
            return final_results[:top_k]
            
        except Exception as e:
            print(f"⚠️ Hybrid retrieval failed: {str(e)}")
            return documents[:top_k]
