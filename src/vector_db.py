import os
from typing import Dict, Any, List, Optional, cast
from datetime import datetime
import numpy as np
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

class SimpleVectorDB:
    def __init__(self, db_path="./data/chroma_db"):
        """
        Initialize the Vector Databse that can manage multiple collections
        
        Args:
            db_path: Path to store the ChromaDB database
        """
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client (one database instance)
        self.client = chromadb.PersistentClient(path=db_path)
        self.db_path = db_path
        self.current_collection = None
        self.current_collection_name = None
        
        print(f"Vector DB initialized at: {db_path}")
        print(f"Persistence enabled: Data will be saved to disk")
    
    def create_collection(self, collection_name: str, description: str = ""):
        """
        Create or get an existing collection
        
        Args:
            collection_name: Name of the collection
            description: Description of what this collection contains
            
        Returns:
            The collection object
        """
        metadata = {"description": description} if description else {}
        
        # Load API key from environment
        cohere_api_key = os.getenv("COHERE_API_KEY")

        # Warn the user if API key is missing
        if not cohere_api_key:
            raise EnvironmentError(
                "❌ COHERE_API_KEY not found in environment. "
                 "Please add it to your .env file before creating collections."
            )

        multimodal_cohere_ef = embedding_functions.CohereEmbeddingFunction(
            model_name="embed-v4.0",
            api_key=os.getenv("COHERE_API_KEY"),
        )   

        # collection with distance metric, COSINE similarity
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=multimodal_cohere_ef,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 100 # default. controls how well the index is constructed. 
                }
            }
        )
        
        print(f"Collection '{collection_name}' ready")
        return collection
    
    def use_collection(self, collection_name: str, description: str = ""):
        """
        Set the current working collection
        
        Args:
            collection_name: Name of the collection to use
            description: Description for new collections
        """
        self.current_collection = self.create_collection(collection_name, description)
        self.current_collection_name = collection_name
        print(f"Now using collection: {collection_name}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        
        print(f"Available collections: {collection_names}")
        return collection_names
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about all collections
        
        Returns:
            Dictionary with collection info
        """
        collections = self.client.list_collections()
        info = {}
        
        for col in collections:
            info[col.name] = {
                "count": col.count(),
                "metadata": col.metadata
            }
        
        print(f"Database contains {len(info)} collections:")
        for name, details in info.items():
            print(f"  • {name}: {details['count']} items - {details['metadata'].get('description', 'No description')}")
        
        return info
    
    def add_image_embedding(
        self,
        embedding_id: str,
        image_embedding: List[float],
        question: str,
        answerability: str,
        question_type: str,
        image_url: str,
        crowd_answers: List[str],
        crowd_majority: str,
        collection_name: Optional[str] = None
    ) -> str:
        """
        Add an image embedding to a collection
        
        Args:
            embedding_id: Specific ID for this embedding
            image_embedding: The multimodal embedding of the image
            question: The visual question the user asked
            answerability: Whether the question is answerable
            question_type: Type of question
            image_url: Original VizWiz image URL
            crowd_answers: List of crowd-sourced answers
            crowd_majority: The majority crowd answer
            collection_name: Optional collection name (uses current if None)
            
        Returns:
            The embedding ID that was used
        """
        try:
            # Use specified collection or current collection
            if collection_name:
                collection = self.create_collection(collection_name)
            elif self.current_collection:
                collection = self.current_collection
            else:
                # Default collection for backward compatibility
                collection = self.create_collection("default_embeddings", "Default image embeddings")
                self.current_collection = collection
                self.current_collection_name = "default_embeddings"
            
            # Create metadata with all VizWiz fields
            metadata = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "id": embedding_id,
                "answerability": answerability,
                "question_type": question_type,
                "image_url": image_url,
                "crowd_answers": "|".join(crowd_answers) if crowd_answers else "",
                "crowd_majority": crowd_majority
            }
            
            # Add embedding to collection with the specific ID (FIXED: embeddings must be a list)
            collection.add(
                embeddings=image_embedding,  
                metadatas=[metadata],
                ids=[embedding_id]
            )
            
            collection_name_used = collection_name or self.current_collection_name or "default_embeddings"
            print(f"Added embedding {embedding_id} to collection '{collection_name_used}' (persisted to disk)")
            return embedding_id
            
        except Exception as e:
            print(f"Error adding embedding {embedding_id}: {e}")
            raise
    
    def search_similar_images(
        self, 
        query_embedding: List[float], 
        n_results: int = 5,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find similar image embeddings in a collection
        
        Args:
            query_embedding: The embedding to search for
            n_results: Number of similar images to return
            collection_name: Optional collection name (uses current if None)
            
        Returns:
            Dictionary with search results
        """
        try:
            # Use specified collection or current collection
            if collection_name:
                if collection_name in [col.name for col in self.client.list_collections()]:
                    collection = self.client.get_collection(collection_name)
                else:
                    raise ValueError(f"Collection '{collection_name}' does not exist")
            elif self.current_collection:
                collection = self.current_collection
                collection_name = self.current_collection_name
            else:
                raise ValueError("No active collection. Please call use_collection() first.")
            
            # normalize query embedding for COSINE similarity (COMMENTED DUE TO EMBEDDINGS ALREADY NORMALIZED)
            # norm_query = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            ids = cast(List[List[str]], results.get("ids", [[]]))[0]
            distances = cast(List[List[float]], results.get("distances", [[]]))[0]
            metadatas = cast(List[List[dict]], results.get("metadatas", [[]]))[0]

            # Format results
            similar_images = [
                {"id": _id, "distance": dist, "metadata": meta}
                for _id, dist, meta in zip(ids, distances, metadatas)
            ]

            if not similar_images:
                print("⚠️ No similar images found or collection is empty.")

            return {
                "similar_images": similar_images,
                "count": len(similar_images),
                "collection": collection_name
            }

        except Exception as e:
            print(f"❌ Error searching collection: {e}")
            return {"similar_images": [], "count": 0, "error": str(e)}
    
    def check_if_exists(self, embedding_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Check if an embedding with this ID already exists in a collection
        
        Args:
            embedding_id: The ID to check
            collection_name: Optional collection name (uses current if None)
            
        Returns:
            True if exists, False otherwise
        """
        try:
            # Use specified collection or current collection
            if collection_name:
                collection = self.create_collection(collection_name)
            elif self.current_collection:
                collection = self.current_collection
            else:
                return False
            
            result = collection.get(ids=[embedding_id])
            return len(result["ids"]) > 0
        except:
            return False
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a specific collection
        
        Args:
            collection_name: Optional collection name (uses current if None)
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Use specified collection or current collection
            if collection_name:
                collection = self.create_collection(collection_name)
                name = collection_name
            elif self.current_collection:
                collection = self.current_collection
                name = self.current_collection_name
            else:
                return {"total_images": 0, "collection_name": "None", "error": "No collection specified"}
            
            count = collection.count()
            return {
                "total_images": count,
                "collection_name": name,
                "persisted": True
            }
        except Exception as e:
            return {"total_images": 0, "collection_name": "Error", "error": str(e)}
    
    def verify_persistence(self) -> Dict[str, Any]:
        """
        Verify that data is properly persisted to disk
        
        Returns:
            Dictionary with persistence verification info
        """
        try:
            # Check if database directory exists
            db_exists = os.path.exists(self.db_path)
            
            # Check if there are any files in the database directory
            db_files = []
            if db_exists:
                db_files = os.listdir(self.db_path)
            
            # Get collection info
            collections = self.list_collections()
            
            result = {
                "database_path": self.db_path,
                "database_exists": db_exists,
                "database_files": db_files,
                "collections_count": len(collections),
                "collections": collections,
                "persistence_working": db_exists and len(db_files) > 0
            }
            
            print(f"Persistence Status:")
            print(f"  Database path: {self.db_path}")
            print(f"  Database exists: {db_exists}")
            print(f"  Files in database: {len(db_files)}")
            print(f"  Collections: {len(collections)}")
            print(f"  Persistence working: {result['persistence_working']}")
            
            return result
            
        except Exception as e:
            print(f"Error verifying persistence: {e}")
            return {"error": str(e), "persistence_working": False}
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection to delete
        """
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name} (change persisted to disk)")
            
            # Reset current collection if it was deleted
            if self.current_collection_name == collection_name:
                self.current_collection = None
                self.current_collection_name = None
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")


if __name__ == "__main__":
    # Quick test of multi-collection functionality and persistence
    db = SimpleVectorDB()
    
    # Verify persistence is working
    db.verify_persistence()
    
    # Show available collections
    db.list_collections()
    
    print("\n✅ Multi-collection Vector DB ready with persistence! 💾🗂️")