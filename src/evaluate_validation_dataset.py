import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json
import time
from tqdm import tqdm
from datetime import datetime
import importlib

from vector_db import SimpleVectorDB

# Load prompt from YAML file
_prompts_path = Path(__file__).resolve().parents[1] / "configs" / "prompts.yml"
with open(_prompts_path, "r", encoding="utf-8") as _f:
    _prompts = yaml.safe_load(_f)

SYSTEM_PROMPT: str = _prompts["be_my_ai_prompt"]

# Evaluation Configuration
EVALUATION_CONFIG = {
    "embedding_provider": "cohere",  # Choose 'cohere' or 'openclip'
    "with_context": True,
    "without_context": True,
    "top_k_similar": 4,
    "max_validation_samples": None,  # Changed to None since we'll specify exact ID
    "specific_validation_id": "295",  # Added specific ID to evaluate
}

# Model Configurations
MODEL_CONFIGS: List[Dict[str, str]] = [
    {"name": "gemini-2.5-pro", "provider": "gemini", "model": "gemini-2.5-pro"},
]

class ValidationEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_provider = self.config["embedding_provider"]
        
        self.base_path = Path(__file__).resolve().parents[1]
        self.results_path = self.base_path / "notebooks" / "data" / "results"
        self.results_path.mkdir(exist_ok=True)
        
        chroma_path = self.base_path / "notebooks" / "data" / "chroma_db"
        self.db = SimpleVectorDB(db_path=str(chroma_path))
        
        ## Change later to adapt based on embeddings provider
        self.train_collection_name = f"vizwiz_500_sample_cosine"
        
        print(f"Using embedding provider: {self.embedding_provider.upper()}")
        print(f"Train collection: {self.train_collection_name}")
        
        self.validation_embeddings = self._load_validation_embeddings()
        if not self.validation_embeddings:
            raise FileNotFoundError("Could not load validation embeddings file.")

        self.validation_ids = [str(item['id']) for item in self.validation_embeddings.get("items", [])]
        
        sys.path.append(os.path.dirname(__file__))
        
        visual_interpreter = importlib.import_module("visual_interpreter")
        self.models = visual_interpreter.create_models(MODEL_CONFIGS)
        
        print(f"Initialized evaluator with {len(self.validation_ids)} validation samples.")
        print(f"Available models: {list(self.models.keys())}")

    def _load_validation_embeddings(self) -> Optional[Dict]:
        """Loads precomputed validation embeddings based on the provider."""
        file_name = f"lf_vqa_validation_embeddings_{self.embedding_provider}.json"
        emb_path = self.base_path / "notebooks" / "data" / "embeddings" / file_name
        print(f"Loading validation embeddings from: {emb_path}")
        if emb_path.exists():
            with open(emb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"⚠️ Embeddings file not found at {emb_path}")
        return None

    def _get_similar_images(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Gets similar images for a validation ID."""
        v_item = next((item for item in self.validation_embeddings.get("items", []) if str(item.get("id")) == str(validation_id)), None)
        if not v_item:
            return None
        
        raw_embedding = v_item["embedding"]
        emb_vec = raw_embedding[0] if isinstance(raw_embedding[0], list) else raw_embedding
        
        try:
            return self.db.search_similar_images(
                emb_vec,
                n_results=self.config["top_k_similar"],
                collection_name=self.train_collection_name
            )
        except Exception as e:
            print(f"Similarity search failed for ID {validation_id}: {e}")
            return None

    def _build_context_prompt(self, similar_images: Dict[str, Any]) -> str:
        """Builds context prompt from similar images."""
        prompt = "Your goal is to optimize your first response by generating a brief, but detailed description of the picture and prioritize what the user most likely needs.\n\n"
        prompt += "We have retrieved pictures with similar visual context. In these pictures, users asked the following questions:"
        
        for res in similar_images["similar_images"]:
            prompt += f"\n - {res['metadata'].get('question', 'No question available')}"
        
        prompt += "\n\nUse these questions as a guide for what kind of information is important to users."
        prompt += "\nIf the past questions conflict with the visual information, ignore them and prioritize describing the image's most prominent features."
        prompt += "\nHere is the first picture that you must give a description of."
        return prompt

    def _evaluate_single_sample(self, validation_id: str, model_name: str, model: Any, with_context: bool) -> Dict[str, Any]:
        """Evaluates a single validation sample."""
        v_item = next((item for item in self.validation_embeddings.get("items", []) if str(item.get("id")) == str(validation_id)), None)
        if not v_item:
            return {"error": f"Data for ID {validation_id} not found."}
            
        metadata = v_item.get("metadata", {})
        
        result = {
            "validation_id": validation_id, "model_name": model_name, "with_context": with_context,
            "embedding_provider": self.embedding_provider, "top_k_similar": self.config["top_k_similar"],
            "image_url": metadata.get("image_url", ""), "real_question": metadata.get("question", ""),
            "crowd_majority": metadata.get("crowd_majority", ""), "timestamp": datetime.now().isoformat(),
            "similar_images": [], "prompt_used": "", "llm_response": "", "error": None, "processing_time": 0.0
        }
        
        if not result["image_url"]:
            result["error"] = "No image URL found"
            return result
            
        start_time = time.time()
        
        try:
            prompt = "Your goal is to optimize your first response by generating a brief, but detailed description of the picture and prioritize what the user most likely needs.\nHere is the first picture that you must give a description of."
            
            if with_context:
                
                similar_images_result = self._get_similar_images(validation_id)
                if similar_images_result and similar_images_result["similar_images"]:
                    saved_similar_images = []
                    for res in similar_images_result["similar_images"]:
                        sim_meta = res.get("metadata", {})
                        saved_similar_images.append({
                            "id": res.get("id"),
                            "distance": res.get("distance"),
                            "question": sim_meta.get("question", ""),
                            "image_url": sim_meta.get("image_url", ""),
                            "crowd_majority": sim_meta.get("crowd_majority", "")
                        })
                    result["similar_images"] = saved_similar_images
                    
                    prompt = self._build_context_prompt(similar_images_result)
            
            result["prompt_used"] = prompt
            response, *_ = model.generate(prompt, mode="standard", image_urls=[result["image_url"]], system_prompt=SYSTEM_PROMPT)
            result["llm_response"] = response
            
        except Exception as e:
            result["error"] = str(e)
        
        result["processing_time"] = time.time() - start_time
        return result

    def run_evaluation(self, output_filename: Optional[str] = None) -> str:
        """Runs the full evaluation."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            provider = self.embedding_provider
            if self.config.get("specific_validation_id"):
                output_filename = f"evaluation_{provider}_{timestamp}_id_{self.config['specific_validation_id']}.jsonl"
            else:
                output_filename = f"evaluation_{provider}_{timestamp}.jsonl"
        
        output_path = self.results_path / output_filename
        
        # Check if we need to evaluate a specific ID
        if self.config.get("specific_validation_id"):
            specific_id = str(self.config["specific_validation_id"])
            if specific_id in self.validation_ids:
                validation_samples = [specific_id]
                print(f"Evaluating specific validation ID: {specific_id}")
            else:
                print(f"⚠️ Validation ID {specific_id} not found in available samples.")
                print(f"Available IDs: {self.validation_ids[:10]}...")  # Show first 10 for reference
                return ""
        else:
            validation_samples = self.validation_ids
            if self.config["max_validation_samples"]:
                validation_samples = validation_samples[:self.config["max_validation_samples"]]
        
        print(f"Starting evaluation of {len(validation_samples)} validation samples...")
        print(f"Results will be saved to: {output_path}")
        
        total_iterations = len(validation_samples) * len(self.models)
        if self.config["with_context"] and self.config["without_context"]:
            total_iterations *= 2
        
        with open(output_path, "w", encoding="utf-8") as f, tqdm(total=total_iterations, desc="Evaluating") as pbar:
            for validation_id in validation_samples:
                for model_name, model in self.models.items():
                    if self.config["with_context"]:
                        f.write(json.dumps(self._evaluate_single_sample(validation_id, model_name, model, True)) + "\n")
                        pbar.update(1)
                    
                    if self.config["without_context"]:
                        f.write(json.dumps(self._evaluate_single_sample(validation_id, model_name, model, False)) + "\n")
                        pbar.update(1)
        
        print(f"Evaluation completed. Results saved to: {output_path}")
        return str(output_path)

def main():
    """Main function to run the evaluation."""
    print("Starting VisionRAG Validation Dataset Evaluation")
    print("=" * 50)
    
    evaluator = ValidationEvaluator(config=EVALUATION_CONFIG)
    jsonl_path = evaluator.run_evaluation()
    
    if jsonl_path:
        print("\n" + "=" * 50)
        print("Evaluation completed successfully!")
        print(f"Results saved to: {jsonl_path}")
    else:
        print("\n" + "=" * 50)
        print("Evaluation failed - check the validation ID.")

if __name__ == "__main__":
    main()