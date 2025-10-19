"""
Dataset Cleaning Script for VizWiz Evaluation Data

This script identifies and removes irrelevant question-image pairs from the VizWiz dataset.
It uses an LLM to evaluate whether questions are relevant for training vision-language models.

Process:
1. Load training and validation data from VizWiz
2. Evaluate each question using an LLM (text-only, no image bias)
3. Generate lists of IDs to discard (train_to_discard.json, validation_to_discard.json)
4. Perform manual review of discarded samples
5. Clean the final evaluation JSONL file by removing discarded samples

Usage:
    python clean_dataset.py --mode collect    # Step 1: Identify irrelevant samples
    python clean_dataset.py --mode clean      # Step 2: Clean the evaluation file after manual review
"""

import os
import sys
import json
import time
import argparse
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import yaml
from tqdm import tqdm
from datetime import datetime
import importlib

from vector_db import SimpleVectorDB

# Load configuration
_prompts_path = Path(__file__).resolve().parents[1] / "configs" / "prompts.yml"
with open(_prompts_path, "r", encoding="utf-8") as _f:
    _prompts = yaml.safe_load(_f)

SYSTEM_PROMPT: str = _prompts["be_my_ai_prompt"]

# Configuration
CONFIG = {
    "embedding_provider": "cohere",
    "max_train_samples": None,  # Process ALL training samples
    "max_validation_samples": None,  # Process ALL validation samples
}

# Model Configuration for Gemini
MODEL_CONFIGS: List[Dict[str, str]] = [
    {"name": "gemini-2.5-pro", "provider": "gemini", "model": "gemini-2.5-pro"},
]

RELEVANCE_PROMPT = """
Evaluate if this question is useful for training a vision-language model.

You will ONLY see the question text, not the image. Evaluate based on whether the question:

MARK AS NOT RELEVANT if the question:
- Is a thank you message ("Thanks", "Thank you", "Thanks for your help")
- Is a greeting ("Hello", "Hi")
- Is nonsensical text ("???", random letters, gibberish)
- Is just punctuation (".", "...", "!!!")
- Is a complaint about image quality instead of asking something useful
- Is a statement/comment rather than a question
- Is unclear, vague, or incomprehensible
- Expresses frustration without asking anything specific
- Is a test message or placeholder
- Uses ambiguous pronouns without clear referents ("it", "this", "that" without specifying what)
- Requires context from previous conversation ("Oh so...", "So then...", "But what about...")
- Is a conversational fragment rather than a standalone question
- Doesn't actually ask about visual content that could be seen in an image
- Is incomplete or requires external context to understand
- Asks for external knowledge not visible in images (e.g., "Where can I buy this?", "How many calories?")

MARK AS RELEVANT only if it:
- Asks something specific that would be useful to answer about visual content
- Is a clear, standalone, understandable question
- Could reasonably be answered by looking at an image
- Would help train a vision-language model
- Doesn't require additional context to understand what is being asked

EXAMPLES OF NOT RELEVANT:
- "Thanks for your help"
- "This image is blurry"
- "I can't see anything"
- "This image does not show who the mail is for"
- "Hello there"
- "???"
- "how do I need to do?"
- "What am I doing wrong?" (not about visual content)
- "So then what happens next?" (requires previous context)
- "Where can I buy this?" (requires external knowledge)
- "How many calories are in this?" (requires external knowledge)

EXAMPLES OF RELEVANT:
- "What color is this shirt?"
- "What does the label say?"
- "How many people are in this photo?"
- "What time does the clock show?"

Your response must be EXACTLY in this JSON format (no other text):
{"is_relevant": false, "reason": "explanation"}

Remember: Use true or false (not True/False), include all quotes, no extra formatting.
"""


class DatasetCleaner:
    """Identifies and removes irrelevant question-image pairs from the dataset."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_provider = self.config["embedding_provider"]

        self.base_path = Path(__file__).resolve().parents[1]
        self.results_path = Path(__file__).parent

        # Initialize vector database
        chroma_path = self.base_path / "notebooks" / "data" / "chroma_db"
        self.db = SimpleVectorDB(db_path=str(chroma_path))
        self.train_collection_name = "vizwiz_500_sample_cosine"

        print(f"Using embedding provider: {self.embedding_provider.upper()}")
        print(f"Train collection: {self.train_collection_name}")

        # Load validation embeddings
        self.validation_embeddings = self._load_validation_embeddings()
        if not self.validation_embeddings:
            raise FileNotFoundError("Could not load validation embeddings file.")

        # Initialize Gemini model
        sys.path.append(os.path.dirname(__file__))
        visual_interpreter = importlib.import_module("visual_interpreter")
        self.models = visual_interpreter.create_models(MODEL_CONFIGS)
        self.model = self.models["gemini-2.5-pro"]

        print(f"Initialized dataset cleaner")

    def _load_validation_embeddings(self) -> Optional[Dict]:
        """Load precomputed validation embeddings based on the provider."""
        file_name = f"lf_vqa_validation_embeddings_{self.embedding_provider}.json"
        emb_path = self.base_path / "notebooks" / "data" / "embeddings" / file_name
        print(f"Loading validation embeddings from: {emb_path}")
        if emb_path.exists():
            with open(emb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"⚠️ Embeddings file not found at {emb_path}")
        return None

    def _evaluate_relevance(self, question: str) -> Dict[str, Any]:
        """Evaluate if a question is relevant using only the question text (no image to avoid bias)."""
        print(f"Evaluating: '{question[:50]}...'")
        try:
            full_prompt = f"{RELEVANCE_PROMPT}\n\nQuestion to evaluate: '{question}'"

            # Send only text prompt, no image to avoid bias
            response, *_ = self.model.generate(
                full_prompt,
                mode="standard",
                image_urls=None,  # No image - evaluate only the question text
                system_prompt=SYSTEM_PROMPT
            )

            # Try to parse JSON response
            try:
                # Clean the response - remove any markdown formatting
                json_str = response.strip()

                # Remove markdown code blocks if present
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                # Try to find JSON-like content in the response
                if not json_str.startswith("{"):
                    start_idx = json_str.find("{")
                    if start_idx != -1:
                        end_idx = json_str.rfind("}") + 1
                        if end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                result = json.loads(json_str)

                # Validate expected fields
                if not all(key in result for key in ["is_relevant", "reason"]):
                    raise ValueError("Missing required fields in response")

                print(f"  -> Relevant: {result['is_relevant']}")

                return {
                    "success": True,
                    "is_relevant": result["is_relevant"],
                    "reason": result["reason"],
                    "raw_response": response
                }

            except (json.JSONDecodeError, ValueError) as e:
                print(f"  -> JSON parsing failed, using fallback heuristic")

                # FALLBACK: If JSON parsing fails, use simple heuristic
                question_lower = question.lower().strip()

                irrelevant_patterns = [
                    "thank", "thanks", "hello", "hi there", "test", "???",
                    ".", "..", "...", "this image", "can't see", "blurry",
                    "not clear", "poor quality", "doesn't show", "does not show",
                    "oh so", "so then", "but what", "what am i doing",
                    "am i doing wrong", "where can i buy", "how many calories"
                ]

                is_irrelevant = any(pattern in question_lower for pattern in irrelevant_patterns)
                is_irrelevant = is_irrelevant or len(question.strip()) < 3

                # Check if it's a statement rather than a question
                if not question.strip().endswith('?') and len(question.split()) > 5:
                    is_irrelevant = True

                print(f"  -> Fallback decision - Relevant: {not is_irrelevant}")

                return {
                    "success": True,
                    "is_relevant": not is_irrelevant,
                    "reason": f"Fallback heuristic decision - parsing failed: {e}",
                    "raw_response": response
                }

        except Exception as e:
            print(f"Error evaluating relevance: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _process_train_data(self) -> List[Dict[str, Any]]:
        """Process training data from ChromaDB and identify irrelevant samples."""
        print("Processing training data...")

        self.db.use_collection(self.train_collection_name, "500 random VizWiz samples")

        collection_snapshot = self.db.current_collection.get()
        all_ids = collection_snapshot["ids"]
        all_metadatas = collection_snapshot["metadatas"]

        print(f"Found {len(all_ids)} training samples")

        if self.config["max_train_samples"]:
            all_ids = all_ids[:self.config["max_train_samples"]]
            all_metadatas = all_metadatas[:self.config["max_train_samples"]]
            print(f"Limited to {len(all_ids)} samples for processing")

        irrelevant_samples = []

        for sample_id, metadata in tqdm(zip(all_ids, all_metadatas),
                                        desc="Processing train samples",
                                        total=len(all_ids)):
            image_url = metadata.get("image_url", "")
            question = metadata.get("question", "")

            if not image_url or not question:
                print(f"Skipping sample {sample_id}: missing image_url or question")
                continue

            evaluation = self._evaluate_relevance(question)

            if evaluation.get("success") and not evaluation.get("is_relevant"):
                irrelevant_samples.append({
                    "id": sample_id,
                    "image_url": image_url,
                    "question": question,
                    "crowd_majority": metadata.get("crowd_majority", ""),
                    "evaluation": {
                        "is_relevant": evaluation.get("is_relevant"),
                        "reason": evaluation.get("reason"),
                        "method": "text_only_evaluation"
                    }
                })

            time.sleep(0.1)  # Rate limiting

        print(f"Found {len(irrelevant_samples)} irrelevant training samples out of {len(all_ids)}")
        return irrelevant_samples

    def _process_validation_data(self) -> List[Dict[str, Any]]:
        """Process validation data and identify irrelevant samples."""
        print("Processing validation data...")

        validation_items = self.validation_embeddings.get("items", [])
        print(f"Found {len(validation_items)} validation samples")

        if self.config["max_validation_samples"]:
            validation_items = validation_items[:self.config["max_validation_samples"]]
            print(f"Limited to {len(validation_items)} samples for processing")

        irrelevant_samples = []

        for item in tqdm(validation_items, desc="Processing validation samples"):
            metadata = item.get("metadata", {})
            image_url = metadata.get("image_url", "")
            question = metadata.get("question", "")
            sample_id = str(item.get("id", ""))

            if not image_url or not question:
                print(f"Skipping validation sample {sample_id}: missing image_url or question")
                continue

            evaluation = self._evaluate_relevance(question)

            if evaluation.get("success") and not evaluation.get("is_relevant"):
                irrelevant_samples.append({
                    "id": sample_id,
                    "image_url": image_url,
                    "question": question,
                    "crowd_majority": metadata.get("crowd_majority", ""),
                    "evaluation": {
                        "is_relevant": evaluation.get("is_relevant"),
                        "reason": evaluation.get("reason"),
                        "method": "text_only_evaluation"
                    }
                })

            time.sleep(0.1)  # Rate limiting

        print(f"Found {len(irrelevant_samples)} irrelevant validation samples out of {len(validation_items)}")
        return irrelevant_samples

    def collect_irrelevant_samples(self) -> None:
        """Run the collection process to identify irrelevant samples."""
        print("=" * 60)
        print("STEP 1: Identifying Irrelevant Samples")
        print("=" * 60)

        train_irrelevant = self._process_train_data()
        validation_irrelevant = self._process_validation_data()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_output = self.results_path / "train_to_discard.json"
        validation_output = self.results_path / "validation_to_discard.json"

        with open(train_output, "w", encoding="utf-8") as f:
            json.dump(train_irrelevant, f, indent=2, ensure_ascii=False)

        with open(validation_output, "w", encoding="utf-8") as f:
            json.dump(validation_irrelevant, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("Collection Completed!")
        print(f"Training samples to discard: {len(train_irrelevant)} -> {train_output}")
        print(f"Validation samples to discard: {len(validation_irrelevant)} -> {validation_output}")
        print("\n📋 NEXT STEP: Manually review the discarded samples")
        print("   Edit manual_review_ids.txt with any IDs you want to keep")
        print("   Then run: python clean_dataset.py --mode clean")

    def _load_discard_ids(self, file_path: Path) -> Set[str]:
        """Load IDs from discard JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            discard_data = json.load(f)
        return {str(item['id']) for item in discard_data}

    def clean_evaluation_file(self) -> None:
        """Clean the evaluation JSONL file by removing discarded samples."""
        print("=" * 60)
        print("STEP 2: Cleaning Evaluation File")
        print("=" * 60)

        validation_discard_ids = self._load_discard_ids(self.results_path / "validation_to_discard.json")
        train_discard_ids = self._load_discard_ids(self.results_path / "train_to_discard.json")

        print(f"Loaded {len(validation_discard_ids)} validation IDs to discard")
        print(f"Loaded {len(train_discard_ids)} train IDs to discard")

        # Input file (original evaluation results)
        input_file = self.base_path / "data" / "evaluation_gemini_2_5_pro_20250713_203535.jsonl"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.base_path / "data" / f"evaluation_gemini_2_5_pro_cleaned_{timestamp}.jsonl"

        cleaned_evaluations = []
        eliminated_count = 0
        total_lines = 0

        print(f"\nProcessing evaluation file...")

        with jsonlines.open(input_file, 'r') as reader:
            for line in reader:
                total_lines += 1

                validation_id = line.get('validation_id')
                similar_images = line.get('similar_images', [])

                # Skip if validation_id should be eliminated
                if validation_id in validation_discard_ids:
                    eliminated_count += 1
                    print(f"   ❌ Eliminating validation_id {validation_id}")
                    continue

                # Filter out contaminated train samples from similar_images
                clean_similar_images = [
                    img for img in similar_images
                    if str(img.get('id', '')) not in train_discard_ids
                ]

                # Update the line with cleaned similar_images
                line['similar_images'] = clean_similar_images
                cleaned_evaluations.append(line)

        # Save cleaned evaluations
        with jsonlines.open(output_file, 'w') as writer:
            for evaluation in cleaned_evaluations:
                writer.write(evaluation)

        print("\n" + "=" * 60)
        print("Cleaning Completed!")
        print(f"Total evaluations processed: {total_lines}")
        print(f"Evaluations eliminated: {eliminated_count}")
        print(f"Evaluations in cleaned file: {len(cleaned_evaluations)}")
        print(f"\nCleaned file saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean VizWiz evaluation dataset")
    parser.add_argument(
        "--mode",
        choices=["collect", "clean"],
        required=True,
        help="Mode: 'collect' to identify irrelevant samples, 'clean' to remove them"
    )

    args = parser.parse_args()

    cleaner = DatasetCleaner(config=CONFIG)

    if args.mode == "collect":
        cleaner.collect_irrelevant_samples()
    elif args.mode == "clean":
        cleaner.clean_evaluation_file()


if __name__ == "__main__":
    main()
