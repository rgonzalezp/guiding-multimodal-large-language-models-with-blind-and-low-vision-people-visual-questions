# Source Code Documentation

This directory contains the main source code for the VizWiz evaluation project, which evaluates vision-language models (VLMs) with and without retrieval-augmented generation (RAG) context.

## Project Overview

This project evaluates how VLMs perform when given contextual information from similar images (RAG approach) compared to without context. The workflow involves:

1. **Generating evaluations** - Running VLMs on validation samples with/without context
2. **Cleaning the dataset** - Removing irrelevant questions that aren't suitable for VLM training
3. **Analyzing results** - Comparing performance with and without RAG context

## Directory Structure

```
src/
├── data/                                    # Data files
│   ├── evaluation_gemini_2_5_pro_20250713_203535.jsonl           # Original evaluations
│   └── evaluation_gemini_2_5_pro_cleaned_20250728_022449.jsonl   # Cleaned evaluations
├── validation_cleaning/                     # Dataset cleaning
│   ├── clean_dataset.py                    # Main cleaning script (collect + clean modes)
│   ├── train_to_discard.json              # Training samples to discard (9 samples)
│   ├── validation_to_discard.json         # Validation samples to discard (8 samples)
│   ├── manual_review_ids.txt              # Manual review results
│   ├── garbage_collection_report_*.md      # Detailed discard report
│   └── README.md                          # Cleaning process documentation
├── evaluate_validation_dataset.py          # Main evaluation script
├── vector_db.py                            # ChromaDB vector database interface
├── embeddings_utils.py                     # Cohere embedding utilities
└── visual_interpreter.py                   # Multi-model VLM interface
```

## Main Scripts

### 1. evaluate_validation_dataset.py

**Purpose:** Main script to generate evaluation results by running VLMs on validation samples.

**What it does:**
- Loads validation samples from pre-computed embeddings
- For each sample, retrieves K similar training images from ChromaDB (RAG context)
- Runs VLM in two modes:
  - **With context:** Includes questions from similar images as context
  - **Without context:** No additional context, just the base prompt
- Saves results to JSONL file with all evaluation metadata

**Configuration:**
```python
EVALUATION_CONFIG = {
    "embedding_provider": "cohere",      # Embedding model used
    "with_context": True,                # Evaluate with RAG context
    "without_context": True,             # Evaluate without context
    "top_k_similar": 4,                  # Number of similar images to retrieve
    "max_validation_samples": None,      # Limit samples (None = all)
    "specific_validation_id": "295",     # Evaluate specific ID (optional)
}

MODEL_CONFIGS = [
    {"name": "gemini-2.5-pro", "provider": "gemini", "model": "gemini-2.5-pro"},
]
```

**Usage:**
```bash
cd src
python evaluate_validation_dataset.py
```

**Output:** Creates `evaluation_{provider}_{timestamp}.jsonl` in `notebooks/data/results/`

**Output format (JSONL):**
```json
{
  "validation_id": "295",
  "model_name": "gemini-2.5-pro",
  "with_context": true,
  "embedding_provider": "cohere",
  "top_k_similar": 4,
  "image_url": "https://...",
  "real_question": "What color is the shirt?",
  "crowd_majority": "blue",
  "similar_images": [
    {"id": "123", "distance": 0.15, "question": "What is the color?", ...},
    ...
  ],
  "prompt_used": "Your goal is to optimize...",
  "llm_response": "The shirt in the image is blue...",
  "timestamp": "2025-07-28T01:30:00",
  "processing_time": 2.5
}
```

### 2. vector_db.py

**Purpose:** Interface for ChromaDB vector database to store and retrieve image embeddings.

**Key Features:**
- Persistent storage of multimodal embeddings
- Multiple collection support (train, validation, etc.)
- Cosine similarity search
- Built on Cohere's embed-v4.0 model

**Main Methods:**
```python
db = SimpleVectorDB(db_path="./data/chroma_db")

# Create/use collection
db.use_collection("vizwiz_500_sample_cosine", "Training samples")

# Add embedding
db.add_image_embedding(
    embedding_id="123",
    image_embedding=[0.1, 0.2, ...],  # 1024-dim vector
    question="What color is this?",
    image_url="https://...",
    crowd_majority="blue",
    ...
)

# Search similar
results = db.search_similar_images(
    query_embedding=[0.1, 0.2, ...],
    n_results=4
)
```

### 3. embeddings_utils.py

**Purpose:** Generate multimodal embeddings using Cohere's embed-v4.0 model.

**Usage:**
```python
from embeddings_utils import cohere_generate_image_embedding

# From URL
embedding = cohere_generate_image_embedding("https://example.com/image.jpg")

# From local path
embedding = cohere_generate_image_embedding("./images/photo.jpg")
```

**Returns:** 1024-dimensional float vector

### 4. visual_interpreter.py

**Purpose:** Unified interface for multiple vision-language models.

**Supported Models:**
- Google Gemini (gemini-2.5-pro, gemini-1.5-pro, etc.)
- OpenAI models (with vision support)
- Anthropic Claude (with vision support)

**Key Features:**
- Rate limiting (RPM management)
- Multimodal support (text + images)
- Conversation history tracking
- Standardized API across providers

**Usage:**
```python
from visual_interpreter import create_models

# Create models
models = create_models([
    {"name": "gemini-2.5-pro", "provider": "gemini", "model": "gemini-2.5-pro"}
])

model = models["gemini-2.5-pro"]

# Generate response
response, metadata, raw, messages = model.generate(
    prompt="Describe this image in detail",
    mode="standard",
    image_urls=["https://example.com/image.jpg"],
    system_prompt="You are a helpful assistant"
)
```

## Validation Cleaning Process

See `validation_cleaning/README.md` for complete documentation on the dataset cleaning process.

**Quick summary:**
1. **Identify irrelevant samples** using LLM evaluation (text-only, no image bias)
2. **Manual review** of flagged samples
3. **Clean evaluation file** by removing discarded samples

**Run cleaning:**
```bash
cd src/validation_cleaning

# Step 1: Identify irrelevant samples
python clean_dataset.py --mode collect

# Step 2: After manual review, clean the file
python clean_dataset.py --mode clean
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    1. DATA PREPARATION                       │
│  - VizWiz train (500 samples) → embeddings → ChromaDB       │
│  - VizWiz validation (100 samples) → embeddings → JSON      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. EVALUATION GENERATION                  │
│  evaluate_validation_dataset.py:                            │
│  - For each validation sample:                              │
│    - Retrieve K=4 similar train images (RAG context)        │
│    - Run VLM with context                                   │
│    - Run VLM without context                                │
│  - Save to evaluation_*.jsonl (200 lines)                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. DATASET CLEANING                       │
│  validation_cleaning/clean_dataset.py:                      │
│  - LLM identifies irrelevant questions                      │
│  - Manual review confirms decisions                         │
│  - Remove 8 validation + filter train context              │
│  - Save to evaluation_*_cleaned.jsonl (184 lines)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    4. ANALYSIS                               │
│  - Compare performance: with_context vs without_context     │
│  - Analyze on cleaned dataset                               │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Files

All configuration is stored in `../configs/`:

### prompts.yml
Contains all prompts used in the project:
- `be_my_ai_prompt` - System prompt for VLMs
- Context building prompts
- Evaluation criteria prompts

### api_keys.yml
API keys for all services:
```yaml
gemini:
  api_key: "your-gemini-key"
cohere:
  api_key: "your-cohere-key"
openai:
  api_key: "your-openai-key"
anthropic:
  api_key: "your-anthropic-key"
```

**Note:** Never commit this file to git. Add to `.gitignore`.

### models.yml & paid_models.yml
Model configurations for different providers.

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create `configs/api_keys.yml`:
```yaml
gemini:
  api_key: "your-gemini-api-key"
cohere:
  api_key: "your-cohere-api-key"
```

Or use `.env` file:
```
GEMINI_API_KEY=your-key
COHERE_API_KEY=your-key
```

### 3. Verify Setup

```bash
python -c "from src.visual_interpreter import create_models; print('Setup OK')"
```

## Key Dependencies

- **chromadb** - Vector database for embeddings
- **cohere** - Multimodal embeddings (embed-v4.0)
- **google-genai** - Gemini models
- **openai** - OpenAI models
- **anthropic** - Claude models
- **numpy** - Numerical operations
- **tqdm** - Progress bars
- **pyyaml** - Configuration files

## Common Tasks

### Evaluate a Specific Sample

Edit `evaluate_validation_dataset.py`:
```python
EVALUATION_CONFIG = {
    ...
    "specific_validation_id": "295",  # Your sample ID
}
```

Then run:
```bash
python evaluate_validation_dataset.py
```

### Change Number of Context Examples

Edit `evaluate_validation_dataset.py`:
```python
EVALUATION_CONFIG = {
    ...
    "top_k_similar": 4,  # Change to 2, 4, 8, etc.
}
```

### Use Different Model

Edit `evaluate_validation_dataset.py`:
```python
MODEL_CONFIGS = [
    {"name": "gemini-1.5-pro", "provider": "gemini", "model": "gemini-1.5-pro"},
]
```

### Clean Dataset with Different Criteria

Edit `validation_cleaning/clean_dataset.py` and modify `RELEVANCE_PROMPT` to adjust evaluation criteria.

## Important Notes

### Rate Limiting
- Visual interpreter implements per-model RPM limits
- Default: 10 requests/minute
- Adjust in `visual_interpreter.py`:
  ```python
  BaseModel.set_rate_limit("gemini-2.5-pro", 60)  # 60 RPM
  ```

### Embeddings
- All embeddings are pre-computed and stored
- Train embeddings: in ChromaDB (`notebooks/data/chroma_db/`)
- Validation embeddings: in JSON (`notebooks/data/embeddings/lf_vqa_validation_embeddings_cohere.json`)

### Persistence
- ChromaDB automatically persists to disk
- No need to manually save collections
- Data survives script restarts

### Error Handling
- All scripts handle API errors gracefully
- Failed samples are logged but don't stop execution
- Check output for `[ERROR]` or warning messages

## Troubleshooting

### "Could not load validation embeddings file"
- Ensure embeddings file exists at `notebooks/data/embeddings/lf_vqa_validation_embeddings_cohere.json`
- Check `embedding_provider` matches the file name

### "COHERE_API_KEY not found"
- Add to `configs/api_keys.yml` or `.env` file
- Restart your script after adding

### "Collection not found"
- Ensure ChromaDB is initialized with data
- Check `notebooks/data/chroma_db/` directory exists and has data

### Rate limit errors
- Increase wait time or decrease RPM limit
- Check API quotas in your provider dashboard

## Citation

If you use this code, please cite our paper:

```bibtex
@article{your-paper-2025,
  title={Guiding Multimodal Large Language Models with Blind and Low Vision People's Visual Questions},
  author={Your Name and Others},
  journal={Your Conference/Journal},
  year={2025}
}
```

## License

See LICENSE file in project root.
