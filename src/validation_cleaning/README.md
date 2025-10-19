# Dataset Cleaning Process

This directory contains the script and documentation for cleaning the VizWiz evaluation dataset used in our research project. The cleaning process identifies and removes question-image pairs with irrelevant or poorly-formed questions that are not suitable for training vision-language models.

## Overview

The VizWiz dataset contains user-submitted questions about images. Some questions are:
- Not related to the image content (e.g., "Where can I buy this?")
- Incomplete or require conversational context (e.g., "Oh so then what?")
- Generic greetings or thanks (e.g., "Thanks for your help")
- Statements rather than questions (e.g., "This image is blurry")

These samples are not useful for training vision-language models and were removed from our evaluation dataset.

## Process

### Step 1: Automatic Collection of Irrelevant Samples

Run the LLM-based collection to identify potentially irrelevant samples:

```bash
python clean_dataset.py --mode collect
```

This will:
1. Load all training samples from the ChromaDB vector database
2. Load all validation samples from the embeddings file
3. Use Gemini 2.5 Pro to evaluate each question (text-only, no image shown to avoid bias)
4. Generate two files:
   - `train_to_discard.json` - Training samples identified as irrelevant
   - `validation_to_discard.json` - Validation samples identified as irrelevant

**Note:** The LLM evaluates only the question text, not the image, to avoid bias. This ensures questions are evaluated based on whether they are clear, standalone, and suitable for vision-language training.

### Step 2: Manual Review

After automatic collection, manually review the identified samples to confirm they should be discarded:

1. Review the samples in `train_to_discard.json` and `validation_to_discard.json`
2. Check `garbage_collection_report_*.md` for detailed explanations
3. Document any IDs you want to keep despite being flagged in `manual_review_ids.txt`

**Our manual review results:** See `manual_review_ids.txt` for the IDs reviewed and confirmed for removal.

### Step 3: Clean the Evaluation File

Once manual review is complete, clean the final evaluation JSONL file:

```bash
python clean_dataset.py --mode clean
```

This will:
1. Load the discard lists from Step 1
2. Process the original evaluation file (`evaluation_gemini_2_5_pro_20250713_203535.jsonl`)
3. Remove evaluations where `validation_id` is in the discard list
4. Remove contaminated train samples from `similar_images` context
5. Save the cleaned file with timestamp

## Files in This Directory

- **clean_dataset.py** - Main script with two modes:
  - `--mode collect`: Identify irrelevant samples using LLM
  - `--mode clean`: Clean evaluation file after manual review
- **manual_review_ids.txt** - Manual review results confirming which IDs to discard
- **train_to_discard.json** - Training samples identified as irrelevant (9 samples)
- **validation_to_discard.json** - Validation samples identified as irrelevant (8 samples)
- **garbage_collection_report_*.md** - Detailed report with explanations for each discarded sample
- **README.md** - This documentation file

## Results

**Original dataset:**
- Training samples: 500
- Validation samples: 100
- Total evaluation lines: 200

**After cleaning:**
- Training samples removed: 9
- Validation samples removed: 8
- Final evaluation lines: 184 (16 lines removed)

**Note:** The evaluation file contains 200 lines initially, but after removing 8 validation samples and filtering contaminated train samples from context, we ended up with 184 evaluation lines in the cleaned file.

## Evaluation Criteria

Questions were marked as **NOT RELEVANT** if they:
- Are greetings or thanks ("Hello", "Thanks")
- Are nonsensical or just punctuation ("???", "...")
- Complain about image quality instead of asking something useful
- Are statements/comments rather than questions
- Use ambiguous pronouns without clear referents ("it", "this", "that")
- Require context from previous conversation ("Oh so...", "So then...")
- Are conversational fragments rather than standalone questions
- Ask for external knowledge not visible in images ("Where can I buy this?", "How many calories?")
- Are incomplete or vague

Questions were marked as **RELEVANT** if they:
- Ask something specific about visual content
- Are clear and standalone
- Could reasonably be answered by looking at an image
- Would help train a vision-language model

## Examples of Discarded Questions

**Training samples discarded:**
1. "How many calories are in a slice of pepperoni and cheese pizza?" (requires external knowledge)
2. "Where do you think I could get a power cord?" (requires external knowledge)
3. "I know but how long do I leave it in the oven for?" (conversational fragment)
4. "This image does not show who the mail is for." (statement, not a question)
5. "What color is..." (incomplete)
6. "Oh so do I leave it in for twelve months then do I?" (conversational fragment)

**Validation samples discarded:**
1. "Where can I buy this from?" (requires external knowledge)
2. "Is this picture better?" (requires previous context)
3. "Where is it made?" (ambiguous pronoun)
4. "Which ocean has the most hurricanes in the world?" (general knowledge, not about image)
5. "I need some help using this application." (general request, not a question)
6. "Do you think my wife's decorating is any good?" (subjective opinion)

## Citation

If you use this cleaning methodology, please cite our paper:

```bibtex
@article{your-paper-2025,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```
