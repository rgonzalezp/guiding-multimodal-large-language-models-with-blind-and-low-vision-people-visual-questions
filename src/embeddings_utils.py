import os
import base64
import requests
from typing import List
import cohere
from dotenv import load_dotenv

# load environment variables (for COHERE_API_KEY)
load_dotenv()

# initialize cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise EnvironmentError(
        "❌ COHERE_API_KEY not found. Please add it to your .env file."
    )

co = cohere.ClientV2(api_key=COHERE_API_KEY)


def cohere_generate_image_embedding(image_path: str) -> List[float]:
    """
    Generate a float embedding for an image via Cohere's embed-v4.0 model.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        List[float]: The embedding vector.
    """
    try:
        # handle both remote URLs and local paths
        if image_path.startswith(("http://", "https://")):
            resp = requests.get(image_path, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
            mime = resp.headers.get("Content-Type", "image/jpeg")
        else:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            mime = "image/jpeg"

        # convert image to data URI (base64)
        data_uri = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

        # request embedding from Cohere
        resp = co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[data_uri],
        )

        # extract embeddings
        embeddings = getattr(resp.embeddings, "float", [])
        if not embeddings:
            raise ValueError("No float embeddings returned from Cohere API.")

        return embeddings[0] if isinstance(embeddings[0], list) else embeddings

    except Exception as e:
        print(f"❌ Error generating image embedding: {e}")
        return []