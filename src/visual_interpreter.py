import openai
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import Content, Part, Tool, GenerateContentConfig, GoogleSearch
from google.genai import errors
import requests
import yaml
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import time, threading
from collections import deque
import base64
import mimetypes


def load_api_keys():
    """Load API keys from the api_keys.yml file."""
    api_keys_path = Path(__file__).parent.parent / "configs" / "api_keys.yml"
    if not api_keys_path.exists():
        raise FileNotFoundError(
            "api_keys.yml not found. Please add api_keys.yml to configs "
            "and fill in your API keys."
        )
    with open(api_keys_path) as f:
        return yaml.safe_load(f)

# Load API keys once at module level
API_KEYS = load_api_keys()

# Base interface
class BaseModel:

    _LOCK          = threading.Lock()
    _WINDOWS       = {}   
    _RATE_LIMITS   = {}


    @classmethod
    def set_rate_limit(cls, model_name: str, rpm: int):
        """Register or update a requests-per-minute limit for a model slug."""
        with cls._LOCK:
            cls._RATE_LIMITS[model_name] = rpm
            cls._WINDOWS.setdefault(model_name, deque())

    def __init__(self, name: str):
        self.name = name
        BaseModel.set_rate_limit(self.name, BaseModel._RATE_LIMITS.get(self.name, 10))

    def _block_if_needed(self):
        """
        Ensure this model stays below its RPM.
        Called immediately before every network request.
        """
        rpm   = BaseModel._RATE_LIMITS[self.name]
        win   = BaseModel._WINDOWS[self.name]
        now   = time.time()

        # drop timestamps older than 60 s
        while win and now - win[0] >= 60:
            win.popleft()

        if len(win) >= rpm:
            sleep_for = 60 - (now - win[0]) + 0.01  # small buffer
            print(f"[{self.name}] ⏳  rate limit reached. Sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            # after sleep, purge again
            now = time.time()
            while win and now - win[0] >= 60:
                win.popleft()

        win.append(now)

    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        mode: str,
        image_urls: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict, Dict, List[Dict[str, str]]]:
        """
        Accept either a prompt string or a message-history list.
        Normalize to a messages list, call _call_model, then append the assistant response.
        
        Args:
            prompt_or_messages: Either a string prompt or a list of message dictionaries
            mode: The generation mode (e.g., "standard", "vision")
            image_urls: Optional list of image URLs to include in the prompt
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            Tuple containing:
                - text: The generated text response
                - metadata: Dictionary with metadata about the generation
                - raw_response: Raw response from the model provider
                - updated_messages: The updated message history including the new response
        """

        self._block_if_needed()

        # Normalize input
        if isinstance(prompt_or_messages, str):
            # Delegate to specialised helper so subclasses can customise message
            messages: List[Dict[str, object]] = [
                self._build_user_message(prompt_or_messages, image_urls)
            ]
        else:
            # copy so we don't mutate the caller's list
            messages = list(prompt_or_messages)

            # Warn if caller tries to attach images when a full message history is supplied
            if image_urls:
                print("[WARN] image_urls were provided but prompt_or_messages is already a list. Ignoring image_urls.")

        # Call the provider-specific implementation
        text, metadata, raw = self._call_model(messages, mode, system_prompt)

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": text})

        return text, metadata, raw, messages

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict, Dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Extension hook – can be overridden by subclasses that require
    # provider-specific message formats (e.g. OpenAI vision with images).
    # ------------------------------------------------------------------
    def _build_user_message(
        self,
        text: str,
        image_urls: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """Return a properly structured *user* message.

        The default implementation handles plain-text messages. Subclasses
        may override to inject multimodal payloads (e.g. images).
        """

        # Default: plain text only.
        return {"role": "user", "content": text}

# OpenAI implementation
class OpenAIModel(BaseModel):
    def __init__(self, name: str, model: str):
        super().__init__(name)
        openai.api_key = API_KEYS["openai"]["api_key"]
        self.model = model

    # --------------------------------------------------------------
    # Provider-specific message builder to support image inputs.
    # --------------------------------------------------------------
    def _build_user_message(
        self,
        text: str,
        image_urls: Optional[List[str]] = None
    ) -> Dict[str, object]:
        # When images are supplied, construct a multimodal message as per
        # OpenAI Vision guidelines. Otherwise fall back to the default
        # behaviour.
        if image_urls:
            content: List[Dict[str, object]] = [{"type": "input_text", "text": text}]
            for url in image_urls:
                content.append({
                    "type": "input_image",
                    "image_url":  url
                })
            return {"role": "user", "content": content}

        return super()._build_user_message(text)

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict, Dict]:
        client = OpenAI()

        params: Dict[str, object] = {
            "model": self.model,
            "input": messages,  # `input` accepts list of message dicts (text or multimodal)
            "temperature": 0,
            "user": "1",
            "max_output_tokens": 1024,
        }

        # Attach a system prompt via the `instructions` field when provided
        if system_prompt:
            params["instructions"] = system_prompt

        # include web-search tool if requested (feature-gated; may require beta access)
        if mode == "web-search":
            params["tools"] = [{"type": "web_search_preview"}]

        try:
            resp = client.responses.create(**params)
        except Exception as e:
            return f"[ERROR] OpenAI request failed ({e})", {}, {}

        text = getattr(resp, "output_text", None)
        if not text:
            return "[ERROR] Empty response from OpenAI", {}, resp.__dict__

        metadata = getattr(resp, "output", {})
        return text.strip(), metadata, resp

class AnthropicModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        
        self.client = anthropic.Anthropic( api_key = API_KEYS["anthropic"]["api_key"])
        self.model  = model

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict, Dict]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = self.client.messages.create(**kwargs)
        metadata = {
            "model":     getattr(resp, "model", None),
            "usage":     getattr(resp, "usage", {}),
            "id":        getattr(resp, "id", None),
        }

        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        ).strip()

        return text, metadata, resp

    # --------------------------------------------------------------
    # Provider-specific message builder to support image inputs.
    # --------------------------------------------------------------
    def _build_user_message(
        self,
        text: str,
        image_urls: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """Create a user message compatible with Anthropic Messages API.

        The API expects `content` to be an array of blocks where each block
        has a `type` field. For images we embed base64-encoded payloads. If
        callers pass raw URLs we fetch and convert them here. If they pass a
        string beginning with `data:` or already base64 we embed directly.
        """

        # Start with an empty list and add text block only if provided
        content_blocks: List[Dict[str, object]] = []

        if text and text.strip():
            content_blocks.append({"type": "text", "text": text})

        if image_urls:
            for url in image_urls:
                try:
                    if url.startswith("data:"):
                        # Already a data URI – extract metadata.
                        header, data_part = url.split(",", 1)
                        media_type = header.split(";")[0][5:] if ";" in header else "image/jpeg"
                        encoded_data = data_part
                    elif url.startswith("http"):
                        # Fetch the remote image and encode.
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        media_type = response.headers.get("Content-Type", "image/jpeg")
                        encoded_data = base64.b64encode(response.content).decode("utf-8")
                    else:
                        # Assume it's a local file path.
                        with open(url, "rb") as f:
                            data = f.read()
                        media_type = "image/jpeg"  # best guess; callers should prefer data URI
                        encoded_data = base64.b64encode(data).decode("utf-8")

                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_data,
                        },
                    })
                except Exception as e:
                    print(f"[WARN] Failed to process image '{url}' – {e}. Skipping.")

        return {"role": "user", "content": content_blocks}

class GeminiModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.client = genai.Client(api_key=API_KEYS["gemini"]["api_key"])
        self.model  = model

    # --------------------------------------------------------------
    # Message builder to add multimodal (image) support.
    # --------------------------------------------------------------
    def _build_user_message(
        self,
        text: str,
        image_urls: Optional[List[str]] = None
    ) -> Dict[str, object]:
        # If no images, fallback to default implementation (plain text)
        if not image_urls:
            return super()._build_user_message(text)

        parts: List[Part] = [Part(text=text)]

        for url in image_urls:
            try:
                # Determine byte payload and mime type
                if url.startswith("data:"):
                    header, data_part = url.split(",", 1)
                    mime_type = header.split(";")[0][5:]
                    image_bytes = base64.b64decode(data_part)
                elif url.startswith("http"):
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    mime_type = response.headers.get("Content-Type", "image/jpeg")
                    image_bytes = response.content
                else:
                    # local file path
                    mime_type, _ = mimetypes.guess_type(url)
                    mime_type = mime_type or "image/jpeg"
                    with open(url, "rb") as f:
                        image_bytes = f.read()

                parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
            except Exception as e:
                print(f"[WARN] Gemini: failed to process image '{url}' – {e}. Skipping.")

        # Gemini Content will be created later in _call_model
        return {"role": "user", "parts": parts}

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict, Dict]:
        tools = []

        # Convert our lightweight message dicts into Gemini Content objects
        content_objects: List[Content] = []

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg:
                print(f"Warning: Skipping invalid message format at index {i}: {msg}")
                continue

            role = 'model' if msg.get('role', 'user').lower() in ['assistant', 'model'] else 'user'

            if 'parts' in msg and isinstance(msg['parts'], list):
                parts_list = msg['parts']
            else:
                content_text = msg.get('content', "")
                if not isinstance(content_text, str):
                    content_text = str(content_text)
                parts_list = [Part(text=content_text)]

            try:
                content_objects.append(Content(role=role, parts=parts_list))
            except Exception as e:
                print(f"Error building Content for Gemini at index {i}: {e}")
                return f"Error: SDK object creation failed ({e})", {}, {}

        # Enable web search tool when needed
        if mode == "standard":
            tools = []
        elif mode == "web-search" or len(messages) > 1:
            tools = [Tool(google_search=GoogleSearch())]

        config_kwargs = {
            "tools": tools,
            "response_modalities": ["TEXT"],
            "max_output_tokens": 8000,
            "temperature": 0,
        }
        if system_prompt:
            # Gemini SDK uses `system_instruction` for system prompts
            config_kwargs["system_instruction"] = system_prompt

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=content_objects,
                config=GenerateContentConfig(**config_kwargs)
            )
        except errors.APIError as e:
            print(e.code)
            print(e.message)
            return f"[ERROR] Gemini APIError {e.code}: {e.message}", {}, {}

        # ----------- Parse response -------------
        if not getattr(response, "candidates", None):
            return "[ERROR] Gemini returned no candidates", {}, response

        cand = response.candidates[0]
        if not getattr(cand, "content", None):
            return "[ERROR] Candidate had no content", {}, response

        parts = getattr(cand.content, "parts", None)
        if not parts:
            return "[ERROR] Candidate content.parts empty", {}, response

        text_chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        if not text_chunks:
            return "[ERROR] No text parts in response", {}, response

        text = "".join(text_chunks).strip()
        metadata = {}
        if hasattr(cand, 'grounding_metadata') and hasattr(cand.grounding_metadata, 'search_entry_point'):
            metadata['search_content'] = cand
        return text, metadata, response



def create_models(model_configs):
    models = {}
    for cfg in model_configs:
        name, prov = cfg["name"], cfg["provider"]
        if prov == "openai":
            models[name] = OpenAIModel(name, cfg["model"])
        elif prov == "anthropic":
            models[name] = AnthropicModel(name, cfg["model"])
        elif prov == "gemini":
            models[name] = GeminiModel(name, cfg["model"])
        else:
            raise ValueError(f"Unknown provider: {prov}")
    return models
