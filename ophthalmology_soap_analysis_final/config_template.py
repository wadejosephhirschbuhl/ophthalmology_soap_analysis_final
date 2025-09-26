"""
Configuration template for the ophthalmology SOAP note analysis pipeline.

Copy this file to `config.py` and fill in your API keys and model name.
The notebooks and scripts will import `config` and use these values.
"""

# API key for the Dartmouth gateway (required to call the GPT‑5 model)
DARTMOUTH_API_KEY = ""

# Chat API key for multi‑turn interactions (optional but recommended)
DARTMOUTH_CHAT_API_KEY = ""

# Name of the GPT‑5 model to use for note generation
SELECTED_MODEL = "openai.gpt-5-mini-2025-08-07"