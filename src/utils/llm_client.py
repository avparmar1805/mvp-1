import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

class LLMClient:
    """Client for interacting with LLM APIs (OpenAI or Gemini)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.provider = "none"
        self.openai_client = None
        self.gemini_model = None
        
        # 1. Try OpenAI first (more reliable for text generation)
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key and OPENAI_AVAILABLE:
            self.provider = "openai"
            self.client = OpenAI(api_key=openai_key)
            self.model = model or "gpt-4o"
            return

        # 2. Fallback to Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and GEMINI_AVAILABLE:
            self.provider = "gemini"
            genai.configure(api_key=gemini_key)
            self.model = model or "gemini-1.5-flash"
            self.gemini_model = genai.GenerativeModel(self.model)
            return
        
        # No valid provider found
        print("Warning: No valid LLM provider configured (OpenAI or Gemini).")

    def generate_text(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Generate a simple text response"""
        if self.provider == "gemini" and self.gemini_model:
            try:
                # Gemini doesn't have a direct system prompt in `generate_content`, 
                # usually it's prepended or passed in config, but prepending is safer for basic use.
                full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
                response = self.gemini_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                print(f"Error calling Gemini: {e}")
                return ""
        
        elif self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI: {e}")
                return ""
        return ""

    def generate_structured_output(self, prompt: str, response_schema: Dict[str, Any], system_prompt: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """Generate a structured JSON response matching the given schema"""
        schema_json = json.dumps(response_schema, indent=2)
        
        if self.provider == "gemini" and self.gemini_model:
            try:
                 # Gemini Pro often outputs Markdown JSON, so we ask for it and strip formatting
                full_prompt = f"{system_prompt}\n\nTask: Output valid JSON matching this schema:\n{schema_json}\n\nUser Request: {prompt}"
                
                # Use generation config to enforce JSON if possible, or just prompt engineering
                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0
                    )
                )
                text = response.text
                # Clean markdown code blocks if present
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                    
                return json.loads(text.strip())
            except Exception as e:
                print(f"Error calling Gemini JSON: {e}")
                return {}

        elif self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"{system_prompt} Respond in structured JSON format matching this schema:\n{schema_json}"},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                print(f"Error calling OpenAI or parsing JSON: {e}")
                return {}
        return {}
