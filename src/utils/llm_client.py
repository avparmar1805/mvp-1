import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    """Client for interacting with LLM APIs (OpenAI)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # For testing without credentials, we might want to warn or mock
            # raise ValueError("OPENAI_API_KEY is not set")
            pass
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_text(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Generate a simple text response"""
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
            print(f"Error calling LLM: {e}")
            return ""

    def generate_structured_output(self, prompt: str, response_schema: Dict[str, Any], system_prompt: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """Generate a structured JSON response matching the given schema"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{system_prompt} Respond in structured JSON format matching this schema:\n{json.dumps(response_schema, indent=2)}"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Error calling LLM or parsing JSON: {e}")
            return {}
