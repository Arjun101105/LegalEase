#!/usr/bin/env python3
"""
LLM Integration Module for Enhanced Legal Text Simplification
Supports multiple LLM providers for better, more natural simplification
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from pathlib import Path

class LLMSimplifier:
    """Enhanced simplification using Large Language Models"""
    
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load LLM configuration"""
        return {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo",
                "base_url": "https://api.openai.com/v1"
            },
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": "claude-3-haiku-20240307"
            },
            "local": {
                "model": "ollama/llama2:7b",
                "base_url": "http://localhost:11434"
            },
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY"),
                "model": "llama3-8b-8192",
                "base_url": "https://api.groq.com/openai/v1"
            }
        }
    
    def create_legal_prompt(self, legal_text: str) -> str:
        """Create an effective prompt for legal text simplification"""
        return f"""You are a legal expert who explains complex Indian court documents to common citizens. 

TASK: Transform this legal text into simple, clear English that any Indian citizen can understand.

RULES:
1. Use everyday words instead of legal jargon
2. Explain what happened in simple terms
3. Break down complex sentences
4. Use "The court said..." instead of "The court held..."
5. Explain legal terms in parentheses when first used
6. Keep it conversational but respectful

LEGAL TEXT TO SIMPLIFY:
{legal_text}

SIMPLIFIED VERSION:"""

    def simplify_with_openai(self, legal_text: str) -> Optional[str]:
        """Simplify using OpenAI GPT models"""
        if not self.config["openai"]["api_key"]:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.config['openai']['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config["openai"]["model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal expert who explains court documents to Indian citizens in simple, clear language."
                    },
                    {
                        "role": "user", 
                        "content": self.create_legal_prompt(legal_text)
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.config['openai']['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"OpenAI API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return None
    
    def simplify_with_groq(self, legal_text: str) -> Optional[str]:
        """Simplify using Groq (Fast Llama models)"""
        if not self.config["groq"]["api_key"]:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.config['groq']['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config["groq"]["model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal expert who explains Indian court documents in simple, clear language for common citizens."
                    },
                    {
                        "role": "user",
                        "content": self.create_legal_prompt(legal_text)
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.config['groq']['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Groq API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Groq Error: {e}")
            return None
    
    def simplify_with_local_llm(self, legal_text: str) -> Optional[str]:
        """Simplify using local Ollama model"""
        try:
            data = {
                "model": "llama2:7b",
                "prompt": self.create_legal_prompt(legal_text),
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.config['local']['base_url']}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"].strip()
            else:
                print(f"Local LLM Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Local LLM Error: {e}")
            return None
    
    def enhance_simplification(self, legal_text: str, fallback_simple: str) -> str:
        """Try LLM enhancement, fallback to current system"""
        print("🤖 Attempting LLM-enhanced simplification...")
        
        # Try different providers in order of preference
        providers = [
            ("Groq (Fast & Affordable)", self.simplify_with_groq),
            ("OpenAI GPT", self.simplify_with_openai),
            ("Local Ollama", self.simplify_with_local_llm)
        ]
        
        for provider_name, provider_func in providers:
            print(f"   Trying {provider_name}...")
            result = provider_func(legal_text)
            if result:
                print(f"✅ Successfully enhanced with {provider_name}")
                return self.post_process_llm_output(result)
        
        print("⚠️  LLM enhancement failed, using current system")
        return fallback_simple
    
    def post_process_llm_output(self, llm_output: str) -> str:
        """Clean and format LLM output"""
        # Remove any remaining legal jargon
        replacements = {
            "impugned": "challenged",
            "per contra": "on the other hand",
            "inter alia": "among other things",
            "vide": "see",
            "vis-a-vis": "compared to",
            "suo moto": "on its own",
            "prima facie": "at first glance"
        }
        
        result = llm_output
        for legal_term, simple_term in replacements.items():
            result = result.replace(legal_term, simple_term)
        
        return result

# Demo function to show LLM vs current system
def demo_llm_enhancement():
    """Demonstrate LLM enhancement vs current system"""
    
    current_output = """In simple terms: The appellants have submitted this appeal according to Article 136 of the Indian Constitution of India challenging the judgment and order dated 15.06.2024 passed by the High Court of Judicature at Allahabad in formal legal request No. 12345/2023."""
    
    llm_simplifier = LLMSimplifier()
    
    sample_text = """The appellants have submitted this appeal under Article 136 of the Constitution of India challenging the judgment and order dated 15.06.2024 passed by the High Court of Judicature at Allahabad in Writ Petition No. 12345/2023."""
    
    print("🔄 CURRENT SYSTEM OUTPUT:")
    print(current_output)
    print("\n" + "="*60)
    
    enhanced = llm_simplifier.enhance_simplification(sample_text, current_output)
    print("\n🚀 LLM-ENHANCED OUTPUT:")
    print(enhanced)

if __name__ == "__main__":
    demo_llm_enhancement()