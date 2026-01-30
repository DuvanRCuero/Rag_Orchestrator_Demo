"""
Demonstration of Factory Pattern for Multi-LLM Support

This script demonstrates how to use the new LLM Factory Pattern to switch
between different LLM providers without modifying existing code.
"""

import os
import asyncio

# Set environment variables
os.environ['ENVIRONMENT'] = 'testing'
os.environ['OPENAI_API_KEY'] = 'your-openai-key'
os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'  # Optional
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'  # Optional

from src.infrastructure.llm import LLMFactory


def demonstrate_factory():
    """Demonstrate the Factory Pattern usage."""
    
    print("=" * 70)
    print("FACTORY PATTERN DEMONSTRATION")
    print("=" * 70)
    print()
    
    # 1. List available providers
    print("1. Available LLM Providers:")
    providers = LLMFactory.available_providers()
    for provider in providers:
        print(f"   • {provider}")
    print()
    
    # 2. Create OpenAI provider
    print("2. Creating OpenAI Provider:")
    openai_llm = LLMFactory.create('openai')
    print(f"   ✓ Provider: {type(openai_llm).__name__}")
    print(f"   ✓ Model: {openai_llm.model}")
    print()
    
    # 3. Create Local LLM provider
    print("3. Creating Local LLM Provider:")
    local_llm = LLMFactory.create('local')
    print(f"   ✓ Provider: {type(local_llm).__name__}")
    print(f"   ✓ Base URL: {local_llm.base_url}")
    print(f"   ✓ Model: {local_llm.model}")
    print()
    
    # 4. Switch provider via environment
    print("4. Switching Provider via Environment:")
    print("   Set LLM_PROVIDER=anthropic in environment")
    print("   Then call: LLMFactory.create()")
    print()
    
    print("=" * 70)
    print("How to Add a New Provider:")
    print("=" * 70)
    print("""
1. Create a new class implementing LLMService interface:
   
   class MyCustomLLM(LLMService):
       async def generate(self, messages, **kwargs):
           # Your implementation
           pass
       
       async def generate_json(self, messages, **kwargs):
           # Your implementation
           pass
       
       async def stream_generation(self, messages, **kwargs):
           # Your implementation
           pass
       
       async def get_token_usage(self, text):
           # Your implementation
           pass

2. Register it with the factory in src/infrastructure/llm/__init__.py:
   
   from src.infrastructure.llm.my_custom_llm import MyCustomLLM
   LLMFactory.register("custom", MyCustomLLM)

3. Use it:
   
   export LLM_PROVIDER=custom
   # Or in code:
   llm = LLMFactory.create("custom")
""")
    print()
    
    print("=" * 70)
    print("Usage Examples:")
    print("=" * 70)
    print("""
# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Anthropic Claude
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
export ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Local/Ollama
export LLM_PROVIDER=local
export OLLAMA_BASE_URL=http://localhost:11434
export LOCAL_MODEL=llama2
""")


async def demonstrate_async_usage():
    """Demonstrate async usage of LLM providers."""
    
    print("=" * 70)
    print("ASYNC USAGE EXAMPLE")
    print("=" * 70)
    print()
    
    # Create a provider
    llm = LLMFactory.create('openai')
    print(f"Created provider: {type(llm).__name__}")
    print()
    
    # In a real scenario, you would call:
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": "Hello!"}
    # ]
    # response = await llm.generate(messages)
    # print(f"Response: {response}")
    
    print("Note: To actually make API calls, provide valid API keys")
    print("=" * 70)


if __name__ == "__main__":
    # Run synchronous demonstration
    demonstrate_factory()
    
    # Run async demonstration
    asyncio.run(demonstrate_async_usage())
