# Factory Pattern for Multi-LLM Support

## Overview

This implementation follows the **Open/Closed Principle** - the system is open for extension (adding new LLM providers) but closed for modification (no need to change existing code).

## Architecture

```
┌─────────────────────────────────────────────────┐
│           LLMFactory (Registry)                 │
│  - register(name, provider_class)               │
│  - create(provider_name) -> LLMService          │
│  - available_providers() -> List[str]           │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼─────┐  ┌──────▼──────┐  ┌────▼────────┐
│  OpenAI     │  │  Anthropic  │  │  Local LLM  │
│  Service    │  │  Service    │  │  Service    │
└─────────────┘  └─────────────┘  └─────────────┘
        │               │               │
        └───────────────┴───────────────┘
                        │
            ┌───────────▼───────────┐
            │   LLMService          │
            │   (Interface)         │
            │   - generate()        │
            │   - generate_json()   │
            │   - stream_generation()│
            │   - get_token_usage() │
            └───────────────────────┘
```

## Supported Providers

### 1. OpenAI (Default)
- **Model**: GPT-3.5-turbo, GPT-4, etc.
- **Features**: Chat completion, streaming, JSON mode
- **Configuration**:
  ```bash
  export LLM_PROVIDER=openai
  export OPENAI_API_KEY=your-key
  export OPENAI_MODEL=gpt-3.5-turbo
  ```

### 2. Anthropic Claude
- **Model**: Claude 3 Sonnet, Opus, Haiku
- **Features**: Chat completion, streaming, vision support
- **Configuration**:
  ```bash
  export LLM_PROVIDER=anthropic
  export ANTHROPIC_API_KEY=your-key
  export ANTHROPIC_MODEL=claude-3-sonnet-20240229
  ```

### 3. Local LLM (Ollama)
- **Model**: Llama 2, Mistral, CodeLlama, etc.
- **Features**: Local inference, privacy-preserving
- **Configuration**:
  ```bash
  export LLM_PROVIDER=local  # or 'ollama'
  export OLLAMA_BASE_URL=http://localhost:11434
  export LOCAL_MODEL=llama2
  ```

## Usage

### Basic Usage

```python
from src.infrastructure.llm import LLMFactory

# Create a provider (uses LLM_PROVIDER from settings)
llm = LLMFactory.create()

# Or specify explicitly
llm = LLMFactory.create("openai")
llm = LLMFactory.create("anthropic")
llm = LLMFactory.create("local")

# Generate text
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
]
response = await llm.generate(messages)
```

### List Available Providers

```python
from src.infrastructure.llm import LLMFactory

providers = LLMFactory.available_providers()
print(providers)  # ['openai', 'anthropic', 'local', 'ollama']
```

### Via API Endpoint

```bash
# List available providers
curl http://localhost:8000/api/v1/monitoring/providers

# Response:
{
  "available_providers": ["openai", "anthropic", "local", "ollama"],
  "current_provider": "openai"
}
```

## Adding a New Provider

### Step 1: Implement the Interface

Create a new file `src/infrastructure/llm/my_provider_client.py`:

```python
from typing import Any, AsyncIterator, Dict, List, Optional
from src.domain.interfaces.llm_service import LLMService
from src.core.config import settings
from src.core.exceptions import GenerationError


class MyProviderService(LLMService):
    
    def __init__(self):
        self.api_key = settings.MY_PROVIDER_API_KEY
        self.model = settings.MY_PROVIDER_MODEL
        # Initialize your client here
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        # Implement generation logic
        pass
    
    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        # Implement JSON generation
        pass
    
    async def stream_generation(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        # Implement streaming
        pass
    
    async def get_token_usage(self, text: str) -> Dict[str, int]:
        # Implement token counting
        pass
    
    @property
    def langchain_llm(self):
        # Optional: Return LangChain-compatible LLM
        pass
```

### Step 2: Register the Provider

Update `src/infrastructure/llm/__init__.py`:

```python
# Register your provider (optional dependency)
try:
    from src.infrastructure.llm.my_provider_client import MyProviderService
    LLMFactory.register("myprovider", MyProviderService)
except ImportError:
    pass  # Package not installed
```

### Step 3: Add Configuration

Update `src/core/config.py`:

```python
# My Provider (optional)
MY_PROVIDER_API_KEY: Optional[str] = None
MY_PROVIDER_MODEL: str = "default-model"
```

### Step 4: Update Dependencies

Add to `requirements.txt`:

```
# Optional LLM providers
my-provider-sdk>=1.0.0  # For My Provider
```

### Step 5: Use Your Provider

```bash
export LLM_PROVIDER=myprovider
export MY_PROVIDER_API_KEY=your-key

# The application will now use your provider!
```

## Benefits

### 1. Open/Closed Principle
- ✅ Add new providers without modifying existing code
- ✅ Existing providers remain unchanged and stable
- ✅ Easy to maintain and extend

### 2. Runtime Flexibility
- ✅ Switch providers via environment variable
- ✅ No code redeployment required
- ✅ Easy A/B testing between providers

### 3. Graceful Degradation
- ✅ Optional providers don't break the system
- ✅ Falls back to available providers
- ✅ Clear error messages for missing dependencies

### 4. Better Testing
- ✅ Easy to mock providers
- ✅ Can test with multiple providers
- ✅ Isolated testing of each provider

### 5. Cost Optimization
- ✅ Use cheap providers for development
- ✅ Use expensive providers for production
- ✅ Route by workload characteristics

## Configuration Examples

### Development Environment
```bash
# Use local LLM for free development
export LLM_PROVIDER=local
export OLLAMA_BASE_URL=http://localhost:11434
export LOCAL_MODEL=llama2
```

### Testing Environment
```bash
# Use OpenAI with cheaper model
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=gpt-3.5-turbo
```

### Production Environment
```bash
# Use best model for production
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
export ANTHROPIC_MODEL=claude-3-opus-20240229
```

## Testing

### Unit Tests
```bash
pytest tests/unit/test_llm_factory.py -v
```

### Integration Tests
```bash
pytest tests/integration/test_api_endpoints.py::TestAPIEndpoints::test_list_providers -v
```

### Manual Testing
```bash
python3 demo_factory_pattern.py
```

## Troubleshooting

### Provider Not Available
```
ValueError: Unknown LLM provider: anthropic. Available: ['openai', 'local', 'ollama']
```
**Solution**: Install the provider package: `pip install anthropic`

### Import Error
```
ImportError: anthropic package required. Install with: pip install anthropic
```
**Solution**: Install the missing dependency or switch to an available provider

### API Key Missing
```
ValidationError: OPENAI_API_KEY is required when using OpenAI LLM
```
**Solution**: Set the required API key in your environment

## Best Practices

1. **Always check available providers first**:
   ```python
   providers = LLMFactory.available_providers()
   if "anthropic" not in providers:
       print("Anthropic not available, falling back to OpenAI")
   ```

2. **Handle provider-specific errors**:
   ```python
   try:
       llm = LLMFactory.create("anthropic")
   except (ImportError, ValueError) as e:
       logger.warning(f"Failed to create Anthropic: {e}")
       llm = LLMFactory.create("openai")  # Fallback
   ```

3. **Test with multiple providers**:
   - Ensure your code works with any provider
   - Don't rely on provider-specific features
   - Use the common LLMService interface

4. **Document provider-specific requirements**:
   - API keys needed
   - Special configuration
   - Feature limitations

## Future Enhancements

Potential additions to the factory pattern:

- **Google PaLM**: Add Google's LLM support
- **Azure OpenAI**: Add Azure-hosted OpenAI
- **Cohere**: Add Cohere's LLM support
- **HuggingFace**: Add HuggingFace Inference API
- **Custom Models**: Add support for fine-tuned models
- **Provider Routing**: Intelligent routing based on task type
- **Fallback Chain**: Automatic failover between providers
- **Cost Tracking**: Track costs per provider
- **Performance Metrics**: Compare provider performance

## References

- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)
- [Open/Closed Principle](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
