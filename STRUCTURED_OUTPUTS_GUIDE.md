# Azure OpenAI Structured Outputs Guide

## Overview

**Always use structured outputs** when working with Azure OpenAI to ensure predictable, reliable responses that follow a specific format. This is critical for production applications where you need consistent data structures.

## Why Use Structured Outputs?

- **Guaranteed Format**: Unlike JSON mode, structured outputs ensure strict adherence to your schema
- **Type Safety**: Responses match your defined data structures exactly
- **Better Error Handling**: Invalid responses are caught at the API level
- **Production Ready**: Eliminates parsing errors and unexpected response formats

## Supported Models

Structured outputs are supported in these Azure OpenAI models:
- `gpt-4o` (versions: `2024-08-06`, `2024-11-20`)
- `gpt-4o-mini` (version: `2024-07-18`)
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` (version: `2025-04-14`)
- `o1` (version: `2024-12-17`)
- `o3`, `o3-mini`, `o4-mini` (latest versions)

## API Requirements

- **Minimum API Version**: `2024-08-01-preview`
- **Recommended**: Use latest GA API version `2024-10-21` or newer
- **Python Libraries**: `openai >= 1.42.0` and `pydantic >= 2.8.2`

## Implementation Patterns

### 1. Basic Structured Response

```python
from pydantic import BaseModel
from openai import AzureOpenAI

class ResponseFormat(BaseModel):
    answer: str
    sources: list[str]
    confidence: str

# Use beta.chat.completions.parse for structured outputs
completion = client.beta.chat.completions.parse(
    model="your-deployment-name",
    messages=[...],
    response_format=ResponseFormat
)

# Access parsed response
response = completion.choices[0].message.parsed
```

### 2. Function Calling with Structured Outputs

```python
import openai
from pydantic import BaseModel

class SearchQuery(BaseModel):
    keywords: str
    max_results: int

tools = [openai.pydantic_function_tool(SearchQuery)]

response = client.chat.completions.create(
    model="your-deployment-name",
    messages=[...],
    tools=tools,
    parallel_tool_calls=False  # Required for structured outputs
)
```

## Schema Requirements

### ✅ Must Include
- `"additionalProperties": false` on all objects
- All fields must be in `"required"` array
- Maximum 100 object properties total
- Maximum 5 levels of nesting

### ✅ Supported Types
- String, Number, Boolean, Integer
- Object, Array, Enum
- `anyOf` (not for root objects)
- Recursive schemas with `$ref`

### ❌ Not Supported
- Optional fields (use union with `null` instead)
- String validation (`minLength`, `maxLength`, `pattern`)
- Number constraints (`minimum`, `maximum`)
- Additional properties

## Best Practices

### 1. Always Define Source URLs
```python
class ChatbotResponse(BaseModel):
    answer: str
    sources: list[str]  # Always include at least one source URL
    timestamp: str
```

### 2. Use Enums for Controlled Values
```python
from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class Response(BaseModel):
    answer: str
    confidence: ConfidenceLevel
    sources: list[str]
```

### 3. Handle Optional Data with Unions
```python
from typing import Union

class Response(BaseModel):
    answer: str
    sources: list[str]
    additional_info: Union[str, None]  # Optional field using union
```

### 4. Validate Response Structure
```python
try:
    completion = client.beta.chat.completions.parse(
        model="deployment-name",
        messages=messages,
        response_format=ResponseFormat
    )
    
    if completion.choices[0].message.parsed:
        response = completion.choices[0].message.parsed
        # Response is guaranteed to match schema
    else:
        # Handle refusal or parsing error
        pass
        
except Exception as e:
    # Handle API errors
    pass
```

## Common Patterns for Web Search Chatbot

### Search Response Format
```python
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class ChatbotResponse(BaseModel):
    answer: str
    sources: list[SearchResult]  # Detailed source information
    search_keywords: str
    response_type: str  # "informational", "factual", "opinion"
```

### Error Response Format
```python
class ErrorResponse(BaseModel):
    error: str
    suggestion: str
    sources: list[str]  # Empty list for errors
```

## Remember

1. **Always use structured outputs** - Never use basic JSON mode
2. **Include source URLs** - Every response must have at least one source
3. **Validate schemas** - Test your Pydantic models thoroughly
4. **Handle errors gracefully** - Plan for parsing failures and refusals
5. **Use appropriate API version** - Minimum `2024-08-01-preview`

## Migration from JSON Mode

If you're currently using JSON mode:
```python
# ❌ Old way (JSON mode)
response_format={"type": "json_object"}

# ✅ New way (Structured outputs)
response_format=YourPydanticModel
```

Structured outputs provide stronger guarantees and should always be preferred over JSON mode for production applications.
