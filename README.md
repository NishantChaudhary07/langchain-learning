# Shopping Agent

A learning project that uses LangChain, Ollama, and tool calling to build a simple shopping assistant.

The agent can:

- Look up product prices
- Apply percentage discounts
- Use LangChain tools instead of calculating values directly
- Run locally using an Ollama model

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

## Setup

Install the project dependencies:

```powershell
uv sync

Pull the Ollama model used by the agent:
   ollama pull qwen3:8b

Make sure Ollama is running before starting the agent:
   ollama serve

Run
Run the shopping agent with:
   uv run shopping-agent

You can also run the module directly:
   uv run python -m shopping_agent.shopping_agent


upported Products
The current demo uses mock prices for:

Laptop: $999.99
Smartphone: $699.99
Headphones: $199.99
Monitor: $299.99
Prices are currently defined directly in the source code and are not retrieved from a real store or database.


