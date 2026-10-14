<a id="readme-top"></a>

<div align="center">

# LangGraph Chatbot Template

**A beginner-friendly template for building AI assistants with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)**

Clone it, add your keys, and run your chatbot with guardrails, routing, tools, RAG and memory.<br>

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C?logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv&logoColor=white)
![License](https://img.shields.io/github/license/f-fakeus3r/langgraph-chatbot-template)
![GitHub stars](https://img.shields.io/github/stars/f-fakeus3r/langgraph-chatbot-template?style=social&logo=github)

</div>

---

## 📚 Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [LangGraph Framework](#langgraph-framework)
- [Requirements](#requirements)
- [Prerequisites](#prerequisites)
- [Getting started](#-getting-started)
- [Using the API](#using-the-api)
- [Configuration](#configuration)
- [Customizing the template](#customizing-the-template)
- [Project structure](#project-structure)
- [Debugging tips](#debugging-tips)
- [Troubleshooting](#troubleshooting)
- [Before you deploy](#before-you-deploy)
- [External Documentation](#external-documentation)
- [License](#license)

---

## Overview

This template ships with a complete, working conversation flow. Each piece is small and commented, so you can read it, understand it and change it.

| Feature | What it does |
| --- | --- |
| **Input and output guardrails** | An LLM checks every user message *before* it reaches an agent, and every answer *before* it reaches the user. Blocked turns get a safe fallback reply. If a guardrail call errors, it **fails closed** (blocks), so unchecked content never goes through. |
| **Classifier router** | A (small) model reads the message and decides which agent should answer. |
| **General agent** | Handles small talk and general questions. No tools. |
| **Tool agent subgraph** | A tool-calling loop: the model calls tools, reads the results, and repeats until it can answer. Ships with a **RAG tool** (Qdrant vector search + reranking) and example tools to copy. |
| **Per-session memory** | Conversation state is saved in Postgres (Supabase) and keyed by `session_id`. Send the same `session_id` and the bot remembers. Each agent also sees the recent turns the user had with the *other* agents. |
| **Swappable LLM provider** | Models are called through [OpenRouter](https://openrouter.ai) (one key, hundreds of models), but any LangChain chat model works. |
| **Live graph diagram** | `GET /chatbot/get-graph` returns a Mermaid diagram of the graph that is actually running. |

---

## How it works

Every message the user sends is one **turn**. A turn walks through the graph from `start` to `end`:

```mermaid
graph TD;
    start([start]) --> init_agents
    init_agents --> input_guardrail
    input_guardrail -. pass .-> input_classifier
    input_guardrail -. fail .-> fallback
    input_classifier -. general .-> general_agent
    input_classifier -. tools .-> update_prompt
    general_agent --> output_guardrail
    subgraph tool_subgraph
        update_prompt --> tool_agent
        tool_agent -. tool call .-> tools
        tools --> tool_agent
    end
    tool_agent -. answer .-> output_guardrail
    output_guardrail -. pass .-> finalize_turn
    output_guardrail -. fail .-> fallback
    fallback --> finalize_turn
    finalize_turn --> finish([end])
```

> Solid arrows always happen. Dotted arrows are **choices**: the graph takes only one of them, depending on a result.

Step by step:

1. **`init_agents`** resets the per-turn values (route, last answer, guardrail results) and refreshes every agent's system prompt.
2. **`input_guardrail`** asks the guardrail model: *"Is this user message allowed?"* It returns `PASS` or `FAIL` (plus a reason).
   - `FAIL` -> jump to **`fallback`**, which replies with a polite refusal (`OUTPUT_BLOCKED_MESSAGE`).
3. **`input_classifier`** asks the router model: *"Is this `general` or does it need `tools`?"*
4. The message goes to one agent:
   - **`general_agent`** answers directly, or
   - the **`tool_subgraph`**: `update_prompt` sets the tool agent's prompt, then **`tool_agent`** and **`tools`** loop. The agent asks for a tool, the `tools` node runs it, the result goes back to the agent, and so on until the agent answers without calling a tool.
5. **`output_guardrail`** checks the answer.
   - `FAIL` -> **`fallback`**.
6. **`finalize_turn`** saves the turn in the shared history, so the other agents know what was said.
7. The API returns the answer, which agent answered, and both guardrail results.

---

## LangGraph Framework

The [LangGraph](https://www.langchain.com/langgraph) Framework works with graphs and nodes. This is a summary of the most important concepts and files in this code.

| Concept | Plain-English meaning | File |
| --- | --- | --- |
| **Graph** | Your chatbot's flowchart. You guide the steps with it. Boxes (nodes) connected by arrows (edges). | [`src/graph/graph_builder.py`](src/graph/graph_builder.py) |
| **State** | A Python dictionary that runs throughout the graph. Every node reads it and returns the keys it wants to update. | [`src/graph/state.py`](src/graph/state.py) |
| **Node** | A plain Python function: `def my_node(state) -> dict`. It does one job (call an LLM, check a guardrail...) and returns a partial update to the state. | [`src/graph/agents/`](src/graph/agents/) |
| **Edge** | Links one node to another. Added with `add_edge("A", "B")`. | `graph_builder.py` |
| **Conditional edge** | Links onde node to another DEPENDING on the result of something. eg "After node A, run this *router* function to decide where to go." Added with `add_conditional_edges(...)`. | [`src/graph/routers/`](src/graph/routers/) |
| **Subgraph** | A whole graph used as a single node inside another graph. Good for packaging a self-contained piece, like the tool loop. | [`src/graph/subgraphs/tool_subgraph/`](src/graph/subgraphs/tool_subgraph/) |
| **Checkpointer** | Saves the state after every step to a database. Next time the same conversation calls the graph, the state is loaded back. That's the bot's memory. Here it's `PostgresSaver`. | [`src/services/graph_service.py`](src/services/graph_service.py) |
| **Tool** | A Python function decorated with `@tool`. Its **name, arguments and docstring** are shown to the LLM, which decides when to call it. | [`src/tools/`](src/tools/) |
| **`ToolNode` + `tools_condition`** | Prebuilt LangGraph helpers. `ToolNode` runs the tools the model asked for; `tools_condition` checks "did the model ask for a tool?" to decide between looping or finishing. | `tool_subgraph/builder.py` |

### How memory works on this template

Instead of one big message list, **each agent keeps its own message history** in the state (`general_agent`, `tool_agent`, `input_classifier`, ...). This keeps each prompt focused on its own job.

So that agents don't lose context when the user changes topic, every answered turn is also written to `shared_history`. Each agent's system prompt includes the last few turns the user had with the *other* agents (`MAX_SHARED_HISTORY_TURNS = 6`). To keep prompts small, only the last 20 messages of each history are sent to the LLM (`MAX_HISTORY_MESSAGES = 20` in [`memory_helper.py`](src/utils/memory_helper.py)).

---

## Requirements

All dependencies are declared in [`pyproject.toml`](pyproject.toml) and pinned in `uv.lock`. 
You don't install them one by one: `uv sync` installs everything.

| Package | Version |
| --- | --- |
| [`langgraph`](https://pypi.org/project/langgraph/) | `>=1.2.12` |
| [`langchain-core`](https://pypi.org/project/langchain-core/) | `>=1.6.5` |
| [`langgraph-checkpoint-postgres`](https://pypi.org/project/langgraph-checkpoint-postgres/) | `>=3.1.2` |
| [`langchain-openrouter`](https://pypi.org/project/langchain-openrouter/) | `>=0.2.9` |
| [`openrouter`](https://pypi.org/project/openrouter/) | `>=0.11.46` |
| [`qdrant-client[fastembed]`](https://pypi.org/project/qdrant-client/) | `>=1.19.1` |
| [`fastapi`](https://pypi.org/project/fastapi/) | `>=0.141.1` |
| [`uvicorn`](https://pypi.org/project/uvicorn/) | `>=0.54.0` |
| [`psycopg[binary,pool]`](https://pypi.org/project/psycopg/) | `>=3.3.6` |
| [`pydantic`](https://pypi.org/project/pydantic/) | `>=2.12.5` |
| [`python-dotenv`](https://pypi.org/project/python-dotenv/) | `>=1.2.3` |
| [`ruff`](https://pypi.org/project/ruff/) *(dev)* | `>=0.16.9` |

**External services**

| Service | Usage |
| --- | --- |
| [OpenRouter](https://openrouter.ai) | LLM and embedding provider: one API key for models from OpenAI, Google, Anthropic, Meta and others. |
| [Supabase](https://supabase.com) | Hosted Postgres database for conversation memory. The free tier is enough to start. |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector database for the RAG tool. Has a free tier. |

---

## Prerequisites

Before you start, make sure you have:

- **[Git](https://git-scm.com/downloads)** to clone the repo.
- **[uv](https://docs.astral.sh/uv/)**, a fast Python package and project manager. You don't need to install Python yourself: uv reads `.python-version` and downloads **Python 3.13** if it's missing.
- An **OpenRouter API key**: create one at [openrouter.ai/keys](https://openrouter.ai/keys). Add a few credits or use the free models (with limitation rates).
- A **Supabase project**: create one at [supabase.com](https://supabase.com) (free). You'll need its database connection string.
- A **Qdrant Cloud cluster** at [cloud.qdrant.io](https://cloud.qdrant.io) (free tier), for the RAG tool to work.

Works on Windows, macOS and Linux.

---

## 🚀 Getting started

### 1. Get your own copy

Click **Use this template -> Create a new repository** at the top of the GitHub page, then clone your new repo:

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

> Keeping it simple? `git clone https://github.com/f-fakeus3r/langgraph-chatbot-template.git` works too.

### 2. Install uv

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

Close and reopen your terminal, then check it works with `uv --version`.

### 3. Install the dependencies

Inside the project's folder:

```bash
uv sync
```

This creates a `.venv` folder with Python 3.13 and every package from the [Requirements](#requirements) table.

### 4. Create your `.env` file

Copy the example file:

```bash
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows (PowerShell)
Copy-Item .env.example .env
```

Then open `.env` and fill in at least the two **required** values:

- **`OPENROUTER_API_KEY`**: your key from [openrouter.ai/keys](https://openrouter.ai/keys).
- **`SUPABASE_POSTGRES_URI`**: in your Supabase project, click **Connect** (top of the dashboard) and copy the **Session pooler** URI (or **Direct connection**). Replace `[YOUR-PASSWORD]` with your database password. It looks like this:

  ```
  postgresql://postgres.abcdefgh:your-password@aws-0-us-east-1.pooler.supabase.com:5432/postgres
  ```

You don't need to create any tables: LangGraph creates its checkpoint tables on the first start. Every variable is described in the [Configuration](#configuration) section.

### 5. *(Optional)* Load sample documents for the RAG tool

Fill in `QDRANT_URL` and `QDRANT_API_KEY` in `.env`, then run:

```bash
uv run python -m scripts.seed_qdrant
```

This creates the collection (if missing) and adds a few sample documents about this template, so the RAG tool has something to find. Use `--recreate` to delete the collection and start over (**this erases its data**). Replace `SAMPLE_DOCUMENTS` in [`scripts/seed_qdrant.py`](scripts/seed_qdrant.py) with your own text when you're ready.

> Skipping this step is fine. The bot still works; the RAG tool just replies "The knowledge base is unavailable right now." to the agent.

### 6. Start the API

```bash
uv run main.py
```

When you see `Uvicorn running on http://0.0.0.0:8000`, it means ir ran successfully.

### 7. Say hi 👋


From Insomnia/Postman or a terminal:

<details>
<summary><b>macOS / Linux (curl)</b></summary>

```bash
curl -X POST http://localhost:8000/chatbot/message \
  -H "Content-Type: application/json" \
  -d '{"input_message": "what can you do?", "session_id": "0001", "username": "user1"}'
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chatbot/message `
  -ContentType "application/json" `
  -Body '{"input_message": "what can you do?", "session_id": "0001", "username": "user1"}'
```

</details>

You'll get back:

```json
{
  "input_message": "what can you do?",
  "output_message": "Hello! I'm here to help you with...",
  "route": "general_agent",
  "input_guardrail": "PASS",
  "output_guardrail": "PASS"
}
```

Send another message with the **same `session_id`** and the bot remembers the conversation. Use a new `session_id` to start fresh.

---

## Using the API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/chatbot/message` | Send a message and get the bot's answer. |
| `GET` | `/chatbot/get-graph` | The current graph as a Mermaid diagram. Paste it into [mermaid.live](https://mermaid.live) to see it. |
| `GET` | `/health` | Health check. Returns `{"status": "ok"}`. |
| `GET` | `/docs` | Interactive API docs (Swagger UI). |

**Request body** of `POST /chatbot/message`

| Field | Type | Description |
| --- | --- | --- |
| `input_message` | string | The user's message. Can't be empty. |
| `session_id` | string | Identifies the conversation. Same id = same memory. Can't be empty. |
| `username` | string | Who is talking. Saved in the run metadata as `user_id` (useful for tracing). |

**Response**

| Field | Description |
| --- | --- |
| `input_message` | The message you sent. |
| `output_message` | The bot's answer. |
| `route` | Who answered: `general_agent`, `tool_agent` or `fallback` (input blocked). |
| `input_guardrail` | `PASS` or `FAIL`. |
| `output_guardrail` | `PASS` or `FAIL`. Empty when the input guardrail already failed (no answer was generated). |

---

## Configuration

All settings come from environment variables, loaded from `.env` by [`src/config/settings.py`](src/config/settings.py). That file is the single place where they're read.

| Variable | Default | Description |
| --- | --- | --- |
| `SUPABASE_POSTGRES_URI` | | Postgres connection string for conversation memory. |
| `POSTGRES_POOL_MIN_SIZE` | `1` | Minimum open database connections. |
| `POSTGRES_POOL_MAX_SIZE` | `10` | Maximum open database connections. |
| `OPENROUTER_API_KEY` | | Your OpenRouter key. Used for chat and embeddings. |
| `LLM_MODEL` | `google/gemini-3.5-flash-lite` *(in `.env.example`)* | Main model used by the agents. Browse ids at [openrouter.ai/models](https://openrouter.ai/models). |
| `LLM_GUARDRAIL_MODEL` | same as `LLM_MODEL` | Model for the guardrails. A smaller, cheaper model is usually enough. |
| `LLM_ROUTER_MODEL` | same as `LLM_MODEL` | Model for the classifier router. Also a good place for a small model. |
| `QDRANT_URL` | | Your Qdrant cluster URL. |
| `QDRANT_API_KEY` | | Your Qdrant API key. |
| `QDRANT_COLLECTION` | `example_collection` | Name of the collection the RAG tool searches. |
| `LOG_LEVEL` | `INFO` | `DEBUG` shows every message sent to each agent. |

> ⚠️ The guardrail and router models must support **structured outputs** (JSON schema). Filter for them [here](https://openrouter.ai/models?supported_parameters=structured_outputs).

**RAG settings** live in code, in [`src/config/rag_config.py`](src/config/rag_config.py): embedding model (`openai/text-embedding-3-small`, 1536 dimensions), vector name, the payload field holding the text (`text`), the reranker model, and how many chunks to fetch (`12`) and keep (`4`). The embedding model and dimensions **must match** how your collection was built.

---

## Customizing the template

### Where to change what

| To change... | Look at |
| --- | --- |
| The flow: nodes, edges, routing | [`src/graph/graph_builder.py`](src/graph/graph_builder.py) |
| Prompts, language, behaviour and fallback messages | [`src/graph/prompts.py`](src/graph/prompts.py) |
| The tool agent's prompt (describe your tools and documents here) | [`src/graph/subgraphs/tool_subgraph/prompt.py`](src/graph/subgraphs/tool_subgraph/prompt.py) |
| Tools available to the tool agent | [`src/tools/__init__.py`](src/tools/__init__.py) (`ALL_TOOLS`) |
| LLM provider and models | [`src/services/chat_models.py`](src/services/chat_models.py) and the `LLM_*` variables in `.env` |
| Embedding provider | [`src/services/embeddings.py`](src/services/embeddings.py) |
| RAG: embedding model, collection layout, reranker | [`src/config/rag_config.py`](src/config/rag_config.py) |
| How much history each agent sees | [`src/utils/memory_helper.py`](src/utils/memory_helper.py) |
| Environment variables | [`src/config/settings.py`](src/config/settings.py) and `.env.example` |


### Using another LLM provider

Every node only uses three methods of the chat model: `invoke`, `bind_tools` and `with_structured_output`. Any LangChain chat model supports them. Replace `ChatOpenRouter` in [`src/services/chat_models.py`](src/services/chat_models.py) with, for example, `ChatOpenAI` (`langchain-openai`), `ChatAnthropic` (`langchain-anthropic`) or `ChatOllama` (`langchain-ollama`, for local models). Add the package with `uv add <package>`.

### Using your own documents for RAG

1. Replace `SAMPLE_DOCUMENTS` in [`scripts/seed_qdrant.py`](scripts/seed_qdrant.py) with your text chunks (or use your own ingestion pipeline).
2. Run `uv run python -m scripts.seed_qdrant --recreate`.
3. Update the `qdrant_vector_search` docstring in [`src/tools/qdrant_rag_tool.py`](src/tools/qdrant_rag_tool.py) and the tool prompt to describe *your* documents.

If you already have a Qdrant collection, set `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `VECTOR_NAME` and `TEXT_PAYLOAD_KEY` in [`rag_config.py`](src/config/rag_config.py) to match it.

---

## Project structure

```
.
├── main.py                       # Entry point: starts uvicorn on port 8000
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Exact versions installed by `uv sync`
├── .env.example                  # Template for your .env (every variable explained)
├── scripts/
│   └── seed_qdrant.py            # Loads sample documents into Qdrant
└── src/
    ├── api/                      # HTTP layer
    │   ├── app.py                # FastAPI app, startup/shutdown, /health
    │   ├── schemas.py            # Request and response models
    │   └── routes/
    │       └── chatbot_route.py  # /chatbot/message and /chatbot/get-graph
    ├── config/
    │   ├── settings.py           # Reads every environment variable
    │   └── rag_config.py         # RAG settings (embeddings, reranker, collection)
    ├── graph/                    # The LangGraph graph folder
    │   ├── graph_builder.py      # The main graph: nodes and edges
    │   ├── state.py              # The shared State
    │   ├── prompts.py            # System prompts and fallback messages
    │   ├── schemas.py            # Guardrail structured output
    │   ├── init_agents.py        # First node of every turn
    │   ├── finalize_turn.py      # Last node of every turn
    │   ├── agents/               # general, tool, classifier and fallback nodes
    │   ├── guardrails/           # input and output guardrails
    │   ├── routers/              # functions used by conditional edges
    │   └── subgraphs/
    │       └── tool_subgraph/    # tool-calling loop (builder + prompt)
    ├── services/
    │   ├── graph_service.py      # Compiles the graph with the Postgres checkpointer and runs it
    │   ├── chat_models.py        # Creates the LLM clients
    │   └── embeddings.py         # Embedding calls for RAG
    ├── tools/                    # Tools available to the tool agent (ALL_TOOLS)
    └── utils/                    # Memory, Qdrant and reranker helpers
```

**Suggested reading order**: `state.py` -> `graph_builder.py` -> `agents/general_agent.py` -> `subgraphs/tool_subgraph/builder.py` -> `services/graph_service.py`.

---

## Debugging tips

- **See what each agent receives**: set `LOG_LEVEL=DEBUG` in `.env`. The logs show the exact messages sent to every agent and each response, including tool calls.
- **Follow routing decisions**: at the default `INFO` level the logs already show each guardrail result and which agent the classifier picked.
- **Visualize the graph**: open `GET /chatbot/get-graph` and paste the output into [mermaid.live](https://mermaid.live).
- **Inspect the memory**: in Supabase's table editor, the `checkpoints` tables hold the saved state for each `thread_id` (your `session_id`).
- **Lint and format**: `uv run ruff check .` and `uv run ruff format .`

> ⚠️ Use a tracing tool such as [LangSmith](https://smith.langchain.com/). No code changes are needed: LangChain and LangGraph send traces automatically once these variables are set. Create an API key in LangSmith and add to your `.env`.

---

## Troubleshooting

<details>
<summary><b><code>RuntimeError: Missing environment variables: ...</code></b></summary>

The API checks for the required variables on startup. Make sure `.env`, has `OPENROUTER_API_KEY` and `SUPABASE_POSTGRES_URI` filled in.

</details>

<details>
<summary><b>The API hangs or fails to connect to Supabase on startup</b></summary>

- Use the **Session pooler** URI. Supabase's **Direct connection** is IPv6-only unless you have the IPv4 add-on, so it fails on networks without IPv6.
- Check you replaced `[YOUR-PASSWORD]` with the real password. If the password has special characters (`@`, `#`, `/`...), [URL-encode](https://www.urlencoder.org/) it.
- A paused (inactive) free Supabase project needs to be restored from the dashboard first.

</details>

<details>
<summary><b>Every message gets "Sorry, I can't help with that request."</b></summary>

The input guardrail is failing. If it's not your message's fault, the guardrail call itself is probably erroring (guardrails fail closed). Check the logs: the most common cause is a guardrail model that doesn't support structured outputs, or an invalid model id. Try another `LLM_GUARDRAIL_MODEL`.

</details>

<details>
<summary><b>The RAG tool says "The knowledge base is unavailable right now."</b></summary>

Check the logs for the real error. Usually `QDRANT_URL` isn't set, the collection doesn't exist yet (run the seed script), or the collection's vector size doesn't match `EMBEDDING_DIMENSIONS` in `rag_config.py`.

</details>

<details>
<summary><b>The first RAG question is slow</b></summary>

On first use the reranker model is downloaded from Hugging Face and cached. Later calls are fast.

</details>

<details>
<summary><b>The bot doesn't remember what I said</b></summary>

Memory is per `session_id`. Make sure you're sending the same one on every request of the conversation.

</details>

<details>
<summary><b>Port 8000 is already in use</b></summary>

Change `port=8000` in [`main.py`](main.py), or stop the other process using that port.

> ⚠️ Check what is running before you stop it. Often it's an earlier run of this API left open in another terminal (a Python process), but it could be another app you need. In that case, change the port instead.

**Windows (PowerShell)**

```powershell
# 1. See what is using port 8000 (note the Id and ProcessName columns)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess

# 2. Stop it, replacing <PID> with the Id from step 1
Stop-Process -Id <PID>
```

**macOS / Linux**

```bash
# 1. See what is using port 8000 (note the PID and COMMAND columns)
lsof -iTCP:8000 -sTCP:LISTEN

# 2. Stop it, replacing <PID> with the PID from step 1
kill <PID>
```

</details>

---

## Before you deploy

- **Add authentication.** The API trusts the `session_id` and `username` it receives, so anyone who knows a `session_id` can read and continue that conversation. Put the API behind auth and derive the `session_id` from the authenticated user.
- **Never commit `.env`.** It's already in `.gitignore`. Use your host's secret manager in production.
- **Review the guardrails for your use case.** They reduce risk but don't remove it. Adjust the rules in [`src/graph/prompts.py`](src/graph/prompts.py).
- **Set spending limits** on your OpenRouter key.

---

## External documentation

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Use the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) and [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [uv documentation](https://docs.astral.sh/uv/)

---
## Like and share!!!

If this template helped you, consider giving it a ⭐!

---

## Created by

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/f-fakeus3r">
        <img src="https://github.com/f-fakeus3r.png" width="100px;" alt="Leonardo Biazom's photo"/><br>
        <sub><b>Leonardo Biazom</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
