# 🔍 HolmesGPT Chat UI

A Streamlit-based chat interface for [HolmesGPT](https://github.com/HolmesGPT/holmesgpt) — the CNCF AI-powered SRE agent.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Full SSE streaming** — Real-time response streaming with all 7 HolmesGPT event types
- **Live tool execution** — See tool calls as they happen with inline status cards
- **Reasoning display** — Shows model's intermediate reasoning when available
- **Token usage tracking** — Real-time token consumption per response
- **Tool approval flow** — Approve/reject tool calls when enabled
- **Conversation history** — Multi-turn investigations with context
- **Quick investigations** — One-click buttons for common checks
- **Dark theme** — Clean, dark-friendly UI

## SSE Events Handled

| Event | Description |
|-------|-------------|
| `ai_message` | Streamed text + reasoning chunks |
| `start_tool_calling` | Tool execution started |
| `tool_calling_result` | Tool execution result |
| `token_count` | Token usage update |
| `ai_answer_end` | Final analysis + conversation history |
| `approval_required` | Tool needs user approval |
| `error` | Error with code and description |

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt

# Point to your HolmesGPT instance
export HOLMES_URL=http://localhost:8080

streamlit run app.py
```

Open http://localhost:8501

### Docker

```bash
docker build -t holmesgpt-chat-ui .

docker run -p 8501:8501 \
  -e HOLMES_URL=http://your-holmes:8080 \
  holmesgpt-chat-ui
```

### Kubernetes

1. **Update the Holmes URL** in `k8s-deployment.yaml`:

```yaml
env:
  - name: HOLMES_URL
    value: "http://holmesgpt-holmes.holmesgpt.svc:80"  # adjust to your Holmes service
```

2. **Deploy:**

```bash
kubectl apply -f k8s-deployment.yaml -n holmesgpt
```

3. **Access** via port-forward:

```bash
kubectl port-forward svc/holmesgpt-chat-ui 8501:80 -n holmesgpt
```

Open http://localhost:8501

### Using the pre-built image

```bash
# Pull from GHCR
docker pull ghcr.io/henrikrexed/holmesgpt-chat-ui:latest

# Or in k8s-deployment.yaml
image: ghcr.io/henrikrexed/holmesgpt-chat-ui:latest
```

## Configuration

All configuration is done via the sidebar in the UI:

| Setting | Description | Default |
|---------|-------------|---------|
| Holmes API URL | HolmesGPT `/api/chat` endpoint | `http://localhost:8080` |
| Model | Model name from Holmes `modelList` | `qwen3-32b` |
| Stream responses | Enable SSE streaming | `true` |
| Show tool calls | Display tool execution details | `true` |
| Show token usage | Display token consumption | `true` |
| Additional system prompt | Extra instructions for the model | empty |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOLMES_URL` | HolmesGPT API base URL | `http://localhost:8080` |

## Connecting to HolmesGPT

The Chat UI connects to HolmesGPT's `/api/chat` endpoint. Make sure:

1. **HolmesGPT is accessible** from where the UI runs
2. **The model** specified in the sidebar matches a key in Holmes' `modelList` config
3. **CORS is not an issue** — Streamlit makes server-side requests, so CORS doesn't apply

### Helm-deployed Holmes

If Holmes was deployed via Helm chart:

```bash
# Find the Holmes service
kubectl get svc -n holmesgpt | grep holmes

# Typical service name: <release>-holmes
# URL: http://<release>-holmes.<namespace>.svc:80
```

Set `HOLMES_URL=http://holmesgpt-holmes.holmesgpt.svc:80` in the deployment.

## Architecture

```
┌─────────────┐     SSE/HTTP      ┌──────────────┐     LLM + Tools     ┌─────────────┐
│  Chat UI    │ ──────────────── │  HolmesGPT   │ ──────────────────  │  Ollama /    │
│  (Streamlit)│   /api/chat      │  (Server)     │                     │  OpenAI /    │
│  :8501      │                  │  :8080        │                     │  LiteLLM     │
└─────────────┘                  └──────┬────────┘                     └─────────────┘
                                        │
                                  MCP Protocol
                                        │
                               ┌────────┴────────┐
                               │  MCP Servers     │
                               │  (k8s-networking,│
                               │   dynatrace,     │
                               │   otel-collector) │
                               └─────────────────┘
```

## License

MIT
