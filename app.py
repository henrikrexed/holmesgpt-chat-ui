"""HolmesGPT Chat UI — Streamlit frontend for HolmesGPT investigations.

Supports both streaming (SSE) and non-streaming modes, with real-time
tool call display, token usage tracking, and tool approval workflows.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
import sseclient
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HolmesGPT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark-theme friendly
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* Tool call cards */
.tool-card {
    border: 1px solid rgba(128,128,128,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
}
.tool-card-header {
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.tool-card-body {
    color: rgba(200,200,200,0.85);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85em;
    line-height: 1.4;
}
.tool-status-success { color: #4caf50; }
.tool-status-error { color: #f44336; }
.tool-status-running { color: #ff9800; }
.tool-status-approval { color: #2196f3; }

/* Token usage bar */
.token-bar {
    font-size: 0.75em;
    color: rgba(180,180,180,0.7);
    padding: 2px 0;
    font-family: monospace;
}

/* Reasoning block */
.reasoning-block {
    border-left: 3px solid rgba(100,100,255,0.4);
    padding: 6px 12px;
    margin: 6px 0;
    font-size: 0.85em;
    color: rgba(180,180,220,0.85);
    font-style: italic;
}

/* Approval card */
.approval-card {
    border: 2px solid #2196f3;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    background: rgba(33,150,243,0.05);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Tracks a single tool call through its lifecycle."""

    tool_name: str
    tool_call_id: str
    status: str = "running"  # running | success | error | no_data | approval_required
    result_data: Optional[str] = None
    description: str = ""
    params: Optional[dict] = None


@dataclass
class PendingApproval:
    """A tool call awaiting user approval."""

    tool_call_id: str
    tool_name: str
    description: str
    params: dict


@dataclass
class TokenUsage:
    """Accumulated token usage info."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    max_tokens: int = 0
    cost: float = 0.0


@dataclass
class ChatMessage:
    """A message in the conversation."""

    role: str  # user | assistant | system
    content: str
    tool_calls: list = field(default_factory=list)
    reasoning: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    pending_approvals: list = field(default_factory=list)
    is_error: bool = False


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = None
    if "pending_approvals" not in st.session_state:
        st.session_state.pending_approvals = []
    if "holmes_url" not in st.session_state:
        st.session_state.holmes_url = "http://localhost:8080"
    if "stream_mode" not in st.session_state:
        st.session_state.stream_mode = True
    if "enable_tool_approval" not in st.session_state:
        st.session_state.enable_tool_approval = False
    if "model" not in st.session_state:
        st.session_state.model = None


init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.holmes_url = st.text_input(
        "HolmesGPT URL",
        value=st.session_state.holmes_url,
        help="Base URL of the HolmesGPT server",
    )

    st.session_state.stream_mode = st.toggle(
        "Streaming mode",
        value=st.session_state.stream_mode,
        help="Stream responses in real-time via SSE",
    )

    st.session_state.enable_tool_approval = st.toggle(
        "Tool approval",
        value=st.session_state.enable_tool_approval,
        help="Require approval before executing dangerous tools",
    )

    model_input = st.text_input(
        "Model override",
        value=st.session_state.model or "",
        help="Leave empty for server default",
        placeholder="e.g. anthropic/claude-sonnet-4-5-20250929",
    )
    st.session_state.model = model_input if model_input.strip() else None

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = None
        st.session_state.pending_approvals = []
        st.rerun()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_tool_card(tc: ToolCall):
    """Render a tool call as an inline card."""
    status_class = {
        "running": "tool-status-running",
        "success": "tool-status-success",
        "error": "tool-status-error",
        "no_data": "tool-status-error",
        "approval_required": "tool-status-approval",
    }.get(tc.status, "tool-status-running")

    icon = {
        "running": "⏳",
        "success": "✅",
        "error": "❌",
        "no_data": "🔍",
        "approval_required": "🔐",
    }.get(tc.status, "⏳")

    body_html = ""
    if tc.result_data:
        # Escape HTML in result data
        escaped = (
            tc.result_data[:500]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if len(tc.result_data) > 500:
            escaped += "\n... (truncated)"
        body_html = f'<div class="tool-card-body">{escaped}</div>'

    return f"""<div class="tool-card">
<div class="tool-card-header">
    {icon} <span class="{status_class}">{tc.tool_name}</span>
    <span style="opacity:0.5;font-size:0.8em">({tc.status})</span>
</div>
{body_html}
</div>"""


def render_token_usage(usage: TokenUsage) -> str:
    """Render token usage as a compact bar."""
    parts = []
    if usage.prompt_tokens:
        parts.append(f"in:{usage.prompt_tokens:,}")
    if usage.completion_tokens:
        parts.append(f"out:{usage.completion_tokens:,}")
    if usage.total_tokens:
        parts.append(f"total:{usage.total_tokens:,}")
    if usage.max_tokens:
        pct = min(100, int(usage.total_tokens / usage.max_tokens * 100)) if usage.max_tokens else 0
        parts.append(f"ctx:{pct}%")
    if usage.cost > 0:
        parts.append(f"${usage.cost:.4f}")
    return " · ".join(parts)


def render_message(msg: ChatMessage):
    """Render a single chat message."""
    if msg.role == "user":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.role == "assistant":
        with st.chat_message("assistant"):
            if msg.reasoning:
                st.markdown(
                    f'<div class="reasoning-block">💭 {msg.reasoning}</div>',
                    unsafe_allow_html=True,
                )
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    st.markdown(render_tool_card(tc), unsafe_allow_html=True)
            if msg.content:
                st.markdown(msg.content)
            if msg.token_usage:
                usage_str = render_token_usage(msg.token_usage)
                if usage_str:
                    st.markdown(
                        f'<div class="token-bar">📊 {usage_str}</div>',
                        unsafe_allow_html=True,
                    )
            if msg.is_error:
                st.error(msg.content)
            if msg.pending_approvals:
                render_approval_ui(msg.pending_approvals)


def render_approval_ui(approvals: list):
    """Render approval buttons for pending tool calls."""
    for approval in approvals:
        pa = PendingApproval(**approval) if isinstance(approval, dict) else approval
        st.markdown(
            f"""<div class="approval-card">
🔐 <b>{pa.tool_name}</b> requires approval<br/>
<small>{pa.description}</small><br/>
<code>{json.dumps(pa.params, indent=2)[:300]}</code>
</div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                f"✅ Approve {pa.tool_name}",
                key=f"approve_{pa.tool_call_id}",
                use_container_width=True,
            ):
                handle_approval_decision(pa.tool_call_id, approved=True)
        with col2:
            if st.button(
                f"❌ Reject {pa.tool_name}",
                key=f"reject_{pa.tool_call_id}",
                use_container_width=True,
                type="secondary",
            ):
                handle_approval_decision(pa.tool_call_id, approved=False)


# ---------------------------------------------------------------------------
# Approval handling
# ---------------------------------------------------------------------------


def handle_approval_decision(tool_call_id: str, approved: bool):
    """Send approval decision back to HolmesGPT and continue the conversation."""
    decision = {"tool_call_id": tool_call_id, "approved": approved}

    # Remove from pending
    st.session_state.pending_approvals = [
        pa
        for pa in st.session_state.pending_approvals
        if (pa.get("tool_call_id") if isinstance(pa, dict) else pa.tool_call_id) != tool_call_id
    ]

    # Send continuation request with the decision
    payload = {
        "ask": "",  # Empty — the server continues from conversation_history
        "conversation_history": st.session_state.conversation_history,
        "stream": st.session_state.stream_mode,
        "enable_tool_approval": st.session_state.enable_tool_approval,
        "tool_decisions": [decision],
    }
    if st.session_state.model:
        payload["model"] = st.session_state.model

    url = f"{st.session_state.holmes_url.rstrip('/')}/api/chat"

    if st.session_state.stream_mode:
        process_streaming_response(url, payload)
    else:
        process_non_streaming_response(url, payload)

    st.rerun()


# ---------------------------------------------------------------------------
# SSE streaming response handler
# ---------------------------------------------------------------------------


def process_streaming_response(url: str, payload: dict):
    """Process a streaming SSE response from HolmesGPT."""
    assistant_msg = ChatMessage(role="assistant", content="")
    tool_calls_map: dict[str, ToolCall] = {}

    # Add placeholder message and render live
    st.session_state.messages.append(assistant_msg)
    msg_idx = len(st.session_state.messages) - 1

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
            timeout=300,
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)

        for event in client.events():
            event_type = event.event
            try:
                data = json.loads(event.data) if event.data else {}
            except json.JSONDecodeError:
                continue

            if event_type == "ai_message":
                # Intermediate AI text and reasoning
                content = data.get("content")
                reasoning = data.get("reasoning")
                if content:
                    assistant_msg.content += content
                if reasoning:
                    assistant_msg.reasoning = (assistant_msg.reasoning or "") + reasoning

            elif event_type == "start_tool_calling":
                # Tool execution starting
                tool_name = data.get("tool_name", "unknown")
                tool_id = data.get("id", "")
                tc = ToolCall(tool_name=tool_name, tool_call_id=tool_id)
                tool_calls_map[tool_id] = tc
                assistant_msg.tool_calls.append(tc)

            elif event_type == "tool_calling_result":
                # Tool execution completed
                tool_id = data.get("tool_call_id", "")
                tc = tool_calls_map.get(tool_id)
                if tc is None:
                    # Tool result without a start event — create one
                    tc = ToolCall(
                        tool_name=data.get("name", "unknown"),
                        tool_call_id=tool_id,
                    )
                    tool_calls_map[tool_id] = tc
                    assistant_msg.tool_calls.append(tc)

                result = data.get("result", {})
                tc.status = result.get("status", "success")
                tc.description = data.get("description", "")

                # Extract result data
                result_data = result.get("data")
                if result_data:
                    tc.result_data = (
                        result_data if isinstance(result_data, str) else json.dumps(result_data, indent=2)
                    )
                elif result.get("error"):
                    tc.result_data = result["error"]

            elif event_type == "token_count":
                # Token usage update
                metadata = data.get("metadata", {})
                usage = metadata.get("usage", {})
                costs = metadata.get("costs", {})
                assistant_msg.token_usage = TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    max_tokens=metadata.get("max_tokens", 0),
                    cost=costs.get("total_cost", 0.0),
                )

            elif event_type == "ai_answer_end":
                # Final response
                analysis = data.get("analysis", "")
                if analysis:
                    assistant_msg.content = analysis
                st.session_state.conversation_history = data.get("conversation_history")
                # Capture final metadata token usage
                final_meta = data.get("metadata", {})
                final_usage = final_meta.get("usage", {})
                final_costs = final_meta.get("costs", {})
                if final_usage:
                    assistant_msg.token_usage = TokenUsage(
                        prompt_tokens=final_usage.get("prompt_tokens", 0),
                        completion_tokens=final_usage.get("completion_tokens", 0),
                        total_tokens=final_usage.get("total_tokens", 0),
                        max_tokens=final_meta.get("max_tokens", 0),
                        cost=final_costs.get("total_cost", 0.0),
                    )

            elif event_type == "approval_required":
                # Tools need user approval
                approvals = data.get("pending_approvals", [])
                assistant_msg.pending_approvals = approvals
                st.session_state.pending_approvals = approvals
                st.session_state.conversation_history = data.get("conversation_history")

            elif event_type == "conversation_history_compacted":
                pass  # No UI action needed

            elif event_type == "error":
                error_msg = data.get("description") or data.get("msg", "Unknown error")
                assistant_msg.content = f"**Error:** {error_msg}"
                assistant_msg.is_error = True

            # Update the message in session state
            st.session_state.messages[msg_idx] = assistant_msg

    except requests.exceptions.ConnectionError:
        assistant_msg.content = f"**Connection error:** Cannot reach HolmesGPT at `{st.session_state.holmes_url}`"
        assistant_msg.is_error = True
        st.session_state.messages[msg_idx] = assistant_msg
    except requests.exceptions.Timeout:
        assistant_msg.content = "**Timeout:** The request took too long."
        assistant_msg.is_error = True
        st.session_state.messages[msg_idx] = assistant_msg
    except Exception as e:
        assistant_msg.content = f"**Error:** {type(e).__name__}: {e}"
        assistant_msg.is_error = True
        st.session_state.messages[msg_idx] = assistant_msg


# ---------------------------------------------------------------------------
# Non-streaming response handler
# ---------------------------------------------------------------------------


def process_non_streaming_response(url: str, payload: dict):
    """Process a non-streaming JSON response from HolmesGPT."""
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        assistant_msg = ChatMessage(role="assistant", content=data.get("analysis", ""))

        # Parse tool calls
        for tc_data in data.get("tool_calls", []):
            result = tc_data.get("result", {})
            result_data = result.get("data")
            tc = ToolCall(
                tool_name=tc_data.get("name", "unknown"),
                tool_call_id=tc_data.get("tool_call_id", ""),
                status=result.get("status", "success"),
                description=tc_data.get("description", ""),
                result_data=(
                    result_data
                    if isinstance(result_data, str)
                    else json.dumps(result_data, indent=2) if result_data else None
                ),
            )
            assistant_msg.tool_calls.append(tc)

        # Token usage from metadata
        metadata = data.get("metadata", {})
        usage = metadata.get("usage", {})
        costs = metadata.get("costs", {})
        if usage:
            assistant_msg.token_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                max_tokens=metadata.get("max_tokens", 0),
                cost=costs.get("total_cost", 0.0),
            )

        # Pending approvals
        pending = data.get("pending_approvals")
        if pending:
            assistant_msg.pending_approvals = pending
            st.session_state.pending_approvals = pending

        st.session_state.conversation_history = data.get("conversation_history")
        st.session_state.messages.append(assistant_msg)

    except requests.exceptions.ConnectionError:
        st.session_state.messages.append(
            ChatMessage(
                role="assistant",
                content=f"**Connection error:** Cannot reach HolmesGPT at `{st.session_state.holmes_url}`",
                is_error=True,
            )
        )
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("detail", str(e))
        except Exception:
            error_detail = str(e)
        st.session_state.messages.append(
            ChatMessage(
                role="assistant",
                content=f"**HTTP {e.response.status_code}:** {error_detail}",
                is_error=True,
            )
        )
    except Exception as e:
        st.session_state.messages.append(
            ChatMessage(
                role="assistant",
                content=f"**Error:** {type(e).__name__}: {e}",
                is_error=True,
            )
        )


# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

st.title("🔍 HolmesGPT")

# Render conversation history
for msg in st.session_state.messages:
    render_message(msg)

# Chat input
if prompt := st.chat_input("Ask HolmesGPT a question..."):
    # Add user message
    user_msg = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(user_msg)

    # Build request payload
    payload: dict[str, Any] = {
        "ask": prompt,
        "stream": st.session_state.stream_mode,
        "enable_tool_approval": st.session_state.enable_tool_approval,
    }
    if st.session_state.conversation_history:
        payload["conversation_history"] = st.session_state.conversation_history
    if st.session_state.model:
        payload["model"] = st.session_state.model

    url = f"{st.session_state.holmes_url.rstrip('/')}/api/chat"

    if st.session_state.stream_mode:
        process_streaming_response(url, payload)
    else:
        process_non_streaming_response(url, payload)

    st.rerun()
