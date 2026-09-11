"""
Chat & Intelligence Component
Provides real-time streaming code reasoning via NVIDIA NIM / Groq,
with dual triggers for Markdown report generation and chat export.
"""
import json
import os
import time
import requests
import streamlit as st
from typing import Generator, List, Dict


def stream_sse_tokens(api_base_url: str, session_id: str, messages: List[Dict], model: str, api_key: str, provider: str) -> Generator[str, None, None]:
    """Consumes Server-Sent Events from the FastAPI backend and yields text tokens."""
    url = f"{api_base_url}/api/chat/stream"
    payload = {
        "session_id": session_id,
        "messages": messages,
        "model": model,
        "api_key": api_key,
        "provider": provider
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                yield f"⚠️ API Error ({resp.status_code}): {resp.text}"
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data_str)
                        if "token" in parsed:
                            yield parsed["token"]
                        elif "error" in parsed:
                            yield f"\n\n❌ {parsed['error']}"
                    except Exception:
                        continue

    except Exception as e:
        yield f"\n\n❌ Connection Error: Could not reach backend ({str(e)}). Is the FastAPI server running?"


def export_chat_to_markdown(messages: List[Dict]) -> str:
    """Formats the current conversation into a clean Markdown document."""
    lines = [
        "# DocuStream Codebase Intelligence Report\n",
        f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n"
    ]
    for msg in messages:
        role = "### 👤 User" if msg["role"] == "user" else "### 🤖 Assistant"
        lines.append(f"{role}\n\n{msg['content']}\n\n---\n")
    return "\n".join(lines)


def render_chat_interface(api_base_url: str = "http://localhost:8000"):
    """Renders the AI chat interface and streaming reasoning view."""
    st.subheader("🤖 Codebase Chat & Reasoning")

    # API Configuration Bar
    col_prov, col_mod, col_key = st.columns([1, 1.8, 1.8])
    with col_prov:
        provider = st.selectbox(
            "Provider",
            options=["NVIDIA NIM (build.nvidia.com)", "Groq Cloud", "Anthropic (Claude)", "OpenAI"],
            index=0,
            key="llm_provider"
        )
        p_lower = provider.lower()
        if "nvidia" in p_lower:
            provider_code = "nvidia"
        elif "groq" in p_lower:
            provider_code = "groq"
        elif "anthropic" in p_lower or "claude" in p_lower:
            provider_code = "anthropic"
        else:
            provider_code = "openai"

    with col_mod:
        preset_models = {
            "nvidia": [
                "nvidia/nemotron-3.5-lightning-30b-a3b",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "kimi-k3",
                "meta/llama-3.1-70b-instruct"
            ],
            "groq": [
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "deepseek-r1-distill-llama-70b"
            ],
            "anthropic": [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022"
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "o3-mini"
            ]
        }
        options = preset_models.get(provider_code, ["gpt-4o"]) + ["Custom model..."]
        selected_option = st.selectbox("Model", options=options, index=0, key="llm_model_select")

        if selected_option == "Custom model...":
            selected_model = st.text_input("Enter Model Name", placeholder="e.g. meta/llama-3.1-405b", key="custom_model_input")
        else:
            selected_model = selected_option

    with col_key:
        env_keys = {
            "nvidia": os.getenv("NVIDIA_API_KEY", ""),
            "groq": os.getenv("GROQ_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", "")
        }
        env_key = env_keys.get(provider_code, "")
        current_val = st.session_state.get(f"llm_api_key_{provider_code}") or env_key

        placeholders = {
            "nvidia": "nvapi-...",
            "groq": "gsk_...",
            "anthropic": "sk-ant-...",
            "openai": "sk-..."
        }
        api_key = st.text_input(
            f"{provider_code.title()} API Key",
            type="password",
            value=current_val,
            placeholder=f"{placeholders.get(provider_code, 'API Key')} (reads from .env if set)",
            help=f"Set {provider_code.upper()}_API_KEY in .env or enter here for the active session.",
            key=f"api_key_input_{provider_code}"
        )
        if api_key:
            st.session_state[f"llm_api_key_{provider_code}"] = api_key.strip()
            st.session_state.llm_api_key = api_key.strip()

    # Chat Actions Header (Dual triggers for report & export)
    col_act1, col_act2, _ = st.columns([1.2, 1.2, 2])
    with col_act1:
        gen_report = st.button("📊 Generate Architecture Report", type="secondary", use_container_width=True)
    with col_act2:
        chat_history = st.session_state.get("chat_messages", [])
        if chat_history:
            md_doc = export_chat_to_markdown(chat_history)
            st.download_button(
                label="📥 Export Chat (.md)",
                data=md_doc.encode('utf-8'),
                file_name="docustream_chat_export.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display message history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle "Generate Report" action trigger
    prompt = None
    if gen_report:
        prompt = "Provide a comprehensive architectural analysis of the indexed codebase, including core modules, dependency flows, key patterns, and potential refactorings."

    user_input = st.chat_input("Ask a question about the repository or code...")
    if user_input:
        prompt = user_input

    if prompt:
        active_key = st.session_state.get("llm_api_key") or ""
        if not active_key:
            st.warning("Please enter your Provider API Key above to start reasoning.")
            return

        session_id = st.session_state.get("backend_session_id")
        if not session_id:
            st.info("⚠️ No codebase has been ingested yet. Ingest a GitHub repository or files from the sidebar first.")
            return

        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream response
        with st.chat_message("assistant"):
            stream_gen = stream_sse_tokens(
                api_base_url=api_base_url,
                session_id=session_id,
                messages=st.session_state.chat_messages,
                model=selected_model,
                api_key=active_key,
                provider=provider_code
            )
            response_text = st.write_stream(stream_gen)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
