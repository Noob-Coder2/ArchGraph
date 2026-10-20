"""
ArchGraph — Obsidian + Technical Typography Design Tokens
Styles the application with dark canvas tokens, emerald highlights, and clean monospace fonts.
Uses native Streamlit sidebar transitions to prevent fixed ghost margins or layout freezing.
"""
import streamlit as st


def apply_custom_styles():
    """Injects Obsidian design tokens and typography into Streamlit."""
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Obsidian Palette Tokens */
        :root {
            --canvas-base: #121212;
            --canvas-surface: #1a1a1a;
            --canvas-subtle: #151515;
            --canvas-input: #1e1e1e;
            --canvas-hover: #262626;

            --border-subtle: #2a2a2a;
            --border-hover: #3d3d3d;
            --border-active: rgba(34, 197, 94, 0.45);

            --accent-emerald: #22c55e;
            --accent-emerald-hover: #16a34a;
            --accent-emerald-dim: rgba(34, 197, 94, 0.15);
            --accent-emerald-border: rgba(34, 197, 94, 0.25);

            --accent-amber: #eab308;
            --accent-amber-dim: rgba(234, 179, 8, 0.15);

            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        /* Base Streamlit App Overrides */
        .stApp {
            background-color: var(--canvas-base) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--canvas-base) !important;
        }

        [data-testid="stHeader"] {
            background-color: var(--canvas-surface) !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            height: 44px !important;
        }

        /* Natural Sidebar Layout — No Fixed Ghost Margins */
        [data-testid="stSidebar"] {
            background-color: var(--canvas-surface) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            background-color: var(--canvas-surface) !important;
        }

        /* Global Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }

        .mono {
            font-family: 'JetBrains Mono', Consolas, monospace !important;
        }

        /* Scrollbars */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
        ::-webkit-scrollbar-track { background: transparent; }

        /* Inputs, Selectboxes, Dropdowns */
        .stTextInput > div > div,
        .stTextArea > div > div,
        [data-baseweb="select"] > div {
            background-color: var(--canvas-input) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
        }

        .stTextInput > div > div:focus-within,
        [data-baseweb="select"] > div:focus-within {
            border-color: var(--accent-emerald) !important;
            box-shadow: 0 0 0 1px var(--accent-emerald) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            border: 1px solid var(--border-subtle) !important;
            background-color: var(--canvas-input) !important;
            color: var(--text-primary) !important;
            transition: all 0.15s ease !important;
        }

        .stButton > button:hover {
            border-color: var(--border-hover) !important;
            background-color: var(--canvas-hover) !important;
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--accent-emerald) !important;
            color: #000000 !important;
            border: 1px solid var(--accent-emerald) !important;
            font-weight: 600 !important;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: var(--accent-emerald-hover) !important;
            color: #000000 !important;
        }

        /* Telemetry Pills */
        .obsidian-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background-color: var(--canvas-input);
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            padding: 2px 8px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .obsidian-pill b {
            color: var(--text-primary);
            font-weight: 600;
        }

        .obsidian-pill.emerald {
            background-color: var(--accent-emerald-dim);
            border-color: var(--accent-emerald-border);
            color: var(--accent-emerald);
        }

        .obsidian-pill.amber {
            background-color: var(--accent-amber-dim);
            border-color: rgba(234, 179, 8, 0.3);
            color: var(--accent-amber);
        }

        .pulse-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
            background-color: var(--accent-emerald);
            box-shadow: 0 0 6px #22c55e;
        }

        /* Streamlit Chat Components */
        [data-testid="stChatMessage"] {
            background-color: var(--canvas-surface) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background-color: var(--canvas-input) !important;
            border-left: 3px solid var(--accent-emerald) !important;
        }

        [data-testid="stChatInput"] > div {
            background-color: var(--canvas-input) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )