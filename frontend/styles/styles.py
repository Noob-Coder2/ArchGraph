"""
ArchGraph — Obsidian + Timeline Design System
Combines the Obsidian dark theme (#121212 / #1a1a1a), emerald accents (#22c55e),
technical monospace typography (JetBrains Mono), and responsive multi-column workspace layout.
"""
import streamlit as st


def apply_custom_styles():
    """Injects the Obsidian + Timeline design tokens and layout styling into Streamlit."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* =========================================================================
       1. Core Obsidian Design Tokens
       ========================================================================= */
    :root {
        --canvas-base: #121212;
        --canvas-surface: #1a1a1a;
        --canvas-subtle: #151515;
        --canvas-input: #1e1e1e;
        --canvas-hover: #252525;
        
        --border-subtle: #2a2a2a;
        --border-hover: #3a3a3a;
        --border-active: rgba(34, 197, 94, 0.5);
        
        --accent-emerald: #22c55e;
        --accent-emerald-hover: #16a34a;
        --accent-emerald-dim: rgba(34, 197, 94, 0.15);
        --accent-emerald-border: rgba(34, 197, 94, 0.25);
        --accent-amber: #eab308;
        --accent-amber-dim: rgba(234, 179, 8, 0.15);
        
        --text-primary: #e5e5e5;
        --text-secondary: #a3a3a3;
        --text-muted: #666666;
        --text-dim: #444444;
    }

    /* =========================================================================
       2. Root Background, Fonts & Streamlit Overrides
       ========================================================================= */
    .stApp {
        background-color: var(--canvas-base) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--canvas-base) !important;
    }

    [data-testid="stHeader"] {
        background-color: var(--canvas-surface) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        height: 48px !important;
    }

    [data-testid="stToolbar"] {
        right: 1rem !important;
        top: 0.35rem !important;
    }

    /* Global typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }

    .mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
    ::-webkit-scrollbar-track { background: transparent; }

    /* =========================================================================
       3. Obsidian Sidebar (Explorer)
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: var(--canvas-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
        min-width: 280px !important;
        max-width: 320px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        background-color: var(--canvas-surface) !important;
    }

    /* =========================================================================
       4. Inputs, Selectboxes, File Uploaders
       ========================================================================= */
    .stTextInput > div > div,
    .stTextArea > div > div,
    [data-baseweb="select"] > div {
        background-color: var(--canvas-input) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent-emerald) !important;
        box-shadow: 0 0 0 1px var(--accent-emerald), 0 0 12px var(--accent-emerald-dim) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--canvas-subtle) !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-emerald-border) !important;
        background: rgba(34, 197, 94, 0.03) !important;
    }

    /* =========================================================================
       5. Obsidian Buttons
       ========================================================================= */
    .stButton > button {
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.8rem !important;
        border: 1px solid var(--border-subtle) !important;
        background-color: var(--canvas-input) !important;
        color: var(--text-primary) !important;
        transition: all 0.18s ease !important;
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

    /* Download button */
    .stDownloadButton > button {
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: 1px solid var(--accent-emerald) !important;
        background-color: transparent !important;
        color: var(--accent-emerald) !important;
        transition: all 0.18s ease !important;
    }

    .stDownloadButton > button:hover {
        background-color: var(--accent-emerald-dim) !important;
        color: var(--accent-emerald) !important;
    }

    /* =========================================================================
       6. Telemetry Badges & Pills
       ========================================================================= */
    .obsidian-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: var(--canvas-input);
        border: 1px solid var(--border-subtle);
        border-radius: 9999px;
        padding: 3px 9px;
        font-size: 11px;
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
        animation: pulseDot 2s infinite alternate;
    }

    @keyframes pulseDot {
        from { opacity: 0.6; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1.15); box-shadow: 0 0 8px #22c55e; }
    }

    .grid-bg {
        background-image: radial-gradient(rgba(34, 197, 94, 0.08) 1px, transparent 1px);
        background-size: 22px 22px;
    }

    /* =========================================================================
       7. Chat Feed & Tool Timeline Styles
       ========================================================================= */
    [data-testid="stChatMessage"] {
        background-color: var(--canvas-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.6rem !important;
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

    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent-emerald) !important;
        box-shadow: 0 0 0 1px var(--accent-emerald) !important;
    }

    /* Popover / Dialog */
    [data-testid="stPopoverBody"] {
        background-color: var(--canvas-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.75) !important;
    }
</style>
""", unsafe_allow_html=True)
