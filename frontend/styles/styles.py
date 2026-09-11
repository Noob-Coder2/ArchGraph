"""
DocuStream Modern UI Styling
Glassmorphic design system, gradient accents, responsive cards, and enhanced micro-interactions.
"""
import streamlit as st


def apply_custom_styles():
    """Applies clean, unified custom CSS to the Streamlit app."""
    st.markdown("""
<style>
    /* ==================== Design System Tokens ==================== */
    :root {
        --glass-bg-light: rgba(255, 255, 255, 0.7);
        --glass-bg-dark: rgba(20, 24, 38, 0.75);
        --glass-border-light: rgba(255, 255, 255, 0.25);
        --glass-border-dark: rgba(255, 255, 255, 0.08);
        --shadow-light: rgba(0, 0, 0, 0.08);
        --shadow-dark: rgba(0, 0, 0, 0.35);
        --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --accent-gradient-hover: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        --accent-cyan: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }

    /* ==================== Typography & Base ==================== */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    h1 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* ==================== Glassmorphism Cards ==================== */
    .metric-card, .glass-card {
        background: rgba(102, 126, 234, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(102, 126, 234, 0.18);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover, .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.35);
    }

    /* ==================== File Uploader Dropzone ==================== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.4) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        background: rgba(102, 126, 234, 0.04) !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8) !important;
        background: rgba(102, 126, 234, 0.08) !important;
    }

    /* ==================== Buttons & Interactive Elements ==================== */
    .stButton > button {
        border-radius: 12px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }

    /* ==================== Tabs Styling ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 6px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
    }

    /* ==================== Form Inputs Focus ==================== */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25) !important;
    }

    /* ==================== Custom Tag & Badge Styles ==================== */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)
