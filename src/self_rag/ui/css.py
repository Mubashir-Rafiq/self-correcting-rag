"""Custom styling and CSS theme for the Self-Correcting RAG Gradio application."""

from __future__ import annotations

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #8b5cf6 100%);
    --card-bg: rgba(15, 23, 42, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --glow-cyan: rgba(6, 182, 212, 0.25);
}

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                #0b0f17 !important;
    color: var(--text-primary) !important;
}

/* Header Banner */
.app-header {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-subtle);
}

.app-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem !important;
    letter-spacing: -0.025em;
}

.app-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.95rem !important;
    font-weight: 400;
}

/* Glassmorphic Tabs & Cards */
.tabs {
    border-radius: 16px !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--card-bg) !important;
    backdrop-filter: blur(16px);
    overflow: hidden;
    padding: 8px !important;
}

.tab-nav {
    border-bottom: 1px solid var(--border-subtle) !important;
    margin-bottom: 12px !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    transition: all 0.25s ease-in-out !important;
}

.tab-nav button.selected {
    background: var(--primary-gradient) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px var(--glow-cyan) !important;
}

/* Chatbot container */
.chatbot-container {
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    background: rgba(11, 15, 23, 0.6) !important;
    backdrop-filter: blur(8px);
}

/* Buttons */
button.primary-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

button.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
}

/* Status Badges */
.thread-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Document List Box */
.sources-box {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 0.85rem !important;
    border-radius: 12px !important;
    background: rgba(2, 6, 23, 0.8) !important;
    border: 1px solid var(--border-subtle) !important;
}
"""
