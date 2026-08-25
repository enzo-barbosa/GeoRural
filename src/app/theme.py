"""Tema visual e CSS customizado para a aplicação."""

PRIMARY_COLOR = "#2e7d32"
SECONDARY_COLOR = "#1565c0"
BACKGROUND_COLOR = "#f5f5f5"
TEXT_COLOR = "#212121"
ACCENT_COLOR = "#ff6f00"
ERROR_COLOR = "#c62828"
SUCCESS_COLOR = "#2e7d32"
WARNING_COLOR = "#f9a825"

CLASS_COLORS = {
    "Agricultura": "#2ecc71",
    "Pastagem": "#27ae60",
    "Vegetação Nativa": "#1a5276",
    "Solo Exposto": "#f39c12",
    "Área Construída": "#e74c3c",
}

CLASS_NAMES = list(CLASS_COLORS.keys())


def get_css() -> str:
    """Retorna CSS customizado para injetar na aplicação."""
    return f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}

    .main-header {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }}

    .main-header h1 {{
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }}

    .main-header p {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }}

    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid {PRIMARY_COLOR};
    }}

    .feature-card h3 {{
        color: {PRIMARY_COLOR};
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }}

    .feature-card p {{
        color: {TEXT_COLOR};
        margin: 0;
        line-height: 1.6;
    }}

    .step-indicator {{
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }}

    .step-number {{
        background: {SECONDARY_COLOR};
        color: white;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
        flex-shrink: 0;
    }}

    .step-content {{
        flex: 1;
    }}

    .step-content strong {{
        color: {SECONDARY_COLOR};
    }}

    .legend-container {{
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }}

    .legend-item {{
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }}

    .legend-color {{
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 0.25rem;
        margin-right: 0.75rem;
        border: 1px solid rgba(0,0,0,0.1);
    }}

    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid {PRIMARY_COLOR};
    }}

    .metric-card .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin: 0.5rem 0;
    }}

    .metric-card .metric-label {{
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}

    .status-badge.success {{
        background: #e8f5e9;
        color: {SUCCESS_COLOR};
    }}

    .status-badge.warning {{
        background: #fff8e1;
        color: {WARNING_COLOR};
    }}

    .status-badge.error {{
        background: #ffebee;
        color: {ERROR_COLOR};
    }}

    .footer {{
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }}

    .info-box {{
        background: #e3f2fd;
        border-left: 4px solid {SECONDARY_COLOR};
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }}

    .info-box strong {{
        color: {SECONDARY_COLOR};
    }}
    </style>
    """
