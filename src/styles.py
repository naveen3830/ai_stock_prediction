# styles.py - Modern UI Styling Module
"""
Custom CSS styling with dark/light theme support,
glassmorphism effects, and animated components.
"""

def get_theme_css(is_dark_mode: bool = True) -> str:
    """Generate CSS based on theme selection."""
    
    if is_dark_mode:
        colors = {
            'bg_primary': '#0e1117',
            'bg_secondary': '#1a1f2e',
            'bg_card': 'rgba(26, 31, 46, 0.8)',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b8c4',
            'accent_primary': '#00d4aa',
            'accent_secondary': '#667eea',
            'accent_gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'success': '#00d4aa',
            'warning': '#ffa726',
            'danger': '#ef5350',
            'border': 'rgba(255, 255, 255, 0.1)',
            'glass_bg': 'rgba(26, 31, 46, 0.7)',
            'shadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
        }
    else:
        colors = {
            'bg_primary': '#f5f7fa',
            'bg_secondary': '#ffffff',
            'bg_card': 'rgba(255, 255, 255, 0.9)',
            'text_primary': '#1a1f2e',
            'text_secondary': '#64748b',
            'accent_primary': '#059669',
            'accent_secondary': '#6366f1',
            'accent_gradient': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            'success': '#059669',
            'warning': '#d97706',
            'danger': '#dc2626',
            'border': 'rgba(0, 0, 0, 0.1)',
            'glass_bg': 'rgba(255, 255, 255, 0.7)',
            'shadow': '0 8px 32px rgba(0, 0, 0, 0.1)',
        }
    
    return f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Main Header Styling */
        .main-header {{
            text-align: center;
            padding: 2rem 1rem;
            background: {colors['accent_gradient']};
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: {colors['shadow']};
        }}
        
        .main-header h1 {{
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .main-header p {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }}
        
        /* Glassmorphism Card */
        .glass-card {{
            background: {colors['glass_bg']};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid {colors['border']};
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: {colors['shadow']};
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }}
        
        /* Metric Cards */
        .metric-container {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 1.5rem 0;
        }}
        
        .metric-card {{
            flex: 1;
            min-width: 200px;
            background: {colors['glass_bg']};
            backdrop-filter: blur(10px);
            border: 1px solid {colors['border']};
            border-radius: 16px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: {colors['accent_gradient']};
        }}
        
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        
        .metric-card .label {{
            color: {colors['text_secondary']};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}
        
        .metric-card .value {{
            color: {colors['text_primary']};
            font-size: 1.75rem;
            font-weight: 700;
        }}
        
        .metric-card .delta {{
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        
        .metric-card .delta.positive {{
            color: {colors['success']};
        }}
        
        .metric-card .delta.negative {{
            color: {colors['danger']};
        }}
        
        /* Section Headers */
        .section-header {{
            color: {colors['text_primary']};
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {colors['accent_primary']};
            display: inline-block;
        }}
        
        /* Tab Navigation */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: {colors['glass_bg']};
            padding: 0.5rem;
            border-radius: 12px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {colors['accent_gradient']} !important;
            color: white !important;
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: {colors['bg_secondary']};
            border-right: 1px solid {colors['border']};
        }}
        
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}
        
        /* Button Styling */
        .stButton > button {{
            background: {colors['accent_gradient']};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        /* Select Box Styling */
        .stSelectbox > div > div {{
            background: {colors['glass_bg']};
            border: 1px solid {colors['border']};
            border-radius: 10px;
        }}
        
        /* Slider Styling */
        .stSlider > div > div > div {{
            background: {colors['accent_gradient']};
        }}
        
        /* Info/Success/Warning Boxes */
        .stAlert {{
            border-radius: 12px;
            border: none;
        }}
        
        /* Stock Symbol Badge */
        .stock-badge {{
            display: inline-block;
            background: {colors['accent_gradient']};
            color: white;
            padding: 0.5rem 1.25rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        /* Price Display */
        .price-display {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {colors['text_primary']};
        }}
        
        .price-change {{
            font-size: 1.1rem;
            font-weight: 500;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            display: inline-block;
            margin-left: 0.5rem;
        }}
        
        .price-change.up {{
            background: rgba(0, 212, 170, 0.2);
            color: {colors['success']};
        }}
        
        .price-change.down {{
            background: rgba(239, 83, 80, 0.2);
            color: {colors['danger']};
        }}
        
        /* Chart Container */
        .chart-container {{
            background: {colors['glass_bg']};
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid {colors['border']};
        }}
        
        /* Indicator Pills */
        .indicator-pill {{
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0.25rem;
        }}
        
        .indicator-pill.bullish {{
            background: rgba(0, 212, 170, 0.2);
            color: {colors['success']};
            border: 1px solid {colors['success']};
        }}
        
        .indicator-pill.bearish {{
            background: rgba(239, 83, 80, 0.2);
            color: {colors['danger']};
            border: 1px solid {colors['danger']};
        }}
        
        .indicator-pill.neutral {{
            background: rgba(255, 167, 38, 0.2);
            color: {colors['warning']};
            border: 1px solid {colors['warning']};
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            color: {colors['text_secondary']};
            border-top: 1px solid {colors['border']};
        }}
        
        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .animate-fade-in {{
            animation: fadeInUp 0.5s ease forwards;
        }}
        
        /* Loading Spinner Override */
        .stSpinner > div {{
            border-top-color: {colors['accent_primary']} !important;
        }}
        
        /* Hide Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Responsive Adjustments */
        @media (max-width: 768px) {{
            .main-header h1 {{
                font-size: 1.75rem;
            }}
            
            .metric-card {{
                min-width: 150px;
            }}
        }}
    </style>
    """


def render_header(title: str, subtitle: str = "") -> str:
    """Render the main application header."""
    return f"""
    <div class="main-header animate-fade-in">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def render_metric_card(label: str, value: str, delta: str = None, delta_positive: bool = True) -> str:
    """Render an animated metric card."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        delta_symbol = "▲" if delta_positive else "▼"
        delta_html = f'<div class="delta {delta_class}">{delta_symbol} {delta}</div>'
    
    return f"""
    <div class="metric-card animate-fade-in">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def render_stock_badge(symbol: str, company: str) -> str:
    """Render a stock symbol badge."""
    return f"""
    <div style="margin: 1rem 0;">
        <span class="stock-badge">{symbol}</span>
        <span style="margin-left: 0.5rem; font-size: 1.1rem;">{company}</span>
    </div>
    """


def render_price_display(price: float, change: float, change_pct: float) -> str:
    """Render the current price with change indicator."""
    change_class = "up" if change >= 0 else "down"
    change_symbol = "+" if change >= 0 else ""
    
    return f"""
    <div style="margin: 1rem 0;">
        <span class="price-display">${price:,.2f}</span>
        <span class="price-change {change_class}">
            {change_symbol}{change:,.2f} ({change_symbol}{change_pct:.2f}%)
        </span>
    </div>
    """


def render_indicator_signal(name: str, signal: str) -> str:
    """Render a technical indicator signal pill."""
    signal_lower = signal.lower()
    if signal_lower in ['bullish', 'buy', 'oversold']:
        signal_class = 'bullish'
    elif signal_lower in ['bearish', 'sell', 'overbought']:
        signal_class = 'bearish'
    else:
        signal_class = 'neutral'
    
    return f'<span class="indicator-pill {signal_class}">{name}: {signal}</span>'


def render_section_header(title: str, emoji: str = "") -> str:
    """Render a styled section header."""
    return f'<h2 class="section-header">{emoji} {title}</h2>'


def get_plotly_theme(is_dark_mode: bool = True) -> dict:
    """Get Plotly chart theme configuration."""
    if is_dark_mode:
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#ffffff',
            'gridcolor': 'rgba(255,255,255,0.1)',
            'colors': ['#667eea', '#00d4aa', '#ffa726', '#ef5350', '#ab47bc', '#42a5f5'],
            'increasing_color': '#00d4aa',
            'decreasing_color': '#ef5350',
        }
    else:
        return {
            'template': 'plotly_white',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#1a1f2e',
            'gridcolor': 'rgba(0,0,0,0.1)',
            'colors': ['#6366f1', '#059669', '#d97706', '#dc2626', '#9333ea', '#0284c7'],
            'increasing_color': '#059669',
            'decreasing_color': '#dc2626',
        }
