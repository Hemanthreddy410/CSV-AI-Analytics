import streamlit as st

def load_custom_css(theme="light"):
    """Load custom CSS based on the selected theme"""
    
    # Common CSS for all themes
    common_css = """
    <style>
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 15px;
        border-radius: 4px 4px 0px 0px;
    }
    
    div.block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        width: 100%;
    }
    
    .section-divider {
        height: 3px;
        margin: 1.5rem 0;
    }
    
    /* Header styling */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4CAF50;
        margin: 0;
    }
    
    .app-version {
        font-size: 0.8rem;
        color: #888;
    }
    </style>
    """
    
    # Light theme CSS
    light_theme_css = """
    <style>
    .data-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .section-divider {
        background-color: #f0f2f6;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """
    
    # Dark theme CSS
    dark_theme_css = """
    <style>
    .data-info {
        background-color: #262730;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .section-divider {
        background-color: #262730;
    }
    
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    </style>
    """
    
    # Apply common CSS
    st.markdown(common_css, unsafe_allow_html=True)
    
    # Apply theme-specific CSS
    if theme.lower() == "dark":
        st.markdown(dark_theme_css, unsafe_allow_html=True)
    else:
        st.markdown(light_theme_css, unsafe_allow_html=True)

def create_header(version="1.0.0", title="CSV AI Analytics", subtitle="Analyze and visualize your data with ease"):
    """Create application header with version information"""
    
    header_html = f"""
    <div class="app-header">
        <div>
            <h1 class="app-title">{title}</h1>
            <p>{subtitle}</p>
        </div>
        <div class="app-version">
            Version {version}
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Optional: Add a horizontal separator after the header
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)