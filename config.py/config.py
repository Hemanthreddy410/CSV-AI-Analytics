import streamlit as st

def load_custom_css():
    """Load custom CSS styling for the application"""
    st.markdown("""
    <style>
    /* Main color palette */
    :root {
        --primary-color: #4A6FE3;
        --secondary-color: #2E4172;
        --accent-color: #6C63FF;
        --background-color: #F8F9FA;
        --card-bg-color: #FFFFFF;
        --text-color: #333333;
        --light-text-color: #6E6E6E;
        --border-color: #E0E0E0;
        --hover-color: #EEF2FF;
        --positive-color: #28A745;
        --warning-color: #FFC107;
        --negative-color: #DC3545;
    }
    
    /* Global styles */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--secondary-color);
        font-weight: 600;
    }
    
    /* Hide sidebar collapse button and footer */
    .st-emotion-cache-1dp5vir {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    /* NavBar styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: var(--card-bg-color);
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    .navbar-brand {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--secondary-color);
        display: flex;
        align-items: center;
    }
    
    .navbar-brand img {
        height: 32px;
        margin-right: 0.5rem;
    }
    
    .navbar-links {
        display: flex;
        gap: 1.5rem;
    }
    
    .navbar-link {
        color: var(--text-color);
        text-decoration: none;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        transition: all 0.3s;
    }
    
    .navbar-link:hover, .navbar-link.active {
        color: var(--primary-color);
        background-color: var(--hover-color);
    }
    
    /* Card styling */
    .card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin: 0;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: var(--card-bg-color);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .welcome-title {
        font-size: 3.5rem;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.5rem;
        color: var(--light-text-color);
        margin-bottom: 3rem;
    }
    
    .features-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background-color: var(--background-color);
        border-radius: 12px;
        padding: 2rem;
        width: 220px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s, box-shadow 0.3s;
        text-align: center;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-card h3 {
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
        color: var(--secondary-color);
    }
    
    .feature-card p {
        font-size: 0.9rem;
        color: var(--light-text-color);
    }
    
    .welcome-instruction {
        font-size: 1.2rem;
        color: var(--primary-color);
        margin-top: 3rem;
        font-weight: 500;
    }
    
    /* Metrics styling */
    .metrics-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1.5rem;
        flex: 1;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: var(--light-text-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe-container {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: var(--background-color);
        padding: 0.25rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: var(--card-bg-color);
        border-radius: 8px;
        margin: 0.25rem;
        font-weight: 500;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Chat interface styling */
    .chat-container {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        height: 400px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        margin-bottom: 1rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .user-message {
        flex-direction: row-reverse;
    }
    
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        max-width: 80%;
    }
    
    .user-bubble {
        background-color: var(--primary-color);
        color: white;
        border-top-right-radius: 0;
    }
    
    .assistant-bubble {
        background-color: var(--background-color);
        border-top-left-radius: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg-color);
    }
    
    [data-testid="stSidebarNav"] {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebarNav"] > div {
        gap: 0.5rem;
    }
    
    [data-testid="stSidebarNav"] button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    [data-testid="stSidebarNav"] button:hover {
        background-color: var(--hover-color);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--light-text-color);
    }
    </style>
    """, unsafe_allow_html=True)