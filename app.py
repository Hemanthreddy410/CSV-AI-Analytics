import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import matplotlib

# Import modules
from modules.project_manager import ProjectManager
from modules.data_analyzer import DataAnalyzer
from modules.visualization import EnhancedVisualizer
from modules.ai_assistant import AIAssistant
from modules.data_processor import DataProcessor
from modules.gen_ai_assistant import GenAIAssistant
from modules.data_explorer import DataExplorer  # Import the new DataExplorer class
from modules.utils import load_custom_css, add_logo, encode_image, fix_arrow_dtypes
from modules.ui_components import create_header, create_footer

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'projects' not in st.session_state:
    st.session_state.projects = {}

if 'current_project' not in st.session_state:
    st.session_state.current_project = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = None

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []

# Set OpenAI API key from environment variable if available
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

def main():
    # Page config
    st.set_page_config(
        page_title="DataInsightHub",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Add logo
    add_logo()
    
    # Create header
    create_header()
    
    # Initialize project manager
    project_manager = ProjectManager()
    
    # Sidebar
    with st.sidebar:
        st.title("DataInsightHub")
        
        # Project management section
        project_manager.render_sidebar()
        
        st.sidebar.divider()
        
        # Upload section (only if project is selected)
        if st.session_state.current_project is not None:
            st.sidebar.header("📂 Data Source")
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Process and display the uploaded file
                    with st.spinner("Processing your data..."):
                        df = pd.read_csv(uploaded_file)
                        
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.projects[st.session_state.current_project]['data'] = df
                        
                        # Success message
                        st.sidebar.success(f"✅ File loaded successfully: {uploaded_file.name}")
                        
                        # Reset visualizations when new file is uploaded
                        st.session_state.visualizations = []
                        
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
    
    # Main content
    if st.session_state.current_project is None:
        # Welcome screen
        st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
        st.markdown("<h1 class='welcome-title'>Welcome to DataInsightHub</h1>", unsafe_allow_html=True)
        st.markdown("<p class='welcome-subtitle'>Your all-in-one platform for data analysis and visualization</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='features-container'>
            <div class='feature-card'>
                <div class='feature-icon'>📊</div>
                <h3>Interactive Visualizations</h3>
                <p>Create beautiful charts and graphs with just a few clicks</p>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <h3>AI-Powered Analysis</h3>
                <p>Get instant insights with our intelligent assistant</p>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>📁</div>
                <h3>Project Management</h3>
                <p>Organize your data analyses into separate projects</p>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🔍</div>
                <h3>Data Exploration</h3>
                <p>Discover patterns and relationships in your data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p class='welcome-instruction'>To get started, create a new project using the sidebar</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Display current project name
        st.markdown(f"<h1 class='project-title'>Project: {st.session_state.current_project}</h1>", unsafe_allow_html=True)
        
        # Get the dataframe from session state
        df = st.session_state.df
        
        # Create tabs for different functionality
        tabs = st.tabs([
            "📋 Data Explorer", 
            "📝 Data Processing",
            "📊 Visualization Studio", 
            "🔍 Analysis Hub",
            "🤖 Data Assistant",
            "🧠 GenAI Assistant"
        ])

        # Data Explorer Tab
        with tabs[0]:
            explorer = DataExplorer(df)
            explorer.render_interface()

        # Data Processing Tab 
        with tabs[1]:
            processor = DataProcessor(df)
            processor.render_interface()

        # Visualization Studio Tab
        with tabs[2]:
            visualizer = EnhancedVisualizer(df)
            visualizer.render_interface()

        # Analysis Hub Tab
        with tabs[3]:
            analyzer = DataAnalyzer(df)
            analyzer.render_interface()

        # Data Assistant Tab
        with tabs[4]:
            assistant = AIAssistant(df)
            assistant.render_interface()

        # GenAI Assistant Tab
        with tabs[5]:
            gen_ai_assistant = GenAIAssistant(df)
            gen_ai_assistant.render_interface()
    
    # Create footer
    create_footer()

if __name__ == "__main__":
    main()