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
from modules.utils import load_custom_css, add_logo, encode_image, fix_arrow_dtypes
from modules.ui_components import create_header, create_footer
from modules.data_workflow import DataWorkflowManager

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

if 'original_df' not in st.session_state:
    st.session_state.original_df = None

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []

# Set OpenAI API key from environment variable if available
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Flag to prevent infinite rerun loops
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

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
    
    # Initialize data workflow manager
    workflow_manager = DataWorkflowManager()
    
    # Sidebar
    with st.sidebar:
        st.title("DataInsightHub")
        
        # Project management section
        project_manager.render_sidebar()
        
        st.sidebar.divider()
        
        # Upload section (only if project is selected)
        if st.session_state.current_project is not None:
            st.sidebar.header("📂 Data Source")
            
            # Show upload widget
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            
            # Only process upload if not currently processing (prevents loops)
            if uploaded_file is not None and not st.session_state.is_processing:
                try:
                    # Set processing flag to prevent rerun loops
                    st.session_state.is_processing = True
                    
                    # Process and display the uploaded file
                    with st.spinner("Processing your data..."):
                        # Read CSV directly without using workflow manager
                        df = pd.read_csv(uploaded_file)
                        
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        
                        # Store in project data
                        if st.session_state.current_project in st.session_state.projects:
                            st.session_state.projects[st.session_state.current_project]['data'] = df
                        
                        # Mark as processed in workflow manager (simplified approach)
                        if 'data_workflow' not in st.session_state:
                            st.session_state.data_workflow = {
                                'original_file': uploaded_file.name,
                                'processed_file': None,
                                'processing_done': True,
                                'temp_dir': None,
                                'processing_timestamp': None
                            }
                        else:
                            st.session_state.data_workflow['original_file'] = uploaded_file.name
                            st.session_state.data_workflow['processing_done'] = True
                        
                        # Success message
                        st.sidebar.success(f"✅ File loaded successfully: {uploaded_file.name}")
                        
                        # Reset visualizations
                        st.session_state.visualizations = []
                        
                        # Reset processing flag
                        st.session_state.is_processing = False
                    
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
                    st.session_state.is_processing = False
            
            # Show data modification status if data exists
            if 'df' in st.session_state and st.session_state.df is not None:
                # Always show data as loaded
                st.sidebar.success(f"✅ Data loaded successfully")
    
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
        
        # Make sure we have data to work with
        if 'df' not in st.session_state or st.session_state.df is None:
            st.info("Please upload a dataset using the sidebar to get started.")
        else:
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
                st.header("Data Explorer")
                
                # Always use the session state DataFrame
                df = st.session_state.df
                
                # Display dataset info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isna().sum().sum())
                with col4:
                    st.metric("Duplicates", df.duplicated().sum())
                
                # Data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(fix_arrow_dtypes(df.head(10)), use_container_width=True)
                
                # Column information
                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isna().sum(),
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(fix_arrow_dtypes(col_info), use_container_width=True)
                
                # Data statistics
                with st.expander("Data Statistics"):
                    if df.select_dtypes(include=['number']).columns.tolist():
                        st.dataframe(df.describe(), use_container_width=True)
                    else:
                        st.info("No numeric columns found for statistics")
            
            # Data Processing Tab
            with tabs[1]:
                try:
                    processor = DataProcessor(st.session_state.df, workflow_manager)
                    processor.render_interface()
                except Exception as e:
                    st.error(f"Error in Data Processing tab: {str(e)}")
            
            # Just use the current DataFrame for all other tabs
            current_df = st.session_state.df
            
            try:
                # Visualization Studio Tab
                with tabs[2]:
                    try:
                        visualizer = EnhancedVisualizer(current_df)
                        visualizer.render_interface()
                    except Exception as e:
                        st.error(f"Error in Visualization Studio: {str(e)}")
                        st.info("This may be due to missing the kaleido package. You can install it with: pip install -U kaleido")
                
                # Analysis Hub Tab
                with tabs[3]:
                    try:
                        analyzer = DataAnalyzer(current_df)
                        analyzer.render_interface()
                    except Exception as e:
                        st.error(f"Error in Analysis Hub: {str(e)}")
                
                # Data Assistant Tab
                with tabs[4]:
                    try:
                        assistant = AIAssistant(current_df)
                        assistant.render_interface()
                    except Exception as e:
                        st.error(f"Error in Data Assistant: {str(e)}")
                
                # GenAI Assistant Tab
                with tabs[5]:
                    try:
                        gen_ai_assistant = GenAIAssistant(current_df)
                        gen_ai_assistant.render_interface()
                    except Exception as e:
                        st.error(f"Error in GenAI Assistant: {str(e)}")
            except Exception as e:
                st.error(f"General error in tabs: {str(e)}")
    
    # Create footer
    create_footer()

if __name__ == "__main__":
    main()