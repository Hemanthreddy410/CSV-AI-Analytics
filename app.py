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
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Process and display the uploaded file
                    with st.spinner("Processing your data..."):
                        # Use workflow manager to handle the upload
                        df = workflow_manager.handle_upload(uploaded_file)
                        
                        if df is not None:
                            # Store in session state
                            st.session_state.df = df
                            st.session_state.original_df = df.copy()  # Keep a copy of the original data
                            st.session_state.projects[st.session_state.current_project]['data'] = df
                            
                            # Reset processing history when new file is uploaded
                            if 'processing_history' in st.session_state:
                                st.session_state.processing_history = []
                            
                            # Success message
                            st.sidebar.success(f"✅ File loaded successfully: {uploaded_file.name}")
                            
                            # Reset visualizations when new file is uploaded
                            st.session_state.visualizations = []
                            
                            # Reset workflow state (processing not done yet)
                            workflow_manager.reset_workflow()
                        
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
            
            # Show data modification status if data exists
            if st.session_state.df is not None:
                # Display workflow status
                status = workflow_manager.get_workflow_status()
                
                if status['processing_done']:
                    st.sidebar.success(f"✅ Data processing completed")
                    if status['processing_timestamp']:
                        st.sidebar.info(f"Last processed: {status['processing_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    if 'processing_history' in st.session_state and len(st.session_state.processing_history) > 0:
                        st.sidebar.warning(f"📝 Data partially processed ({len(st.session_state.processing_history)} changes)")
                    else:
                        st.sidebar.info("⚠️ Please complete data processing before using other modules")
    
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
        if st.session_state.df is None:
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
                processor = DataProcessor(st.session_state.df, workflow_manager)  # Pass session state df and workflow manager
                processor.render_interface()
            
            # Check if data processing is completed before allowing access to other tabs
            status = workflow_manager.get_workflow_status()
            if not status['processing_done']:
                # For remaining tabs, show a message prompting to complete data processing first
                for i in range(2, 6):
                    with tabs[i]:
                        st.warning("⚠️ Please complete data processing before using this module.")
                        st.info("Go to the 'Data Processing' tab and save your processed data.")
            else:
                # Get the processed data from workflow manager
                processed_df = workflow_manager.get_data(require_processed=True)
                
                if processed_df is None:
                    st.error("Error loading processed data. Please try processing your data again.")
                else:
                    # Visualization Studio Tab
                    with tabs[2]:
                        visualizer = EnhancedVisualizer(processed_df)  # Pass processed df
                        visualizer.render_interface()
                    
                    # Analysis Hub Tab
                    with tabs[3]:
                        analyzer = DataAnalyzer(processed_df)  # Pass processed df
                        analyzer.render_interface()
                    
                    # Data Assistant Tab
                    with tabs[4]:
                        assistant = AIAssistant(processed_df)  # Pass processed df
                        assistant.render_interface()
                    
                    # GenAI Assistant Tab
                    with tabs[5]:
                        gen_ai_assistant = GenAIAssistant(processed_df)  # Pass processed df
                        gen_ai_assistant.render_interface()
    
    # Create footer
    create_footer()

if __name__ == "__main__":
    main()