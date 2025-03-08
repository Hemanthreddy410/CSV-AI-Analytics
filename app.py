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
from modules.visualization import Visualizer
from modules.ai_assistant import AIAssistant
from modules.data_processor import DataProcessor
from modules.gen_ai_assistant import GenAIAssistant  # Add GenAI module
from modules.utils import load_custom_css, add_logo, encode_image
from modules.ui_components import create_header, create_footer
from modules.utils import load_custom_css, add_logo, encode_image, fix_arrow_dtypes

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
        
        # Check if data is loaded
        if st.session_state.df is None:
            st.info("👆 Please upload a CSV file using the sidebar to get started")
        else:
            df = st.session_state.df
            
            # Create tabs for different functionality
            tabs = st.tabs([
                "📋 Data Explorer", 
                "📊 Visualization Studio", 
                "🔍 Analysis Hub",
                "🤖 Data Assistant",
                "🧠 GenAI Assistant",  # New GenAI tab
                "📝 Data Processing"
            ])
            
            # Data Explorer Tab
            with tabs[0]:
                st.header("Data Explorer")
                
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
                
                # Column details
                with st.expander("Column Details"):
                    selected_column = st.selectbox("Select a column for details:", df.columns)
                    
                    if selected_column:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Column:** {selected_column}")
                            st.write(f"**Type:** {df[selected_column].dtype}")
                            st.write(f"**Missing Values:** {df[selected_column].isna().sum()}")
                            st.write(f"**Unique Values:** {df[selected_column].nunique()}")
                            
                            if df[selected_column].dtype in ['int64', 'float64']:
                                st.write(f"**Min:** {df[selected_column].min()}")
                                st.write(f"**Max:** {df[selected_column].max()}")
                                st.write(f"**Mean:** {df[selected_column].mean()}")
                                st.write(f"**Median:** {df[selected_column].median()}")
                        
                        with col2:
                            # Show value counts or histogram depending on data type
                            if df[selected_column].dtype in ['int64', 'float64']:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
                                plt.title(f'Distribution of {selected_column}')
                                st.pyplot(fig)
                            else:
                                # Show top 10 value counts for non-numeric columns
                                value_counts = df[selected_column].value_counts().head(10)
                                fig, ax = plt.subplots(figsize=(10, 6))
                                value_counts.plot(kind='bar', ax=ax)
                                plt.title(f'Top 10 Values in {selected_column}')
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
            
            # Visualization Studio Tab
            with tabs[1]:
                visualizer = Visualizer(df)
                visualizer.render_interface()
            
            # Analysis Hub Tab
            with tabs[2]:
                analyzer = DataAnalyzer(df)
                analyzer.render_interface()
            
            # Data Assistant Tab
            with tabs[3]:
                assistant = AIAssistant(df)
                assistant.render_interface()
            
            # NEW: GenAI Assistant Tab
            with tabs[4]:
                gen_ai_assistant = GenAIAssistant(df)
                gen_ai_assistant.render_interface()
            
            # Data Processing Tab
            with tabs[5]:
                processor = DataProcessor(df)
                processor.render_interface()
    
    # Create footer
    create_footer()

if __name__ == "__main__":
    main()