# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from io import StringIO, BytesIO
# import base64
# import re
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from modules.gen_ai_assistant import GenAIAssistant

# # Import custom modules
# from modules.ui_helper import load_custom_css, create_header
# from modules.data_manager import (
#     initialize_session_state, 
#     load_file, 
#     load_sample_dataset, 
#     save_to_project, 
#     load_from_project,
#     get_dataframe_info
# )
# from modules.theme_manager import ThemeManager
# from modules.data_exporter import DataExporter
# from modules.dashboard_generator import DashboardGenerator
# from modules.file_handler import handle_uploaded_file, save_uploaded_file, load_sample_data
# from modules.data_overview import render_data_overview
# from modules.data_processor import render_data_processing
# from modules.visualization import render_visualization
# from modules.analysis import render_analysis
# from modules.reports import render_reports
# # from modules.dashboard import render_dashboard
# from modules.dashboard import render_dashboard, render_exploratory_report, render_trend_analysis, render_distribution_analysis
# # App configuration
# APP_TITLE = "CSV AI Analytics"
# APP_SUBTITLE = "Analyze and visualize your data with ease"
# APP_VERSION = "1.0.0"

# def main():
#     """Main application entry point"""
#     # Initialize session state
#     initialize_session_state()
    
#     # Setup page configuration
#     st.set_page_config(
#         page_title=APP_TITLE,
#         page_icon="📊",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Initialize theme manager
#     theme_manager = ThemeManager()
#     theme_manager.apply_theme()
    
#     # Load custom CSS
#     load_custom_css(theme=st.session_state.theme)
    
#     # Create header
#     create_header(version=APP_VERSION, title=APP_TITLE, subtitle=APP_SUBTITLE)
    
#     # Create sidebar and main content
#     render_sidebar()
#     render_main_content()

# def render_sidebar():
#     """Render the sidebar with app controls"""
#     with st.sidebar:
#         st.title("📊 CSV AI Analytics")
#         st.markdown("Explore, analyze, and visualize your data")
        
#         st.markdown("---")
        
#         # Data upload section
#         st.header("Data Upload")
        
#         uploaded_file = st.file_uploader("Upload your data file", type=["csv", "excel", "xlsx", "xls", "json", "txt"])
        
#         if uploaded_file is not None:
#             if st.button("Process File", key="process_file_sidebar", use_container_width=True):
#                 if load_file(uploaded_file):
#                     st.success(f"Successfully loaded {uploaded_file.name}")
#                     st.rerun()
        
#         st.markdown("---")
        
#         # Sample data section
#         st.header("Sample Datasets")
#         sample_data = st.selectbox(
#             "Load a sample dataset",
#             ["Choose a dataset...", "Iris", "Titanic", "Boston Housing", "Wine Quality", "Diabetes"]
#         )
        
#         if sample_data != "Choose a dataset...":
#             if st.button("Load Sample Dataset", key="load_sample_sidebar", use_container_width=True):
#                 if load_sample_dataset(sample_data):
#                     st.success(f"Successfully loaded {sample_data} dataset")
#                     st.rerun()
        
#         st.markdown("---")
        
#         # Project management section
#         render_project_management()
        
#         # Show current data info
#         if st.session_state.df is not None:
#             st.markdown("---")
            
#             st.subheader("Current Data")
            
#             # Display dataset info
#             df_info = get_dataframe_info()
            
#             st.caption(f"Rows: {df_info['rows']} | Columns: {df_info['columns']}")
            
#             if st.session_state.file_details:
#                 st.caption(f"Source: {st.session_state.file_details['filename']}")
            
#             if st.session_state.current_project:
#                 st.caption(f"Project: {st.session_state.current_project}")
            
#             # Data preview
#             if st.button("Show Data Preview", key="data_preview_sidebar", use_container_width=True):
#                 st.dataframe(st.session_state.df.head(), use_container_width=True)
            
#             # Quick export
#             exporter = DataExporter(st.session_state.df)
#             st.markdown("#### Quick Export")
#             exporter.quick_export_widget("Download CSV")
        
#         # Theme settings
#         st.markdown("---")
        
#         with st.expander("Theme Settings", expanded=False):
#             theme_manager = ThemeManager()
#             theme_manager.render_theme_selector()
        
#         # App info footer
#         st.markdown("---")
#         st.markdown(
#             f"""
#             <div style="text-align: center;">
#             <p style="font-size: 0.8em; color: gray;">
#             {APP_TITLE} v{APP_VERSION}<br>
#             © 2023
#             </p>
#             </div>
#             """, 
#             unsafe_allow_html=True
#         )

# def render_project_management():
#     """Render project management section in sidebar"""
#     st.header("Projects")
    
#     # Project actions
#     project_action = st.selectbox(
#         "Project Action",
#         ["Select Project", "Create New Project", "Rename Project", "Delete Project"]
#     )
    
#     if project_action == "Select Project":
#         # Get list of projects
#         project_list = list(st.session_state.projects.keys())
        
#         if project_list:
#             # Add "None" option
#             project_list = ["None"] + project_list
            
#             # Select project
#             selected_project = st.selectbox(
#                 "Select a Project",
#                 project_list,
#                 index=0 if st.session_state.current_project is None else project_list.index(st.session_state.current_project)
#             )
            
#             if st.button("Load Project", key="load_project_button", use_container_width=True):
#                 if selected_project == "None":
#                     st.session_state.current_project = None
#                     st.success("Project selection cleared.")
#                 else:
#                     # Load project data
#                     if load_from_project(selected_project):
#                         st.success(f"Project '{selected_project}' loaded successfully!")
#                         st.rerun()
#         else:
#             st.info("No projects available. Create a new project first.")
    
#     elif project_action == "Create New Project":
#         # Get project name
#         project_name = st.text_input("New Project Name")
        
#         if project_name and st.button("Create Project", key="create_project_button", use_container_width=True):
#             if project_name in st.session_state.projects:
#                 st.error(f"Project '{project_name}' already exists!")
#             else:
#                 # Create new project
#                 if save_to_project(project_name):
#                     st.success(f"Project '{project_name}' created successfully!")
#                     st.rerun()
    
#     elif project_action == "Rename Project":
#         # Get list of projects
#         project_list = list(st.session_state.projects.keys())
        
#         if project_list:
#             # Select project to rename
#             old_name = st.selectbox("Select Project to Rename", project_list)
#             new_name = st.text_input("New Project Name")
            
#             if old_name and new_name and st.button("Rename Project", key="rename_project_button", use_container_width=True):
#                 if new_name in st.session_state.projects:
#                     st.error(f"Project '{new_name}' already exists!")
#                 else:
#                     # Rename project
#                     st.session_state.projects[new_name] = st.session_state.projects[old_name]
#                     del st.session_state.projects[old_name]
                    
#                     # Update current project if needed
#                     if st.session_state.current_project == old_name:
#                         st.session_state.current_project = new_name
                    
#                     st.success(f"Project renamed from '{old_name}' to '{new_name}' successfully!")
#                     st.rerun()
#         else:
#             st.info("No projects available to rename.")
    
#     elif project_action == "Delete Project":
#         # Get list of projects
#         project_list = list(st.session_state.projects.keys())
        
#         if project_list:
#             # Select project to delete
#             project_to_delete = st.selectbox("Select Project to Delete", project_list)
            
#             # Confirmation
#             confirm = st.checkbox("I understand this action cannot be undone.")
            
#             if project_to_delete and confirm and st.button("Delete Project", key="delete_project_button", use_container_width=True):
#                 # Delete project
#                 del st.session_state.projects[project_to_delete]
                
#                 # Update current project if needed
#                 if st.session_state.current_project == project_to_delete:
#                     st.session_state.current_project = None
                    
#                     # Clear dataframes if they belonged to the deleted project
#                     st.session_state.df = None
#                     st.session_state.original_df = None
#                     st.session_state.file_details = None
#                     st.session_state.processing_history = []
                
#                 st.success(f"Project '{project_to_delete}' deleted successfully!")
#                 st.rerun()
#         else:
#             st.info("No projects available to delete.")

# def render_main_content():
#     """Render the main content area"""
#     if st.session_state.df is not None:
#         # Create tabs for different data exploration tasks
#         tabs = st.tabs([
#             "📋 Data Overview", 
#             "🧹 Data Processing", 
#             "📊 Visualization", 
#             "📈 Analysis",
#             "📝 Reports",
#             "🎛️ Dashboard",
#             "🤖 Gen AI Assistant"
#         ])
        
#         # Data Overview Tab
#         with tabs[0]:
#             render_data_overview()
        
#         # Data Processing Tab
#         with tabs[1]:
#             render_data_processing()
        
#         # Visualization Tab
#         with tabs[2]:
#             render_visualization()
        
#         # Analysis Tab
#         with tabs[3]:
#             render_analysis()
        
#         # Reports Tab
#         with tabs[4]:
#             render_reports()
        
#         # Dashboard Tab
#         with tabs[5]:
#             render_dashboard()
            
#         # Gen AI Assistant Tab
#         with tabs[6]:
#             render_genai_assistant()
#     else:
#         # Show welcome message and data upload options
#         render_welcome()

# def render_welcome():
#     """Render welcome page when no data is loaded"""
#     st.title("Welcome to CSV AI Analytics 📊")
    
#     # Create columns for better layout
#     col1, col2 = st.columns([3, 2])
    
#     with col1:
#         st.markdown("""
#         ### Explore, Process, and Visualize Your Data
        
#         This application provides a comprehensive set of tools for data analysis:
        
#         🔍 **Data Overview**: Get a quick summary of your dataset
        
#         🧹 **Data Processing**: Clean, transform, and engineer your data
        
#         📊 **Visualization**: Create informative charts and plots
        
#         📈 **Analysis**: Perform statistical analysis and detect patterns
        
#         📝 **Reports**: Generate and export reports with your findings
        
#         🎛️ **Dashboard**: Build interactive dashboards for your data
        
#         ### Getting Started
        
#         Upload your data file using the sidebar, or try one of our sample datasets.
#         """)
    
#     with col2:
#         # Show a decorative image or icon
#         st.markdown("""
#         <div style="text-align: center; padding: 20px;">
#             <span style="font-size: 120px; color: #4CAF50;">📊</span>
#             <h3>Ready to analyze your data</h3>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Quick links
#         st.markdown("### Quick Links")
        
#         # Create buttons for quick actions
#         if st.button("📂 Upload Data", key="upload_data_welcome", use_container_width=True):
#             # This just highlights the uploader
#             st.info("Use the file uploader in the sidebar to upload your data.")
        
#         if st.button("🔢 Try Sample Dataset", key="try_sample_welcome", use_container_width=True):
#             # This just highlights the sample dataset selector
#             st.info("Select a sample dataset from the dropdown in the sidebar.")
        
#         if st.button("📝 Create New Project", key="create_project_welcome", use_container_width=True):
#             # This just highlights the project creation
#             st.info("Use the project management section in the sidebar to create a new project.")

# def render_genai_assistant():
#     """Render GenAI assistant interface"""
#     if st.session_state.df is not None:
#         # Create GenAI assistant instance
#         assistant = AIAssistant(st.session_state.df)
        
#         # Render the interface
#         assistant.render_interface()
#     else:
#         st.warning("Please upload a dataset first to use the GenAI Assistant.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

# Import custom modules
from modules.ui_helper import load_custom_css, create_header
from modules.data_manager import (
    initialize_session_state, 
    load_file, 
    load_sample_dataset, 
    save_to_project, 
    load_from_project,
    get_dataframe_info
)
from modules.theme_manager import ThemeManager
from modules.data_exporter import DataExporter
from modules.dashboard_generator import DashboardGenerator
from modules.file_handler import handle_uploaded_file, save_uploaded_file, load_sample_data
from modules.data_overview import render_data_overview
from modules.data_processor import render_data_processing
from modules.visualization import render_visualization
from modules.analysis import render_analysis
from modules.reports import render_reports
from modules.dashboard import render_dashboard, render_exploratory_report, render_trend_analysis, render_distribution_analysis
from modules.ai_assistant import AIAssistant
from modules.gen_ai_assistant import GenAIAssistant

# App configuration
APP_TITLE = "CSV AI Analytics"
APP_SUBTITLE = "Analyze and visualize your data with ease"
APP_VERSION = "1.0.0"

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Setup page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize theme manager
    theme_manager = ThemeManager()
    theme_manager.apply_theme()
    
    # Load custom CSS
    load_custom_css(theme=st.session_state.theme)
    
    # Create header
    create_header(version=APP_VERSION, title=APP_TITLE, subtitle=APP_SUBTITLE)
    
    # Create sidebar and main content
    render_sidebar()
    render_main_content()

def render_sidebar():
    """Render the sidebar with app controls"""
    with st.sidebar:
        st.title("📊 CSV AI Analytics")
        st.markdown("Explore, analyze, and visualize your data")
        
        st.markdown("---")
        
        # Data upload section
        st.header("Data Upload")
        
        uploaded_file = st.file_uploader("Upload your data file", type=["csv", "excel", "xlsx", "xls", "json", "txt"])
        
        if uploaded_file is not None:
            if st.button("Process File", key="process_file_sidebar", use_container_width=True):
                if load_file(uploaded_file):
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    st.rerun()
        
        st.markdown("---")
        
        # Sample data section
        st.header("Sample Datasets")
        sample_data = st.selectbox(
            "Load a sample dataset",
            ["Choose a dataset...", "Iris", "Titanic", "Boston Housing", "Wine Quality", "Diabetes"]
        )
        
        if sample_data != "Choose a dataset...":
            if st.button("Load Sample Dataset", key="load_sample_sidebar", use_container_width=True):
                if load_sample_dataset(sample_data):
                    st.success(f"Successfully loaded {sample_data} dataset")
                    st.rerun()
        
        st.markdown("---")
        
        # Project management section
        render_project_management()
        
        # Show current data info
        if st.session_state.df is not None:
            st.markdown("---")
            
            st.subheader("Current Data")
            
            # Display dataset info
            df_info = get_dataframe_info()
            
            st.caption(f"Rows: {df_info['rows']} | Columns: {df_info['columns']}")
            
            if st.session_state.file_details:
                st.caption(f"Source: {st.session_state.file_details['filename']}")
            
            if st.session_state.current_project:
                st.caption(f"Project: {st.session_state.current_project}")
            
            # Data preview
            if st.button("Show Data Preview", key="data_preview_sidebar", use_container_width=True):
                st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # Quick export
            exporter = DataExporter(st.session_state.df)
            st.markdown("#### Quick Export")
            exporter.quick_export_widget("Download CSV")
        
        # Theme settings
        st.markdown("---")
        
        with st.expander("Theme Settings", expanded=False):
            theme_manager = ThemeManager()
            theme_manager.render_theme_selector()
        
        # App info footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style="text-align: center;">
            <p style="font-size: 0.8em; color: gray;">
            {APP_TITLE} v{APP_VERSION}<br>
            © 2023
            </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

def render_project_management():
    """Render project management section in sidebar"""
    st.header("Projects")
    
    # Project actions
    project_action = st.selectbox(
        "Project Action",
        ["Select Project", "Create New Project", "Rename Project", "Delete Project"]
    )
    
    if project_action == "Select Project":
        # Get list of projects
        project_list = list(st.session_state.projects.keys())
        
        if project_list:
            # Add "None" option
            project_list = ["None"] + project_list
            
            # Select project
            selected_project = st.selectbox(
                "Select a Project",
                project_list,
                index=0 if st.session_state.current_project is None else project_list.index(st.session_state.current_project)
            )
            
            if st.button("Load Project", key="load_project_button", use_container_width=True):
                if selected_project == "None":
                    st.session_state.current_project = None
                    st.success("Project selection cleared.")
                else:
                    # Load project data
                    if load_from_project(selected_project):
                        st.success(f"Project '{selected_project}' loaded successfully!")
                        st.rerun()
        else:
            st.info("No projects available. Create a new project first.")
    
    elif project_action == "Create New Project":
        # Get project name
        project_name = st.text_input("New Project Name")
        
        if project_name and st.button("Create Project", key="create_project_button", use_container_width=True):
            if project_name in st.session_state.projects:
                st.error(f"Project '{project_name}' already exists!")
            else:
                # Create new project
                if save_to_project(project_name):
                    st.success(f"Project '{project_name}' created successfully!")
                    st.rerun()
    
    elif project_action == "Rename Project":
        # Get list of projects
        project_list = list(st.session_state.projects.keys())
        
        if project_list:
            # Select project to rename
            old_name = st.selectbox("Select Project to Rename", project_list)
            new_name = st.text_input("New Project Name")
            
            if old_name and new_name and st.button("Rename Project", key="rename_project_button", use_container_width=True):
                if new_name in st.session_state.projects:
                    st.error(f"Project '{new_name}' already exists!")
                else:
                    # Rename project
                    st.session_state.projects[new_name] = st.session_state.projects[old_name]
                    del st.session_state.projects[old_name]
                    
                    # Update current project if needed
                    if st.session_state.current_project == old_name:
                        st.session_state.current_project = new_name
                    
                    st.success(f"Project renamed from '{old_name}' to '{new_name}' successfully!")
                    st.rerun()
        else:
            st.info("No projects available to rename.")
    
    elif project_action == "Delete Project":
        # Get list of projects
        project_list = list(st.session_state.projects.keys())
        
        if project_list:
            # Select project to delete
            project_to_delete = st.selectbox("Select Project to Delete", project_list)
            
            # Confirmation
            confirm = st.checkbox("I understand this action cannot be undone.")
            
            if project_to_delete and confirm and st.button("Delete Project", key="delete_project_button", use_container_width=True):
                # Delete project
                del st.session_state.projects[project_to_delete]
                
                # Update current project if needed
                if st.session_state.current_project == project_to_delete:
                    st.session_state.current_project = None
                    
                    # Clear dataframes if they belonged to the deleted project
                    st.session_state.df = None
                    st.session_state.original_df = None
                    st.session_state.file_details = None
                    st.session_state.processing_history = []
                
                st.success(f"Project '{project_to_delete}' deleted successfully!")
                st.rerun()
        else:
            st.info("No projects available to delete.")

def render_main_content():
    """Render the main content area"""
    if st.session_state.df is not None:
        # Create tabs for different data exploration tasks
        tabs = st.tabs([
            "📋 Data Overview", 
            "🧹 Data Processing", 
            "📊 Visualization", 
            "📈 Analysis",
            "📝 Reports",
            "🎛️ Dashboard",
            "🤖 AI Assistant",
            "💡 GenAI Assistant"
        ])
        
        # Data Overview Tab
        with tabs[0]:
            render_data_overview()
        
        # Data Processing Tab
        with tabs[1]:
            render_data_processing()
        
        # Visualization Tab
        with tabs[2]:
            render_visualization()
        
        # Analysis Tab
        with tabs[3]:
            render_analysis()
        
        # Reports Tab
        with tabs[4]:
            render_reports()
        
        # Dashboard Tab
        with tabs[5]:
            render_dashboard()
            
        # AI Assistant Tab (rule-based)
        with tabs[6]:
            render_ai_assistant()
            
        # GenAI Assistant Tab (OpenAI-powered)
        with tabs[7]:
            render_genai_assistant()
    else:
        # Show welcome message and data upload options
        render_welcome()

def render_welcome():
    """Render welcome page when no data is loaded"""
    st.title("Welcome to CSV AI Analytics 📊")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Explore, Process, and Visualize Your Data
        
        This application provides a comprehensive set of tools for data analysis:
        
        🔍 **Data Overview**: Get a quick summary of your dataset
        
        🧹 **Data Processing**: Clean, transform, and engineer your data
        
        📊 **Visualization**: Create informative charts and plots
        
        📈 **Analysis**: Perform statistical analysis and detect patterns
        
        📝 **Reports**: Generate and export reports with your findings
        
        🎛️ **Dashboard**: Build interactive dashboards for your data
        
        🤖 **AI Assistant**: Get data insights through a rule-based AI assistant
        
        💡 **GenAI Assistant**: Leverage the power of OpenAI for advanced data analysis
        """)
    
    with col2:
        # Show a decorative image or icon
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span style="font-size: 120px; color: #4CAF50;">📊</span>
            <h3>Ready to analyze your data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick links
        st.markdown("### Quick Links")
        
        # Create buttons for quick actions
        if st.button("📂 Upload Data", key="upload_data_welcome", use_container_width=True):
            # This just highlights the uploader
            st.info("Use the file uploader in the sidebar to upload your data.")
        
        if st.button("🔢 Try Sample Dataset", key="try_sample_welcome", use_container_width=True):
            # This just highlights the sample dataset selector
            st.info("Select a sample dataset from the dropdown in the sidebar.")
        
        if st.button("📝 Create New Project", key="create_project_welcome", use_container_width=True):
            # This just highlights the project creation
            st.info("Use the project management section in the sidebar to create a new project.")

def render_ai_assistant():
    """Render AI assistant interface (rule-based)"""
    if st.session_state.df is not None:
        # Create AI assistant instance
        assistant = AIAssistant(st.session_state.df)
        
        # Render the interface
        assistant.render_interface()
    else:
        st.warning("Please upload a dataset first to use the AI Assistant.")

def render_genai_assistant():
    """Render GenAI assistant interface (OpenAI-powered)"""
    if st.session_state.df is not None:
        # Create GenAI assistant instance
        assistant = GenAIAssistant(st.session_state.df)
        
        # Render the interface
        assistant.render_interface()
    else:
        st.warning("Please upload a dataset first to use the GenAI Assistant.")

if __name__ == "__main__":
    main()