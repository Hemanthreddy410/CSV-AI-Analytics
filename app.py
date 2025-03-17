import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
import gc

# Import optimized modules
from modules.data_workflow import DataWorkflowManager
from modules.memorymanager import MemoryManager
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
# from modules.gen_ai_assistant import GenAIAssistant

# Import other modules
from modules.ui_helper import load_custom_css, create_header


# App configuration
APP_TITLE = "CSV AI Analytics"
APP_SUBTITLE = "Analyze and visualize your data with ease"
APP_VERSION = "1.1.0 (Optimized for Large Datasets)"

def main():
    """Main application entry point"""
    # Start timing for performance monitoring
    start_time = time.time()
    
    # Initialize session state
    initialize_session_state()
    
    # Setup page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
    # Load custom CSS
    load_custom_css(theme=st.session_state.theme)
    
    # Create header
    create_header(version=APP_VERSION, title=APP_TITLE, subtitle=APP_SUBTITLE)
    
    # Create sidebar and main content
    render_sidebar()
    render_main_content()
    
    # Display memory stats in the sidebar (if enabled)
    if st.session_state.get('show_memory_stats', False):
        MemoryManager.display_memory_stats()
    
    # Performance monitoring
    if st.session_state.get('show_performance_stats', False):
        total_time = time.time() - start_time
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Performance Stats")
        st.sidebar.info(f"Page render time: {total_time:.2f} seconds")

def initialize_session_state():
    """Initialize session state with default values"""
    # Core data
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    
    # File details
    if 'file_details' not in st.session_state:
        st.session_state.file_details = None
    
    # Project management
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    
    # Processing history
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    # Performance settings
    if 'show_memory_stats' not in st.session_state:
        st.session_state.show_memory_stats = True
    
    if 'show_performance_stats' not in st.session_state:
        st.session_state.show_performance_stats = True
    
    # Data loading settings
    if 'use_data_sampling' not in st.session_state:
        st.session_state.use_data_sampling = True
    
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = 50000
    
    # Theme setting
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"

def render_sidebar():
    """Render the sidebar with app controls"""
    with st.sidebar:
        st.title("📊 CSV AI Analytics")
        st.markdown("Explore, analyze, and visualize your data")
        
        st.markdown("---")
        
        # Performance settings
        with st.expander("Performance Settings", expanded=False):
            st.checkbox("Show memory statistics", value=st.session_state.show_memory_stats, 
                       key="show_memory_stats_checkbox", 
                       on_change=lambda: setattr(st.session_state, 'show_memory_stats', st.session_state.show_memory_stats_checkbox))
            
            st.checkbox("Show performance statistics", value=st.session_state.show_performance_stats, 
                       key="show_performance_stats_checkbox",
                       on_change=lambda: setattr(st.session_state, 'show_performance_stats', st.session_state.show_performance_stats_checkbox))
            
            st.checkbox("Use data sampling for large files", value=st.session_state.use_data_sampling,
                       key="use_data_sampling_checkbox",
                       on_change=lambda: setattr(st.session_state, 'use_data_sampling', st.session_state.use_data_sampling_checkbox))
            
            if st.session_state.use_data_sampling:
                st.slider("Sample size (rows):", 10000, 100000, st.session_state.sample_size, 10000, 
                         key="sample_size_slider",
                         on_change=lambda: setattr(st.session_state, 'sample_size', st.session_state.sample_size_slider))
        
        # Data upload section
        st.header("Data Upload")
        
        # Show current memory usage before upload to help user gauge capacity
        if st.session_state.show_memory_stats:
            mem_usage = MemoryManager.get_memory_usage()
            st.caption(f"Current memory usage: {mem_usage:.1f} MB")
            
            # Check available memory
            sufficient_mem, available_mem = MemoryManager.check_available_memory()
            if not sufficient_mem:
                st.warning(f"⚠️ Low available memory ({available_mem:.1f} MB). Large files may cause performance issues.")
        
        # File uploader with options
        uploaded_file = st.file_uploader("Upload your data file", type=["csv", "excel", "xlsx", "xls", "json", "txt"])
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"File size: {file_size_mb:.2f} MB")
            
            # Show preview & import options button
            if st.button("Preview & Configure Import", key="preview_file", use_container_width=True):
                # Create a file details expander
                with st.expander("File Details", expanded=True):
                    st.info(f"Filename: {uploaded_file.name}")
                    st.info(f"Size: {file_size_mb:.2f} MB")
                    
                    # For large files, show a warning/info
                    if file_size_mb > 100:
                        st.warning("This is a large file. Consider using data sampling for better performance.")
                        
                        # Allow sampling configuration
                        use_sampling = st.checkbox("Use data sampling", value=st.session_state.use_data_sampling)
                        if use_sampling:
                            sample_size = st.slider("Sample size (rows):", 10000, 100000, st.session_state.sample_size, 10000)
                        else:
                            sample_size = None
                    else:
                        sample_size = None
                    
                    # Show a preview and import options
                    sample = None
                    try:
                        # Try to read the first few rows for preview
                        if uploaded_file.type == "text/csv":
                            sample = pd.read_csv(uploaded_file, nrows=5)
                        elif "excel" in uploaded_file.type:
                            sample = pd.read_excel(uploaded_file, nrows=5)
                        
                        if sample is not None:
                            st.subheader("Data Preview")
                            st.dataframe(sample, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error previewing file: {str(e)}")
            
            # Process file button
            if st.button("Process File", key="process_file_sidebar", use_container_width=True):
                with st.spinner("Processing file..."):
                    # Use the optimized file handler with sampling if enabled
                    if st.session_state.use_data_sampling and uploaded_file.size > 100 * 1024 * 1024:  # > 100MB
                        sample_size = st.session_state.sample_size
                    else:
                        sample_size = None
                    
                    # Process the file
                    df, file_details = handle_uploaded_file(uploaded_file, sample_size=sample_size)
                    
                    if df is not None:
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.session_state.file_details = file_details
                        
                        # Optimize memory usage
                        optimized_df, savings_pct = MemoryManager.optimize_dataframe(df)
                        
                        if savings_pct > 5:  # Only replace if significant savings
                            st.session_state.df = optimized_df
                            st.session_state.original_df = optimized_df.copy()
                            st.success(f"Successfully loaded {uploaded_file.name} with {savings_pct:.1f}% memory optimization")
                        else:
                            st.success(f"Successfully loaded {uploaded_file.name}")
                        
                        # Force garbage collection after loading
                        gc.collect()
                        
                        st.rerun()
                    else:
                        st.error("Failed to process file. Please check the file format and try again.")
        
        st.markdown("---")
        
        # Sample data section
        st.header("Sample Datasets")
        
        # Show a list of sample datasets with descriptions
        sample_options = [
            {"name": "Iris", "description": "Iris flower dataset (150 rows, 5 columns)"},
            {"name": "Titanic", "description": "Titanic passengers (891 rows, 12 columns)"},
            {"name": "Boston Housing", "description": "Boston housing data (506 rows, 14 columns)"},
            {"name": "Wine Quality", "description": "Wine quality analysis (1599 rows, 12 columns)"},
            {"name": "Diabetes", "description": "Diabetes progression (442 rows, 11 columns)"}
        ]
        
        sample_name = st.selectbox(
            "Select a sample dataset:",
            ["Choose a dataset..."] + [sample["name"] for sample in sample_options],
            format_func=lambda x: next((sample["description"] for sample in sample_options if sample["name"] == x), x)
        )
        
        if sample_name != "Choose a dataset...":
            if st.button("Load Sample Dataset", key="load_sample_sidebar", use_container_width=True):
                with st.spinner(f"Loading {sample_name} dataset..."):
                    df, file_details = load_sample_data(sample_name)
                    
                    if df is not None:
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.session_state.file_details = file_details
                        st.success(f"Successfully loaded {sample_name} dataset")
                        st.rerun()
                    else:
                        st.error(f"Failed to load {sample_name} dataset")
        
        st.markdown("---")
        
        # Project management section
        render_project_management()
        
        # Show current data info
        if st.session_state.df is not None:
            st.markdown("---")
            
            st.subheader("Current Data")
            
            # Display dataset info
            df_info = get_dataframe_info()
            
            # Indicate if data is sampled
            if df_info.get('is_sampled', False):
                st.caption(f"Rows: {df_info['rows']:,} (sampled) | Columns: {df_info['columns']}")
                st.caption(f"Original size: {df_info['original_rows']:,} rows")
            else:
                st.caption(f"Rows: {df_info['rows']:,} | Columns: {df_info['columns']}")
            
            if st.session_state.file_details:
                st.caption(f"Source: {st.session_state.file_details['filename']}")
            
            if st.session_state.current_project:
                st.caption(f"Project: {st.session_state.current_project}")
            
            # Data preview
            if st.button("Show Data Preview", key="data_preview_sidebar", use_container_width=True):
                st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # Memory usage
            if st.session_state.show_memory_stats:
                try:
                    # Calculate memory usage
                    memory_usage = st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                    st.caption(f"Memory usage: {memory_usage:.2f} MB")
                except:
                    pass
        
        # App info footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style="text-align: center;">
            <p style="font-size: 0.8em; color: gray;">
            {APP_TITLE} v{APP_VERSION}<br>
            © 2023-2025
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
            "💡 Gen AI"
        ])
        
        # Data Overview Tab
        with tabs[0]:
            render_data_overview()
        
        # Data Processing Tab
        with tabs[1]:
            render_data_processing()
        
        # Visualization Tab - Using the imported function
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

def render_data_overview():
    """Render data overview tab with optimizations for large datasets"""
    st.header("Data Overview")
    
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    # Get basic information about the dataset
    rows, cols = st.session_state.df.shape
    
    # Create a summary container
    summary_container = st.container()
    
    with summary_container:
        # Display summary metrics in a clean layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{rows:,}")
        
        with col2:
            st.metric("Columns", cols)
        
        with col3:
            # Count data types
            dtypes = st.session_state.df.dtypes.value_counts()
            num_numeric = sum(dtype.kind in 'bifc' for dtype in st.session_state.df.dtypes)
            st.metric("Numeric Columns", num_numeric)
        
        with col4:
            # Calculate missing values percentage
            missing_cells = st.session_state.df.isna().sum().sum()
            total_cells = rows * cols
            missing_percent = (missing_cells / total_cells * 100) if total_cells > 0 else 0
            st.metric("Missing Values", f"{missing_percent:.1f}%")
    
    # Create tabs for different overview sections
    overview_tabs = st.tabs([
        "📊 Data Preview", 
        "📋 Data Types", 
        "🔍 Missing Values",
        "📈 Summary Statistics"
    ])
    
    # Data Preview Tab
    with overview_tabs[0]:
        # For large datasets, limit preview rows and add sampling
        is_large = rows > 10000
        
        if is_large:
            st.info(f"Dataset has {rows:,} rows. Showing a sample for better performance.")
            
            preview_options = st.radio(
                "Preview options:",
                ["Head (first rows)", "Tail (last rows)", "Random Sample"],
                horizontal=True
            )
            
            sample_size = st.slider("Sample size:", 5, 100, 10)
            
            if preview_options == "Head (first rows)":
                st.dataframe(st.session_state.df.head(sample_size), use_container_width=True)
            elif preview_options == "Tail (last rows)":
                st.dataframe(st.session_state.df.tail(sample_size), use_container_width=True)
            else:  # Random Sample
                st.dataframe(st.session_state.df.sample(sample_size), use_container_width=True)
        else:
            # For smaller datasets, show more rows
            st.dataframe(st.session_state.df.head(20), use_container_width=True)
    
    # Data Types Tab
    with overview_tabs[1]:
        # Create a cleaned up datatype display
        dtypes_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': [str(st.session_state.df[col].dtype) for col in st.session_state.df.columns],
            'Sample Values': [str(st.session_state.df[col].dropna().head(3).tolist()) for col in st.session_state.df.columns]
        })
        
        st.dataframe(dtypes_df, use_container_width=True)
    
    # Missing Values Tab
    with overview_tabs[2]:
        # Create a missing values analysis
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Values': st.session_state.df.isna().sum().values,
            'Percentage': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
        })
        
        # Sort by missing values (descending)
        missing_df = missing_df.sort_values('Missing Values', ascending=False)
        
        # Only show columns with missing values
        if missing_df['Missing Values'].sum() > 0:
            # Check if plotly is available and imported before creating visualization
            if 'px' in globals():
                # Create a visualization for missing values
                fig = px.bar(
                    missing_df[missing_df['Missing Values'] > 0], 
                    x='Column', 
                    y='Percentage',
                    title='Missing Values by Column (%)',
                    color='Percentage',
                    color_continuous_scale='Reds',
                    text_auto='.1f'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Visualization library (plotly) not imported. Showing tabular data only.")
            
            # Show the table
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Summary Statistics Tab
    with overview_tabs[3]:
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if num_cols:
            # For large datasets, calculate statistics more efficiently
            if is_large:
                with st.spinner("Calculating statistics for large dataset..."):
                    # Use optimized describe to handle large datasets better
                    stats_df = st.session_state.df[num_cols].describe().T
                    
                    # Add additional stats
                    stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
                    stats_df['missing_pct'] = (stats_df['missing'] / len(st.session_state.df) * 100).round(2)
            else:
                # For smaller datasets, include more statistics
                stats_df = st.session_state.df[num_cols].describe().T
                
                # Add additional stats
                stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
                stats_df['missing_pct'] = (stats_df['missing'] / len(st.session_state.df) * 100).round(2)
                
                try:
                    # Add skewness and kurtosis
                    stats_df['skew'] = st.session_state.df[num_cols].skew()
                    stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis()
                except:
                    pass
            
            # Display statistics
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numeric columns found for statistics.")

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
    try:
        # Import the GenAIAssistant class - making this import optional
        from modules.gen_ai_assistant import GenAIAssistant
        
        if st.session_state.df is not None:
            # Create GenAI assistant instance
            assistant = GenAIAssistant(st.session_state.df)
            
            # Render the interface
            assistant.render_interface()
        else:
            st.warning("Please upload a dataset first to use the GenAI Assistant.")
    except ImportError:
        st.warning("GenAI Assistant is not available. Please check if the required dependencies are installed.")
        st.info("You may need to uncomment the import statement and install OpenAI dependencies.")

def render_welcome():
    """Render welcome page when no data is loaded"""
    st.title("Welcome to CSV AI Analytics 📊")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
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
        
        💡 **GenAI Assistant**: Leverage the power of AI for advanced data analysis
        ### 🚀 Optimized for Large Datasets
        
        """)
        
        st.markdown("""        
        This application now handles large datasets efficiently with:
        
        ✅ **Smart Data Loading**: Chunks and samples large files for better memory usage
        
        ✅ **Performance Optimizations**: Monitors memory usage and improves processing speed
        
        ✅ **Progressive Visualization**: Renders large datasets without performance impact
        
        ✅ **Memory Management**: Automatically optimizes data types to reduce memory usage
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

def get_dataframe_info():
    """Get information about the current dataframe"""
    df_info = {
        'rows': len(st.session_state.df),
        'columns': len(st.session_state.df.columns),
        'is_sampled': False,
        'original_rows': len(st.session_state.df)
    }
    
    # Check if this is a sampled dataset
    if 'file_details' in st.session_state and st.session_state.file_details:
        if 'is_sample' in st.session_state.file_details and st.session_state.file_details['is_sample']:
            df_info['is_sampled'] = True
            
            # Get original size if available
            if 'rows_in_original' in st.session_state.file_details:
                df_info['original_rows'] = st.session_state.file_details['rows_in_original']
    
    return df_info

def load_file(uploaded_file):
    """Process and load a file into the app (optimized version)"""
    if uploaded_file is None:
        return False
    
    try:
        # Use the optimized file handler
        sample_size = st.session_state.sample_size if st.session_state.use_data_sampling else None
        df, file_details = handle_uploaded_file(uploaded_file, sample_size=sample_size)
        
        if df is not None:
            # Store in session state
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.file_details = file_details
            
            # Optimize memory usage
            optimized_df, savings_pct = MemoryManager.optimize_dataframe(df)
            
            if savings_pct > 5:  # Only replace if significant savings
                st.session_state.df = optimized_df
                st.session_state.original_df = optimized_df.copy()
            
            # Force garbage collection
            gc.collect()
            
            return True
        else:
            return False
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return False

def load_sample_dataset(sample_name):
    """Load a sample dataset into the app (optimized version)"""
    if not sample_name or sample_name == "Choose a dataset...":
        return False
    
    try:
        # Use the optimized sample data loader
        df, file_details = load_sample_data(sample_name)
        
        if df is not None:
            # Store in session state
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.file_details = file_details
            
            # No need to optimize further as sample datasets are already small
            
            return True
        else:
            return False
    
    except Exception as e:
        st.error(f"Error loading sample dataset: {str(e)}")
        return False

def save_to_project(project_name):
    """Save current data to a project"""
    if not project_name:
        return False
    
    if not st.session_state.df is not None:
        st.warning("No data available to save to project")
        return False
    
    try:
        # Create project data
        project_data = {
            'df': st.session_state.df.copy(),
            'original_df': st.session_state.original_df.copy() if st.session_state.original_df is not None else None,
            'file_details': st.session_state.file_details,
            'processing_history': st.session_state.processing_history.copy(),
            'created_at': datetime.datetime.now(),
            'last_modified': datetime.datetime.now()
        }
        
        # Store in projects dictionary
        st.session_state.projects[project_name] = project_data
        
        # Set as current project
        st.session_state.current_project = project_name
        
        return True
    
    except Exception as e:
        st.error(f"Error saving to project: {str(e)}")
        return False

def load_from_project(project_name):
    """Load data from a project"""
    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found")
        return False
    
    try:
        # Get project data
        project_data = st.session_state.projects[project_name]
        
        # Load project data into session state
        st.session_state.df = project_data['df'].copy()
        st.session_state.original_df = project_data['original_df'].copy() if project_data['original_df'] is not None else None
        st.session_state.file_details = project_data['file_details']
        st.session_state.processing_history = project_data['processing_history'].copy()
        
        # Set as current project
        st.session_state.current_project = project_name
        
        # Update last accessed timestamp
        st.session_state.projects[project_name]['last_accessed'] = datetime.datetime.now()
        
        return True
    
    except Exception as e:
        st.error(f"Error loading from project: {str(e)}")
        return False

if __name__ == "__main__":
    main()