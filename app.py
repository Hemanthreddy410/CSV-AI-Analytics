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
            "🎛️ Dashboard"
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
        
        ### Getting Started
        
        Upload your data file using the sidebar, or try one of our sample datasets.
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

def render_data_overview():
    """Render data overview section"""
    st.header("Data Overview")
    
    # Display file information
    if st.session_state.file_details is not None:
        # Create an info box
        st.markdown('<div class="data-info">', unsafe_allow_html=True)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Filename:** {st.session_state.file_details['filename']}")
            st.markdown(f"**Type:** {st.session_state.file_details['type']}")
            
            # Format size nicely
            size = st.session_state.file_details['size']
            if size is not None:
                if size < 1024:
                    size_str = f"{size} bytes"
                elif size < 1024 ** 2:
                    size_str = f"{size / 1024:.2f} KB"
                elif size < 1024 ** 3:
                    size_str = f"{size / (1024 ** 2):.2f} MB"
                else:
                    size_str = f"{size / (1024 ** 3):.2f} GB"
                
                st.markdown(f"**Size:** {size_str}")
            
            # Show additional info if available
            if 'info' in st.session_state.file_details:
                st.markdown(f"**Info:** {st.session_state.file_details['info']}")
        
        with col2:
            st.markdown(f"**Rows:** {st.session_state.df.shape[0]:,}")
            st.markdown(f"**Columns:** {st.session_state.df.shape[1]}")
            
            # Format datetime properly
            last_modified = st.session_state.file_details['last_modified']
            if isinstance(last_modified, datetime.datetime):
                formatted_date = last_modified.strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"**Last Modified:** {formatted_date}")
            
            # Show current project if available
            if st.session_state.current_project is not None:
                st.markdown(f"**Project:** {st.session_state.current_project}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different overview sections
    overview_tabs = st.tabs([
        "Data Preview", 
        "Column Info", 
        "Summary Statistics", 
        "Missing Values", 
        "Duplicates"
    ])
    
    # Data Preview Tab
    with overview_tabs[0]:
        st.subheader("Data Preview")
        
        # Create columns for controls
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        
        with control_col1:
            # Number of rows to display
            n_rows = st.slider("Number of rows:", 5, 100, 10, key="preview_rows")
        
        with control_col2:
            # View options
            view_option = st.radio("View:", ["Head", "Tail", "Sample"], horizontal=True)
        
        with control_col3:
            # Column filter
            if st.checkbox("Select columns", value=False, key="select_columns_preview"):
                selected_cols = st.multiselect(
                    "Columns to display:",
                    st.session_state.df.columns.tolist(),
                    default=st.session_state.df.columns.tolist()[:5],
                    key="selected_cols_preview"
                )
            else:
                selected_cols = st.session_state.df.columns.tolist()
        
        # Display the data
        if selected_cols:
            if view_option == "Head":
                st.dataframe(st.session_state.df[selected_cols].head(n_rows), use_container_width=True)
            elif view_option == "Tail":
                st.dataframe(st.session_state.df[selected_cols].tail(n_rows), use_container_width=True)
            else:  # Sample
                st.dataframe(st.session_state.df[selected_cols].sample(min(n_rows, len(st.session_state.df))), use_container_width=True)
        else:
            st.info("Please select at least one column to display.")
    
    # Column Info Tab
    with overview_tabs[1]:
        st.subheader("Column Information")
        
        # Create a dataframe with column information - FIXED to ensure all arrays have the same length
        df = st.session_state.df
        columns_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%',
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Memory Usage': [df[col].memory_usage(deep=True) for col in df.columns]
        })
        
        # Format memory usage
        columns_df['Memory Usage'] = columns_df['Memory Usage'].apply(
            lambda x: f"{x} bytes" if x < 1024 else (
                f"{x/1024:.2f} KB" if x < 1024**2 else f"{x/(1024**2):.2f} MB"
            )
        )
        
        # Display the dataframe
        st.dataframe(columns_df, use_container_width=True)
        
        # Download button for column info
        csv = columns_df.to_csv(index=False)
        st.download_button(
            label="Download Column Info",
            data=csv,
            file_name="column_info.csv",
            mime="text/csv",
            key="download_column_info",
            use_container_width=True
        )
        
        # Display column type distribution
        st.subheader("Column Type Distribution")
        
        # Create a dataframe with type counts
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display type counts
            st.dataframe(type_counts, use_container_width=True)
        
        with col2:
            # Create a pie chart of column types
            fig = px.pie(
                type_counts, 
                values='Count', 
                names='Data Type', 
                title='Column Type Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics Tab
    with overview_tabs[2]:
        st.subheader("Summary Statistics")
        
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if num_cols:
            # Calculate statistics
            stats_df = st.session_state.df[num_cols].describe().T
            
            # Round to 3 decimal places
            stats_df = stats_df.round(3)
            
            # Add more statistics
            stats_df['median'] = st.session_state.df[num_cols].median()
            stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
            stats_df['missing_pct'] = (st.session_state.df[num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
            
            try:
                stats_df['skew'] = st.session_state.df[num_cols].skew().round(3)
                stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis().round(3)
            except:
                pass
            
            # Reorder columns
            stats_cols = ['count', 'missing', 'missing_pct', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
            extra_cols = ['skew', 'kurtosis']
            
            # Only include available columns
            avail_cols = [col for col in stats_cols + extra_cols if col in stats_df.columns]
            
            stats_df = stats_df[avail_cols]
            
            # Display the dataframe
            st.dataframe(stats_df, use_container_width=True)
            
            # Download button for statistics
            csv = stats_df.to_csv()
            st.download_button(
                label="Download Statistics",
                data=csv,
                file_name="summary_statistics.csv",
                mime="text/csv",
                key="download_statistics",
                use_container_width=True
            )
            
            # Distribution visualization
            st.subheader("Distributions")
            
            # Select column for distribution
            dist_col = st.selectbox("Select column for distribution:", num_cols, key="dist_col_select")
            
            # Create histogram with KDE
            fig = px.histogram(
                st.session_state.df, 
                x=dist_col,
                marginal="box",
                title=f"Distribution of {dist_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for summary statistics.")
    
    # Missing Values Tab
    with overview_tabs[3]:
        st.subheader("Missing Values Analysis")
        
        # Calculate missing values - FIXED to use .values
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isna().sum().values,
            'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
        })
        
        # Sort by missing count (descending)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        # Display the dataframe
        st.dataframe(missing_df, use_container_width=True)
        
        # Create visualization
        fig = px.bar(
            missing_df, 
            x='Column', 
            y='Missing %',
            title='Missing Values by Column (%)',
            color='Missing %',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values heatmap
        st.subheader("Missing Values Heatmap")
        st.write("This visualization shows patterns of missing data across the dataset.")
        
        # Limit to columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        
        if cols_with_missing:
            # Create sample for heatmap (limit to 100 rows for performance)
            sample_size = min(100, len(st.session_state.df))
            sample_df = st.session_state.df[cols_with_missing].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[cols_with_missing]
            
            # Create heatmap
            fig = px.imshow(
                sample_df.isna(),
                labels=dict(x="Column", y="Row", color="Missing"),
                color_continuous_scale=["blue", "red"],
                title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Duplicates Tab
    with overview_tabs[4]:
        st.subheader("Duplicate Rows Analysis")
        
        # Calculate duplicates
        duplicates = st.session_state.df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_pct = (duplicate_count / len(st.session_state.df) * 100).round(2)
        
        # Display duplicates summary
        st.write(f"**Total duplicate rows:** {duplicate_count} ({duplicate_pct}% of all rows)")
        
        # Show duplicates if any
        if duplicate_count > 0:
            # Get the duplicate rows
            duplicate_rows = st.session_state.df[duplicates]
            
            # Show options
            show_option = st.radio(
                "Display options:",
                ["Show sample of duplicates", "Show all duplicates", "Show duplicate values counts"],
                key="dup_display_options",
                horizontal=True
            )
            
            if show_option == "Show sample of duplicates":
                # Show sample of duplicates
                sample_size = min(10, len(duplicate_rows))
                st.dataframe(duplicate_rows.head(sample_size), use_container_width=True)
                
            elif show_option == "Show all duplicates":
                # Show all duplicates
                st.dataframe(duplicate_rows, use_container_width=True)
                
            else:  # Show counts
                # Count occurrences of each duplicate combination
                dup_counts = st.session_state.df.groupby(list(st.session_state.df.columns)).size().reset_index(name='count')
                dup_counts = dup_counts[dup_counts['count'] > 1].sort_values('count', ascending=False)
                
                st.dataframe(dup_counts, use_container_width=True)
            
            # Download button for duplicates
            csv = duplicate_rows.to_csv(index=False)
            st.download_button(
                label="Download Duplicate Rows",
                data=csv,
                file_name="duplicate_rows.csv",
                mime="text/csv",
                key="download_duplicates",
                use_container_width=True
            )
            
            # Option to remove duplicates - with unique key
            if st.button("Remove Duplicate Rows", key="remove_duplicates_overview", use_container_width=True):
                # Remove duplicates
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Removed {duplicate_count} duplicate rows",
                    "timestamp": datetime.datetime.now(),
                    "type": "remove_duplicates",
                    "details": {
                        "rows_removed": int(duplicate_count)
                    }
                })
                
                # Success message
                st.success(f"Removed {duplicate_count} duplicate rows!")
                st.rerun()
        else:
            st.success("No duplicate rows found in the dataset!")

def render_data_processing():
    """Render data processing section"""
    st.header("Data Processing")
    
    # Create tabs for different processing tasks
    processing_tabs = st.tabs([
        "Data Cleaning",
        "Data Transformation",
        "Feature Engineering",
        "Data Filtering",
        "Column Management"
    ])
    
    # Data Cleaning Tab
    with processing_tabs[0]:
        render_data_cleaning()
    
    # Data Transformation Tab
    with processing_tabs[1]:
        render_data_transformation()
    
    # Feature Engineering Tab
    with processing_tabs[2]:
        render_feature_engineering()
    
    # Data Filtering Tab
    with processing_tabs[3]:
        render_data_filtering()
    
    # Column Management Tab
    with processing_tabs[4]:
        render_column_management()
    
    # Processing History and Export Options
    if hasattr(st.session_state, 'processing_history') and st.session_state.processing_history:
        st.header("Processing History")

        # Create collapsible section for history
        with st.expander("View Processing Steps", expanded=False):
            for i, step in enumerate(st.session_state.processing_history):
                st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Reset and Export buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Reset to Original Data", key="reset_data_proc", use_container_width=True):
                if hasattr(st.session_state, 'original_df') and st.session_state.original_df is not None:
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state.processing_history = []
                    st.success("Data reset to original state!")
                    st.rerun()
        with col2:
            if st.button("Save Processed Data", key="save_data_proc", use_container_width=True):
                if st.session_state.current_project:
                    if save_to_project(st.session_state.current_project):
                        st.success("✅ Processed data saved to project successfully!")
                else:
                    # Show export options
                    exporter = DataExporter(st.session_state.df)
                    exporter.render_export_options()
        with col3:
            # Quick export as CSV
            if st.session_state.df is not None and len(st.session_state.df) > 0:
                exporter = DataExporter(st.session_state.df)
                exporter.quick_export_widget("Quick Download CSV")

def render_data_cleaning():
    """Render data cleaning interface"""
    st.subheader("Data Cleaning")
    
    # Create columns for organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Missing Values")
        
        # Show missing value statistics
        missing_vals = st.session_state.df.isna().sum()
        total_missing = missing_vals.sum()
        
        if total_missing == 0:
            st.success("No missing values found in the dataset!")
        else:
            st.warning(f"Found {total_missing} missing values across {(missing_vals > 0).sum()} columns")
            
            # Display columns with missing values
            cols_with_missing = st.session_state.df.columns[missing_vals > 0].tolist()
            
            # Create a dataframe to show missing value statistics
            missing_df = pd.DataFrame({
                'Column': cols_with_missing,
                'Missing Values': [missing_vals[col] for col in cols_with_missing],
                'Percentage': [missing_vals[col] / len(st.session_state.df) * 100 for col in cols_with_missing]
            })
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Options for handling missing values
            st.markdown("#### Handle Missing Values")
            
            # Add option to handle all columns at once
            handle_all = st.checkbox("Apply to all columns with missing values", value=False, key="handle_all_missing")
            
            if handle_all:
                col_to_handle = "All columns with missing values"
            else:
                col_to_handle = st.selectbox(
                    "Select column to handle:",
                    cols_with_missing,
                    key="col_to_handle_missing"
                )
            
            handling_method = st.selectbox(
                "Select handling method:",
                [
                    "Drop rows",
                    "Fill with mean",
                    "Fill with median",
                    "Fill with mode",
                    "Fill with constant value",
                    "Fill with forward fill",
                    "Fill with backward fill",
                    "Fill with interpolation"
                ],
                key="handling_method_missing"
            )
            
            # Additional input for constant value if selected
            if handling_method == "Fill with constant value":
                constant_value = st.text_input("Enter constant value:", key="constant_value_missing")
            
            # Apply button
            if st.button("Apply Missing Value Treatment", key="apply_missing_cleaning", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Determine columns to process
                    columns_to_process = cols_with_missing if handle_all else [col_to_handle]
                    
                    # Apply the selected method
                    if handling_method == "Drop rows":
                        st.session_state.df = st.session_state.df.dropna(subset=columns_to_process)
                        
                        # Add to processing history
                        rows_removed = orig_shape[0] - st.session_state.df.shape[0]
                        st.session_state.processing_history.append({
                            "description": f"Dropped {rows_removed} rows with missing values in {len(columns_to_process)} column(s)",
                            "timestamp": datetime.datetime.now(),
                            "type": "missing_values",
                            "details": {
                                "columns": columns_to_process,
                                "method": "drop",
                                "rows_affected": rows_removed
                            }
                        })
                        
                        st.success(f"Dropped {rows_removed} rows with missing values")
                        
                    elif handling_method == "Fill with mean":
                        for col in columns_to_process:
                            if st.session_state.df[col].dtype.kind in 'bifc':  # Check if numeric
                                mean_val = st.session_state.df[col].mean()
                                st.session_state.df[col] = st.session_state.df[col].fillna(mean_val)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col}' with mean ({mean_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col,
                                        "method": "mean",
                                        "value": mean_val
                                    }
                                })
                            else:
                                st.warning(f"Column '{col}' is not numeric. Skipping mean imputation.")
                        
                        st.success(f"Filled missing values with mean in {len(columns_to_process)} column(s)")
                            
                    elif handling_method == "Fill with median":
                        for col in columns_to_process:
                            if st.session_state.df[col].dtype.kind in 'bifc':  # Check if numeric
                                median_val = st.session_state.df[col].median()
                                st.session_state.df[col] = st.session_state.df[col].fillna(median_val)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col}' with median ({median_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col,
                                        "method": "median",
                                        "value": median_val
                                    }
                                })
                            else:
                                st.warning(f"Column '{col}' is not numeric. Skipping median imputation.")
                        
                        st.success(f"Filled missing values with median in {len(columns_to_process)} column(s)")
                            
                    elif handling_method == "Fill with mode":
                        for col in columns_to_process:
                            mode_val = st.session_state.df[col].mode()[0]
                            st.session_state.df[col] = st.session_state.df[col].fillna(mode_val)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col}' with mode ({mode_val})",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col,
                                    "method": "mode",
                                    "value": mode_val
                                }
                            })
                        
                        st.success(f"Filled missing values with mode in {len(columns_to_process)} column(s)")
                        
                    elif handling_method == "Fill with constant value":
                        if constant_value:
                            for col in columns_to_process:
                                # Try to convert to appropriate type
                                try:
                                    if st.session_state.df[col].dtype.kind in 'bifc':  # Numeric
                                        constant_val = float(constant_value)
                                    elif st.session_state.df[col].dtype.kind == 'b':  # Boolean
                                        constant_val = constant_value.lower() in ['true', 'yes', '1', 't', 'y']
                                    else:
                                        constant_val = constant_value
                                        
                                    st.session_state.df[col] = st.session_state.df[col].fillna(constant_val)
                                    
                                    # Add to processing history
                                    st.session_state.processing_history.append({
                                        "description": f"Filled missing values in '{col}' with constant ({constant_val})",
                                        "timestamp": datetime.datetime.now(),
                                        "type": "missing_values",
                                        "details": {
                                            "column": col,
                                            "method": "constant",
                                            "value": constant_val
                                        }
                                    })
                                except ValueError:
                                    st.warning(f"Could not convert '{constant_value}' to appropriate type for column '{col}'. Skipping.")
                                    
                            st.success(f"Filled missing values with constant in {len(columns_to_process)} column(s)")
                        else:
                            st.error("Please enter a constant value")
                            
                    elif handling_method == "Fill with forward fill":
                        for col in columns_to_process:
                            st.session_state.df[col] = st.session_state.df[col].fillna(method='ffill')
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col}' with forward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col,
                                    "method": "ffill"
                                }
                            })
                        
                        st.success(f"Filled missing values with forward fill in {len(columns_to_process)} column(s)")
                        
                    elif handling_method == "Fill with backward fill":
                        for col in columns_to_process:
                            st.session_state.df[col] = st.session_state.df[col].fillna(method='bfill')
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col}' with backward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col,
                                    "method": "bfill"
                                }
                            })
                        
                        st.success(f"Filled missing values with backward fill in {len(columns_to_process)} column(s)")
                        
                    elif handling_method == "Fill with interpolation":
                        for col in columns_to_process:
                            if st.session_state.df[col].dtype.kind in 'bifc':  # Check if numeric
                                st.session_state.df[col] = st.session_state.df[col].interpolate(method='linear')
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col}' with interpolation",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col,
                                        "method": "interpolation"
                                    }
                                })
                            else:
                                st.warning(f"Column '{col}' is not numeric. Skipping interpolation.")
                        
                        st.success(f"Filled missing values with interpolation in {len(columns_to_process)} column(s)")
                
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"Error handling missing values: {str(e)}")
    
    with col2:
        st.markdown("### Duplicate Rows")
        
        # Check for duplicates
        dup_count = st.session_state.df.duplicated().sum()
        
        if dup_count == 0:
            st.success("No duplicate rows found in the dataset!")
        else:
            st.warning(f"Found {dup_count} duplicate rows in the dataset")
            
            # Display sample of duplicates
            if st.checkbox("Show sample of duplicates", key="show_duplicates_cleaning"):
                duplicates = st.session_state.df[st.session_state.df.duplicated(keep='first')]
                st.dataframe(duplicates.head(5), use_container_width=True)
            
            # Button to remove duplicates - with unique key
            if st.button("Remove Duplicate Rows", key="remove_duplicates_cleaning", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Remove duplicates
                    st.session_state.df = st.session_state.df.drop_duplicates()
                    
                    # Add to processing history
                    rows_removed = orig_shape[0] - st.session_state.df.shape[0]
                    st.session_state.processing_history.append({
                        "description": f"Removed {rows_removed} duplicate rows",
                        "timestamp": datetime.datetime.now(),
                        "type": "duplicates",
                        "details": {
                            "rows_removed": rows_removed
                        }
                    })
                    
                    st.success(f"Removed {rows_removed} duplicate rows")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error removing duplicates: {str(e)}")
        
        # Outlier Detection and Handling
        st.markdown("### Outlier Detection")
        
        # Get numeric columns for outlier detection
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("No numeric columns found for outlier detection.")
        else:
            col_for_outliers = st.selectbox(
                "Select column for outlier detection:",
                num_cols,
                key="col_for_outliers"
            )
            
            # Calculate outlier bounds using IQR method
            Q1 = st.session_state.df[col_for_outliers].quantile(0.25)
            Q3 = st.session_state.df[col_for_outliers].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = st.session_state.df[(st.session_state.df[col_for_outliers] < lower_bound) | (st.session_state.df[col_for_outliers] > upper_bound)]
            outlier_count = len(outliers)
            
            # Visualize the distribution with outlier bounds
            fig = px.box(st.session_state.df, y=col_for_outliers, title=f"Distribution of {col_for_outliers} with Outlier Bounds")
            fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower bound")
            fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper bound")
            st.plotly_chart(fig, use_container_width=True)
            
            if outlier_count == 0:
                st.success(f"No outliers found in column '{col_for_outliers}'")
            else:
                st.warning(f"Found {outlier_count} outliers in column '{col_for_outliers}'")
                st.write(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Display sample of outliers
                if st.checkbox("Show sample of outliers", key="show_outliers"):
                    st.dataframe(outliers.head(5), use_container_width=True)
                
                # Options for handling outliers
                outlier_method = st.selectbox(
                    "Select handling method:",
                    [
                        "Remove outliers",
                        "Cap outliers (winsorize)",
                        "Replace with NaN",
                        "Replace with median"
                    ],
                    key="outlier_method"
                )
                
                # Button to handle outliers
                if st.button("Handle Outliers", key="handle_outliers", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = st.session_state.df.shape
                        
                        if outlier_method == "Remove outliers":
                            # Remove rows with outliers
                            st.session_state.df = st.session_state.df[
                                (st.session_state.df[col_for_outliers] >= lower_bound) & 
                                (st.session_state.df[col_for_outliers] <= upper_bound)
                            ]
                            
                            # Add to processing history
                            rows_removed = orig_shape[0] - st.session_state.df.shape[0]
                            st.session_state.processing_history.append({
                                "description": f"Removed {rows_removed} outliers from column '{col_for_outliers}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "outliers",
                                "details": {
                                    "column": col_for_outliers,
                                    "method": "remove",
                                    "rows_affected": rows_removed
                                }
                            })
                            
                            st.success(f"Removed {outlier_count} outliers from column '{col_for_outliers}'")
                            
                        elif outlier_method == "Cap outliers (winsorize)":
                            # Cap values at the bounds
                            st.session_state.df[col_for_outliers] = st.session_state.df[col_for_outliers].clip(lower=lower_bound, upper=upper_bound)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Capped {outlier_count} outliers in column '{col_for_outliers}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "outliers",
                                "details": {
                                    "column": col_for_outliers,
                                    "method": "cap",
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound,
                                    "values_affected": outlier_count
                                }
                            })
                            
                            st.success(f"Capped {outlier_count} outliers in column '{col_for_outliers}'")
                            
                        elif outlier_method == "Replace with NaN":
                            # Replace outliers with NaN
                            st.session_state.df.loc[
                                (st.session_state.df[col_for_outliers] < lower_bound) | 
                                (st.session_state.df[col_for_outliers] > upper_bound),
                                col_for_outliers
                            ] = np.nan
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "outliers",
                                "details": {
                                    "column": col_for_outliers,
                                    "method": "replace_nan",
                                    "values_affected": outlier_count
                                }
                            })
                            
                            st.success(f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'")
                            
                        elif outlier_method == "Replace with median":
                            # Get median value
                            median_val = st.session_state.df[col_for_outliers].median()
                            
                            # Replace outliers with median
                            st.session_state.df.loc[
                                (st.session_state.df[col_for_outliers] < lower_bound) | 
                                (st.session_state.df[col_for_outliers] > upper_bound),
                                col_for_outliers
                            ] = median_val
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Replaced {outlier_count} outliers with median in column '{col_for_outliers}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "outliers",
                                "details": {
                                    "column": col_for_outliers,
                                    "method": "replace_median",
                                    "values_affected": outlier_count,
                                    "median": median_val
                                }
                            })
                            
                            st.success(f"Replaced {outlier_count} outliers with median ({median_val:.2f}) in column '{col_for_outliers}'")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error handling outliers: {str(e)}")

def render_data_transformation():
    """Render data transformation interface"""
    st.subheader("Data Transformation")
    
    # Create columns for organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Numeric Transformations")
        
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("No numeric columns found for transformation.")
        else:
            # Column selection
            col_to_transform = st.selectbox(
                "Select column to transform:",
                num_cols,
                key="num_transform_col"
            )
            
            # Transformation method selection
            transform_method = st.selectbox(
                "Select transformation method:",
                [
                    "Standardization (Z-score)",
                    "Min-Max Scaling",
                    "Robust Scaling",
                    "Log Transform",
                    "Square Root Transform",
                    "Box-Cox Transform",
                    "Binning/Discretization",
                    "Power Transform",
                    "Normalization"
                ],
                key="transform_method"
            )
            
            # Additional parameters for specific transforms
            if transform_method == "Binning/Discretization":
                num_bins = st.slider("Number of bins:", 2, 20, 5)
                bin_strategy = st.selectbox("Binning strategy:", ["uniform", "quantile", "kmeans"])
                bin_labels = st.text_input("Bin labels (comma-separated, leave empty for default):")
            
            elif transform_method == "Power Transform":
                power = st.slider("Power value:", -3.0, 3.0, 1.0, 0.1)
            
            # Create new column or replace
            create_new_col = st.checkbox("Create new column", value=True)
            
            # Apply button
            if st.button("Apply Transformation", key="apply_transform", use_container_width=True):
                try:
                    # Determine output column name
                    if create_new_col:
                        output_col = f"{col_to_transform}_{transform_method.split(' ')[0].lower()}"
                    else:
                        output_col = col_to_transform
                    
                    # Apply transformation
                    if transform_method == "Standardization (Z-score)":
                        # Z-score standardization: (x - mean) / std
                        data = st.session_state.df[col_to_transform].values.reshape(-1, 1)
                        scaler = StandardScaler()
                        transformed_data = scaler.fit_transform(data).flatten()
                        
                        # Handle potential NaN values
                        transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
                        
                        # Store transformation parameters
                        mean = scaler.mean_[0]
                        std = scaler.scale_[0]
                        
                        st.session_state.df[output_col] = transformed_data
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied standardization to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "standardization",
                                "new_column": output_col if create_new_col else None,
                                "mean": mean,
                                "std": std
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with standardized values")
                    
                    elif transform_method == "Min-Max Scaling":
                        # Min-Max scaling: (x - min) / (max - min)
                        data = st.session_state.df[col_to_transform].values.reshape(-1, 1)
                        scaler = MinMaxScaler()
                        transformed_data = scaler.fit_transform(data).flatten()
                        
                        # Handle potential NaN values
                        transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
                        
                        # Store transformation parameters
                        min_val = scaler.data_min_[0]
                        max_val = scaler.data_max_[0]
                        
                        st.session_state.df[output_col] = transformed_data
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied min-max scaling to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "min_max_scaling",
                                "new_column": output_col if create_new_col else None,
                                "min": min_val,
                                "max": max_val
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with scaled values (0-1)")
                        
                    elif transform_method == "Robust Scaling":
                        # Robust scaling using median and IQR (less sensitive to outliers)
                        data = st.session_state.df[col_to_transform].values.reshape(-1, 1)
                        scaler = RobustScaler()
                        transformed_data = scaler.fit_transform(data).flatten()
                        
                        # Handle potential NaN values
                        transformed_data = np.nan_to_num(transformed_data, nan=np.nanmedian(transformed_data))
                        
                        st.session_state.df[output_col] = transformed_data
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied robust scaling to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "robust_scaling",
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with robust scaling")
                    
                    elif transform_method == "Log Transform":
                        # Check for non-positive values
                        min_val = st.session_state.df[col_to_transform].min()
                        
                        if min_val <= 0:
                            # Add a constant to make all values positive
                            const = abs(min_val) + 1
                            st.session_state.df[output_col] = np.log(st.session_state.df[col_to_transform] + const)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied log transform to '{col_to_transform}' with constant {const}",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "log_transform",
                                    "constant": const,
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with log transform (added constant {const})")
                        else:
                            st.session_state.df[output_col] = np.log(st.session_state.df[col_to_transform])
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied log transform to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "log_transform",
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with log transform")
                    
                    elif transform_method == "Square Root Transform":
                        # Check for negative values
                        min_val = st.session_state.df[col_to_transform].min()
                        
                        if min_val < 0:
                            # Add a constant to make all values non-negative
                            const = abs(min_val) + 1
                            st.session_state.df[output_col] = np.sqrt(st.session_state.df[col_to_transform] + const)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied square root transform to '{col_to_transform}' with constant {const}",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "sqrt_transform",
                                    "constant": const,
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with square root transform (added constant {const})")
                        else:
                            st.session_state.df[output_col] = np.sqrt(st.session_state.df[col_to_transform])
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied square root transform to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "sqrt_transform",
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with square root transform")
                    
                    elif transform_method == "Box-Cox Transform":
                        # Check if all values are positive
                        min_val = st.session_state.df[col_to_transform].min()
                        
                        if min_val <= 0:
                            # Add a constant to make all values positive
                            const = abs(min_val) + 1
                            
                            from scipy import stats
                            transformed_data, lambda_val = stats.boxcox(st.session_state.df[col_to_transform] + const)
                            st.session_state.df[output_col] = transformed_data
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied Box-Cox transform to '{col_to_transform}' with constant {const}",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "boxcox_transform",
                                    "constant": const,
                                    "lambda": lambda_val,
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with Box-Cox transform (lambda={lambda_val:.4f}, added constant {const})")
                        else:
                            from scipy import stats
                            transformed_data, lambda_val = stats.boxcox(st.session_state.df[col_to_transform])
                            st.session_state.df[output_col] = transformed_data
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied Box-Cox transform to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "boxcox_transform",
                                    "lambda": lambda_val,
                                    "new_column": output_col if create_new_col else None
                                }
                            })
                            
                            st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with Box-Cox transform (lambda={lambda_val:.4f})")
                    
                    elif transform_method == "Binning/Discretization":
                        # Create bins based on strategy
                        if bin_strategy == "uniform":
                            # Uniform binning
                            bins = pd.cut(
                                st.session_state.df[col_to_transform], 
                                bins=num_bins, 
                                labels=bin_labels.split(',') if bin_labels else None
                            )
                        elif bin_strategy == "quantile":
                            # Quantile-based binning
                            bins = pd.qcut(
                                st.session_state.df[col_to_transform], 
                                q=num_bins, 
                                labels=bin_labels.split(',') if bin_labels else None
                            )
                        elif bin_strategy == "kmeans":
                            # K-means clustering binning
                            from sklearn.cluster import KMeans
                            
                            # Reshape data for KMeans
                            data = st.session_state.df[col_to_transform].values.reshape(-1, 1)
                            
                            # Fit KMeans
                            kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(data)
                            
                            # Get cluster labels
                            clusters = kmeans.labels_
                            
                            # Create custom bin labels if provided
                            if bin_labels:
                                label_map = {i: label for i, label in enumerate(bin_labels.split(',')[:num_bins])}
                                bins = pd.Series([label_map.get(c, f"Cluster {c}") for c in clusters])
                            else:
                                bins = pd.Series([f"Cluster {c}" for c in clusters])
                        
                        st.session_state.df[output_col] = bins
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied binning to '{col_to_transform}' with {num_bins} bins using {bin_strategy} strategy",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "binning",
                                "num_bins": num_bins,
                                "strategy": bin_strategy,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with {num_bins} bins using {bin_strategy} strategy")
                    
                    elif transform_method == "Power Transform":
                        # Apply power transform: x^power
                        st.session_state.df[output_col] = np.power(st.session_state.df[col_to_transform], power)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied power transform to '{col_to_transform}' with power {power}",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "power_transform",
                                "power": power,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with power transform (power={power})")
                    
                    elif transform_method == "Normalization":
                        # Normalize to sum to 1
                        total = st.session_state.df[col_to_transform].sum()
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform] / total
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied normalization to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "normalization",
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with normalization")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying transformation: {str(e)}")
    
    with col2:
        st.markdown("### Categorical Transformations")
        
        # Get categorical columns
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            st.info("No categorical columns found for transformation.")
        else:
            # Column selection
            col_to_transform = st.selectbox(
                "Select column to transform:",
                cat_cols,
                key="cat_transform_col"
            )
            
            # Transformation method selection
            transform_method = st.selectbox(
                "Select transformation method:",
                [
                    "One-Hot Encoding",
                    "Label Encoding",
                    "Frequency Encoding",
                    "Target Encoding",
                    "Binary Encoding",
                    "Count Encoding",
                    "Ordinal Encoding",
                    "String Cleaning"
                ],
                key="cat_transform_method"
            )
            
            # Additional parameters for specific transforms
            if transform_method == "Target Encoding":
                # Target column selection
                target_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                if target_cols:
                    target_col = st.selectbox("Select target column:", target_cols)
                else:
                    st.warning("No numeric columns available for target encoding.")
            
            elif transform_method == "Ordinal Encoding":
                # Get unique values
                unique_vals = st.session_state.df[col_to_transform].dropna().unique().tolist()
                st.write("Unique values:", ", ".join([str(val) for val in unique_vals]))
                
                # Let user specify order
                order_input = st.text_area(
                    "Enter ordered values (one per line, first = lowest):",
                    value="\n".join([str(val) for val in unique_vals])
                )
            
            elif transform_method == "String Cleaning":
                # String cleaning options
                cleaning_options = st.multiselect(
                    "Select cleaning operations:",
                    ["Lowercase", "Remove special characters", "Remove numbers", "Remove extra spaces", "Remove HTML tags"]
                )
            
            # Create new column or overwrite
            create_new_col = st.checkbox("Create new columns", value=True, key="cat_new_col")
            
            # Apply button
            if st.button("Apply Transformation", key="apply_cat_transform", use_container_width=True):
                try:
                    if transform_method == "One-Hot Encoding":
                        # Get dummies (one-hot encoding)
                        prefix = col_to_transform if create_new_col else ""
                        dummies = pd.get_dummies(st.session_state.df[col_to_transform], prefix=prefix)
                        
                        # Add to dataframe
                        if create_new_col:
                            st.session_state.df = pd.concat([st.session_state.df, dummies], axis=1)
                        else:
                            # Drop original column and add dummies
                            st.session_state.df = st.session_state.df.drop(columns=[col_to_transform])
                            st.session_state.df = pd.concat([st.session_state.df, dummies], axis=1)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied one-hot encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "one_hot_encoding",
                                "new_columns": dummies.columns.tolist(),
                                "drop_original": not create_new_col
                            }
                        })
                        
                        st.success(f"Created {dummies.shape[1]} new columns with one-hot encoding for '{col_to_transform}'")
                    
                    elif transform_method == "Label Encoding":
                        # Output column name
                        output_col = f"{col_to_transform}_encoded" if create_new_col else col_to_transform
                        
                        # Map each unique value to an integer
                        unique_values = st.session_state.df[col_to_transform].dropna().unique()
                        mapping = {val: i for i, val in enumerate(unique_values)}
                        
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform].map(mapping)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied label encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "label_encoding",
                                "mapping": mapping,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with label encoding")
                    
                    elif transform_method == "Frequency Encoding":
                        # Output column name
                        output_col = f"{col_to_transform}_freq" if create_new_col else col_to_transform
                        
                        # Replace each category with its frequency
                        freq = st.session_state.df[col_to_transform].value_counts(normalize=True)
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform].map(freq)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied frequency encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "frequency_encoding",
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with frequency encoding")
                    
                    elif transform_method == "Target Encoding":
                        # Check if target column exists and is numeric
                        if 'target_col' not in locals():
                            st.error("No target column selected")
                            return
                            
                        # Output column name
                        output_col = f"{col_to_transform}_target" if create_new_col else col_to_transform
                        
                        # Calculate mean target value for each category
                        target_means = st.session_state.df.groupby(col_to_transform)[target_col].mean()
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform].map(target_means)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied target encoding to '{col_to_transform}' using '{target_col}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "target_encoding",
                                "target_column": target_col,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with target encoding")
                    
                    elif transform_method == "Binary Encoding":
                        # Output column prefix
                        prefix = f"{col_to_transform}_bin" if create_new_col else col_to_transform
                        
                        # Label encode first
                        label_encoded = pd.Series(
                            st.session_state.df[col_to_transform].astype('category').cat.codes, 
                            index=st.session_state.df.index
                        )
                        
                        # Convert to binary and create columns
                        for i in range(label_encoded.max().bit_length()):
                            bit_val = label_encoded.apply(lambda x: (x >> i) & 1)
                            st.session_state.df[f"{prefix}_{i}"] = bit_val
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied binary encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "binary_encoding",
                                "prefix": prefix
                            }
                        })
                        
                        st.success(f"Created binary encoding columns with prefix '{prefix}'")
                    
                    elif transform_method == "Count Encoding":
                        # Output column name
                        output_col = f"{col_to_transform}_count" if create_new_col else col_to_transform
                        
                        # Replace each category with its count
                        counts = st.session_state.df[col_to_transform].value_counts()
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform].map(counts)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied count encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "count_encoding",
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with count encoding")
                    
                    elif transform_method == "Ordinal Encoding":
                        # Output column name
                        output_col = f"{col_to_transform}_ordinal" if create_new_col else col_to_transform
                        
                        # Parse the ordered values
                        ordered_values = [val.strip() for val in order_input.split('\n') if val.strip()]
                        
                        # Create mapping
                        mapping = {val: i for i, val in enumerate(ordered_values)}
                        
                        # Apply mapping
                        st.session_state.df[output_col] = st.session_state.df[col_to_transform].map(mapping)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied ordinal encoding to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "ordinal_encoding",
                                "mapping": mapping,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with ordinal encoding")
                    
                    elif transform_method == "String Cleaning":
                        # Output column name
                        output_col = f"{col_to_transform}_clean" if create_new_col else col_to_transform
                        
                        # Apply selected cleaning operations
                        cleaned_series = st.session_state.df[col_to_transform].astype(str)
                        
                        for op in cleaning_options:
                            if op == "Lowercase":
                                cleaned_series = cleaned_series.str.lower()
                            elif op == "Remove special characters":
                                cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True)
                            elif op == "Remove numbers":
                                cleaned_series = cleaned_series.str.replace(r'\d+', '', regex=True)
                            elif op == "Remove extra spaces":
                                cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True).str.strip()
                            elif op == "Remove HTML tags":
                                cleaned_series = cleaned_series.str.replace(r'<.*?>', '', regex=True)
                        
                        st.session_state.df[output_col] = cleaned_series
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied string cleaning to '{col_to_transform}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "transformation",
                            "details": {
                                "column": col_to_transform,
                                "method": "string_cleaning",
                                "operations": cleaning_options,
                                "new_column": output_col if create_new_col else None
                            }
                        })
                        
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with string cleaning")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying transformation: {str(e)}")
            
            # Date Transformations
            st.markdown("### Date/Time Transformations")
            
            # Get datetime columns and potential date columns
            datetime_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
            potential_date_cols = []
            
            # Look for columns that might contain dates but aren't datetime type
            for col in st.session_state.df.columns:
                if col in datetime_cols:
                    continue
                    
                # Look for columns with "date", "time", "year", "month", "day" in the name
                if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                    potential_date_cols.append(col)
                    
                # Check sample values if it's a string column
                elif st.session_state.df[col].dtype == 'object':
                    # Check first non-null value
                    sample_val = st.session_state.df[col].dropna().iloc[0] if not st.session_state.df[col].dropna().empty else None
                    if sample_val and isinstance(sample_val, str):
                        # Try common date patterns
                        date_patterns = [
                            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                        ]
                        if any(re.search(pattern, sample_val) for pattern in date_patterns):
                            potential_date_cols.append(col)
            
            # Combine all columns that might be dates
            all_date_cols = datetime_cols + potential_date_cols
            
            if not all_date_cols:
                st.info("No datetime columns found for transformation.")
            else:
                # Column selection
                date_col = st.selectbox(
                    "Select date/time column:",
                    all_date_cols
                )
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' is not currently a datetime type. It will be converted.")
                    date_format = st.text_input(
                        "Enter date format (e.g., '%Y-%m-%d', leave blank for auto-detection):",
                        key="date_format"
                    )
                    
                    if st.button("Convert to Datetime", key="convert_date", use_container_width=True):
                        try:
                            if date_format and date_format.strip():
                                st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], format=date_format, errors='coerce')
                            else:
                                st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], errors='coerce')
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Converted '{date_col}' to datetime format",
                                "timestamp": datetime.datetime.now(),
                                "type": "date_conversion",
                                "details": {
                                    "column": date_col,
                                    "format": date_format if date_format and date_format.strip() else "auto-detected"
                                }
                            })
                            
                            st.success(f"Converted '{date_col}' to datetime format")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error converting to datetime: {str(e)}")
                
                # Date feature extraction
                st.markdown("#### Extract Date Features")
                
                date_features = st.multiselect(
                    "Select date features to extract:",
                    ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Quarter", "Week of Year", "Is Weekend", "Is Month End", "Is Month Start"]
                )
                
                if date_features and st.button("Extract Date Features", use_container_width=True):
                    try:
                        # Ensure the column is datetime type
                        if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[date_col]):
                            st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], errors='coerce')
                        
                        # Extract each selected feature
                        for feature in date_features:
                            if feature == "Year":
                                st.session_state.df[f"{date_col}_year"] = st.session_state.df[date_col].dt.year
                            elif feature == "Month":
                                st.session_state.df[f"{date_col}_month"] = st.session_state.df[date_col].dt.month
                            elif feature == "Day":
                                st.session_state.df[f"{date_col}_day"] = st.session_state.df[date_col].dt.day
                            elif feature == "Hour":
                                st.session_state.df[f"{date_col}_hour"] = st.session_state.df[date_col].dt.hour
                            elif feature == "Minute":
                                st.session_state.df[f"{date_col}_minute"] = st.session_state.df[date_col].dt.minute
                            elif feature == "Second":
                                st.session_state.df[f"{date_col}_second"] = st.session_state.df[date_col].dt.second
                            elif feature == "Day of Week":
                                st.session_state.df[f"{date_col}_dayofweek"] = st.session_state.df[date_col].dt.dayofweek
                            elif feature == "Quarter":
                                st.session_state.df[f"{date_col}_quarter"] = st.session_state.df[date_col].dt.quarter
                            elif feature == "Week of Year":
                                st.session_state.df[f"{date_col}_weekofyear"] = st.session_state.df[date_col].dt.isocalendar().week
                            elif feature == "Is Weekend":
                                st.session_state.df[f"{date_col}_is_weekend"] = st.session_state.df[date_col].dt.dayofweek.isin([5, 6])
                            elif feature == "Is Month End":
                                st.session_state.df[f"{date_col}_is_month_end"] = st.session_state.df[date_col].dt.is_month_end
                            elif feature == "Is Month Start":
                                st.session_state.df[f"{date_col}_is_month_start"] = st.session_state.df[date_col].dt.is_month_start
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Extracted {len(date_features)} date features from '{date_col}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "feature_extraction",
                            "details": {
                                "column": date_col,
                                "features": date_features
                            }
                        })
                        
                        st.success(f"Extracted {len(date_features)} date features from '{date_col}'")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error extracting date features: {str(e)}")
                
                # Time difference calculation
                st.markdown("#### Calculate Time Difference")
                
                if len(all_date_cols) >= 2:
                    date_col2 = st.selectbox(
                        "Select second date column for time difference:",
                        [col for col in all_date_cols if col != date_col]
                    )
                    
                    time_units = st.selectbox(
                        "Select time difference unit:",
                        ["Days", "Hours", "Minutes", "Seconds"]
                    )
                    
                    result_name = st.text_input(
                        "Result column name:",
                        value=f"{date_col}_to_{date_col2}_diff_{time_units.lower()}"
                    )
                    
                    if st.button("Calculate Time Difference", use_container_width=True):
                        try:
                            # Ensure both columns are datetime type
                            for col in [date_col, date_col2]:
                                if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]):
                                    st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                            
                            # Calculate time difference
                            time_diff = st.session_state.df[date_col] - st.session_state.df[date_col2]
                            
                            # Convert to selected unit
                            if time_units == "Days":
                                st.session_state.df[result_name] = time_diff.dt.total_seconds() / (24 * 3600)
                            elif time_units == "Hours":
                                st.session_state.df[result_name] = time_diff.dt.total_seconds() / 3600
                            elif time_units == "Minutes":
                                st.session_state.df[result_name] = time_diff.dt.total_seconds() / 60
                            else:  # Seconds
                                st.session_state.df[result_name] = time_diff.dt.total_seconds()
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Calculated time difference between '{date_col}' and '{date_col2}' in {time_units.lower()}",
                                "timestamp": datetime.datetime.now(),
                                "type": "time_difference",
                                "details": {
                                    "column1": date_col,
                                    "column2": date_col2,
                                    "unit": time_units.lower(),
                                    "result_column": result_name
                                }
                            })
                            
                            st.success(f"Calculated time difference in {time_units.lower()} as '{result_name}'")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error calculating time difference: {str(e)}")
                else:
                    st.info("Need at least two datetime columns to calculate time difference.")

def render_feature_engineering():
    """Render feature engineering interface with enhanced capabilities"""
    st.subheader("Feature Engineering")
    
    # Create tabs for different feature engineering tasks
    fe_tabs = st.tabs([
        "Mathematical Operations",
        "Text Features",
        "Interaction Features",
        "Advanced Features",
        "Custom Formula"
    ])
    
    # Mathematical Operations tab
    with fe_tabs[0]:
        st.markdown("### Mathematical Operations")
        st.write("Create new columns by applying mathematical operations to existing numeric columns.")
        
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("No numeric columns found for mathematical operations.")
            return
        
        # Column selection
        col1 = st.selectbox("Select first column:", num_cols, key="math_col1")
        
        # Operation selection
        operations = [
            "Addition (+)",
            "Subtraction (-)",
            "Multiplication (*)",
            "Division (/)",
            "Modulo (%)",
            "Power (^)",
            "Absolute Value (|x|)",
            "Logarithm (log)",
            "Exponential (exp)",
            "Round",
            "Floor",
            "Ceiling"
        ]
        
        operation = st.selectbox("Select operation:", operations)
        
        # Second column for binary operations
        binary_ops = ["Addition (+)", "Subtraction (-)", "Multiplication (*)", "Division (/)", "Modulo (%)", "Power (^)"]
        
        if operation in binary_ops:
            # Allow second column or constant value
            use_constant = st.checkbox("Use constant value instead of second column", value=False)
            
            if use_constant:
                constant = st.number_input("Enter constant value:", value=1.0)
                second_operand = constant
                operand_desc = str(constant)
            else:
                col2 = st.selectbox("Select second column:", [c for c in num_cols if c != col1], key="math_col2")
                second_operand = st.session_state.df[col2]
                operand_desc = col2
        
        # New column name
        if operation in binary_ops:
            op_symbols = {
                "Addition (+)": "+",
                "Subtraction (-)": "-",
                "Multiplication (*)": "*",
                "Division (/)": "/",
                "Modulo (%)": "%",
                "Power (^)": "^"
            }
            default_name = f"{col1}_{op_symbols[operation]}_{operand_desc}"
        else:
            op_names = {
                "Absolute Value (|x|)": "abs",
                "Logarithm (log)": "log",
                "Exponential (exp)": "exp",
                "Round": "round",
                "Floor": "floor",
                "Ceiling": "ceil"
            }
            default_name = f"{op_names[operation]}_{col1}"
            
        new_col_name = st.text_input("New column name:", value=default_name)
        
        # Preview button
        if st.button("Preview Result", key="math_preview"):
            try:
                # Calculate result based on operation
                if operation == "Addition (+)":
                    result = st.session_state.df[col1] + second_operand
                elif operation == "Subtraction (-)":
                    result = st.session_state.df[col1] - second_operand
                elif operation == "Multiplication (*)":
                    result = st.session_state.df[col1] * second_operand
                elif operation == "Division (/)":
                    # Avoid division by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot divide by zero!")
                        return
                    result = st.session_state.df[col1] / second_operand
                elif operation == "Modulo (%)":
                    # Avoid modulo by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot perform modulo by zero!")
                        return
                    result = st.session_state.df[col1] % second_operand
                elif operation == "Power (^)":
                    result = st.session_state.df[col1] ** second_operand
                elif operation == "Absolute Value (|x|)":
                    result = st.session_state.df[col1].abs()
                elif operation == "Logarithm (log)":
                    # Check for non-positive values
                    if (st.session_state.df[col1] <= 0).any():
                        st.warning("Column contains non-positive values. Will apply log to positive values only and return NaN for others.")
                    result = np.log(st.session_state.df[col1])
                elif operation == "Exponential (exp)":
                    result = np.exp(st.session_state.df[col1])
                elif operation == "Round":
                    result = st.session_state.df[col1].round()
                elif operation == "Floor":
                    result = np.floor(st.session_state.df[col1])
                elif operation == "Ceiling":
                    result = np.ceil(st.session_state.df[col1])
                
                # Preview the first few rows
                preview_df = pd.DataFrame({
                    col1: st.session_state.df[col1].head(5),
                    "Result": result.head(5)
                })
                
                # Add second column to preview if applicable
                if operation in binary_ops and not use_constant:
                    preview_df.insert(1, col2, st.session_state.df[col2].head(5))
                
                st.write("Preview (first 5 rows):")
                st.dataframe(preview_df)
                
                # Statistics about the result
                st.write("Result Statistics:")
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Mean", f"{result.mean():.2f}")
                with stat_cols[1]:
                    st.metric("Min", f"{result.min():.2f}")
                with stat_cols[2]:
                    st.metric("Max", f"{result.max():.2f}")
                with stat_cols[3]:
                    st.metric("Missing Values", f"{result.isna().sum()}")
                
            except Exception as e:
                st.error(f"Error previewing result: {str(e)}")
        
        # Apply button
        if st.button("Create New Column", key="math_apply", use_container_width=True):
            try:
                # Calculate result based on operation
                if operation == "Addition (+)":
                    st.session_state.df[new_col_name] = st.session_state.df[col1] + second_operand
                    op_desc = f"Added {col1} and {operand_desc}"
                elif operation == "Subtraction (-)":
                    st.session_state.df[new_col_name] = st.session_state.df[col1] - second_operand
                    op_desc = f"Subtracted {operand_desc} from {col1}"
                elif operation == "Multiplication (*)":
                    st.session_state.df[new_col_name] = st.session_state.df[col1] * second_operand
                    op_desc = f"Multiplied {col1} by {operand_desc}"
                elif operation == "Division (/)":
                    # Avoid division by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot divide by zero!")
                        return
                    st.session_state.df[new_col_name] = st.session_state.df[col1] / second_operand
                    op_desc = f"Divided {col1} by {operand_desc}"
                elif operation == "Modulo (%)":
                    # Avoid modulo by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot perform modulo by zero!")
                        return
                    st.session_state.df[new_col_name] = st.session_state.df[col1] % second_operand
                    op_desc = f"Calculated {col1} modulo {operand_desc}"
                elif operation == "Power (^)":
                    st.session_state.df[new_col_name] = st.session_state.df[col1] ** second_operand
                    op_desc = f"Raised {col1} to the power of {operand_desc}"
                elif operation == "Absolute Value (|x|)":
                    st.session_state.df[new_col_name] = st.session_state.df[col1].abs()
                    op_desc = f"Calculated absolute value of {col1}"
                elif operation == "Logarithm (log)":
                    st.session_state.df[new_col_name] = np.log(st.session_state.df[col1])
                    op_desc = f"Calculated natural logarithm of {col1}"
                elif operation == "Exponential (exp)":
                    st.session_state.df[new_col_name] = np.exp(st.session_state.df[col1])
                    op_desc = f"Calculated exponential of {col1}"
                elif operation == "Round":
                    st.session_state.df[new_col_name] = st.session_state.df[col1].round()
                    op_desc = f"Rounded {col1} to nearest integer"
                elif operation == "Floor":
                    st.session_state.df[new_col_name] = np.floor(st.session_state.df[col1])
                    op_desc = f"Applied floor function to {col1}"
                elif operation == "Ceiling":
                    st.session_state.df[new_col_name] = np.ceil(st.session_state.df[col1])
                    op_desc = f"Applied ceiling function to {col1}"
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Created new column '{new_col_name}': {op_desc}",
                    "timestamp": datetime.datetime.now(),
                    "type": "feature_engineering",
                    "details": {
                        "operation": operation,
                        "columns_used": [col1] if operation not in binary_ops or use_constant else [col1, col2],
                        "result_column": new_col_name
                    }
                })
                
                st.success(f"Created new column '{new_col_name}'")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating new column: {str(e)}")
    
    # Text Features tab
    with fe_tabs[1]:
        st.markdown("### Text Features")
        st.write("Extract features from text columns.")
        
        # Get text columns
        text_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_cols:
            st.info("No text columns found for feature extraction.")
            return
        
        # Column selection
        text_col = st.selectbox("Select text column:", text_cols)
        
        # Feature type selection
        feature_type = st.selectbox(
            "Select feature to extract:",
            [
                "Text Length",
                "Word Count",
                "Character Count",
                "Contains Specific Text",
                "Extract Pattern",
                "Count Specific Pattern",
                "Sentiment Analysis"
            ]
        )
        
        # Additional inputs based on feature type
        if feature_type == "Contains Specific Text":
            search_text = st.text_input("Text to search for:")
            case_sensitive = st.checkbox("Case sensitive search", value=False)
            
            # Default name
            default_name = f"{text_col}_contains_{search_text.lower().replace(' ', '_')}"
            
        elif feature_type == "Extract Pattern":
            pattern = st.text_input("Regular expression pattern:", 
                                   placeholder="e.g., \\d+ for numbers")
            st.caption("Use regex pattern to extract matching content. First match will be used.")
            
            # Default name
            default_name = f"{text_col}_pattern_extract"
            
        elif feature_type == "Count Specific Pattern":
            pattern = st.text_input("Regular expression pattern:", 
                                   placeholder="e.g., \\w+ for words")
            st.caption("Use regex pattern to count occurrences.")
            
            # Default name
            default_name = f"{text_col}_pattern_count"
            
        elif feature_type == "Sentiment Analysis":
            st.info("Note: Basic sentiment analysis uses a simple dictionary approach and may not be accurate for all contexts.")
            
            # Default name
            default_name = f"{text_col}_sentiment"
            
        else:
            # Default names for simpler features
            if feature_type == "Text Length":
                default_name = f"{text_col}_length"
            elif feature_type == "Word Count":
                default_name = f"{text_col}_word_count"
            elif feature_type == "Character Count":
                default_name = f"{text_col}_char_count"
        
        # New column name
        new_col_name = st.text_input("New column name:", value=default_name)
        
        # Apply button
        if st.button("Create Text Feature", key="text_feature_apply", use_container_width=True):
            try:
                # Apply the selected feature extraction
                if feature_type == "Text Length":
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(len)
                    desc = f"Extracted text length from '{text_col}'"
                
                elif feature_type == "Word Count":
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                        lambda x: len(str(x).split())
                    )
                    desc = f"Counted words in '{text_col}'"
                
                elif feature_type == "Character Count":
                    # Count only alphabetic characters
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                        lambda x: sum(c.isalpha() for c in str(x))
                    )
                    desc = f"Counted alphabetic characters in '{text_col}'"
                
                elif feature_type == "Contains Specific Text":
                    if not search_text:
                        st.error("Please enter text to search for")
                        return
                    
                    if case_sensitive:
                        st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                            lambda x: search_text in str(x)
                        )
                    else:
                        st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                            lambda x: search_text.lower() in str(x).lower()
                        )
                    
                    desc = f"Checked if '{text_col}' contains '{search_text}'"
                
                elif feature_type == "Extract Pattern":
                    if not pattern:
                        st.error("Please enter a regex pattern")
                        return
                    
                    # Extract first match of the pattern
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                        lambda x: re.search(pattern, str(x)).group(0) if re.search(pattern, str(x)) else None
                    )
                    
                    desc = f"Extracted pattern '{pattern}' from '{text_col}'"
                
                elif feature_type == "Count Specific Pattern":
                    if not pattern:
                        st.error("Please enter a regex pattern")
                        return
                    
                    # Count occurrences of the pattern
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].astype(str).apply(
                        lambda x: len(re.findall(pattern, str(x)))
                    )
                    
                    desc = f"Counted occurrences of pattern '{pattern}' in '{text_col}'"
                
                elif feature_type == "Sentiment Analysis":
                    # Simple sentiment analysis using common words
                    # This is a very basic approach and should be improved for production use
                    positive_words = [
                        'good', 'great', 'excellent', 'positive', 'happy', 'best', 'perfect',
                        'love', 'awesome', 'fantastic', 'nice', 'amazing', 'wonderful', 'enjoy'
                    ]
                    
                    negative_words = [
                        'bad', 'poor', 'negative', 'terrible', 'worst', 'horrible', 'hate',
                        'awful', 'disappointing', 'dislike', 'sad', 'unhappy', 'problem', 'fail'
                    ]
                    
                    def simple_sentiment(text):
                        text = str(text).lower()
                        words = re.findall(r'\b\w+\b', text)
                        
                        pos_count = sum(word in positive_words for word in words)
                        neg_count = sum(word in negative_words for word in words)
                        
                        if pos_count > neg_count:
                            return 'positive'
                        elif neg_count > pos_count:
                            return 'negative'
                        else:
                            return 'neutral'
                    
                    st.session_state.df[new_col_name] = st.session_state.df[text_col].apply(simple_sentiment)
                    
                    desc = f"Applied simple sentiment analysis to '{text_col}'"
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Created text feature '{new_col_name}': {desc}",
                    "timestamp": datetime.datetime.now(),
                    "type": "feature_engineering",
                    "details": {
                        "column": text_col,
                        "feature_type": feature_type,
                        "result_column": new_col_name
                    }
                })
                
                st.success(f"Created new column '{new_col_name}'")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating text feature: {str(e)}")
    
    # Interaction Features tab
    with fe_tabs[2]:
        st.markdown("### Interaction Features")
        st.write("Create features that capture interactions between multiple columns.")
        
        # Get columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Interaction type selection
        interaction_type = st.selectbox(
            "Select interaction type:",
            [
                "Numerical Interaction",
                "Groupby Statistics",
                "Polynomial Features",
                "Ratio/Proportion"
            ]
        )
        
        if interaction_type == "Numerical Interaction":
            st.write("Create interaction terms between numeric variables (e.g., multiplication of features).")
            
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for numerical interactions.")
                return
                
            # Select columns for interaction
            cols_for_interaction = st.multiselect(
                "Select columns for interaction (max 3 recommended):", 
                num_cols,
                default=num_cols[:min(2, len(num_cols))]
            )
            
            if len(cols_for_interaction) < 2:
                st.info("Please select at least two columns for interaction.")
                return
                
            # Interaction operation
            interaction_op = st.selectbox(
                "Interaction operation:",
                ["Multiplication", "Addition", "Subtraction", "Division"]
            )
            
            # New column name
            if interaction_op == "Multiplication":
                default_name = "_".join(cols_for_interaction) + "_product"
            elif interaction_op == "Addition":
                default_name = "_".join(cols_for_interaction) + "_sum"
            elif interaction_op == "Subtraction":
                default_name = "_".join(cols_for_interaction) + "_diff"
            else:  # Division
                default_name = "_".join(cols_for_interaction) + "_ratio"
                
            new_col_name = st.text_input("New column name:", value=default_name)
            
            # Apply button
            if st.button("Create Interaction Feature", key="num_interaction", use_container_width=True):
                try:
                    # Create the interaction feature
                    if interaction_op == "Multiplication":
                        result = st.session_state.df[cols_for_interaction[0]].copy()
                        for col in cols_for_interaction[1:]:
                            result *= st.session_state.df[col]
                        st.session_state.df[new_col_name] = result
                        desc = f"Multiplication of {', '.join(cols_for_interaction)}"
                    elif interaction_op == "Addition":
                        st.session_state.df[new_col_name] = sum(st.session_state.df[col] for col in cols_for_interaction)
                        desc = f"Sum of {', '.join(cols_for_interaction)}"
                    elif interaction_op == "Subtraction":
                        if len(cols_for_interaction) != 2:
                            st.warning("Subtraction works best with exactly 2 columns. Using first two selected columns.")
                        result = st.session_state.df[cols_for_interaction[0]] - st.session_state.df[cols_for_interaction[1]]
                        st.session_state.df[new_col_name] = result
                        desc = f"Subtraction: {cols_for_interaction[0]} - {cols_for_interaction[1]}"
                    else:  # Division
                        if len(cols_for_interaction) != 2:
                            st.warning("Division works best with exactly 2 columns. Using first two selected columns.")
                        # Avoid division by zero by replacing zeros with NaN
                        denominator = st.session_state.df[cols_for_interaction[1]].replace(0, np.nan)
                        st.session_state.df[new_col_name] = st.session_state.df[cols_for_interaction[0]] / denominator
                        desc = f"Ratio: {cols_for_interaction[0]} / {cols_for_interaction[1]}"
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created interaction feature '{new_col_name}': {desc}",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "columns": cols_for_interaction,
                            "operation": interaction_op,
                            "result_column": new_col_name
                        }
                    })
                    
                    st.success(f"Created new interaction column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating interaction feature: {str(e)}")
        
        elif interaction_type == "Groupby Statistics":
            st.write("Create statistical features based on grouping by categorical variables.")
            
            if not cat_cols:
                st.info("Need at least one categorical column for groupby statistics.")
                return
                
            if not num_cols:
                st.info("Need at least one numeric column for groupby statistics.")
                return
                
            # Select groupby column(s)
            groupby_cols = st.multiselect(
                "Select grouping column(s):",
                cat_cols,
                default=[cat_cols[0]] if cat_cols else []
            )
            
            if not groupby_cols:
                st.info("Please select at least one grouping column.")
                return
                
            # Select value column and aggregation
            value_col = st.selectbox("Select value column:", num_cols)
            
            agg_funcs = st.multiselect(
                "Select aggregation functions:",
                ["Mean", "Median", "Sum", "Count", "Min", "Max", "Std", "Var"],
                default=["Mean"]
            )
            
            if not agg_funcs:
                st.info("Please select at least one aggregation function.")
                return
                
            # Apply button
            if st.button("Create Groupby Features", key="groupby_features", use_container_width=True):
                try:
                    # Translate function names to pandas aggregation functions
                    agg_map = {
                        "Mean": "mean",
                        "Median": "median", 
                        "Sum": "sum",
                        "Count": "count",
                        "Min": "min",
                        "Max": "max",
                        "Std": "std",
                        "Var": "var"
                    }
                    
                    # Selected aggregation functions
                    selected_aggs = [agg_map[func] for func in agg_funcs]
                    
                    # Create aggregation dictionary
                    agg_map = {col: selected_aggs for col in [value_col]}
                    
                    # Calculate grouped statistics
                    grouped_stats = st.session_state.df.groupby(groupby_cols)[value_col].agg(selected_aggs).reset_index()
                    
                    # Flatten MultiIndex if needed
                    if isinstance(grouped_stats.columns, pd.MultiIndex):
                        grouped_stats.columns = [f"{col}_{agg}" if isinstance(col, tuple) else col for col in grouped_stats.columns]
                    
                    # Create feature columns
                    group_key = "_".join(groupby_cols)
                    new_columns = []
                    
                    # Merge aggregated features back to original dataframe
                    for agg_func in selected_aggs:
                        # Create column name
                        col_name = f"{group_key}_{value_col}_{agg_func}"
                        
                        # Add to new columns list
                        new_columns.append(col_name)
                    
                    # Merge back to the original dataframe
                    st.session_state.df = st.session_state.df.merge(grouped_stats, on=groupby_cols, how='left')
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created {len(selected_aggs)} groupby features for {value_col} grouped by {', '.join(groupby_cols)}",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "group_columns": groupby_cols,
                            "value_column": value_col,
                            "aggregations": selected_aggs,
                            "new_columns": new_columns
                        }
                    })
                    
                    st.success(f"Created {len(selected_aggs)} new groupby feature columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating groupby features: {str(e)}")
        
        elif interaction_type == "Polynomial Features":
            st.write("Generate polynomial features from numeric columns.")
            
            if len(num_cols) < 1:
                st.info("Need at least one numeric column for polynomial features.")
                return
                
            # Select columns for polynomial features
            poly_cols = st.multiselect(
                "Select columns for polynomial features:",
                num_cols,
                default=[num_cols[0]] if num_cols else []
            )
            
            if not poly_cols:
                st.info("Please select at least one column.")
                return
            
            # Degree selection
            degree = st.slider("Polynomial degree:", min_value=2, max_value=5, value=2)
            
            # Include interactions
            include_interaction = st.checkbox("Include interaction terms", value=True)
            
            # Apply button
            if st.button("Create Polynomial Features", key="poly_features", use_container_width=True):
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    # Get selected columns data
                    X = st.session_state.df[poly_cols].fillna(0)  # Replace NaNs with 0 for polynomial generation
                    
                    # Create polynomial features
                    poly = PolynomialFeatures(
                        degree=degree, 
                        interaction_only=not include_interaction,
                        include_bias=False
                    )
                    
                    # Transform data
                    poly_features = poly.fit_transform(X)
                    
                    # Get feature names
                    if hasattr(poly, 'get_feature_names_out'):
                        # For newer scikit-learn versions
                        feature_names = poly.get_feature_names_out(input_features=poly_cols)
                    else:
                        # For older scikit-learn versions
                        feature_names = poly.get_feature_names(input_features=poly_cols)
                    
                    # Create dataframe with new features
                    poly_df = pd.DataFrame(
                        poly_features, 
                        columns=feature_names,
                        index=st.session_state.df.index
                    )
                    
                    # Drop the original columns from the polynomial features (first len(poly_cols) columns)
                    poly_df = poly_df.iloc[:, len(poly_cols):]
                    
                    # Add new features to original dataframe
                    for col in poly_df.columns:
                        col_name = col.replace(' ', '_').replace('^', '_pow_')
                        st.session_state.df[col_name] = poly_df[col]
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created polynomial features (degree {degree}) from {', '.join(poly_cols)}",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "base_columns": poly_cols,
                            "degree": degree,
                            "include_interaction": include_interaction,
                            "new_columns_count": poly_df.shape[1]
                        }
                    })
                    
                    st.success(f"Created {poly_df.shape[1]} new polynomial feature columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating polynomial features: {str(e)}")
        
        elif interaction_type == "Ratio/Proportion":
            st.write("Create ratio features from related numeric columns.")
            
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for ratio features.")
                return
                
            # Select numerator and denominator
            numerator_col = st.selectbox("Select numerator column:", num_cols, key="ratio_num")
            denominator_col = st.selectbox(
                "Select denominator column:",
                [col for col in num_cols if col != numerator_col],
                key="ratio_denom"
            )
            
            # Handle zero values in denominator
            handle_zeros = st.selectbox(
                "Handle zeros in denominator:",
                ["Replace with NaN", "Replace with small value (1e-6)", "Add small value (1e-6)"]
            )
            
            # New column name
            new_col_name = st.text_input(
                "New column name:",
                value=f"{numerator_col}_to_{denominator_col}_ratio"
            )
            
            # Apply button
            if st.button("Create Ratio Feature", key="ratio_feature", use_container_width=True):
                try:
                    # Handle zeros in denominator
                    if handle_zeros == "Replace with NaN":
                        denominator = st.session_state.df[denominator_col].replace(0, np.nan)
                    elif handle_zeros == "Replace with small value (1e-6)":
                        denominator = st.session_state.df[denominator_col].replace(0, 1e-6)
                    else:  # Add small value
                        denominator = st.session_state.df[denominator_col] + 1e-6
                    
                    # Calculate ratio
                    st.session_state.df[new_col_name] = st.session_state.df[numerator_col] / denominator
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created ratio feature '{new_col_name}': {numerator_col} / {denominator_col}",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "numerator": numerator_col,
                            "denominator": denominator_col,
                            "zero_handling": handle_zeros,
                            "result_column": new_col_name
                        }
                    })
                    
                    st.success(f"Created new ratio column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating ratio feature: {str(e)}")
    
    # Advanced Features tab
    with fe_tabs[3]:
        st.markdown("### Advanced Features")
        st.write("Create advanced features using machine learning techniques and statistical methods.")
        
        # Feature type selection
        feature_type = st.selectbox(
            "Select advanced feature type:",
            [
                "PCA Components",
                "Clustering Labels",
                "Anomaly Scores",
                "Distance Features",
                "Time-window Statistics"
            ]
        )
        
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Need numeric columns for advanced feature engineering.")
            return
        
        if feature_type == "PCA Components":
            st.write("Create principal components from numeric features.")
            
            # Select columns for PCA
            pca_cols = st.multiselect(
                "Select columns for PCA:",
                num_cols,
                default=num_cols[:min(5, len(num_cols))]
            )
            
            if len(pca_cols) < 2:
                st.info("Need at least two columns for PCA.")
                return
                
            # Number of components
            n_components = st.slider(
                "Number of components:", 
                min_value=1, 
                max_value=min(len(pca_cols), 10),
                value=min(2, len(pca_cols))
            )
            
            # Apply button
            if st.button("Create PCA Features", key="pca_features", use_container_width=True):
                try:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    # Get selected data and handle missing values
                    X = st.session_state.df[pca_cols].fillna(0)
                    
                    # Standardize data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply PCA
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(X_scaled)
                    
                    # Create new columns with PCA results
                    for i in range(n_components):
                        st.session_state.df[f"pca_component_{i+1}"] = pca_result[:, i]
                    
                    # Calculate explained variance
                    explained_variance = pca.explained_variance_ratio_
                    total_variance = sum(explained_variance)
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created {n_components} PCA components from {len(pca_cols)} columns",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "base_columns": pca_cols,
                            "n_components": n_components,
                            "explained_variance": explained_variance.tolist(),
                            "total_variance": total_variance
                        }
                    })
                    
                    # Success message with variance explained
                    st.success(f"Created {n_components} PCA components capturing {total_variance:.2%} of variance")
                    
                    # Show variance explained
                    st.write("Variance explained by each component:")
                    for i, var in enumerate(explained_variance):
                        st.write(f"Component {i+1}: {var:.2%}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating PCA features: {str(e)}")
                
        elif feature_type == "Clustering Labels":
            st.write("Generate cluster labels from numeric features.")
            
            # Select columns for clustering
            cluster_cols = st.multiselect(
                "Select columns for clustering:",
                num_cols,
                default=num_cols[:min(3, len(num_cols))]
            )
            
            if len(cluster_cols) < 1:
                st.info("Need at least one column for clustering.")
                return
                
            # Clustering algorithm
            algorithm = st.selectbox(
                "Clustering algorithm:",
                ["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"]
            )
            
            # Parameters based on algorithm
            if algorithm == "K-Means":
                n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                param_info = {"n_clusters": n_clusters}
                
            elif algorithm == "DBSCAN":
                eps = st.slider("Epsilon (neighborhood distance):", 0.1, 5.0, 0.5)
                min_samples = st.slider("Min samples:", 2, 10, 5)
                param_info = {"eps": eps, "min_samples": min_samples}
                
            else:  # Hierarchical
                n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                linkage = st.selectbox("Linkage:", ["ward", "complete", "average", "single"])
                param_info = {"n_clusters": n_clusters, "linkage": linkage}
            
            # New column name
            new_col_name = st.text_input(
                "New column name:",
                value=f"{algorithm.lower()}_cluster"
            )
            
            # Apply button
            if st.button("Create Cluster Labels", key="cluster_labels", use_container_width=True):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
                    
                    # Get selected data and handle missing values
                    X = st.session_state.df[cluster_cols].fillna(0)
                    
                    # Standardize data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply clustering algorithm
                    if algorithm == "K-Means":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    elif algorithm == "DBSCAN":
                        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                    else:  # Hierarchical
                        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    
                    # Get cluster labels
                    cluster_labels = clusterer.fit_predict(X_scaled)
                    
                    # Add to dataframe
                    st.session_state.df[new_col_name] = cluster_labels
                    
                    # For DBSCAN, count clusters (excluding noise points labeled as -1)
                    if algorithm == "DBSCAN":
                        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        n_noise = list(cluster_labels).count(-1)
                        cluster_info = f"{n_clusters_found} clusters found, {n_noise} noise points"
                    else:
                        cluster_info = f"{n_clusters} clusters"
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created cluster labels using {algorithm} on {len(cluster_cols)} columns ({cluster_info})",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "base_columns": cluster_cols,
                            "algorithm": algorithm,
                            "parameters": param_info,
                            "result_column": new_col_name
                        }
                    })
                    
                    st.success(f"Created cluster labels column '{new_col_name}' ({cluster_info})")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating cluster labels: {str(e)}")
        
        elif feature_type == "Anomaly Scores":
            st.write("Generate anomaly scores to detect outliers in the dataset.")
            
            # Select columns for anomaly detection
            anomaly_cols = st.multiselect(
                "Select columns for anomaly detection:",
                num_cols,
                default=num_cols[:min(3, len(num_cols))]
            )
            
            if len(anomaly_cols) < 1:
                st.info("Need at least one column for anomaly detection.")
                return
                
            # Anomaly detection method
            method = st.selectbox(
                "Anomaly detection method:",
                ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]
            )
            
            # New column name
            new_col_name = st.text_input(
                "New column name:",
                value=f"{method.lower().replace(' ', '_')}_anomaly_score"
            )
            
            # Apply button
            if st.button("Create Anomaly Scores", key="anomaly_scores", use_container_width=True):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.ensemble import IsolationForest
                    from sklearn.neighbors import LocalOutlierFactor
                    from sklearn.svm import OneClassSVM
                    
                    # Get selected data and handle missing values
                    X = st.session_state.df[anomaly_cols].fillna(0)
                    
                    # Standardize data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply anomaly detection
                    if method == "Isolation Forest":
                        detector = IsolationForest(random_state=42, contamination='auto')
                        # For Isolation Forest, scores are in [-1, 1] where -1 is an outlier
                        scores = detector.fit_predict(X_scaled)
                        # Convert to anomaly score (higher is more anomalous)
                        anomaly_scores = 1 - (scores + 1) / 2
                        
                    elif method == "Local Outlier Factor":
                        detector = LocalOutlierFactor(novelty=True)
                        detector.fit(X_scaled)
                        # For LOF, negative scores indicate outliers
                        scores = -detector.decision_function(X_scaled)
                        # Normalize to [0, 1] for consistency
                        anomaly_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else scores
                        
                    else:  # One-Class SVM
                        detector = OneClassSVM(gamma='auto')
                        detector.fit(X_scaled)
                        # For OCSVM, negative scores indicate outliers
                        scores = -detector.decision_function(X_scaled)
                        # Normalize to [0, 1] for consistency
                        anomaly_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else scores
                    
                    # Add to dataframe
                    st.session_state.df[new_col_name] = anomaly_scores
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created anomaly scores using {method} on {len(anomaly_cols)} columns",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "base_columns": anomaly_cols,
                            "method": method,
                            "result_column": new_col_name
                        }
                    })
                    
                    st.success(f"Created anomaly scores column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating anomaly scores: {str(e)}")
        
        elif feature_type == "Distance Features":
            st.write("Calculate the distance between data points and reference points.")
            
            # Select columns for distance calculation
            distance_cols = st.multiselect(
                "Select columns for distance calculation:",
                num_cols,
                default=num_cols[:min(2, len(num_cols))]
            )
            
            if len(distance_cols) < 1:
                st.info("Need at least one column for distance calculation.")
                return
                
            # Distance method
            distance_method = st.selectbox(
                "Distance method:",
                ["Euclidean", "Manhattan", "Cosine similarity"]
            )
            
            # Reference point selection
            reference_type = st.radio(
                "Reference point type:",
                ["Mean", "Median", "Min", "Max", "Custom"]
            )
            
            # For custom reference point
            if reference_type == "Custom":
                ref_point = {}
                st.write("Enter values for reference point:")
                
                for col in distance_cols:
                    col_min = float(st.session_state.df[col].min())
                    col_max = float(st.session_state.df[col].max())
                    ref_point[col] = st.slider(
                        f"Value for {col}:",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min + col_max) / 2
                    )
            
            # New column name
            new_col_name = st.text_input(
                "New column name:",
                value=f"{distance_method.lower()}_distance_to_{reference_type.lower()}"
            )
            
            # Apply button
            if st.button("Create Distance Feature", key="distance_feature", use_container_width=True):
                try:
                    import scipy.spatial.distance as distance
                    
                    # Get selected data and handle missing values
                    X = st.session_state.df[distance_cols].fillna(0)
                    
                    # Determine reference point based on selection
                    if reference_type == "Mean":
                        ref_point = X.mean().to_dict()
                    elif reference_type == "Median":
                        ref_point = X.median().to_dict()
                    elif reference_type == "Min":
                        ref_point = X.min().to_dict()
                    elif reference_type == "Max":
                        ref_point = X.max().to_dict()
                    # For "Custom", ref_point is already defined
                    
                    # Create reference point array in same order as columns
                    ref_array = np.array([ref_point[col] for col in distance_cols])
                    
                    # Calculate distances
                    if distance_method == "Euclidean":
                        distances = np.sqrt(np.sum((X - ref_array) ** 2, axis=1))
                    elif distance_method == "Manhattan":
                        distances = np.sum(np.abs(X - ref_array), axis=1)
                    else:  # Cosine similarity
                        # For cosine similarity, higher values (closer to 1) indicate similarity
                        # We convert to distance (1 - similarity) so higher values indicate dissimilarity, consistent with other metrics
                        similarities = np.array([
                            1 - distance.cosine(row, ref_array) if not np.all(row == 0) and not np.all(ref_array == 0) else 0
                            for _, row in X.iterrows()
                        ])
                        distances = 1 - similarities
                    
                    # Add to dataframe
                    st.session_state.df[new_col_name] = distances
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Created {distance_method} distance to {reference_type} point using {len(distance_cols)} columns",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "base_columns": distance_cols,
                            "distance_method": distance_method,
                            "reference_type": reference_type,
                            "reference_point": ref_point,
                            "result_column": new_col_name
                        }
                    })
                    
                    st.success(f"Created distance feature column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating distance feature: {str(e)}")
        
        elif feature_type == "Time-window Statistics":
            st.write("Calculate statistics over time windows (requires date/time column).")
            
            # Get datetime columns
            datetime_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Also check for other columns that might be dates
            potential_date_cols = []
            for col in st.session_state.df.columns:
                if col in datetime_cols:
                    continue
                
                # Look for date-related names
                if any(term in col.lower() for term in ["date", "time", "year"]):
                    potential_date_cols.append(col)
            
            all_date_cols = datetime_cols + potential_date_cols
            
            if not all_date_cols:
                st.info("Need at least one date/time column for time-window statistics.")
                return
                
            # Select date column
            date_col = st.selectbox("Select date/time column:", all_date_cols)
            
            # Convert to datetime if needed
            if date_col in potential_date_cols:
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[date_col]):
                    st.warning(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid date/time values.")
            
            # Select column for statistics
            stat_col = st.selectbox(
                "Select column for statistics:",
                [col for col in num_cols if col != date_col] if date_col in num_cols else num_cols
            )
            
            # Window type and size
            window_type = st.selectbox("Window type:", ["Rolling", "Expanding"])
            
            if window_type == "Rolling":
                window_size = st.slider("Window size (periods):", 2, 30, 7)
            
            # Statistics to calculate
            stats_to_calc = st.multiselect(
                "Statistics to calculate:",
                ["Mean", "Median", "Min", "Max", "Std", "Count", "Sum"],
                default=["Mean"]
            )
            
            # Apply button
            if st.button("Create Time-window Features", key="timewindow_features", use_container_width=True):
                try:
                    # Ensure date column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[date_col]):
                        st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], errors='coerce')
                    
                    # Sort by date
                    df_sorted = st.session_state.df.sort_values(date_col)
                    
                    # Map stat names to pandas functions
                    stat_map = {
                        "Mean": "mean",
                        "Median": "median",
                        "Min": "min",
                        "Max": "max",
                        "Std": "std",
                        "Count": "count",
                        "Sum": "sum"
                    }
                    
                    # Calculate time-window statistics
                    new_columns = []
                    
                    for stat in stats_to_calc:
                        # Determine column name
                        if window_type == "Rolling":
                            col_name = f"{stat_col}_{window_type.lower()}_{window_size}_{stat.lower()}"
                        else:
                            col_name = f"{stat_col}_{window_type.lower()}_{stat.lower()}"
                        
                        # Calculate statistic
                        if window_type == "Rolling":
                            if stat_map[stat] == "median":
                                # For median, we need to use apply since it's not a direct method
                                stat_result = df_sorted[stat_col].rolling(window=window_size).apply(
                                    lambda x: np.median(x) if len(x) > 0 else np.nan, raw=True
                                )
                            else:
                                # Use direct method for other stats
                                stat_result = getattr(df_sorted[stat_col].rolling(window=window_size), stat_map[stat])()
                        else:  # Expanding
                            if stat_map[stat] == "median":
                                # For median, we need to use apply
                                stat_result = df_sorted[stat_col].expanding().apply(
                                    lambda x: np.median(x) if len(x) > 0 else np.nan, raw=True
                                )
                            else:
                                # Use direct method
                                stat_result = getattr(df_sorted[stat_col].expanding(), stat_map[stat])()
                        
                        # Align result with original dataframe by index
                        st.session_state.df[col_name] = stat_result
                        
                        # Add to new columns list
                        new_columns.append(col_name)
                    
                    # Add to processing history
                    window_desc = f"{window_size}-period rolling" if window_type == "Rolling" else "expanding"
                    
                    st.session_state.processing_history.append({
                        "description": f"Created {len(stats_to_calc)} {window_desc} statistics for '{stat_col}'",
                        "timestamp": datetime.datetime.now(),
                        "type": "feature_engineering",
                        "details": {
                            "date_column": date_col,
                            "value_column": stat_col,
                            "window_type": window_type,
                            "window_size": window_size if window_type == "Rolling" else "N/A",
                            "statistics": stats_to_calc,
                            "new_columns": new_columns
                        }
                    })
                    
                    st.success(f"Created {len(stats_to_calc)} time-window statistic columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating time-window features: {str(e)}")
    
    # Custom Formula tab
    with fe_tabs[4]:
        st.markdown("### Custom Formula")
        st.write("Create new features using custom formulas or expressions.")
        
        # Available columns
        available_cols = st.session_state.df.columns.tolist()
        st.write("**Available columns:**")
        st.text(", ".join(available_cols))
        
        # Custom formula input
        st.markdown("""
        **Formula syntax:**
        - Use column names directly: `column_name`
        - Basic operations: `+`, `-`, `*`, `/`, `**` (power)
        - Functions: `np.log()`, `np.sqrt()`, `np.sin()`, etc.
        - Conditional: `np.where(condition, value_if_true, value_if_false)`
        
        **Examples:**
        - `column_a + column_b`
        - `np.log(column_a) - 2 * column_b`
        - `np.where(column_a > 10, 1, 0)`
        - `(column_a - column_a.mean()) / column_a.std()`
        """)
        
        # Formula input
        formula = st.text_area(
            "Enter your formula:",
            placeholder="e.g., np.log(column_a) + column_b / 100"
        )
        
        # New column name
        new_col_name = st.text_input(
            "New column name:",
            value="custom_feature"
        )
        
        # Apply button
        if st.button("Create Custom Feature", key="custom_formula", use_container_width=True):
            if not formula:
                st.error("Please enter a formula.")
                return
                
            try:
                # Prepare local namespace with dataframe columns
                local_dict = {col: st.session_state.df[col] for col in st.session_state.df.columns}
                
                # Add NumPy for advanced functions
                local_dict['np'] = np
                
                # Evaluate the formula
                result = eval(formula, {"__builtins__": {}}, local_dict)
                
                # Add result to dataframe
                st.session_state.df[new_col_name] = result
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Created custom feature '{new_col_name}' using formula",
                    "timestamp": datetime.datetime.now(),
                    "type": "feature_engineering",
                    "details": {
                        "formula": formula,
                        "result_column": new_col_name
                    }
                })
                
                st.success(f"Created new column '{new_col_name}' using custom formula")
                
                # Show statistics about the result
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Mean", f"{st.session_state.df[new_col_name].mean():.2f}")
                with stat_cols[1]:
                    st.metric("Min", f"{st.session_state.df[new_col_name].min():.2f}")
                with stat_cols[2]:
                    st.metric("Max", f"{st.session_state.df[new_col_name].max():.2f}")
                with stat_cols[3]:
                    st.metric("Missing Values", f"{st.session_state.df[new_col_name].isna().sum()}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error evaluating formula: {str(e)}")
                st.info("Check that your formula uses valid column names and operations.")

def render_data_filtering():
    """Render data filtering interface"""
    st.subheader("Data Filtering")
    
    # Make sure we have data
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("Please upload a dataset to begin data filtering.")
        return
    
    # Create columns for organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Filter by Conditions")
        
        # Get all columns
        all_cols = st.session_state.df.columns.tolist()
        
        # Allow adding multiple filter conditions
        conditions = []
        
        # Add first condition
        st.markdown("#### Add Filter Conditions")
        
        # Container for conditions
        condition_container = st.container()
        
        with condition_container:
            # Condition 1
            col1a, col1b, col1c = st.columns([2, 1, 2])
            with col1a:
                filter_col = st.selectbox("Column", all_cols, key="filter_col_1")
            
            with col1b:
                # Choose operator based on column type
                if st.session_state.df[filter_col].dtype.kind in 'bifc':
                    # Numeric
                    operators = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                elif st.session_state.df[filter_col].dtype.kind in 'M':
                    # Datetime
                    operators = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                else:
                    # String/object
                    operators = ["==", "!=", "contains", "starts with", "ends with", "is null", "is not null", "in"]
                
                filter_op = st.selectbox("Operator", operators, key="filter_op_1")
            
            with col1c:
                # Value input based on operator
                if filter_op == "between":
                    # Range values
                    if st.session_state.df[filter_col].dtype.kind in 'bifc':
                        # Numeric range
                        min_val, max_val = st.slider(
                            "Range", 
                            float(st.session_state.df[filter_col].min()), 
                            float(st.session_state.df[filter_col].max()),
                            (float(st.session_state.df[filter_col].min()), float(st.session_state.df[filter_col].max())),
                            key="filter_range_1"
                        )
                        filter_val = (min_val, max_val)
                    elif st.session_state.df[filter_col].dtype.kind in 'M':
                        # Date range
                        min_date = pd.to_datetime(st.session_state.df[filter_col].min()).date()
                        max_date = pd.to_datetime(st.session_state.df[filter_col].max()).date()
                        
                        # Default to last 30 days if range is large
                        default_min = max_date - pd.Timedelta(days=30)
                        
                        start_date = st.date_input("Start date", default_min, min_value=min_date, max_value=max_date, key="filter_date_start_1")
                        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date, key="filter_date_end_1")
                        filter_val = (pd.Timestamp(start_date), pd.Timestamp(end_date))
                    else:
                        # String input for range
                        filter_val1 = st.text_input("Min value", key="filter_min_1")
                        filter_val2 = st.text_input("Max value", key="filter_max_1")
                        filter_val = (filter_val1, filter_val2)
                elif filter_op in ["is null", "is not null"]:
                    # No value needed
                    filter_val = None
                    st.write("No value needed")
                elif filter_op == "in":
                    # Multiple values
                    filter_val = st.text_input("Values (comma-separated)", key="filter_in_1")
                else:
                    # Single value
                    if st.session_state.df[filter_col].dtype.kind in 'bifc':
                        # Numeric input
                        filter_val = st.number_input("Value", value=0, key="filter_val_1")
                    elif st.session_state.df[filter_col].dtype.kind in 'M':
                        # Date input
                        min_date = pd.to_datetime(st.session_state.df[filter_col].min()).date()
                        max_date = pd.to_datetime(st.session_state.df[filter_col].max()).date()
                        filter_val = st.date_input("Value", max_date, min_value=min_date, max_value=max_date, key="filter_date_1")
                        filter_val = pd.Timestamp(filter_val)
                    else:
                        # Get unique values if not too many
                        unique_vals = st.session_state.df[filter_col].dropna().unique()
                        if len(unique_vals) <= 20:
                            filter_val = st.selectbox("Value", unique_vals, key="filter_dropdown_1")
                        else:
                            filter_val = st.text_input("Value", key="filter_val_1")
            
            # Add this condition
            condition = {
                "column": filter_col,
                "operator": filter_op,
                "value": filter_val
            }
            conditions.append(condition)
        
        # Add more conditions if needed
        add_condition = st.checkbox("Add another condition")
        
        if add_condition:
            # Condition 2
            col2a, col2b, col2c = st.columns([2, 1, 2])
            with col2a:
                filter_col2 = st.selectbox("Column", all_cols, key="filter_col_2")
            
            with col2b:
                # Choose operator based on column type
                if st.session_state.df[filter_col2].dtype.kind in 'bifc':
                    # Numeric
                    operators2 = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                elif st.session_state.df[filter_col2].dtype.kind in 'M':
                    # Datetime
                    operators2 = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                else:
                    # String/object
                    operators2 = ["==", "!=", "contains", "starts with", "ends with", "is null", "is not null", "in"]
                
                filter_op2 = st.selectbox("Operator", operators2, key="filter_op_2")
            
            with col2c:
                # Value input based on operator
                if filter_op2 == "between":
                    # Range values
                    if st.session_state.df[filter_col2].dtype.kind in 'bifc':
                        # Numeric range
                        min_val2, max_val2 = st.slider(
                            "Range", 
                            float(st.session_state.df[filter_col2].min()), 
                            float(st.session_state.df[filter_col2].max()),
                            (float(st.session_state.df[filter_col2].min()), float(st.session_state.df[filter_col2].max())),
                            key="filter_range_2"
                        )
                        filter_val2 = (min_val2, max_val2)
                    elif st.session_state.df[filter_col2].dtype.kind in 'M':
                        # Date range
                        min_date2 = pd.to_datetime(st.session_state.df[filter_col2].min()).date()
                        max_date2 = pd.to_datetime(st.session_state.df[filter_col2].max()).date()
                        
                        # Default to last 30 days if range is large
                        default_min2 = max_date2 - pd.Timedelta(days=30)
                        
                        start_date2 = st.date_input("Start date", default_min2, min_value=min_date2, max_value=max_date2, key="filter_date_start_2")
                        end_date2 = st.date_input("End date", max_date2, min_value=min_date2, max_value=max_date2, key="filter_date_end_2")
                        filter_val2 = (pd.Timestamp(start_date2), pd.Timestamp(end_date2))
                    else:
                        # String input for range
                        filter_val2_1 = st.text_input("Min value", key="filter_min_2")
                        filter_val2_2 = st.text_input("Max value", key="filter_max_2")
                        filter_val2 = (filter_val2_1, filter_val2_2)
                elif filter_op2 in ["is null", "is not null"]:
                    # No value needed
                    filter_val2 = None
                    st.write("No value needed")
                elif filter_op2 == "in":
                    # Multiple values
                    filter_val2 = st.text_input("Values (comma-separated)", key="filter_in_2")
                else:
                    # Single value
                    if st.session_state.df[filter_col2].dtype.kind in 'bifc':
                        # Numeric input
                        filter_val2 = st.number_input("Value", value=0, key="filter_val_2")
                    elif st.session_state.df[filter_col2].dtype.kind in 'M':
                        # Date input
                        min_date2 = pd.to_datetime(st.session_state.df[filter_col2].min()).date()
                        max_date2 = pd.to_datetime(st.session_state.df[filter_col2].max()).date()
                        filter_val2 = st.date_input("Value", max_date2, min_value=min_date2, max_value=max_date2, key="filter_date_2")
                        filter_val2 = pd.Timestamp(filter_val2)
                    else:
                        # Get unique values if not too many
                        unique_vals2 = st.session_state.df[filter_col2].dropna().unique()
                        if len(unique_vals2) <= 20:
                            filter_val2 = st.selectbox("Value", unique_vals2, key="filter_dropdown_2")
                        else:
                            filter_val2 = st.text_input("Value", key="filter_val_2")
            
            # Add this condition
            condition2 = {
                "column": filter_col2,
                "operator": filter_op2,
                "value": filter_val2
            }
            conditions.append(condition2)
        
        # Condition combination
        if len(conditions) > 1:
            combine_op = st.radio("Combine conditions with:", ["AND", "OR"], horizontal=True)
        
        # Apply filters button
        if st.button("Apply Filters", use_container_width=True):
            try:
                # Store original shape for reporting
                orig_shape = st.session_state.df.shape
                
                # Create filter mask for each condition
                masks = []
                for condition in conditions:
                    col = condition["column"]
                    op = condition["operator"]
                    val = condition["value"]
                    
                    # Create the mask based on the operator
                    if op == "==":
                        mask = st.session_state.df[col] == val
                    elif op == "!=":
                        mask = st.session_state.df[col] != val
                    elif op == ">":
                        mask = st.session_state.df[col] > val
                    elif op == ">=":
                        mask = st.session_state.df[col] >= val
                    elif op == "<":
                        mask = st.session_state.df[col] < val
                    elif op == "<=":
                        mask = st.session_state.df[col] <= val
                    elif op == "between":
                        min_val, max_val = val
                        mask = (st.session_state.df[col] >= min_val) & (st.session_state.df[col] <= max_val)
                    elif op == "is null":
                        mask = st.session_state.df[col].isna()
                    elif op == "is not null":
                        mask = ~st.session_state.df[col].isna()
                    elif op == "contains":
                        mask = st.session_state.df[col].astype(str).str.contains(str(val), na=False)
                    elif op == "starts with":
                        mask = st.session_state.df[col].astype(str).str.startswith(str(val), na=False)
                    elif op == "ends with":
                        mask = st.session_state.df[col].astype(str).str.endswith(str(val), na=False)
                    elif op == "in":
                        # Split comma-separated values
                        in_vals = [v.strip() for v in str(val).split(",")]
                        mask = st.session_state.df[col].isin(in_vals)
                    else:
                        st.error(f"Unknown operator: {op}")
                        return
                    
                    masks.append(mask)
                
                # Combine masks
                if len(masks) == 1:
                    final_mask = masks[0]
                else:
                    if combine_op == "AND":
                        final_mask = masks[0]
                        for mask in masks[1:]:
                            final_mask = final_mask & mask
                    else:  # OR
                        final_mask = masks[0]
                        for mask in masks[1:]:
                            final_mask = final_mask | mask
                
                # Apply the filter
                st.session_state.df = st.session_state.df[final_mask]
                
                # Calculate how many rows were filtered out
                rows_filtered = orig_shape[0] - st.session_state.df.shape[0]
                
                # Add to processing history
                filter_details = {
                    "conditions": conditions,
                    "combine_operator": combine_op if len(conditions) > 1 else None,
                    "rows_filtered": rows_filtered,
                    "rows_remaining": st.session_state.df.shape[0]
                }
                
                st.session_state.processing_history.append({
                    "description": f"Filtered data: {rows_filtered} rows removed, {st.session_state.df.shape[0]} remaining",
                    "timestamp": datetime.datetime.now(),
                    "type": "filter",
                    "details": filter_details
                })
                
                st.success(f"Filter applied: {rows_filtered} rows removed, {st.session_state.df.shape[0]} remaining")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error applying filter: {str(e)}")
    
    with col2:
        st.markdown("### Sampling & Subset")
        
        # Sample methods
        sample_method = st.selectbox(
            "Sampling method:",
            ["Random Sample", "Stratified Sample", "Systematic Sample", "First/Last N Rows"]
        )
        
        if sample_method == "Random Sample":
            # Random sampling options
            sample_size_type = st.radio(
                "Sample size as:",
                ["Percentage", "Number of rows"],
                horizontal=True
            )
            
            if sample_size_type == "Percentage":
                sample_pct = st.slider("Percentage to sample:", 1, 100, 10)
                sample_size = int(len(st.session_state.df) * sample_pct / 100)
            else:
                sample_size = st.number_input(
                    "Number of rows to sample:",
                    min_value=1,
                    max_value=len(st.session_state.df),
                    value=min(100, len(st.session_state.df))
                )
            
            random_state = st.number_input("Random seed (for reproducibility):", value=42)
            
            if st.button("Apply Random Sampling", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Apply random sampling
                    st.session_state.df = st.session_state.df.sample(n=sample_size, random_state=random_state)
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Applied random sampling: {sample_size} rows selected",
                        "timestamp": datetime.datetime.now(),
                        "type": "sampling",
                        "details": {
                            "method": "random",
                            "sample_size": sample_size,
                            "original_rows": orig_shape[0],
                            "random_state": random_state
                        }
                    })
                    
                    st.success(f"Applied random sampling: {sample_size} rows selected")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying sampling: {str(e)}")
        
        elif sample_method == "Stratified Sample":
            # Stratified sampling options
            strat_col = st.selectbox(
                "Select column to stratify by:",
                st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            )
            
            sample_pct = st.slider("Percentage to sample from each stratum:", 1, 100, 10)
            
            if st.button("Apply Stratified Sampling", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Apply stratified sampling
                    sampled_dfs = []
                    for value, group in st.session_state.df.groupby(strat_col):
                        sample_size = int(len(group) * sample_pct / 100)
                        if sample_size > 0:
                            sampled_dfs.append(group.sample(n=min(sample_size, len(group))))
                    
                    # Combine samples
                    st.session_state.df = pd.concat(sampled_dfs)
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Applied stratified sampling by '{strat_col}': {st.session_state.df.shape[0]} rows selected",
                        "timestamp": datetime.datetime.now(),
                        "type": "sampling",
                        "details": {
                            "method": "stratified",
                            "stratify_column": strat_col,
                            "percentage": sample_pct,
                            "original_rows": orig_shape[0],
                            "sampled_rows": st.session_state.df.shape[0]
                        }
                    })
                    
                    st.success(f"Applied stratified sampling: {st.session_state.df.shape[0]} rows selected")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying stratified sampling: {str(e)}")
        
        elif sample_method == "Systematic Sample":
            # Systematic sampling options
            k = st.number_input(
                "Select every kth row:",
                min_value=2,
                max_value=len(st.session_state.df),
                value=min(10, len(st.session_state.df))
            )
            
            if st.button("Apply Systematic Sampling", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Apply systematic sampling
                    indices = range(0, len(st.session_state.df), k)
                    st.session_state.df = st.session_state.df.iloc[indices]
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Applied systematic sampling (every {k}th row): {len(st.session_state.df)} rows selected",
                        "timestamp": datetime.datetime.now(),
                        "type": "sampling",
                        "details": {
                            "method": "systematic",
                            "k": k,
                            "original_rows": orig_shape[0],
                            "sampled_rows": st.session_state.df.shape[0]
                        }
                    })
                    
                    st.success(f"Applied systematic sampling: {len(st.session_state.df)} rows selected")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying systematic sampling: {str(e)}")
        
        elif sample_method == "First/Last N Rows":
            # First/Last N rows options
            select_type = st.radio(
                "Select:",
                ["First N Rows", "Last N Rows"],
                horizontal=True
            )
            
            n_rows = st.number_input(
                "Number of rows:",
                min_value=1,
                max_value=len(st.session_state.df),
                value=min(100, len(st.session_state.df))
            )
            
            if st.button("Apply Selection", use_container_width=True):
                try:
                    # Store original shape for reporting
                    orig_shape = st.session_state.df.shape
                    
                    # Apply selection
                    if select_type == "First N Rows":
                        st.session_state.df = st.session_state.df.head(n_rows)
                        selection_type = "first"
                    else:
                        st.session_state.df = st.session_state.df.tail(n_rows)
                        selection_type = "last"
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Selected {selection_type} {n_rows} rows",
                        "timestamp": datetime.datetime.now(),
                        "type": "sampling",
                        "details": {
                            "method": selection_type,
                            "n_rows": n_rows,
                            "original_rows": orig_shape[0],
                            "sampled_rows": st.session_state.df.shape[0]
                        }
                    })
                    
                    st.success(f"Selected {selection_type} {n_rows} rows")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error selecting rows: {str(e)}")
        
        # Data preview after filtering/sampling
        st.markdown("### Data Preview")
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

def render_column_management():
    """Render column management interface"""
    st.subheader("Column Management")
    
    # Make sure we have data
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("Please upload a dataset to begin column management.")
        return
    
    # Create tabs for different column operations
    col_tabs = st.tabs([
        "Rename Columns",
        "Select/Drop Columns",
        "Reorder Columns",
        "Split & Merge Columns"
    ])
    
    # Rename Columns Tab
    with col_tabs[0]:
        st.markdown("### Rename Columns")
        
        # Column information
        st.markdown("Current column names:")
        
        # Create a dataframe with column information
        col_df = pd.DataFrame({
            'Current Name': st.session_state.df.columns,
            'Type': [str(dtype) for dtype in st.session_state.df.dtypes.values],
            'Non-Null Values': st.session_state.df.count().values,
            'Null Values': st.session_state.df.isna().sum().values
        })
        
        st.dataframe(col_df, use_container_width=True)
        
        # Rename options
        rename_method = st.radio(
            "Rename method:",
            ["Individual Columns", "Bulk Rename with Pattern"],
            horizontal=True
        )
        
        if rename_method == "Individual Columns":
            # Individual column renaming
            with st.form("rename_columns_form"):
                st.markdown("Enter new names for columns:")
                
                # Create multiple rows of column inputs
                rename_dict = {}
                
                # Limit to 5 columns at a time for better UX
                for i in range(min(5, len(st.session_state.df.columns))):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        old_col = st.selectbox(
                            f"Column {i+1}:",
                            [col for col in st.session_state.df.columns if col not in rename_dict.keys()],
                            key=f"old_col_{i}"
                        )
                    
                    with col2:
                        new_col = st.text_input(
                            "New name:",
                            value=old_col,
                            key=f"new_col_{i}"
                        )
                    
                    if old_col and new_col and old_col != new_col:
                        rename_dict[old_col] = new_col
                
                # Submit button
                submit_button = st.form_submit_button("Rename Columns", use_container_width=True)
                
            if submit_button and rename_dict:
                try:
                    # Rename columns
                    st.session_state.df = st.session_state.df.rename(columns=rename_dict)
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Renamed {len(rename_dict)} columns",
                        "timestamp": datetime.datetime.now(),
                        "type": "column_management",
                        "details": {
                            "operation": "rename",
                            "rename_map": rename_dict
                        }
                    })
                    
                    st.success(f"Renamed {len(rename_dict)} columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error renaming columns: {str(e)}")
        
        else:  # Bulk Rename with Pattern
            # Bulk renaming options
            st.markdown("### Bulk Rename Columns")
            
            # Pattern options
            pattern_type = st.selectbox(
                "Pattern type:",
                ["Add Prefix", "Add Suffix", "Remove Text", "Replace Text", "Change Case", "Strip Whitespace"]
            )
            
            if pattern_type == "Add Prefix":
                prefix = st.text_input("Prefix to add:")
                preview = {col: f"{prefix}{col}" for col in st.session_state.df.columns[:3]}
                
            elif pattern_type == "Add Suffix":
                suffix = st.text_input("Suffix to add:")
                preview = {col: f"{col}{suffix}" for col in st.session_state.df.columns[:3]}
                
            elif pattern_type == "Remove Text":
                text_to_remove = st.text_input("Text to remove:")
                preview = {col: col.replace(text_to_remove, "") for col in st.session_state.df.columns[:3]}
                
            elif pattern_type == "Replace Text":
                find_text = st.text_input("Find text:")
                replace_text = st.text_input("Replace with:")
                preview = {col: col.replace(find_text, replace_text) for col in st.session_state.df.columns[:3]}
                
            elif pattern_type == "Change Case":
                case_option = st.selectbox(
                    "Convert to:",
                    ["lowercase", "UPPERCASE", "Title Case", "snake_case", "camelCase"]
                )
                
                # Show preview
                if case_option == "lowercase":
                    preview = {col: col.lower() for col in st.session_state.df.columns[:3]}
                elif case_option == "UPPERCASE":
                    preview = {col: col.upper() for col in st.session_state.df.columns[:3]}
                elif case_option == "Title Case":
                    preview = {col: col.title() for col in st.session_state.df.columns[:3]}
                elif case_option == "snake_case":
                    preview = {col: col.lower().replace(" ", "_") for col in st.session_state.df.columns[:3]}
                elif case_option == "camelCase":
                    preview = {col: ''.join([s.capitalize() if i > 0 else s.lower() 
                                            for i, s in enumerate(col.split())]) 
                              for col in st.session_state.df.columns[:3]}
            
            elif pattern_type == "Strip Whitespace":
                preview = {col: col.strip() for col in st.session_state.df.columns[:3]}
            
            # Show preview of changes
            st.markdown("### Preview (first 3 columns)")
            preview_df = pd.DataFrame({
                'Original': list(preview.keys()),
                'New': list(preview.values())
            })
            st.dataframe(preview_df, use_container_width=True)
            
            # Column selection for bulk rename
            apply_to = st.radio(
                "Apply to:",
                ["All Columns", "Selected Columns"],
                horizontal=True
            )
            
            selected_cols = None
            if apply_to == "Selected Columns":
                selected_cols = st.multiselect(
                    "Select columns to rename:",
                    st.session_state.df.columns.tolist()
                )
            
            # Apply button
            if st.button("Apply Bulk Rename", use_container_width=True):
                try:
                    # Determine which columns to rename
                    cols_to_rename = selected_cols if apply_to == "Selected Columns" else st.session_state.df.columns
                    
                    # Create rename mapping
                    rename_dict = {}
                    
                    for col in cols_to_rename:
                        if pattern_type == "Add Prefix":
                            new_col = f"{prefix}{col}"
                        elif pattern_type == "Add Suffix":
                            new_col = f"{col}{suffix}"
                        elif pattern_type == "Remove Text":
                            new_col = col.replace(text_to_remove, "")
                        elif pattern_type == "Replace Text":
                            new_col = col.replace(find_text, replace_text)
                        elif pattern_type == "Change Case":
                            if case_option == "lowercase":
                                new_col = col.lower()
                            elif case_option == "UPPERCASE":
                                new_col = col.upper()
                            elif case_option == "Title Case":
                                new_col = col.title()
                            elif case_option == "snake_case":
                                new_col = col.lower().replace(" ", "_")
                            elif case_option == "camelCase":
                                new_col = ''.join([s.capitalize() if i > 0 else s.lower() 
                                                  for i, s in enumerate(col.split())])
                        elif pattern_type == "Strip Whitespace":
                            new_col = col.strip()
                        
                        # Only add to rename dict if the name actually changed
                        if col != new_col:
                            rename_dict[col] = new_col
                    
                    if rename_dict:
                        # Rename columns
                        
                        st.session_state.df = st.session_state.df.rename(columns=rename_dict)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Bulk renamed {len(rename_dict)} columns using {pattern_type}",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "bulk_rename",
                                "pattern_type": pattern_type,
                                "columns_affected": len(rename_dict)
                            }
                        })
                        
                        st.success(f"Renamed {len(rename_dict)} columns")
                        st.rerun()
                    else:
                        st.info("No column names were changed. Check your pattern settings.")
                    
                except Exception as e:
                    st.error(f"Error renaming columns: {str(e)}")
    
    # Select/Drop Columns Tab
    with col_tabs[1]:
        st.markdown("### Select or Drop Columns")
        
        # Column selection operation
        operation = st.radio(
            "Operation:",
            ["Keep Selected Columns", "Drop Selected Columns"],
            horizontal=True
        )
        
        # Column selection
        all_cols = st.session_state.df.columns.tolist()
        
        # Group columns by type for easier selection
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        other_cols = [col for col in all_cols if col not in num_cols + datetime_cols + cat_cols]
        
        # Selection method
        selection_method = st.radio(
            "Selection method:",
            ["Manual Selection", "Pattern Selection", "Data Type Selection"],
            horizontal=True
        )
        
        if selection_method == "Manual Selection":
            # Multiselect for columns
            selected_cols = st.multiselect(
                "Select columns:",
                all_cols,
                default=all_cols
            )
            
        elif selection_method == "Pattern Selection":
            # Text pattern for selection
            pattern = st.text_input("Enter pattern (e.g., 'date' or 'amount'):")
            match_case = st.checkbox("Match case", value=False)
            
            if pattern:
                if match_case:
                    selected_cols = [col for col in all_cols if pattern in col]
                else:
                    selected_cols = [col for col in all_cols if pattern.lower() in col.lower()]
                
                st.write(f"Matched columns ({len(selected_cols)}):")
                st.write(", ".join(selected_cols))
            else:
                selected_cols = []
                
        elif selection_method == "Data Type Selection":
            # Select by data type
            data_types = st.multiselect(
                "Select data types:",
                ["Numeric", "Datetime", "Categorical/Text", "Other"],
                default=["Numeric", "Datetime", "Categorical/Text", "Other"]
            )
            
            selected_cols = []
            if "Numeric" in data_types:
                selected_cols.extend(num_cols)
            if "Datetime" in data_types:
                selected_cols.extend(datetime_cols)
            if "Categorical/Text" in data_types:
                selected_cols.extend(cat_cols)
            if "Other" in data_types:
                selected_cols.extend(other_cols)
            
            st.write(f"Selected columns ({len(selected_cols)}):")
            st.write(", ".join(selected_cols))
        
        # Apply button
        if st.button("Apply Column Selection", use_container_width=True):
            if not selected_cols and operation == "Keep Selected Columns":
                st.error("Please select at least one column to keep.")
                return
                
            try:
                # Store original columns for reporting
                orig_cols = st.session_state.df.columns.tolist()
                
                # Apply selection
                if operation == "Keep Selected Columns":
                    st.session_state.df = st.session_state.df[selected_cols]
                    action = "kept"
                else:  # Drop
                    st.session_state.df = st.session_state.df.drop(columns=selected_cols)
                    action = "dropped"
                
                # Calculate how many columns were affected
                if operation == "Keep Selected Columns":
                    affected_cols = len(orig_cols) - len(selected_cols)
                else:
                    affected_cols = len(selected_cols)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"{action.title()} {affected_cols} columns, {len(st.session_state.df.columns)} columns remaining",
                    "timestamp": datetime.datetime.now(),
                    "type": "column_management",
                    "details": {
                        "operation": "keep" if operation == "Keep Selected Columns" else "drop",
                        "selection_method": selection_method,
                        "affected_columns": affected_cols,
                        "remaining_columns": len(st.session_state.df.columns)
                    }
                })
                
                st.success(f"{action.title()} {affected_cols} columns. {len(st.session_state.df.columns)} columns remaining.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error selecting columns: {str(e)}")
    
    # Reorder Columns Tab
    with col_tabs[2]:
        st.markdown("### Reorder Columns")
        
        # Get current columns
        current_cols = st.session_state.df.columns.tolist()
        
        # Reorder methods
        reorder_method = st.radio(
            "Reorder method:",
            ["Manual Reordering", "Alphabetical", "Move to Front/Back"],
            horizontal=True
        )
        
        if reorder_method == "Manual Reordering":
            # Manual drag and drop reordering
            st.markdown("Drag and drop columns to reorder:")
            
            # Since Streamlit doesn't have native drag-drop, simulate with a multiselect in order
            reordered_cols = st.multiselect(
                "Selected columns will appear in this order:",
                current_cols,
                default=current_cols
            )
            
            # Add any missing columns at the end
            missing_cols = [col for col in current_cols if col not in reordered_cols]
            if missing_cols:
                st.info(f"Unselected columns will be added at the end: {', '.join(missing_cols)}")
                final_cols = reordered_cols + missing_cols
            else:
                final_cols = reordered_cols
        
        elif reorder_method == "Alphabetical":
            # Sorting options
            sort_order = st.radio(
                "Sort order:",
                ["Ascending (A-Z)", "Descending (Z-A)"],
                horizontal=True
            )
            
            # Sort columns
            if sort_order == "Ascending (A-Z)":
                final_cols = sorted(current_cols)
            else:
                final_cols = sorted(current_cols, reverse=True)
            
            # Preview
            st.write("Preview of new order:")
            st.write(", ".join(final_cols[:5]) + ("..." if len(final_cols) > 5 else ""))
        
        elif reorder_method == "Move to Front/Back":
            # Move specific columns
            move_direction = st.radio(
                "Move direction:",
                ["Move to Front", "Move to Back"],
                horizontal=True
            )
            
            # Column selection
            move_cols = st.multiselect(
                "Select columns to move:",
                current_cols
            )
            
            if move_cols:
                # Calculate new order
                if move_direction == "Move to Front":
                    remaining_cols = [col for col in current_cols if col not in move_cols]
                    final_cols = move_cols + remaining_cols
                else:
                    remaining_cols = [col for col in current_cols if col not in move_cols]
                    final_cols = remaining_cols + move_cols
                
                # Preview
                st.write("Preview of new order:")
                st.write(", ".join(final_cols[:5]) + ("..." if len(final_cols) > 5 else ""))
            else:
                final_cols = current_cols
                st.info("Please select columns to move.")
        
        # Apply button
        if st.button("Apply Column Reordering", use_container_width=True):
            try:
                # Reorder columns
                st.session_state.df = st.session_state.df[final_cols]
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Reordered columns using {reorder_method}",
                    "timestamp": datetime.datetime.now(),
                    "type": "column_management",
                    "details": {
                        "operation": "reorder",
                        "method": reorder_method
                    }
                })
                
                st.success("Columns reordered successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error reordering columns: {str(e)}")
    
    # Split & Merge Columns Tab
    with col_tabs[3]:
        st.markdown("### Split & Merge Columns")
        
        # Create subtabs for Split and Merge
        split_merge_tabs = st.tabs(["Split Column", "Merge Columns"])
        
        # Split Column Tab
        with split_merge_tabs[0]:
            st.markdown("#### Split one column into multiple columns")
            
            # Column selection
            text_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_cols:
                st.info("No text columns available for splitting.")
            else:
                split_col = st.selectbox(
                    "Select column to split:",
                    text_cols
                )
                
                # Sample values
                st.markdown("Sample values from selected column:")
                sample_values = st.session_state.df[split_col].dropna().head(3).tolist()
                for i, val in enumerate(sample_values):
                    st.text(f"Sample {i+1}: {val}")
                
                # Split method
                split_method = st.selectbox(
                    "Split method:",
                    ["Split by Delimiter", "Split by Position", "Regular Expression"]
                )
                
                if split_method == "Split by Delimiter":
                    delimiter = st.text_input("Delimiter:", ",")
                    max_splits = st.number_input("Maximum number of splits (-1 for all):", value=-1)
                    expand = st.checkbox("Create separate columns for each split", value=True)
                    
                elif split_method == "Split by Position":
                    positions = st.text_input("Enter positions to split at (comma-separated):", "3,6,9")
                    positions = [int(pos.strip()) for pos in positions.split(",") if pos.strip().isdigit()]
                    
                elif split_method == "Regular Expression":
                    regex_pattern = st.text_input("Regular expression pattern:", r"(\w+)")
                    st.caption("Use capture groups () to extract specific parts")
                
                # New column naming
                st.markdown("#### New Column Names")
                
                if split_method == "Split by Delimiter" and expand:
                    # Guess number of resulting columns from sample
                    if sample_values and delimiter:
                        num_cols = max([len(str(val).split(delimiter)) for val in sample_values])
                    else:
                        num_cols = 2
                    
                    prefix = st.text_input("New column prefix:", f"{split_col}_part")
                    st.write(f"Columns will be named: {prefix}0, {prefix}1, ... up to {prefix}{num_cols-1}")
                
                elif split_method == "Split by Position":
                    # One column for each segment
                    prefix = st.text_input("New column prefix:", f"{split_col}_pos")
                    num_cols = len(positions) + 1
                    col_names = [f"{prefix}{i}" for i in range(num_cols)]
                    st.write(f"Columns will be named: {', '.join(col_names)}")
                
                elif split_method == "Regular Expression":
                    # Depends on number of capture groups
                    prefix = st.text_input("New column prefix:", f"{split_col}_match")
                    st.write(f"Columns will be named based on regex capture groups: {prefix}0, {prefix}1, etc.")
                
                # Apply button
                if st.button("Split Column", use_container_width=True):
                    try:
                        if split_method == "Split by Delimiter":
                            # Apply split
                            if expand:
                                split_data = st.session_state.df[split_col].str.split(delimiter, n=max_splits, expand=True)
                                
                                # Rename columns
                                split_data.columns = [f"{prefix}{i}" for i in range(len(split_data.columns))]
                                
                                # Add to dataframe
                                for col in split_data.columns:
                                    st.session_state.df[col] = split_data[col]
                                
                                # Description for history
                                desc = f"Split '{split_col}' by delimiter '{delimiter}' into {len(split_data.columns)} columns"
                                
                            else:
                                # Split into list but don't expand
                                st.session_state.df[f"{split_col}_list"] = st.session_state.df[split_col].str.split(delimiter, n=max_splits)
                                
                                # Description for history
                                desc = f"Split '{split_col}' by delimiter '{delimiter}' into list column"
                            
                        elif split_method == "Split by Position":
                            # Create a function to split string by positions
                            def split_by_positions(text, pos_list):
                                if not isinstance(text, str):
                                    return [None] * (len(pos_list) + 1)
                                
                                result = []
                                start = 0
                                
                                for pos in sorted(pos_list):
                                    result.append(text[start:pos])
                                    start = pos
                                
                                result.append(text[start:])
                                return result
                            
                            # Apply the function
                            split_results = st.session_state.df[split_col].apply(
                                lambda x: split_by_positions(x, positions)
                            )
                            
                            # Create new columns
                            for i in range(len(positions) + 1):
                                col_name = f"{prefix}{i}"
                                st.session_state.df[col_name] = split_results.apply(lambda x: x[i] if len(x) > i else None)
                            
                            # Description for history
                            desc = f"Split '{split_col}' by positions {positions} into {len(positions) + 1} columns"
                            
                        elif split_method == "Regular Expression":
                            # Apply regex extraction
                            import re
                            
                            def extract_groups(text, pattern):
                                if not isinstance(text, str):
                                    return []
                                match = re.search(pattern, text)
                                if match:
                                    return [match.group(0)] + list(match.groups())
                                return []
                            
                            # Extract using regex
                            extracted = st.session_state.df[split_col].apply(
                                lambda x: extract_groups(x, regex_pattern)
                            )
                            
                            # Get max number of groups
                            max_groups = max(extracted.apply(len)) if not extracted.empty else 0
                            
                            # Create new columns
                            for i in range(max_groups):
                                col_name = f"{prefix}{i}"
                                st.session_state.df[col_name] = extracted.apply(lambda x: x[i] if len(x) > i else None)
                            
                            # Description for history
                            desc = f"Split '{split_col}' using regex pattern '{regex_pattern}' into {max_groups} columns"
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": desc,
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "split",
                                "method": split_method,
                                "source_column": split_col
                            }
                        })
                        
                        st.success(desc)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error splitting column: {str(e)}")
        
        # Merge Columns Tab
        with split_merge_tabs[1]:
            st.markdown("#### Merge multiple columns into one")
            
            # Column selection
            merge_cols = st.multiselect(
                "Select columns to merge:",
                st.session_state.df.columns.tolist(),
                default=st.session_state.df.columns[:min(2, len(st.session_state.df.columns))].tolist()
            )
            
            if len(merge_cols) < 2:
                st.info("Please select at least two columns to merge.")
            else:
                # Merge method
                merge_method = st.selectbox(
                    "Merge method:",
                    ["Concatenate with Separator", "Add/Subtract Values", "Join Text Columns"]
                )
                
                if merge_method == "Concatenate with Separator":
                    separator = st.text_input("Separator:", " ")
                    
                    # Preview
                    st.markdown("Preview:")
                    
                    # Create example of concatenation
                    if st.session_state.df.shape[0] > 0:
                        sample_row = st.session_state.df.iloc[0]
                        sample_values = [str(sample_row[col]) for col in merge_cols]
                        preview_value = separator.join(sample_values)
                        st.text(f"First row result: {preview_value}")
                    
                elif merge_method == "Add/Subtract Values":
                    # Check if columns are numeric
                    numeric_cols = st.session_state.df[merge_cols].select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) != len(merge_cols):
                        non_numeric = [col for col in merge_cols if col not in numeric_cols]
                        st.warning(f"Some selected columns are not numeric: {', '.join(non_numeric)}")
                    
                    operation = st.radio(
                        "Operation:",
                        ["Add", "Subtract", "Multiply", "Divide"],
                        horizontal=True
                    )
                    
                    # Preview
                    st.markdown("Preview:")
                    
                    # Create example of operation
                    if st.session_state.df.shape[0] > 0 and numeric_cols:
                        sample_row = st.session_state.df.iloc[0]
                        preview_value = sample_row[numeric_cols[0]]
                        
                        for col in numeric_cols[1:]:
                            if operation == "Add":
                                preview_value += sample_row[col]
                            elif operation == "Subtract":
                                preview_value -= sample_row[col]
                            elif operation == "Multiply":
                                preview_value *= sample_row[col]
                            elif operation == "Divide":
                                if sample_row[col] != 0:
                                    preview_value /= sample_row[col]
                                else:
                                    preview_value = "Division by zero error"
                                    break
                        
                        st.text(f"First row result: {preview_value}")
                
                elif merge_method == "Join Text Columns":
                    word_separator = st.checkbox("Add space between words", value=True)
                    remove_duplicates = st.checkbox("Remove duplicate words", value=False)
                    
                    # Preview
                    st.markdown("Preview:")
                    
                    # Create example of text join
                    if st.session_state.df.shape[0] > 0:
                        sample_row = st.session_state.df.iloc[0]
                        sample_texts = []
                        
                        for col in merge_cols:
                            text = str(sample_row[col])
                            words = text.split()
                            sample_texts.extend(words)
                        
                        if remove_duplicates:
                            sample_texts = list(dict.fromkeys(sample_texts))
                        
                        preview_value = " ".join(sample_texts) if word_separator else "".join(sample_texts)
                        st.text(f"First row result: {preview_value}")
                
                # New column name
                new_col_name = st.text_input(
                    "New column name:",
                    value="_".join([col.split('_')[0] for col in merge_cols[:2]]) + "_merged"
                )
                
                # Apply button
                if st.button("Merge Columns", use_container_width=True):
                    try:
                        if merge_method == "Concatenate with Separator":
                            # Convert all columns to string and concatenate
                            st.session_state.df[new_col_name] = st.session_state.df[merge_cols].astype(str).agg(
                                lambda x: separator.join(x), axis=1
                            )
                            
                            # Description
                            desc = f"Merged {len(merge_cols)} columns with separator '{separator}' into '{new_col_name}'"
                            
                        elif merge_method == "Add/Subtract Values":
                            # Use only numeric columns
                            numeric_cols = st.session_state.df[merge_cols].select_dtypes(include=['number']).columns.tolist()
                            
                            if not numeric_cols:
                                st.error("No numeric columns selected for mathematical operation.")
                                return
                            
                            # Apply selected operation
                            if operation == "Add":
                                st.session_state.df[new_col_name] = st.session_state.df[numeric_cols].sum(axis=1)
                            elif operation == "Subtract":
                                # Start with first column, then subtract the rest
                                result = st.session_state.df[numeric_cols[0]].copy()
                                for col in numeric_cols[1:]:
                                    result -= st.session_state.df[col]
                                st.session_state.df[new_col_name] = result
                            elif operation == "Multiply":
                                st.session_state.df[new_col_name] = st.session_state.df[numeric_cols].prod(axis=1)
                            elif operation == "Divide":
                                # Start with first column, then divide by the rest
                                result = st.session_state.df[numeric_cols[0]].copy()
                                for col in numeric_cols[1:]:
                                    # Avoid division by zero
                                    result = result.div(st.session_state.df[col].replace(0, np.nan))
                                st.session_state.df[new_col_name] = result
                            
                            # Description
                            desc = f"Merged {len(numeric_cols)} columns with {operation.lower()} operation into '{new_col_name}'"
                            
                        elif merge_method == "Join Text Columns":
                            # Function to join text
                            def join_text(row):
                                words = []
                                for col in merge_cols:
                                    text = str(row[col])
                                    words.extend(text.split())
                                
                                if remove_duplicates:
                                    words = list(dict.fromkeys(words))
                                
                                return " ".join(words) if word_separator else "".join(words)
                            
                            # Apply function
                            st.session_state.df[new_col_name] = st.session_state.df.apply(join_text, axis=1)
                            
                            # Description
                            sep_desc = "space-separated" if word_separator else "concatenated"
                            dup_desc = "removed duplicates" if remove_duplicates else "kept duplicates"
                            desc = f"Joined text from {len(merge_cols)} columns ({sep_desc}, {dup_desc}) into '{new_col_name}'"
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": desc,
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "merge",
                                "method": merge_method,
                                "source_columns": merge_cols,
                                "result_column": new_col_name
                            }
                        })
                        
                        st.success(desc)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error merging columns: {str(e)}")

def render_visualization():
    """Render visualization section"""
    st.header("Data Visualization")
    
    # Create visualization tabs
    viz_tabs = st.tabs([
        "Charts",
        "Interactive Plots",
        "Statistical Plots",
        "Geospatial Plots",
        "Custom Visualization"
    ])
    
    # Charts Tab
    with viz_tabs[0]:
        render_charts_tab()
    
    # Interactive Plots Tab
    with viz_tabs[1]:
        render_interactive_plots_tab()
    
    # Statistical Plots Tab
    with viz_tabs[2]:
        render_statistical_plots_tab()
    
    # Geospatial Plots Tab
    with viz_tabs[3]:
        render_geospatial_plots_tab()
    
    # Custom Visualization Tab
    with viz_tabs[4]:
        render_custom_viz_tab()

def render_charts_tab():
    """Render basic charts tab"""
    st.subheader("Basic Charts")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select chart type:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Area Chart", "Heatmap"]
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Column selectors based on chart type
    if chart_type == "Bar Chart":
        x_axis = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist())
        y_axis = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist())
        
        # Optional grouping
        use_grouping = st.checkbox("Group by another column")
        if use_grouping and len(cat_cols) > 1:
            group_col = st.selectbox("Group by:", [col for col in cat_cols if col != x_axis])
            
            # Limit number of categories to prevent overcrowding
            top_n = st.slider("Show top N categories:", 3, 20, 10)
            
            # Count values for each group
            value_counts = st.session_state.df.groupby([x_axis, group_col]).size().reset_index(name='Count')
            
            # Get top N categories for x_axis
            top_categories = st.session_state.df[x_axis].value_counts().nlargest(top_n).index.tolist()
            
            # Filter to only top categories
            value_counts = value_counts[value_counts[x_axis].isin(top_categories)]
            
            # Create grouped bar chart
            fig = px.bar(
                value_counts,
                x=x_axis,
                y='Count',
                color=group_col,
                title=f"Grouped Bar Chart: {x_axis} vs Count, grouped by {group_col}",
                barmode='group'
            )
        else:
            # Simple bar chart
            fig = px.bar(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Bar Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Line Chart":
        if date_cols:
            x_axis = st.selectbox("X-axis (typically date/time):", date_cols + num_cols)
        else:
            x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
        y_axis = st.selectbox("Y-axis (values):", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create line chart with grouping
            fig = px.line(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                color=group_col,
                title=f"Line Chart: {x_axis} vs {y_axis}, grouped by {group_col}",
                labels={x_axis: x_axis, y_axis: y_axis, group_col: group_col}
            )
        else:
            # Simple line chart
            fig = px.line(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Line Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist())
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols)
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols)
        size_col = st.selectbox("Size by:", ["None"] + num_cols)
        
        # Create scatter plot
        scatter_params = {
            "x": x_axis,
            "y": y_axis,
            "title": f"Scatter Plot: {x_axis} vs {y_axis}",
            "labels": {x_axis: x_axis, y_axis: y_axis}
        }
        
        if color_col != "None":
            scatter_params["color"] = color_col
        
        if size_col != "None":
            scatter_params["size"] = size_col
        
        fig = px.scatter(
            st.session_state.df,
            **scatter_params
        )
    
    elif chart_type == "Pie Chart":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist())
        names_col = st.selectbox("Category names:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
        
        # Limit number of categories to prevent overcrowding
        top_n = st.slider("Show top N categories:", 3, 20, 10)
        
        # Calculate values for pie chart
        value_counts = st.session_state.df.groupby(names_col)[value_col].sum().reset_index()
        value_counts = value_counts.sort_values(value_col, ascending=False).head(top_n)
        
        # Create pie chart
        fig = px.pie(
            value_counts,
            values=value_col,
            names=names_col,
            title=f"Pie Chart: {value_col} by {names_col} (Top {top_n})"
        )
    
    elif chart_type == "Histogram":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist())
        
        # Histogram options
        n_bins = st.slider("Number of bins:", 5, 100, 20)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create histogram with grouping
            fig = px.histogram(
                st.session_state.df,
                x=value_col,
                color=group_col,
                nbins=n_bins,
                title=f"Histogram of {value_col}, grouped by {group_col}",
                marginal="box"  # Add box plot to the top
            )
        else:
            # Simple histogram
            fig = px.histogram(
                st.session_state.df,
                x=value_col,
                nbins=n_bins,
                title=f"Histogram of {value_col}",
                marginal="box"  # Add box plot to the top
            )
    
    elif chart_type == "Box Plot":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist())
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create box plot with grouping
            fig = px.box(
                st.session_state.df,
                x=group_col,
                y=value_col,
                title=f"Box Plot of {value_col}, grouped by {group_col}"
            )
        else:
            # Simple box plot
            fig = px.box(
                st.session_state.df,
                y=value_col,
                title=f"Box Plot of {value_col}"
            )
    
    elif chart_type == "Area Chart":
        if date_cols:
            x_axis = st.selectbox("X-axis (typically date/time):", date_cols + num_cols)
        else:
            x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
        y_axis = st.selectbox("Y-axis (values):", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create area chart with grouping
            fig = px.area(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                color=group_col,
                title=f"Area Chart: {x_axis} vs {y_axis}, grouped by {group_col}",
                labels={x_axis: x_axis, y_axis: y_axis, group_col: group_col}
            )
        else:
            # Simple area chart
            fig = px.area(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Area Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Heatmap":
        if len(num_cols) < 1:
            st.warning("Need at least one numeric column for heatmap.")
            return
            
        # Column selections
        x_axis = st.selectbox("X-axis:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
        y_axis = st.selectbox("Y-axis:", [col for col in cat_cols if col != x_axis] if len(cat_cols) > 1 else cat_cols)
        value_col = st.selectbox("Value:", num_cols)
        
        # Limit categories to prevent overcrowding
        top_x = st.slider("Top X categories:", 3, 20, 10, key="heatmap_top_x")
        top_y = st.slider("Top Y categories:", 3, 20, 10, key="heatmap_top_y")
        
        # Get top categories
        top_x_cats = st.session_state.df[x_axis].value_counts().nlargest(top_x).index.tolist()
        top_y_cats = st.session_state.df[y_axis].value_counts().nlargest(top_y).index.tolist()
        
        # Filter data to top categories
        filtered_df = st.session_state.df[
            st.session_state.df[x_axis].isin(top_x_cats) & 
            st.session_state.df[y_axis].isin(top_y_cats)
        ]
        
        # Calculate values for heatmap
        heatmap_data = filtered_df.pivot_table(
            index=y_axis, 
            columns=x_axis, 
            values=value_col,
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x=x_axis, y=y_axis, color=value_col),
            title=f"Heatmap of {value_col} by {x_axis} and {y_axis}"
        )
    
    # Layout settings
    with st.expander("Chart Settings", expanded=False):
        # Title and labels
        chart_title = st.text_input("Chart title:", fig.layout.title.text)
        
        # Size settings
        width = st.slider("Chart width:", 400, 1200, 800)
        height = st.slider("Chart height:", 300, 1000, 500)
        
        # Update chart
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["PNG", "JPEG", "SVG", "HTML"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Chart", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_interactive_plots_tab():
    """Render interactive plots tab"""
    st.subheader("Interactive Plots")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Dynamic Scatter Plot", "Interactive Time Series", "3D Plot", "Multi-axis Plot", "Animated Chart"]
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    if not num_cols:
        st.warning("Need numeric columns for interactive plots.")
        return
    
    if plot_type == "Dynamic Scatter Plot":
        # Column selectors
        x_axis = st.selectbox("X-axis:", num_cols)
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis])
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols)
        size_col = st.selectbox("Size by:", ["None"] + num_cols)
        
        # Animation option
        animate_col = st.selectbox("Animate by:", ["None"] + cat_cols + date_cols)
        
        # Build plot parameters
        plot_params = {
            "x": x_axis,
            "y": y_axis,
            "title": f"Interactive Scatter Plot: {x_axis} vs {y_axis}",
            "labels": {x_axis: x_axis, y_axis: y_axis}
        }
        
        if color_col != "None":
            plot_params["color"] = color_col
        
        if size_col != "None":
            plot_params["size"] = size_col
            
            # Adjust size range
            plot_params["size_max"] = 30
        
        if animate_col != "None":
            plot_params["animation_frame"] = animate_col
        
        # Create the plot
        fig = px.scatter(
            st.session_state.df,
            **plot_params
        )
        
        # Add hover data for interactivity
        fig.update_traces(
            hovertemplate="<br>".join([
                f"{x_axis}: %{{x}}",
                f"{y_axis}: %{{y}}",
                "Click for more info"
            ])
        )
        
        # Make the plot interactive
        fig.update_layout(
            clickmode='event+select'
        )
    
    elif plot_type == "Interactive Time Series":
        if not date_cols:
            st.warning("No datetime columns found. Consider converting a column to datetime first.")
            time_col = st.selectbox("X-axis (time):", st.session_state.df.columns.tolist())
        else:
            time_col = st.selectbox("X-axis (time):", date_cols)
        
        # Select values to plot
        value_cols = st.multiselect(
            "Select values to plot:",
            num_cols,
            default=[num_cols[0]] if num_cols else []
        )
        
        if not value_cols:
            st.warning("Please select at least one value column to plot.")
            return
        
        # Create a figure with secondary y-axis if needed
        fig = go.Figure()
        
        # Add traces for each selected value
        for i, col in enumerate(value_cols):
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[time_col],
                    y=st.session_state.df[col],
                    name=col,
                    mode='lines+markers'
                )
            )
        
        # Add range selector for time series
        fig.update_layout(
            title=f"Interactive Time Series Plot",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
    
    elif plot_type == "3D Plot":
        # Ensure we have at least 3 numeric columns
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns for 3D plot.")
            return
            
        # Column selectors
        x_axis = st.selectbox("X-axis:", num_cols)
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis])
        z_axis = st.selectbox("Z-axis:", [col for col in num_cols if col not in [x_axis, y_axis]])
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols)
        
        # Plot types
        plot3d_type = st.radio(
            "3D Plot type:",
            ["Scatter", "Surface", "Line"],
            horizontal=True
        )
        
        if plot3d_type == "Scatter":
            # Create 3D scatter plot
            plot_params = {
                "x": x_axis,
                "y": y_axis,
                "z": z_axis,
                "title": f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}"
            }
            
            if color_col != "None":
                plot_params["color"] = color_col
            
            fig = px.scatter_3d(
                st.session_state.df,
                **plot_params
            )
        
        elif plot3d_type == "Surface":
            # For surface plot, we need to create a grid
            st.info("Surface plot requires data on a regular grid. Attempting to create grid from data...")
            
            # Try to create a grid
            try:
                # Get unique x and y values
                x_vals = st.session_state.df[x_axis].unique()
                y_vals = st.session_state.df[y_axis].unique()
                
                # Create grid
                grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                
                # Create z values grid
                grid_z = np.zeros(grid_x.shape)
                
                # Fill in the z values
                for i, x_val in enumerate(x_vals):
                    for j, y_val in enumerate(y_vals):
                        mask = (st.session_state.df[x_axis] == x_val) & (st.session_state.df[y_axis] == y_val)
                        if mask.any():
                            grid_z[j, i] = st.session_state.df.loc[mask, z_axis].mean()
                
                # Create surface plot
                fig = go.Figure(data=[go.Surface(z=grid_z, x=x_vals, y=y_vals)])
                fig.update_layout(
                    title=f"3D Surface Plot: {z_axis} as a function of {x_axis} and {y_axis}",
                    scene=dict(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        zaxis_title=z_axis
                    )
                )
            except Exception as e:
                st.error(f"Error creating surface plot: {str(e)}")
                st.info("Surface plot requires data on a regular grid. Try using scatter or line 3D plot instead.")
                return
        
        elif plot3d_type == "Line":
            # Create 3D line plot
            plot_params = {
                "x": x_axis,
                "y": y_axis,
                "z": z_axis,
                "title": f"3D Line Plot: {x_axis} vs {y_axis} vs {z_axis}"
            }
            
            if color_col != "None":
                plot_params["color"] = color_col
            
            fig = px.line_3d(
                st.session_state.df,
                **plot_params
            )
    
    elif plot_type == "Multi-axis Plot":
        # Select a common x-axis
        x_axis = st.selectbox("X-axis:", st.session_state.df.columns.tolist())
        
        # Select values for left y-axis
        left_y = st.selectbox("Left Y-axis:", num_cols)
        
        # Select values for right y-axis
        right_y = st.selectbox("Right Y-axis:", [col for col in num_cols if col != left_y])
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_axis],
                y=st.session_state.df[left_y],
                name=left_y,
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_axis],
                y=st.session_state.df[right_y],
                name=right_y,
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Multi-axis Plot: {left_y} and {right_y} vs {x_axis}"
        )
        
        # Update axes titles
        fig.update_xaxes(title_text=x_axis)
        fig.update_yaxes(title_text=left_y, secondary_y=False)
        fig.update_yaxes(title_text=right_y, secondary_y=True)
    
    elif plot_type == "Animated Chart":
        # Select columns
        x_axis = st.selectbox("X-axis:", num_cols)
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis])
        
        # Animation frame
        if date_cols:
            frame_col = st.selectbox("Animate by:", date_cols + cat_cols)
        else:
            frame_col = st.selectbox("Animate by:", cat_cols)
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols)
        size_col = st.selectbox("Size by:", ["None"] + num_cols)
        
        # Chart type
        anim_chart_type = st.radio(
            "Chart type:",
            ["Scatter", "Bar", "Line"],
            horizontal=True
        )
        
        # Build plot parameters
        plot_params = {
            "x": x_axis,
            "y": y_axis,
            "animation_frame": frame_col,
            "title": f"Animated {anim_chart_type} Chart: {x_axis} vs {y_axis} over {frame_col}"
        }
        
        if color_col != "None":
            plot_params["color"] = color_col
        
        if size_col != "None" and anim_chart_type == "Scatter":
            plot_params["size"] = size_col
            plot_params["size_max"] = 30
        
        # Create the plot
        if anim_chart_type == "Scatter":
            fig = px.scatter(st.session_state.df, **plot_params)
        elif anim_chart_type == "Bar":
            fig = px.bar(st.session_state.df, **plot_params)
        elif anim_chart_type == "Line":
            fig = px.line(st.session_state.df, **plot_params)
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text)
        width = st.slider("Chart width:", 400, 1200, 800)
        height = st.slider("Chart height:", 300, 1000, 600)
        
        # Theme selection
        theme = st.selectbox(
            "Color theme:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
        )
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            template=theme
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Chart", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"interactive_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"interactive_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_statistical_plots_tab():
    """Render statistical plots tab"""
    st.subheader("Statistical Plots")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Distribution Plot", "Correlation Matrix", "Pair Plot", "ECDF Plot", "Q-Q Plot", "Violin Plot", "Residual Plot"]
    )
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not num_cols:
        st.warning("Need numeric columns for statistical plots.")
        return
    
    if plot_type == "Distribution Plot":
        # Column selection
        dist_col = st.selectbox("Select column:", num_cols)
        
        # Plot options
        show_kde = st.checkbox("Show KDE", value=True)
        show_rug = st.checkbox("Show rug plot", value=False)
        
        # Number of bins
        n_bins = st.slider("Number of bins:", 5, 100, 30)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create distribution plot with grouping
            fig = px.histogram(
                st.session_state.df,
                x=dist_col,
                color=group_col,
                nbins=n_bins,
                title=f"Distribution of {dist_col}, grouped by {group_col}",
                marginal="box" if not show_kde else "violin",
            )
        else:
            # Simple distribution plot
            fig = px.histogram(
                st.session_state.df,
                x=dist_col,
                nbins=n_bins,
                title=f"Distribution of {dist_col}",
                marginal="box" if not show_kde else "violin",
            )
            
            if show_kde:
                # Add KDE plot
                try:
                    from scipy import stats
                    
                    # Get data for KDE
                    data = st.session_state.df[dist_col].dropna()
                    
                    # Calculate KDE
                    kde_x = np.linspace(data.min(), data.max(), 1000)
                    kde = stats.gaussian_kde(data)
                    kde_y = kde(kde_x)
                    
                    # Scale KDE to match histogram
                    hist, bin_edges = np.histogram(data, bins=n_bins)
                    scaling_factor = max(hist) / max(kde_y)
                    kde_y = kde_y * scaling_factor
                    
                    # Add KDE line
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        )
                    )
                except Exception as e:
                    st.warning(f"Could not calculate KDE: {str(e)}")
            
            if show_rug:
                # Add rug plot
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df[dist_col],
                        y=np.zeros(len(st.session_state.df)),
                        mode='markers',
                        marker=dict(symbol='line-ns', size=5, color='blue'),
                        name='Data points'
                    )
                )
    
    elif plot_type == "Correlation Matrix":
        # Column selection
        corr_cols = st.multiselect(
            "Select columns for correlation:",
            num_cols,
            default=num_cols[:min(5, len(num_cols))]
        )
        
        if not corr_cols:
            st.warning("Please select at least one column for correlation matrix.")
            return
        
        # Correlation method
        corr_method = st.radio(
            "Correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True
        )
        
        # Calculate correlation matrix
        corr_matrix = st.session_state.df[corr_cols].corr(method=corr_method.lower())
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f"{corr_method} Correlation Matrix",
            labels=dict(color="Correlation")
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
    
    elif plot_type == "Pair Plot":
        # Column selection
        pair_cols = st.multiselect(
            "Select columns for pair plot:",
            num_cols,
            default=num_cols[:min(3, len(num_cols))]
        )
        
        if len(pair_cols) < 2:
            st.warning("Please select at least two columns for pair plot.")
            return
        
        # Optional color column
        use_color = st.checkbox("Color by a column")
        color_col = None
        
        if use_color and cat_cols:
            color_col = st.selectbox("Color by:", cat_cols)
        
        # Create pair plot
        if color_col:
            fig = px.scatter_matrix(
                st.session_state.df,
                dimensions=pair_cols,
                color=color_col,
                title="Pair Plot"
            )
        else:
            fig = px.scatter_matrix(
                st.session_state.df,
                dimensions=pair_cols,
                title="Pair Plot"
            )
        
        # Update layout for better appearance
        fig.update_traces(diagonal_visible=False)
    
    elif plot_type == "ECDF Plot":
        # Column selection
        ecdf_col = st.selectbox("Select column for ECDF:", num_cols)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        group_col = None
        
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
        
        # Create ECDF plot
        if group_col:
            # Get unique groups
            groups = st.session_state.df[group_col].unique()
            
            # Create figure
            fig = go.Figure()
            
            # Add ECDF for each group
            for group in groups:
                data = st.session_state.df[st.session_state.df[group_col] == group][ecdf_col].dropna().sort_values()
                y = np.arange(1, len(data) + 1) / len(data)
                
                fig.add_trace(
                    go.Scatter(
                        x=data,
                        y=y,
                        mode='lines',
                        name=str(group)
                    )
                )
            
            fig.update_layout(
                title=f"ECDF Plot of {ecdf_col}, grouped by {group_col}",
                xaxis_title=ecdf_col,
                yaxis_title="Cumulative Probability"
            )
        else:
            # Single ECDF
            data = st.session_state.df[ecdf_col].dropna().sort_values()
            y = np.arange(1, len(data) + 1) / len(data)
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data,
                    y=y,
                    mode='lines',
                    name=ecdf_col
                )
            )
            
            fig.update_layout(
                title=f"ECDF Plot of {ecdf_col}",
                xaxis_title=ecdf_col,
                yaxis_title="Cumulative Probability"
            )
    
    elif plot_type == "Q-Q Plot":
        # Column selection
        qq_col = st.selectbox("Select column for Q-Q plot:", num_cols)
        
        # Distribution selection
        dist = st.selectbox(
            "Theoretical distribution:",
            ["Normal", "Uniform", "Exponential", "Log-normal"]
        )
        
        # Create Q-Q plot
        try:
            import scipy.stats as stats
            
            # Get data
            data = st.session_state.df[qq_col].dropna()
            
            # Calculate theoretical quantiles
            if dist == "Normal":
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Normal"
            elif dist == "Uniform":
                theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Uniform"
            elif dist == "Exponential":
                theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Exponential"
            elif dist == "Log-normal":
                theoretical_quantiles = stats.lognorm.ppf(np.linspace(0.01, 0.99, len(data)), s=1)
                theoretical_name = "Log-normal"
            
            # Sort the data
            sample_quantiles = np.sort(data)
            
            # Create Q-Q plot
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Data'
                )
            )
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Q-Q Plot: {qq_col} vs {theoretical_name} Distribution",
                xaxis_title=f"Theoretical Quantiles ({theoretical_name})",
                yaxis_title=f"Sample Quantiles ({qq_col})"
            )
            
        except Exception as e:
            st.error(f"Error creating Q-Q plot: {str(e)}")
            return
    
    elif plot_type == "Violin Plot":
        # Column selection
        violin_col = st.selectbox("Select column for values:", num_cols)
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols)
            
            # Create violin plot with grouping
            fig = px.violin(
                st.session_state.df,
                x=group_col,
                y=violin_col,
                box=True,
                points="all",
                title=f"Violin Plot of {violin_col}, grouped by {group_col}"
            )
        else:
            # Simple violin plot
            fig = px.violin(
                st.session_state.df,
                y=violin_col,
                box=True,
                points="all",
                title=f"Violin Plot of {violin_col}"
            )
    
    elif plot_type == "Residual Plot":
        # Column selection
        x_col = st.selectbox("Independent variable (x):", num_cols)
        y_col = st.selectbox("Dependent variable (y):", [col for col in num_cols if col != x_col])
        
        # Create residual plot
        try:
            # Fit linear regression
            from scipy import stats
            
            # Get data
            x = st.session_state.df[x_col].dropna()
            y = st.session_state.df[y_col].dropna()
            
            # Ensure same length
            df_clean = st.session_state.df[[x_col, y_col]].dropna()
            x = df_clean[x_col]
            y = df_clean[y_col]
            
            # Fit regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate predicted values
            y_pred = intercept + slope * x
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Create figure with subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Linear Regression", "Residuals"))
            
            # Add regression plot
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name='Data',
                    marker=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add regression line
            fig.add_trace(
                go.Scatter(
                    x=[x.min(), x.max()],
                    y=[intercept + slope * x.min(), intercept + slope * x.max()],
                    mode='lines',
                    name=f'Regression Line (r²={r_value**2:.3f})',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Add residual plot
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Add zero line to residual plot
            fig.add_trace(
                go.Scatter(
                    x=[x.min(), x.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Residual Plot: {y_col} vs {x_col}",
                height=700
            )
            
            # Update axis labels
            fig.update_xaxes(title_text=x_col, row=2, col=1)
            fig.update_yaxes(title_text=y_col, row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            
        except Exception as e:
            st.error(f"Error creating residual plot: {str(e)}")
            return
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text)
        width = st.slider("Chart width:", 400, 1200, 800)
        height = st.slider("Chart height:", 300, 1000, 600)
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Chart", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"statistical_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"statistical_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_geospatial_plots_tab():
    """Render geospatial plots tab"""
    st.subheader("Geospatial Plots")
    
    # Check if we have potential geospatial data
    has_lat_lon = False
    lat_col = None
    lon_col = None
    
    # Look for latitude/longitude columns
    for col in st.session_state.df.columns:
        if col.lower() in ['lat', 'latitude', 'y']:
            lat_col = col
            has_lat_lon = True
        elif col.lower() in ['lon', 'long', 'longitude', 'x']:
            lon_col = col
            has_lat_lon = True
    
    # Get all numeric columns for potential coordinates
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Scatter Map", "Bubble Map", "Choropleth Map", "Density Map"]
    )
    
    # Column selection for coordinates
    if has_lat_lon:
        st.info(f"Found potential latitude/longitude columns: {lat_col}, {lon_col}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lat_col = st.selectbox(
            "Latitude column:",
            num_cols,
            index=num_cols.index(lat_col) if lat_col in num_cols else 0
        )
    
    with col2:
        lon_col = st.selectbox(
            "Longitude column:",
            [col for col in num_cols if col != lat_col],
            index=[col for col in num_cols if col != lat_col].index(lon_col) if lon_col in num_cols and lon_col != lat_col else 0
        )
    
    # Additional parameters based on plot type
    if plot_type == "Scatter Map":
        color_col = st.selectbox(
            "Color by:",
            ["None"] + st.session_state.df.columns.tolist()
        )
        
        hover_data = st.multiselect(
            "Additional hover data:",
            [col for col in st.session_state.df.columns if col not in [lat_col, lon_col]],
            default=[]
        )
        
        # Create map
        map_params = {
            "lat": lat_col,
            "lon": lon_col,
            "hover_name": hover_data[0] if hover_data else None,
            "title": "Scatter Map"
        }
        
        if color_col != "None":
            map_params["color"] = color_col
        
        if hover_data:
            map_params["hover_data"] = hover_data
        
        fig = px.scatter_mapbox(
            st.session_state.df,
            **map_params,
            zoom=3,
            mapbox_style="carto-positron"
        )
        
    elif plot_type == "Bubble Map":
        size_col = st.selectbox(
            "Size by:",
            num_cols
        )
        
        color_col = st.selectbox(
            "Color by:",
            ["None"] + st.session_state.df.columns.tolist()
        )
        
        hover_data = st.multiselect(
            "Additional hover data:",
            [col for col in st.session_state.df.columns if col not in [lat_col, lon_col, size_col]],
            default=[]
        )
        
        # Create map
        map_params = {
            "lat": lat_col,
            "lon": lon_col,
            "size": size_col,
            "hover_name": hover_data[0] if hover_data else None,
            "title": "Bubble Map"
        }
        
        if color_col != "None":
            map_params["color"] = color_col
        
        if hover_data:
            map_params["hover_data"] = hover_data
        
        fig = px.scatter_mapbox(
            st.session_state.df,
            **map_params,
            zoom=3,
            mapbox_style="carto-positron"
        )
        
    elif plot_type == "Choropleth Map":
        st.warning("Choropleth maps typically require geographic boundary data (GeoJSON) which is not part of this application. Simplified version shown.")
        
        # Get categorical columns
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Select region column
        region_col = st.selectbox(
            "Region column (e.g., country, state):",
            cat_cols if cat_cols else st.session_state.df.columns.tolist()
        )
        
        # Select value column
        value_col = st.selectbox(
            "Value column:",
            num_cols
        )
        
        # Create simplified choropleth
        fig = px.choropleth(
            st.session_state.df,
            locations=region_col,
            locationmode="country names",  # Try to match with country names
            color=value_col,
            title="Choropleth Map",
            labels={value_col: value_col}
        )
        
    elif plot_type == "Density Map":
        # Density maps work best with lots of points
        if len(st.session_state.df) < 100:
            st.warning("Density maps work best with larger datasets (100+ points).")
        
        radius = st.slider("Density radius:", 5, 50, 20)
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color scheme:",
            ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Reds", "Blues", "Greens"]
        )
        
        # Create density map
        fig = px.density_mapbox(
            st.session_state.df,
            lat=lat_col,
            lon=lon_col,
            radius=radius,
            zoom=3,
            mapbox_style="carto-positron",
            title="Density Map",
            color_continuous_scale=color_scheme.lower()
        )
    
    # Map style
    map_style = st.selectbox(
        "Map style:",
        ["carto-positron", "open-street-map", "white-bg", "stamen-terrain", "stamen-toner", "carto-darkmatter"]
    )
    
    # Update map style
    fig.update_layout(mapbox_style=map_style)
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text)
        width = st.slider("Chart width:", 400, 1200, 800)
        height = st.slider("Chart height:", 300, 1000, 600)
        
        # Map center and zoom
        zoom_level = st.slider("Zoom level:", 1, 20, 3)
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            mapbox=dict(
                zoom=zoom_level
            )
        )
    
    # Show the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Map", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"map_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"map_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_custom_viz_tab():
    """Render custom visualization tab using raw Plotly"""
    st.subheader("Custom Visualization")
    st.write("Create a custom visualization using Plotly's extensive capabilities.")
    
    # Create subtabs
    custom_tabs = st.tabs(["Template-based", "Chart Gallery", "Advanced JSON"])
    
    # Template-based tab
    with custom_tabs[0]:
        st.markdown("### Template-based Custom Visualization")
        st.write("Start with a template and customize it.")
        
        # Template selection
        template = st.selectbox(
            "Select template:",
            ["Bar Chart with Error Bars", "Multi-Y Axis", "Sunburst Chart", 
             "Radar Chart", "Candlestick Chart", "Waterfall Chart", "Gauge Chart"]
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Template-specific options
        if template == "Bar Chart with Error Bars":
            # Column selections
            x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Error bar column
            error_col = st.selectbox("Error bar column:", ["None"] + num_cols)
            
            # Create figure
            fig = go.Figure()
            
            if error_col != "None":
                # With error bars
                fig.add_trace(
                    go.Bar(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        error_y=dict(
                            type='data',
                            array=st.session_state.df[error_col],
                            visible=True
                        ),
                        name=y_col
                    )
                )
            else:
                # Without error bars
                fig.add_trace(
                    go.Bar(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        name=y_col
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"Bar Chart with Error Bars: {y_col} by {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col
            )
        
        elif template == "Multi-Y Axis":
            # Column selections
            x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist())
            
            y_cols = st.multiselect(
                "Y-axis columns:",
                num_cols,
                default=num_cols[:min(3, len(num_cols))]
            )
            
            if len(y_cols) < 1:
                st.warning("Please select at least one Y-axis column.")
                return
            
            # Create figure with multiple y-axes
            fig = go.Figure()
            
            # Colors for different traces
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
            
            # Add first trace on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=st.session_state.df[y_cols[0]],
                    name=y_cols[0],
                    line=dict(color=colors[0])
                )
            )
            
            # Add additional traces on secondary y-axes
            for i, y_col in enumerate(y_cols[1:]):
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        name=y_col,
                        yaxis=f"y{i+2}",
                        line=dict(color=colors[(i+1) % len(colors)])
                    )
                )
            
            # Set up the layout with multiple y-axes
            layout = {
                "title": "Multi-Y Axis Chart",
                "xaxis": {"domain": [0.1, 0.9], "title": x_col},
                "yaxis": {"title": y_cols[0], "titlefont": {"color": colors[0]}, "tickfont": {"color": colors[0]}}
            }
            
            # Add secondary y-axes
            for i, y_col in enumerate(y_cols[1:]):
                pos = 0.05 * (i + 1)
                layout[f"yaxis{i+2}"] = {
                    "title": y_col,
                    "titlefont": {"color": colors[(i+1) % len(colors)]},
                    "tickfont": {"color": colors[(i+1) % len(colors)]},
                    "anchor": "x",
                    "overlaying": "y",
                    "side": "right" if i % 2 == 0 else "left",
                    "position": 1 + pos if i % 2 == 0 else 0 - pos
                }
            
            # Update layout
            fig.update_layout(**layout)
        
        elif template == "Sunburst Chart":
            # Column selections for path
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(2, len(cat_cols))]
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create sunburst chart
            fig = px.sunburst(
                st.session_state.df,
                path=path_cols,
                values=value_col,
                title=f"Sunburst Chart: {', '.join(path_cols)} with {value_col}"
            )
        
        elif template == "Radar Chart":
            # Column selections
            cat_col = st.selectbox("Category column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            
            value_cols = st.multiselect(
                "Value columns (radar axes):",
                num_cols,
                default=num_cols[:min(5, len(num_cols))]
            )
            
            if not value_cols:
                st.warning("Please select at least one value column.")
                return
            
            # Number of categories to show
            top_n = st.slider("Show top N categories:", 1, 10, 3)
            
            # Get top categories
            top_categories = st.session_state.df[cat_col].value_counts().nlargest(top_n).index.tolist()
            
            # Create radar chart
            fig = go.Figure()
            
            for category in top_categories:
                # Filter for this category
                df_cat = st.session_state.df[st.session_state.df[cat_col] == category]
                
                # Calculate average for each value column
                values = [df_cat[col].mean() for col in value_cols]
                
                # Add radar trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=value_cols,
                        fill='toself',
                        name=str(category)
                    )
                )
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                title=f"Radar Chart for Top {top_n} {cat_col} Categories"
            )
        
        elif template == "Candlestick Chart":
            # Check if we have enough numeric columns
            if len(num_cols) < 4:
                st.warning("Candlestick charts require at least 4 numeric columns (open, high, low, close).")
                return
            
            # Column selections
            x_col = st.selectbox(
                "Date/category column:", 
                date_cols + cat_cols if date_cols + cat_cols else st.session_state.df.columns.tolist()
            )
            
            open_col = st.selectbox("Open column:", num_cols)
            high_col = st.selectbox("High column:", [col for col in num_cols if col != open_col])
            low_col = st.selectbox("Low column:", [col for col in num_cols if col not in [open_col, high_col]])
            close_col = st.selectbox("Close column:", [col for col in num_cols if col not in [open_col, high_col, low_col]])
            
            # Create candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=st.session_state.df[x_col],
                    open=st.session_state.df[open_col],
                    high=st.session_state.df[high_col],
                    low=st.session_state.df[low_col],
                    close=st.session_state.df[close_col],
                    name="Price"
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f"Candlestick Chart",
                xaxis_title=x_col
            )
        
        elif template == "Waterfall Chart":
            # Column selections
            cat_col = st.selectbox("Category column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Number of steps to show
            top_n = st.slider("Show top N steps:", 3, 20, 10)
            
            # Sort data for waterfall
            sorted_df = st.session_state.df.sort_values(value_col, ascending=False).head(top_n)
            
            # Create a total row
            total_value = sorted_df[value_col].sum()
            
            # Create waterfall chart data
            measure = ["relative"] * len(sorted_df) + ["total"]
            x = sorted_df[cat_col].tolist() + ["Total"]
            y = sorted_df[value_col].tolist() + [total_value]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Waterfall",
                orientation="v",
                measure=measure,
                x=x,
                y=y,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Waterfall Chart: {value_col} by {cat_col}",
                xaxis_title=cat_col,
                yaxis_title=value_col
            )
        
        elif template == "Gauge Chart":
            # Column selection
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Gauge range
            min_val = st.number_input("Minimum value:", value=0.0)
            max_val = st.number_input("Maximum value:", value=100.0)
            
            # Reference levels
            low_thresh = st.slider("Low threshold:", min_val, max_val, (max_val - min_val) * 0.3 + min_val)
            high_thresh = st.slider("High threshold:", low_thresh, max_val, (max_val - min_val) * 0.7 + min_val)
            
            # Calculate average
            mean_val = st.session_state.df[value_col].mean()
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mean_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Average {value_col}"},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [min_val, low_thresh], 'color': "red"},
                        {'range': [low_thresh, high_thresh], 'color': "yellow"},
                        {'range': [high_thresh, max_val], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': mean_val
                    }
                }
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Gauge Chart: Average {value_col}"
            )
    
    # Chart Gallery tab
    with custom_tabs[1]:
        st.markdown("### Chart Gallery")
        st.write("Select from a gallery of pre-built advanced visualizations.")
        
        # Chart gallery selection
        gallery_chart = st.selectbox(
            "Select chart type:",
            ["Treemap", "Parallel Coordinates", "Sankey Diagram", "Icicle Chart", "Funnel Chart", "Timeline"]
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        
        if gallery_chart == "Treemap":
            # Column selections
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(2, len(cat_cols))]
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            color_col = st.selectbox("Color by:", ["None"] + st.session_state.df.columns.tolist())
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create treemap
            treemap_params = {
                "path": path_cols,
                "values": value_col,
                "title": f"Treemap: {', '.join(path_cols)} with {value_col}"
            }
            
            if color_col != "None":
                treemap_params["color"] = color_col
            
            fig = px.treemap(
                st.session_state.df,
                **treemap_params
            )
        
        elif gallery_chart == "Parallel Coordinates":
            # Column selections
            dimensions = st.multiselect(
                "Select dimensions:",
                st.session_state.df.columns.tolist(),
                default=st.session_state.df.columns[:min(5, len(st.session_state.df.columns))].tolist()
            )
            
            color_col = st.selectbox("Color by:", ["None"] + st.session_state.df.columns.tolist())
            
            if not dimensions:
                st.warning("Please select at least one dimension.")
                return
            
            # Create parallel coordinates
            parallel_params = {
                "dimensions": dimensions,
                "title": "Parallel Coordinates Plot"
            }
            
            if color_col != "None":
                parallel_params["color"] = color_col
            
            fig = px.parallel_coordinates(
                st.session_state.df,
                **parallel_params
            )
        
        elif gallery_chart == "Sankey Diagram":
            st.write("Sankey diagrams show flows between nodes.")
            
            # Column selections
            source_col = st.selectbox("Source column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            target_col = st.selectbox("Target column:", [col for col in cat_cols if col != source_col] if len(cat_cols) > 1 else cat_cols)
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Limit number of nodes for readability
            max_nodes = st.slider("Maximum number of sources/targets:", 5, 50, 20)
            
            try:
                # Prepare Sankey data
                # Get top source nodes
                top_sources = st.session_state.df[source_col].value_counts().nlargest(max_nodes).index.tolist()
                
                # Get top target nodes
                top_targets = st.session_state.df[target_col].value_counts().nlargest(max_nodes).index.tolist()
                
                # Filter data to top nodes
                filtered_df = st.session_state.df[
                    st.session_state.df[source_col].isin(top_sources) & 
                    st.session_state.df[target_col].isin(top_targets)
                ]
                
                # Aggregate values
                sankey_df = filtered_df.groupby([source_col, target_col])[value_col].sum().reset_index()
                
                # Create lists for Sankey diagram
                all_nodes = list(set(sankey_df[source_col].tolist() + sankey_df[target_col].tolist()))
                node_indices = {node: i for i, node in enumerate(all_nodes)}
                
                # Convert source and target to indices
                sources = [node_indices[src] for src in sankey_df[source_col]]
                targets = [node_indices[tgt] for tgt in sankey_df[target_col]]
                values = sankey_df[value_col].tolist()
                
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values
                    )
                )])
                
                # Update layout
                fig.update_layout(
                    title=f"Sankey Diagram: {source_col} → {target_col}"
                )
                
            except Exception as e:
                st.error(f"Error creating Sankey diagram: {str(e)}")
                st.info("Try selecting different columns with categorical data or increase the maximum number of nodes.")
                return
        
        elif gallery_chart == "Icicle Chart":
            # Column selections
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(3, len(cat_cols))]
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create icicle chart
            fig = px.icicle(
                st.session_state.df,
                path=path_cols,
                values=value_col,
                title=f"Icicle Chart: {', '.join(path_cols)} with {value_col}"
            )
        
        elif gallery_chart == "Funnel Chart":
            # Column selections
            stage_col = st.selectbox("Stage column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Calculate values for funnel
            funnel_df = st.session_state.df.groupby(stage_col)[value_col].sum().reset_index()
            
            # Sort order
            sort_by = st.radio(
                "Sort stages by:",
                ["Value (descending)", "Value (ascending)", "Custom order"],
                horizontal=True
            )
            
            if sort_by == "Value (descending)":
                funnel_df = funnel_df.sort_values(value_col, ascending=False)
            elif sort_by == "Value (ascending)":
                funnel_df = funnel_df.sort_values(value_col, ascending=True)
            else:  # Custom order
                # Allow user to specify order
                stage_order = st.multiselect(
                    "Arrange stages in order (top to bottom):",
                    funnel_df[stage_col].unique().tolist(),
                    default=funnel_df[stage_col].unique().tolist()
                )
                
                if stage_order:
                    # Create a mapping for sorting
                    stage_order_map = {stage: i for i, stage in enumerate(stage_order)}
                    
                    # Apply sorting
                    funnel_df['sort_order'] = funnel_df[stage_col].map(stage_order_map)
                    funnel_df = funnel_df.sort_values('sort_order').drop('sort_order', axis=1)
            
            # Create funnel chart
            fig = go.Figure(go.Funnel(
                y=funnel_df[stage_col],
                x=funnel_df[value_col],
                textinfo="value+percent initial"
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Funnel Chart: {stage_col} by {value_col}"
            )
        
        elif gallery_chart == "Timeline":
            # Check if we have date columns
            if not date_cols:
                st.warning("Timeline requires at least one date/time column. No datetime columns found.")
                return
            
            # Column selections
            date_col = st.selectbox("Date column:", date_cols)
            event_col = st.selectbox("Event column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            
            # Optional group column
            use_group = st.checkbox("Group events")
            if use_group and len(cat_cols) > 1:
                group_col = st.selectbox("Group by:", [col for col in cat_cols if col != event_col])
                
                # Create timeline with groups
                fig = px.timeline(
                    st.session_state.df,
                    x_start=date_col,
                    y=group_col,
                    color=event_col,
                    hover_name=event_col,
                    title=f"Timeline of {event_col} by {group_col}"
                )
            else:
                # Simple timeline
                # Need to create a fake "y" column for Plotly timeline
                dummy_y = "timeline"
                
                # Create a copy of the dataframe with the dummy column
                timeline_df = st.session_state.df.copy()
                timeline_df[dummy_y] = dummy_y
                
                fig = px.timeline(
                    timeline_df,
                    x_start=date_col,
                    y=dummy_y,
                    color=event_col,
                    hover_name=event_col,
                    title=f"Timeline of {event_col}"
                )
            
            # Update layout
            fig.update_yaxes(autorange="reversed")
    
    # Advanced JSON tab
    with custom_tabs[2]:
        st.markdown("### Advanced JSON Configuration")
        st.write("Use Plotly's JSON configuration for ultimate customization.")
        
        # JSON template examples
        template_options = {
            "None": "{}",
            "Bar Chart": """{
    "data": [
        {
            "type": "bar",
            "x": ["A", "B", "C", "D"],
            "y": [1, 3, 2, 4]
        }
    ],
    "layout": {
        "title": "Bar Chart Example"
    }
}""",
            "Line Chart": """{
    "data": [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": [1, 2, 3, 4, 5],
            "y": [1, 3, 2, 4, 3],
            "name": "Series 1"
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 1, 3, 5],
            "name": "Series 2"
        }
    ],
    "layout": {
        "title": "Line Chart Example"
    }
}""",
            "Pie Chart": """{
    "data": [
        {
            "type": "pie",
            "values": [30, 20, 15, 10, 25],
            "labels": ["Category A", "Category B", "Category C", "Category D", "Category E"]
        }
    ],
    "layout": {
        "title": "Pie Chart Example"
    }
}"""
        }
        
        # Template selection
        template_selection = st.selectbox("Choose a template:", list(template_options.keys()))
        
        # JSON input
        json_input = st.text_area(
            "Enter or modify Plotly JSON:",
            value=template_options[template_selection],
            height=400
        )
        
        # Help text
        st.markdown("""
        **Tips for JSON Configuration:**
        - The JSON must include both `data` and `layout` properties.
        - Use column names from your dataframe to replace example data.
        - For dynamic data, replace sample arrays with column references.
        - Visit the [Plotly JSON Chart Schema](https://plotly.com/chart-studio-help/json-chart-schema/) for more details.
        """)
        
        # Column reference helper
        with st.expander("Column Reference Helper", expanded=False):
            st.write("Select a column to get a JSON data reference snippet:")
            
            # Column selection
            ref_col = st.selectbox("Select column:", st.session_state.df.columns.tolist())
            
            # Generate reference code
            st.code(f'st.session_state.df["{ref_col}"].tolist()')
            
            # Show sample of column data
            st.write(f"Sample values from {ref_col}:")
            st.write(st.session_state.df[ref_col].head(5).tolist())
        
        # Apply JSON button
        if st.button("Apply JSON Configuration", use_container_width=True):
            try:
                # Parse JSON
                import json
                config = json.loads(json_input)
                
                # Validate basic structure
                if "data" not in config or "layout" not in config:
                    st.error("JSON must include both 'data' and 'layout' properties.")
                    return
                
                # Evaluate references to dataframe columns
                def process_data(data_item):
                    for key, value in list(data_item.items()):
                        if isinstance(value, str) and value.startswith("st.session_state.df["):
                            # Evaluate the expression to get actual data
                            try:
                                data_item[key] = eval(value)
                            except Exception as e:
                                st.error(f"Error evaluating {value}: {str(e)}")
                    return data_item
                
                # Process each data item
                for i, data_item in enumerate(config["data"]):
                    config["data"][i] = process_data(data_item)
                
                # Create figure from JSON
                fig = go.Figure(config)
                
            except Exception as e:
                st.error(f"Error parsing JSON configuration: {str(e)}")
                return
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else "")
        width = st.slider("Chart width:", 400, 1200, 800)
        height = st.slider("Chart height:", 300, 1000, 600)
        
        # Theme selection
        theme = st.selectbox(
            "Color theme:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
        )
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            template=theme
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Chart", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name="custom_visualization.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"custom_visualization.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_analysis():
    """Render analysis section"""
    st.header("Data Analysis")
    
    # Create tabs for different analysis types
    analysis_tabs = st.tabs([
        "Summary Statistics",
        "Correlation Analysis", 
        "Distribution Analysis",
        "Trend Analysis",
        "Hypothesis Testing"
    ])
    
    # Summary Statistics Tab
    with analysis_tabs[0]:
        render_summary_statistics()
    
    # Correlation Analysis Tab
    with analysis_tabs[1]:
        render_correlation_analysis()
    
    # Distribution Analysis Tab
    with analysis_tabs[2]:
        render_distribution_analysis()
    
    # Trend Analysis Tab
    with analysis_tabs[3]:
        render_trend_analysis()
    
    # Hypothesis Testing Tab
    with analysis_tabs[4]:
        render_hypothesis_testing()

def render_summary_statistics():
    """Render summary statistics tab"""
    st.subheader("Summary Statistics")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns found for summary statistics.")
        return
    
    # Column selection
    selected_cols = st.multiselect(
        "Select columns for statistics:",
        num_cols,
        default=num_cols
    )
    
    if not selected_cols:
        st.warning("Please select at least one column.")
        return
    
    # Calculate statistics
    stats_df = st.session_state.df[selected_cols].describe().T
    
    # Add more statistics
    stats_df['median'] = st.session_state.df[selected_cols].median()
    stats_df['range'] = st.session_state.df[selected_cols].max() - st.session_state.df[selected_cols].min()
    stats_df['missing'] = st.session_state.df[selected_cols].isna().sum()
    stats_df['missing_pct'] = (st.session_state.df[selected_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
    
    try:
        stats_df['skew'] = st.session_state.df[selected_cols].skew().round(3)
        stats_df['kurtosis'] = st.session_state.df[selected_cols].kurtosis().round(3)
    except:
        pass
    
    # Round numeric columns
    stats_df = stats_df.round(3)
    
    # Display the statistics
    st.dataframe(stats_df, use_container_width=True)
    
    # Visualization of statistics
    st.subheader("Statistical Visualization")
    
    # Statistic to visualize
    stat_type = st.selectbox(
        "Statistic to visualize:",
        ["Mean", "Median", "Standard Deviation", "Range", "Missing Values %"]
    )
    
    # Map statistic to column name
    stat_map = {
        "Mean": "mean",
        "Median": "median",
        "Standard Deviation": "std",
        "Range": "range",
        "Missing Values %": "missing_pct"
    }
    
    # Create visualization
    if stat_type in stat_map:
        fig = px.bar(
            stats_df.reset_index(),
            x="index",
            y=stat_map[stat_type],
            title=f"{stat_type} by Column",
            labels={"index": "Column", "y": stat_type}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # Export statistics to CSV
        csv = stats_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download Statistics as CSV",
            data=csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export chart
        if 'fig' in locals():
            # Format selection
            export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
            
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Chart",
                    data=html_bytes,
                    file_name=f"statistics_{stat_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Chart as {export_format}",
                    data=img_bytes,
                    file_name=f"statistics_{stat_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True
                )

def render_correlation_analysis():
    """Render correlation analysis tab"""
    st.subheader("Correlation Analysis")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for correlation analysis.")
        return
    
    # Column selection
    selected_cols = st.multiselect(
        "Select columns for correlation analysis:",
        num_cols,
        default=num_cols[:min(5, len(num_cols))]
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least two columns.")
        return
    
    # Correlation method
    corr_method = st.radio(
        "Correlation method:",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True
    )
    
    # Calculate correlation matrix
    corr_matrix = st.session_state.df[selected_cols].corr(method=corr_method.lower())
    
    # Round to 3 decimal places
    corr_matrix = corr_matrix.round(3)
    
    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix, use_container_width=True)
    
    # Visualization options
    viz_type = st.radio(
        "Visualization type:",
        ["Heatmap", "Scatter Matrix", "Network Graph"],
        horizontal=True
    )
    
    if viz_type == "Heatmap":
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f"{corr_method} Correlation Heatmap",
            labels=dict(color="Correlation")
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )
        
    elif viz_type == "Scatter Matrix":
        # Create scatter matrix
        fig = px.scatter_matrix(
            st.session_state.df[selected_cols],
            title=f"Scatter Matrix ({corr_method} Correlation)",
            dimensions=selected_cols
        )
        
        # Update traces
        fig.update_traces(diagonal_visible=False)
        
    elif viz_type == "Network Graph":
        # Network graph visualization of correlations
        try:
            import networkx as nx
            
            # Create a graph from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for col in selected_cols:
                G.add_node(col)
            
            # Add edges with correlation values as weights
            for i, col1 in enumerate(selected_cols):
                for col2 in selected_cols[i+1:]:
                    # Only add edges for correlations above a threshold
                    corr_value = abs(corr_matrix.loc[col1, col2])
                    if corr_value > 0.1:  # Ignore very weak correlations
                        G.add_edge(col1, col2, weight=corr_value)
            
            # Get node positions using force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create network graph
            node_x = []
            node_y = []
            for key, value in pos.items():
                node_x.append(value[0])
                node_y.append(value[1])
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='skyblue',
                    line=dict(width=2, color='darkblue')
                ),
                text=list(pos.keys()),
                textposition='top center',
                hoverinfo='text'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Correlation Network Graph ({corr_method})",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
        except Exception as e:
            st.error(f"Error creating network graph: {str(e)}")
            st.info("Try using the heatmap or scatter matrix visualization instead.")
            return
    
    # Show visualization
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    st.subheader("Top Correlations")
    
    # Convert correlation matrix to a list of pairs
    pairs = []
    for i, col1 in enumerate(selected_cols):
        for j, col2 in enumerate(selected_cols):
            if i < j:  # Only include each pair once
                pairs.append({
                    'Column 1': col1,
                    'Column 2': col2,
                    'Correlation': corr_matrix.loc[col1, col2]
                })
    
    # Convert to dataframe
    pairs_df = pd.DataFrame(pairs)
    
    # Sort by absolute correlation
    pairs_df['Abs Correlation'] = pairs_df['Correlation'].abs()
    pairs_df = pairs_df.sort_values('Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
    
    # Display top correlations
    st.dataframe(pairs_df, use_container_width=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # Export correlation matrix to CSV
        csv = corr_matrix.reset_index().to_csv(index=False)
        st.download_button(
            label="Download Correlation Matrix",
            data=csv,
            file_name=f"correlation_matrix_{corr_method.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export visualization
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
        
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Visualization",
                data=html_bytes,
                file_name=f"correlation_{viz_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )

def render_hypothesis_testing():
    """Render hypothesis testing tab"""
    st.subheader("Hypothesis Testing")
    
    # Select test type
    test_type = st.selectbox(
        "Select test type:",
        [
            "One Sample t-test",
            "Two Sample t-test",
            "Paired t-test",
            "One-way ANOVA",
            "Chi-Square Test",
            "Correlation Test"
        ]
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if test_type == "One Sample t-test":
        st.write("Test if the mean of a sample is significantly different from a hypothesized value.")
        
        if not num_cols:
            st.warning("No numeric columns available for t-test.")
            return
        
        # Column selection
        col = st.selectbox("Select column:", num_cols)
        
        # Hypothesized mean
        pop_mean = st.number_input("Hypothesized population mean:", value=0.0)
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05)
        
        # Run the test
        if st.button("Run One Sample t-test", use_container_width=True):
            try:
                from scipy import stats
                
                # Get data
                data = st.session_state.df[col].dropna()
                
                # Run t-test
                t_stat, p_value = stats.ttest_1samp(data, pop_mean)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sample Mean", f"{data.mean():.4f}")
                with col2:
                    st.metric("Hypothesized Mean", f"{pop_mean:.4f}")
                with col3:
                    st.metric("t-statistic", f"{t_stat:.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean of {col} is significantly different from {pop_mean}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean of {col} is significantly different from {pop_mean}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create distribution plot
                fig = go.Figure()
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name="Sample Distribution",
                        opacity=0.7,
                        histnorm="probability density"
                    )
                )
                
                # Add normal curve
                x = np.linspace(data.min(), data.max(), 1000)
                y = stats.norm.pdf(x, data.mean(), data.std())
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name="Normal Distribution",
                        line=dict(color="red")
                    )
                )
                
                # Add vertical lines for means
                fig.add_vline(
                    x=data.mean(),
                    line_dash="solid",
                    line_color="blue",
                    annotation_text="Sample Mean",
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=pop_mean,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Hypothesized Mean",
                    annotation_position="top left"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Distribution of {col} vs. Hypothesized Mean",
                    xaxis_title=col,
                    yaxis_title="Density"
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running t-test: {str(e)}")
    
    elif test_type == "Two Sample t-test":
        st.write("Test if the means of two independent samples are significantly different.")
        
        if not num_cols:
            st.warning("No numeric columns available for t-test.")
            return
        
        if not cat_cols:
            st.warning("No categorical columns available for grouping.")
            return
        
        # Column selections
        value_col = st.selectbox("Value column:", num_cols)
        group_col = st.selectbox("Group column:", cat_cols)
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for two-sample t-test.")
            return
        
        # Group selection
        col1, col2 = st.columns(2)
        
        with col1:
            group1 = st.selectbox("First group:", groups)
        
        with col2:
            group2 = st.selectbox("Second group:", [g for g in groups if g != group1])
        
        # Variance assumption
        equal_var = st.checkbox("Assume equal variances", value=False)
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05)
        
        # Run the test
        if st.button("Run Two Sample t-test", use_container_width=True):
            try:
                from scipy import stats
                
                # Get data for each group
                data1 = st.session_state.df[st.session_state.df[group_col] == group1][value_col].dropna()
                data2 = st.session_state.df[st.session_state.df[group_col] == group2][value_col].dropna()
                
                # Run t-test
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Mean ({group1})", f"{data1.mean():.4f}")
                with col2:
                    st.metric(f"Mean ({group2})", f"{data2.mean():.4f}")
                with col3:
                    st.metric("t-statistic", f"{t_stat:.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean {value_col} is significantly different between {group1} and {group2}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean {value_col} is significantly different between {group1} and {group2}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create box plot
                fig = go.Figure()
                
                # Add box plot for each group
                fig.add_trace(
                    go.Box(
                        y=data1,
                        name=str(group1),
                        boxmean=True
                    )
                )
                
                fig.add_trace(
                    go.Box(
                        y=data2,
                        name=str(group2),
                        boxmean=True
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Comparison of {value_col} between {group1} and {group2}",
                    yaxis_title=value_col
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running t-test: {str(e)}")
    
    elif test_type == "Paired t-test":
        st.write("Test if the mean difference between paired observations is significantly different from zero.")
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for paired t-test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox("First column:", num_cols)
        
        with col2:
            second_col = st.selectbox("Second column:", [col for col in num_cols if col != first_col])
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="paired_alpha")
        
        # Run the test
        if st.button("Run Paired t-test", use_container_width=True):
            try:
                from scipy import stats
                
                # Get data
                # Remove rows with NaN in either column
                valid_data = st.session_state.df[[first_col, second_col]].dropna()
                
                data1 = valid_data[first_col]
                data2 = valid_data[second_col]
                
                # Run paired t-test
                t_stat, p_value = stats.ttest_rel(data1, data2)
                
                # Calculate differences
                differences = data1 - data2
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Mean ({first_col})", f"{data1.mean():.4f}")
                with col2:
                    st.metric(f"Mean ({second_col})", f"{data2.mean():.4f}")
                with col3:
                    st.metric("Mean Difference", f"{differences.mean():.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that there is a significant difference between {first_col} and {second_col}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant difference between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create visualization of paired differences
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Paired Values", "Differences"))
                
                # Add scatter plot with paired values
                fig.add_trace(
                    go.Scatter(
                        x=data1,
                        y=data2,
                        mode="markers",
                        marker=dict(color="blue"),
                        name="Paired Values"
                    ),
                    row=1, col=1
                )
                
                # Add diagonal line y=x
                min_val = min(data1.min(), data2.min())
                max_val = max(data1.max(), data2.max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="y=x"
                    ),
                    row=1, col=1
                )
                
                # Add histogram of differences
                fig.add_trace(
                    go.Histogram(
                        x=differences,
                        marker=dict(color="green"),
                        name="Differences"
                    ),
                    row=1, col=2
                )
                
                # Add vertical line at zero
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="black",
                    row=1, col=2
                )
                
                # Add vertical line at mean difference
                fig.add_vline(
                    x=differences.mean(),
                    line_dash="solid",
                    line_color="red",
                    annotation_text="Mean Difference",
                    annotation_position="top right",
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Paired Comparison: {first_col} vs {second_col}",
                    height=500
                )
                
                # Update axes titles
                fig.update_xaxes(title_text=first_col, row=1, col=1)
                fig.update_yaxes(title_text=second_col, row=1, col=1)
                fig.update_xaxes(title_text="Difference", row=1, col=2)
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running paired t-test: {str(e)}")
    
    elif test_type == "One-way ANOVA":
        st.write("Test if the means of multiple groups are significantly different.")
        
        if not num_cols:
            st.warning("No numeric columns available for ANOVA.")
            return
        
        if not cat_cols:
            st.warning("No categorical columns available for grouping.")
            return
        
        # Column selections
        value_col = st.selectbox("Value column:", num_cols)
        group_col = st.selectbox("Group column:", cat_cols)
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for ANOVA.")
            return
        
        # Select groups to include
        selected_groups = st.multiselect(
            "Select groups to compare:",
            groups,
            default=groups[:min(5, len(groups))]
        )
        
        if len(selected_groups) < 2:
            st.warning("Please select at least 2 groups for comparison.")
            return
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="anova_alpha")
        
        # Run the test
        if st.button("Run One-way ANOVA", use_container_width=True):
            try:
                from scipy import stats
                
                # Get data for each group
                group_data = []
                group_means = []
                group_counts = []
                
                for group in selected_groups:
                    data = st.session_state.df[st.session_state.df[group_col] == group][value_col].dropna()
                    group_data.append(data)
                    group_means.append(data.mean())
                    group_counts.append(len(data))
                
                # Run ANOVA
                f_stat, p_value = stats.f_oneway(*group_data)
                
                # Show results
                st.subheader("Test Results")
                
                # Group statistics
                group_stats = pd.DataFrame({
                    "Group": selected_groups,
                    "Count": group_counts,
                    "Mean": group_means,
                    "Std": [data.std() for data in group_data],
                    "Min": [data.min() for data in group_data],
                    "Max": [data.max() for data in group_data]
                })
                
                st.dataframe(group_stats.round(4), use_container_width=True)
                
                # ANOVA results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F-statistic", f"{f_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    st.metric("# of Groups", len(selected_groups))
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean {value_col} is significantly different among the groups.")
                    
                    # Post-hoc test (Tukey's HSD)
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        import numpy as np
                        
                        # Prepare data for Tukey's test
                        all_data = np.concatenate(group_data)
                        all_groups = np.concatenate([[group] * len(data) for group, data in zip(selected_groups, group_data)])
                        
                        # Run Tukey's test
                        tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=alpha)
                        
                        # Display results
                        st.subheader("Post-hoc: Tukey's HSD Test")
                        st.write("Pairwise comparisons:")
                        
                        # Convert Tukey results to dataframe
                        tukey_df = pd.DataFrame(
                            data=np.column_stack([tukey_result.groupsunique[tukey_result.pairindices[:,0]], 
                                              tukey_result.groupsunique[tukey_result.pairindices[:,1]], 
                                              tukey_result.meandiffs, 
                                              tukey_result.confint, 
                                              tukey_result.pvalues, 
                                              tukey_result.reject]),
                            columns=['group1', 'group2', 'mean_diff', 'lower_bound', 'upper_bound', 'p_value', 'reject']
                        )
                        
                        # Convert numeric columns
                        for col in ['mean_diff', 'lower_bound', 'upper_bound', 'p_value']:
                            tukey_df[col] = pd.to_numeric(tukey_df[col])
                        
                        # Format reject column
                        tukey_df['significant'] = tukey_df['reject'].map({True: 'Yes', False: 'No'})
                        
                        # Round numeric columns
                        tukey_df = tukey_df.round(4)
                        
                        # Display dataframe
                        st.dataframe(
                            tukey_df[['group1', 'group2', 'mean_diff', 'p_value', 'significant']],
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.warning(f"Could not perform post-hoc test: {str(e)}")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean {value_col} is significantly different among the groups.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create box plot
                fig = go.Figure()
                
                # Add box plot for each group
                for i, (group, data) in enumerate(zip(selected_groups, group_data)):
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=str(group),
                            boxmean=True
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Comparison of {value_col} across groups",
                    yaxis_title=value_col
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running ANOVA: {str(e)}")
    
    elif test_type == "Chi-Square Test":
        st.write("Test if there is a significant association between two categorical variables.")
        
        if len(cat_cols) < 2:
            st.warning("Need at least two categorical columns for Chi-Square test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox("First categorical column:", cat_cols)
        
        with col2:
            second_col = st.selectbox("Second categorical column:", [col for col in cat_cols if col != first_col])
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="chi2_alpha")
        
        # Run the test
        if st.button("Run Chi-Square Test", use_container_width=True):
            try:
                from scipy import stats
                
                # Create contingency table
                contingency = pd.crosstab(
                    st.session_state.df[first_col],
                    st.session_state.df[second_col]
                )
                
                # Run chi-square test
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                # Check if expected frequencies are too small
                expected_df = pd.DataFrame(
                    expected,
                    index=contingency.index,
                    columns=contingency.columns
                )
                
                # Check for small expected frequencies
                small_expected = (expected_df < 5).values.sum()
                total_cells = expected_df.size
                
                # Convert expected to dataframe for display
                expected_df_display = expected_df.round(2)
                
                # Show results
                st.subheader("Contingency Table (Observed)")
                st.dataframe(contingency, use_container_width=True)
                
                st.subheader("Expected Frequencies")
                st.dataframe(expected_df_display, use_container_width=True)
                
                # Chi-square results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chi-Square", f"{chi2:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    st.metric("Degrees of Freedom", dof)
                with col4:
                    # Calculate Cramer's V (effect size)
                    n = contingency.values.sum()
                    cramers_v = np.sqrt(chi2 / (n * min(contingency.shape[0]-1, contingency.shape[1]-1)))
                    st.metric("Cramer's V", f"{cramers_v:.4f}")
                
                # Warning for small expected frequencies
                if small_expected > 0:
                    pct_small = (small_expected / total_cells) * 100
                    st.warning(f"{small_expected} cells ({pct_small:.1f}%) have expected frequencies less than 5. Chi-square results may not be reliable.")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that there is a significant association between {first_col} and {second_col}.")
                    
                    # Interpret Cramer's V
                    if cramers_v < 0.1:
                        effect_size = "negligible"
                    elif cramers_v < 0.3:
                        effect_size = "small"
                    elif cramers_v < 0.5:
                        effect_size = "medium"
                    else:
                        effect_size = "large"
                    
                    st.write(f"The strength of the association (Cramer's V = {cramers_v:.4f}) is {effect_size}.")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant association between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create heatmap of observed frequencies
                fig = px.imshow(
                    contingency,
                    labels=dict(
                        x=second_col,
                        y=first_col,
                        color="Frequency"
                    ),
                    text_auto=True,
                    title=f"Contingency Table: {first_col} vs {second_col}"
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Create stacked bar chart
                # Normalize contingency table by rows
                prop_table = contingency.div(contingency.sum(axis=1), axis=0)
                
                # Create stacked bar chart of proportions
                fig_bar = px.bar(
                    prop_table.reset_index().melt(id_vars=first_col),
                    x=first_col,
                    y="value",
                    color="variable",
                    labels={"value": "Proportion", "variable": second_col},
                    title=f"Proportions of {second_col} within each {first_col}",
                    text_auto='.2f'
                )
                
                # Show plot
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running Chi-Square test: {str(e)}")
    
    elif test_type == "Correlation Test":
        st.write("Test if there is a significant correlation between two numeric variables.")
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for correlation test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox("First numeric column:", num_cols)
        
        with col2:
            second_col = st.selectbox("Second numeric column:", [col for col in num_cols if col != first_col])
        
        # Correlation method
        method = st.radio(
            "Correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True
        )
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="corr_alpha")
        
        # Run the test
        if st.button("Run Correlation Test", use_container_width=True):
            try:
                from scipy import stats
                
                # Get data
                # Remove rows with NaN in either column
                valid_data = st.session_state.df[[first_col, second_col]].dropna()
                
                data1 = valid_data[first_col]
                data2 = valid_data[second_col]
                
                # Run correlation test
                if method == "Pearson":
                    corr, p_value = stats.pearsonr(data1, data2)
                elif method == "Spearman":
                    corr, p_value = stats.spearmanr(data1, data2)
                else:  # Kendall
                    corr, p_value = stats.kendalltau(data1, data2)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{method} Correlation", f"{corr:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    # Calculate coefficient of determination (r-squared)
                    r_squared = corr ** 2
                    st.metric("R²", f"{r_squared:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    
                    if corr > 0:
                        direction = "positive"
                    else:
                        direction = "negative"
                    
                    # Interpret correlation strength
                    corr_abs = abs(corr)
                    if corr_abs < 0.3:
                        strength = "weak"
                    elif corr_abs < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    st.write(f"There is enough evidence to suggest that there is a significant {direction} correlation ({strength}) between {first_col} and {second_col}.")
                    st.write(f"The coefficient of determination (R²) indicates that {r_squared:.2%} of the variance in one variable can be explained by the other variable.")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant correlation between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create scatter plot
                fig = px.scatter(
                    valid_data,
                    x=first_col,
                    y=second_col,
                    trendline="ols",
                    labels={
                        first_col: first_col,
                        second_col: second_col
                    },
                    title=f"Scatter Plot: {first_col} vs {second_col}"
                )
                
                # Add annotation with correlation coefficient
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    text=f"{method} correlation: {corr:.4f}<br>p-value: {p_value:.4f}<br>R²: {r_squared:.4f}",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running correlation test: {str(e)}")

def render_reports():
    """Render reports section"""
    st.header("Reports")
    
    # Create tabs for different report types
    report_tabs = st.tabs([
        "Summary Report",
        "Exploratory Analysis",
        "Custom Report",
        "Export Options"
    ])
    
    # Summary Report Tab
    with report_tabs[0]:
        render_summary_report()
    
    # Exploratory Analysis Tab
    with report_tabs[1]:
        render_exploratory_report()
    
    # Custom Report Tab
    with report_tabs[2]:
        render_custom_report()
    
    # Export Options Tab
    with report_tabs[3]:
        render_export_options()

def render_summary_report():
    """Render summary report tab"""
    st.subheader("Summary Report")
    
    # Report configuration
    st.write("Generate a comprehensive summary report of your dataset.")
    
    with st.expander("Report Configuration", expanded=True):
        # Report title
        report_title = st.text_input("Report Title:", "Data Summary Report")
        
        # Include sections
        st.write("Select sections to include:")
        
        include_dataset_info = st.checkbox("Dataset Information", value=True)
        include_column_summary = st.checkbox("Column Summary", value=True)
        include_numeric_summary = st.checkbox("Numeric Data Summary", value=True)
        include_categorical_summary = st.checkbox("Categorical Data Summary", value=True)
        include_missing_data = st.checkbox("Missing Data Analysis", value=True)
        include_charts = st.checkbox("Include Charts", value=True)
    
    # Generate report
    if st.button("Generate Summary Report", use_container_width=True):
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Dataset Information
        if include_dataset_info:
            report += "## Dataset Information\n\n"
            
            # Basic stats
            report += f"* **Rows:** {len(st.session_state.df):,}\n"
            report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
            
            # File details if available
            if hasattr(st.session_state, 'file_details') and st.session_state.file_details is not None:
                report += f"* **Filename:** {st.session_state.file_details['filename']}\n"
                
                # Format size nicely
                size = st.session_state.file_details['size']
                if size is not None:
                    if size < 1024:
                        size_str = f"{size} bytes"
                    elif size < 1024 ** 2:
                        size_str = f"{size / 1024:.2f} KB"
                    elif size < 1024 ** 3:
                        size_str = f"{size / (1024 ** 2):.2f} MB"
                    else:
                        size_str = f"{size / (1024 ** 3):.2f} GB"
                    
                    report += f"* **Size:** {size_str}\n"
            
            # Data types summary
            dtypes = st.session_state.df.dtypes.value_counts().to_dict()
            report += "\n### Data Types\n\n"
            for dtype, count in dtypes.items():
                report += f"* **{dtype}:** {count} columns\n"
            
            report += "\n"
        
        # Column Summary
        if include_column_summary:
            report += "## Column Summary\n\n"
            
            # Create a dataframe with column information
            columns_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes.astype(str).values,
                'Non-Null Count': st.session_state.df.count().values,
                'Null Count': st.session_state.df.isna().sum().values,
                'Null %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).astype(str) + '%',
                'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
            })
            
            # Convert to markdown table
            report += columns_df.to_markdown(index=False)
            report += "\n\n"
        
        # Numeric Data Summary
        if include_numeric_summary:
            report += "## Numeric Data Summary\n\n"
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if num_cols:
                # Calculate statistics
                stats_df = st.session_state.df[num_cols].describe().T.round(3)
                
                # Add more statistics
                stats_df['median'] = st.session_state.df[num_cols].median().round(3)
                stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
                stats_df['missing_pct'] = (st.session_state.df[num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
                
                try:
                    stats_df['skew'] = st.session_state.df[num_cols].skew().round(3)
                    stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis().round(3)
                except:
                    pass
                
                # Convert to markdown table
                report += stats_df.reset_index().rename(columns={'index': 'Column'}).to_markdown(index=False)
                report += "\n\n"
                
                # Include charts for numeric columns
                if include_charts:
                    report += "### Numeric Data Distributions\n\n"
                    report += "See attached visualizations for distributions of numeric columns.\n\n"
            else:
                report += "*No numeric columns found in the dataset.*\n\n"
        
        # Categorical Data Summary
        if include_categorical_summary:
            report += "## Categorical Data Summary\n\n"
            
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                # For each categorical column, show top values
                for col in cat_cols:
                    value_counts = st.session_state.df[col].value_counts().nlargest(10)
                    report += f"### {col}\n\n"
                    report += f"* **Unique Values:** {st.session_state.df[col].nunique()}\n"
                    report += f"* **Missing Values:** {st.session_state.df[col].isna().sum()} ({st.session_state.df[col].isna().sum() / len(st.session_state.df) * 100:.2f}%)\n"
                    report += f"* **Most Common Values:**\n\n"
                    
                    # Create value counts dataframe
                    vc_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(st.session_state.df) * 100).round(2)
                    })
                    
                    # Convert to markdown table
                    report += vc_df.to_markdown(index=False)
                    report += "\n\n"
            else:
                report += "*No categorical columns found in the dataset.*\n\n"
        
        # Missing Data Analysis
        if include_missing_data:
            report += "## Missing Data Analysis\n\n"
            
            # Calculate missing values
            missing_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Missing Count': st.session_state.df.isna().sum().values,
                'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
            })
            
            # Sort by missing count (descending)
            missing_df = missing_df.sort_values('Missing Count', ascending=False)
            
            # Get columns with missing values
            cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
            
            if len(cols_with_missing) > 0:
                report += f"**Total Columns with Missing Values:** {len(cols_with_missing)} out of {len(st.session_state.df.columns)}\n\n"
                
                # Convert to markdown table
                report += cols_with_missing.to_markdown(index=False)
                report += "\n\n"
                
                if include_charts:
                    report += "### Missing Data Visualization\n\n"
                    report += "See attached visualizations for patterns of missing data.\n\n"
            else:
                report += "*No missing values found in the dataset.*\n\n"
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate charts if requested
        if include_charts:
            st.subheader("Report Charts")
            
            # Create a list to store chart images
            charts = []
            
            # Numeric column distributions
            if include_numeric_summary:
                num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                
                if num_cols:
                    # Select a subset of numeric columns for visualization (max 5)
                    selected_num_cols = num_cols[:min(5, len(num_cols))]
                    
                    # Create histogram for each selected column
                    for col in selected_num_cols:
                        fig = px.histogram(
                            st.session_state.df,
                            x=col,
                            title=f"Distribution of {col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Missing data visualization
            if include_missing_data:
                missing_df = pd.DataFrame({
                    'Column': st.session_state.df.columns,
                    'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
                })
                
                # Sort by missing percentage (descending)
                missing_df = missing_df.sort_values('Missing %', ascending=False)
                
                # Get columns with missing values
                cols_with_missing = missing_df[missing_df['Missing %'] > 0]
                
                if len(cols_with_missing) > 0:
                    # Create missing values chart
                    fig = px.bar(
                        cols_with_missing,
                        x='Column',
                        y='Missing %',
                        title='Missing Values by Column (%)',
                        color='Missing %',
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as markdown
            st.download_button(
                label="Download as Markdown",
                data=report,
                file_name=f"{report_title.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            try:
                import markdown
                html = markdown.markdown(report)
                
                st.download_button(
                    label="Download as HTML",
                    data=html,
                    file_name=f"{report_title.replace(' ', '_').lower()}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_custom_report():
    """Render custom report tab"""
    st.subheader("Custom Report")
    
    # Report configuration
    st.write("Build a custom report with selected sections and visualizations.")
    
    with st.expander("Report Settings", expanded=True):
        # Report title and description
        report_title = st.text_input("Report Title:", "Custom Data Analysis Report")
        report_description = st.text_area("Report Description:", "This report provides a custom analysis of the dataset.")
        
        # Author information
        include_author = st.checkbox("Include Author Information", value=False)
        if include_author:
            author_name = st.text_input("Author Name:")
            author_info = st.text_input("Additional Info (e.g., email, organization):")
    
    # Section selection
    st.subheader("Report Sections")
    
    # Create a list to store selected sections
    selected_sections = []
    
    # Dataset Overview section
    include_overview = st.checkbox("Dataset Overview", value=True)
    if include_overview:
        selected_sections.append("overview")
        with st.container():
            st.markdown("##### Dataset Overview Options")
            include_basic_stats = st.checkbox("Basic Statistics", value=True)
            include_data_types = st.checkbox("Data Types Summary", value=True)
            include_sample_data = st.checkbox("Sample Data Preview", value=True)
            
            # Store options
            overview_options = {
                "basic_stats": include_basic_stats,
                "data_types": include_data_types,
                "sample_data": include_sample_data
            }
    else:
        overview_options = {}
    
    # Numeric Data Analysis section
    include_numeric = st.checkbox("Numeric Data Analysis", value=True)
    if include_numeric:
        selected_sections.append("numeric")
        with st.container():
            st.markdown("##### Numeric Analysis Options")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                # Select columns to include
                selected_num_cols = st.multiselect(
                    "Select columns to analyze:",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))]
                )
                
                # Chart types
                include_histograms = st.checkbox("Include Histograms", value=True)
                include_boxplots = st.checkbox("Include Box Plots", value=True)
                include_descriptive = st.checkbox("Include Descriptive Statistics", value=True)
                
                # Store options
                numeric_options = {
                    "columns": selected_num_cols,
                    "histograms": include_histograms,
                    "boxplots": include_boxplots,
                    "descriptive": include_descriptive
                }
    else:
        numeric_options = {}
    
    # Categorical Data Analysis section
    include_categorical = st.checkbox("Categorical Data Analysis", value=True)
    if include_categorical:
        selected_sections.append("categorical")
        with st.container():
            st.markdown("##### Categorical Analysis Options")
            
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found in the dataset.")
            else:
                # Select columns to include
                selected_cat_cols = st.multiselect(
                    "Select columns to analyze:",
                    cat_cols,
                    default=cat_cols[:min(5, len(cat_cols))]
                )
                
                # Chart types
                include_bar_charts = st.checkbox("Include Bar Charts", value=True)
                include_pie_charts = st.checkbox("Include Pie Charts", value=False)
                include_frequency = st.checkbox("Include Frequency Tables", value=True)
                
                # Maximum categories to display
                max_categories = st.slider("Maximum categories per chart:", 5, 30, 10)
                
                # Store options
                categorical_options = {
                    "columns": selected_cat_cols,
                    "bar_charts": include_bar_charts,
                    "pie_charts": include_pie_charts,
                    "frequency": include_frequency,
                    "max_categories": max_categories
                }
    else:
        categorical_options = {}
    
    # Correlation Analysis section
    include_correlation = st.checkbox("Correlation Analysis", value=True)
    if include_correlation:
        selected_sections.append("correlation")
        with st.container():
            st.markdown("##### Correlation Analysis Options")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for correlation analysis.")
            else:
                # Select columns to include
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    num_cols,
                    default=num_cols[:min(7, len(num_cols))]
                )
                
                # Correlation method
                corr_method = st.selectbox(
                    "Correlation method:",
                    ["Pearson", "Spearman", "Kendall"]
                )
                
                # Visualization options
                include_heatmap = st.checkbox("Include Correlation Heatmap", value=True)
                include_corr_table = st.checkbox("Include Correlation Table", value=True)
                
                # Correlation threshold
                corr_threshold = st.slider("Highlight correlations with absolute value above:", 0.0, 1.0, 0.7)
                
                # Store options
                correlation_options = {
                    "columns": selected_corr_cols,
                    "method": corr_method,
                    "heatmap": include_heatmap,
                    "table": include_corr_table,
                    "threshold": corr_threshold
                }
    else:
        correlation_options = {}
    
    # Missing Data Analysis section
    include_missing = st.checkbox("Missing Data Analysis", value=True)
    if include_missing:
        selected_sections.append("missing")
        with st.container():
            st.markdown("##### Missing Data Analysis Options")
            
            # Visualization options
            include_missing_bar = st.checkbox("Include Missing Values Bar Chart", value=True)
            include_missing_heatmap = st.checkbox("Include Missing Values Heatmap", value=False)
            include_missing_table = st.checkbox("Include Missing Values Table", value=True)
            
            # Store options
            missing_options = {
                "bar_chart": include_missing_bar,
                "heatmap": include_missing_heatmap,
                "table": include_missing_table
            }
    else:
        missing_options = {}
    
    # Additional Custom Section
    include_custom = st.checkbox("Custom Text Section", value=False)
    if include_custom:
        selected_sections.append("custom")
        with st.container():
            st.markdown("##### Custom Section Content")
            
            custom_section_title = st.text_input("Section Title:", "Additional Insights")
            custom_section_content = st.text_area("Section Content:", "Enter your custom analysis and insights here.")
            
            # Store options
            custom_options = {
                "title": custom_section_title,
                "content": custom_section_content
            }
    else:
        custom_options = {}
    
    # Generate report
    if st.button("Generate Custom Report", use_container_width=True):
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        
        # Add description
        report += f"{report_description}\n\n"
        
        # Add date generated
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add author information if included
        if include_author and author_name:
            report += f"**Author:** {author_name}\n"
            if author_info:
                report += f"**Contact:** {author_info}\n"
            report += "\n"
        
        # Generate each selected section
        section_num = 1
        
        # Dataset Overview section
        if "overview" in selected_sections:
            report += f"## {section_num}. Dataset Overview\n\n"
            section_num += 1
            
            if overview_options.get("basic_stats", False):
                # Basic statistics
                report += "### Basic Statistics\n\n"
                report += f"* **Rows:** {len(st.session_state.df):,}\n"
                report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
                
                # File details if available
                if hasattr(st.session_state, 'file_details') and st.session_state.file_details is not None:
                    report += f"* **Filename:** {st.session_state.file_details['filename']}\n"
                
                # Missing values
                missing_values = st.session_state.df.isna().sum().sum()
                missing_pct = (missing_values / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100).round(2)
                report += f"* **Missing Values:** {missing_values:,} ({missing_pct}% of all values)\n\n"
            
            if overview_options.get("data_types", False):
                # Data types summary
                report += "### Data Types\n\n"
                
                # Count by type
                type_counts = st.session_state.df.dtypes.astype(str).value_counts().to_dict()
                for dtype, count in type_counts.items():
                    report += f"* **{dtype}:** {count} columns\n"
                
                report += "\n"
            
            if overview_options.get("sample_data", False):
                # Sample data preview
                report += "### Data Preview\n\n"
                
                # Convert first 5 rows to markdown table
                sample_df = st.session_state.df.head(5)
                report += sample_df.to_markdown(index=False)
                report += "\n\n"
        
        # Numeric Data Analysis section
        if "numeric" in selected_sections and numeric_options.get("columns"):
            report += f"## {section_num}. Numeric Data Analysis\n\n"
            section_num += 1
            
            selected_num_cols = numeric_options.get("columns")
            
            if numeric_options.get("descriptive", False):
                # Descriptive statistics
                report += "### Descriptive Statistics\n\n"
                
                # Calculate statistics
                stats_df = st.session_state.df[selected_num_cols].describe().T.round(3)
                
                # Add additional statistics
                stats_df['median'] = st.session_state.df[selected_num_cols].median().round(3)
                stats_df['missing'] = st.session_state.df[selected_num_cols].isna().sum()
                stats_df['missing_pct'] = (st.session_state.df[selected_num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
                
                try:
                    stats_df['skew'] = st.session_state.df[selected_num_cols].skew().round(3)
                    stats_df['kurtosis'] = st.session_state.df[selected_num_cols].kurtosis().round(3)
                except:
                    pass
                
                # Convert to markdown table
                report += stats_df.reset_index().rename(columns={'index': 'Column'}).to_markdown(index=False)
                report += "\n\n"
            
            # Add note about visualizations
            if numeric_options.get("histograms", False) or numeric_options.get("boxplots", False):
                report += "### Visualizations\n\n"
                report += "*The report includes the following visualizations for numeric columns:*\n\n"
                
                if numeric_options.get("histograms", False):
                    report += "* **Histograms** showing the distribution of values for each numeric column\n"
                
                if numeric_options.get("boxplots", False):
                    report += "* **Box plots** showing the central tendency and spread of each numeric column\n"
                
                report += "\n*These visualizations will be included in the exported report.*\n\n"
        
        # Categorical Data Analysis section
        if "categorical" in selected_sections and categorical_options.get("columns"):
            report += f"## {section_num}. Categorical Data Analysis\n\n"
            section_num += 1
            
            selected_cat_cols = categorical_options.get("columns")
            max_categories = categorical_options.get("max_categories", 10)
            
            if categorical_options.get("frequency", False):
                report += "### Frequency Analysis\n\n"
                
                # For each categorical column, show frequency table
                for col in selected_cat_cols:
                    report += f"#### {col}\n\n"
                    report += f"* **Unique Values:** {st.session_state.df[col].nunique()}\n"
                    report += f"* **Missing Values:** {st.session_state.df[col].isna().sum()} ({st.session_state.df[col].isna().sum() / len(st.session_state.df) * 100:.2f}%)\n\n"
                    
                    # Get top categories
                    value_counts = st.session_state.df[col].value_counts().nlargest(max_categories)
                    total_values = len(st.session_state.df)
                    
                    # Create frequency table
                    freq_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / total_values * 100).round(2)
                    })
                    
                    # Check if there are more categories not shown
                    if st.session_state.df[col].nunique() > max_categories:
                        report += f"*Top {max_categories} categories shown (out of {st.session_state.df[col].nunique()} total)*\n\n"
                    
                    # Convert to markdown table
                    report += freq_df.to_markdown(index=False)
                    report += "\n\n"
            
            # Add note about visualizations
            if categorical_options.get("bar_charts", False) or categorical_options.get("pie_charts", False):
                report += "### Visualizations\n\n"
                report += "*The report includes the following visualizations for categorical columns:*\n\n"
                
                if categorical_options.get("bar_charts", False):
                    report += "* **Bar charts** showing the frequency of each category\n"
                
                if categorical_options.get("pie_charts", False):
                    report += "* **Pie charts** showing the proportion of each category\n"
                
                report += "\n*These visualizations will be included in the exported report.*\n\n"
        
        # Correlation Analysis section
        if "correlation" in selected_sections and correlation_options.get("columns") and len(correlation_options.get("columns", [])) >= 2:
            report += f"## {section_num}. Correlation Analysis\n\n"
            section_num += 1
            
            selected_corr_cols = correlation_options.get("columns")
            corr_method = correlation_options.get("method", "Pearson")
            corr_threshold = correlation_options.get("threshold", 0.7)
            
            # Calculate correlation matrix
            corr_matrix = st.session_state.df[selected_corr_cols].corr(method=corr_method.lower()).round(3)
            
            if correlation_options.get("table", False):
                report += f"### {corr_method} Correlation Matrix\n\n"
                
                # Convert to markdown table
                report += corr_matrix.to_markdown()
                report += "\n\n"
            
            # Strong correlations
            report += "### Strong Correlations\n\n"
            
            # Find pairs with strong correlation
            strong_pairs = []
            for i, col1 in enumerate(selected_corr_cols):
                for j, col2 in enumerate(selected_corr_cols):
                    if i < j:  # Only include each pair once
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) >= corr_threshold:
                            strong_pairs.append({
                                'Column 1': col1,
                                'Column 2': col2,
                                'Correlation': corr_val
                            })
            
            if strong_pairs:
                # Convert to dataframe and sort
                strong_df = pd.DataFrame(strong_pairs)
                strong_df = strong_df.sort_values('Correlation', key=abs, ascending=False)
                
                report += f"*Pairs with absolute correlation ≥ {corr_threshold}:*\n\n"
                report += strong_df.to_markdown(index=False)
                report += "\n\n"
            else:
                report += f"*No pairs with absolute correlation ≥ {corr_threshold} found.*\n\n"
            
            # Add note about visualization
            if correlation_options.get("heatmap", False):
                report += "### Correlation Heatmap\n\n"
                report += "*A correlation heatmap will be included in the exported report.*\n\n"
        
        # Missing Data Analysis section
        if "missing" in selected_sections:
            report += f"## {section_num}. Missing Data Analysis\n\n"
            section_num += 1
            
            # Calculate missing values for each column
            missing_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Missing Count': st.session_state.df.isna().sum().values,
                'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
            })
            
            # Sort by missing count (descending)
            missing_df = missing_df.sort_values('Missing Count', ascending=False)
            
            # Get columns with missing values
            cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
            
            if len(cols_with_missing) > 0:
                report += f"**Total Columns with Missing Values:** {len(cols_with_missing)} out of {len(st.session_state.df.columns)}\n\n"
                
                if missing_options.get("table", False):
                    # Convert to markdown table
                    report += cols_with_missing.to_markdown(index=False)
                    report += "\n\n"
                
                # Add note about visualization
                if missing_options.get("bar_chart", False) or missing_options.get("heatmap", False):
                    report += "### Missing Value Visualizations\n\n"
                    report += "*The report includes the following visualizations for missing data:*\n\n"
                    
                    if missing_options.get("bar_chart", False):
                        report += "* **Bar chart** showing the percentage of missing values in each column\n"
                    
                    if missing_options.get("heatmap", False):
                        report += "* **Heatmap** showing patterns of missing values across the dataset\n"
                    
                    report += "\n*These visualizations will be included in the exported report.*\n\n"
            else:
                report += "*No missing values found in the dataset.*\n\n"
        
        # Custom text section
        if "custom" in selected_sections:
            report += f"## {section_num}. {custom_options.get('title', 'Additional Insights')}\n\n"
            section_num += 1
            
            report += custom_options.get('content', '')
            report += "\n\n"
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate visualizations mentioned in the report
        if (("numeric" in selected_sections and (numeric_options.get("histograms", False) or numeric_options.get("boxplots", False))) or
            ("categorical" in selected_sections and (categorical_options.get("bar_charts", False) or categorical_options.get("pie_charts", False))) or
            ("correlation" in selected_sections and correlation_options.get("heatmap", False)) or
            ("missing" in selected_sections and (missing_options.get("bar_chart", False) or missing_options.get("heatmap", False)))):
            
            st.subheader("Report Visualizations")
            
            # Numeric visualizations
            if "numeric" in selected_sections:
                selected_num_cols = numeric_options.get("columns", [])
                
                if numeric_options.get("histograms", False) and selected_num_cols:
                    st.markdown("#### Histograms")
                    # Show one example histogram
                    example_col = selected_num_cols[0]
                    fig = px.histogram(
                        st.session_state.df,
                        x=example_col,
                        title=f"Distribution of {example_col}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(selected_num_cols) > 1:
                        st.info(f"Histograms for all {len(selected_num_cols)} selected numeric columns will be included in the exported report.")
                
                if numeric_options.get("boxplots", False) and selected_num_cols:
                    st.markdown("#### Box Plots")
                    # Create box plots
                    fig = go.Figure()
                    
                    for col in selected_num_cols[:3]:  # Show up to 3 columns
                        fig.add_trace(
                            go.Box(
                                y=st.session_state.df[col],
                                name=col
                            )
                        )
                    
                    fig.update_layout(title="Box Plots of Numeric Columns")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(selected_num_cols) > 3:
                        st.info(f"Box plots for all {len(selected_num_cols)} selected numeric columns will be included in the exported report.")
            
            # Categorical visualizations
            if "categorical" in selected_sections:
                selected_cat_cols = categorical_options.get("columns", [])
                max_categories = categorical_options.get("max_categories", 10)
                
                if categorical_options.get("bar_charts", False) and selected_cat_cols:
                    st.markdown("#### Bar Charts")
                    # Show one example bar chart
                    example_col = selected_cat_cols[0]
                    
                    # Get top categories
                    value_counts = st.session_state.df[example_col].value_counts().nlargest(max_categories)
                    
                    # Create bar chart
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Frequency of {example_col} Categories",
                        labels={"x": example_col, "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(selected_cat_cols) > 1:
                        st.info(f"Bar charts for all {len(selected_cat_cols)} selected categorical columns will be included in the exported report.")
                
                if categorical_options.get("pie_charts", False) and selected_cat_cols:
                    st.markdown("#### Pie Charts")
                    # Show one example pie chart
                    example_col = selected_cat_cols[0]
                    
                    # Get top categories
                    value_counts = st.session_state.df[example_col].value_counts().nlargest(max_categories)
                    
                    # Create pie chart
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {example_col} Categories"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(selected_cat_cols) > 1:
                        st.info(f"Pie charts for all {len(selected_cat_cols)} selected categorical columns will be included in the exported report.")
            
            # Correlation heatmap
            if "correlation" in selected_sections and correlation_options.get("heatmap", False):
                selected_corr_cols = correlation_options.get("columns", [])
                
                if len(selected_corr_cols) >= 2:
                    st.markdown("#### Correlation Heatmap")
                    
                    # Calculate correlation matrix
                    corr_method = correlation_options.get("method", "Pearson")
                    corr_matrix = st.session_state.df[selected_corr_cols].corr(method=corr_method.lower()).round(3)
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        title=f"{corr_method} Correlation Heatmap",
                        labels=dict(color="Correlation")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Missing data visualizations
            if "missing" in selected_sections:
                # Get columns with missing values
                missing_df = pd.DataFrame({
                    'Column': st.session_state.df.columns,
                    'Missing Count': st.session_state.df.isna().sum().values,
                    'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
                })
                
                # Sort by missing count (descending)
                missing_df = missing_df.sort_values('Missing Count', ascending=False)
                
                # Get columns with missing values
                cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
                
                if len(cols_with_missing) > 0:
                    if missing_options.get("bar_chart", False):
                        st.markdown("#### Missing Values Bar Chart")
                        
                        # Create bar chart
                        fig = px.bar(
                            cols_with_missing,
                            x='Column',
                            y='Missing %',
                            title='Missing Values by Column (%)',
                            color='Missing %',
                            color_continuous_scale="Reds"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if missing_options.get("heatmap", False):
                        st.markdown("#### Missing Values Heatmap")
                        
                        # Limit to columns with missing values
                        cols_to_plot = cols_with_missing['Column'].tolist()
                        
                        if cols_to_plot:
                            # Create sample for heatmap (limit to 100 rows for performance)
                            sample_size = min(100, len(st.session_state.df))
                            sample_df = st.session_state.df[cols_to_plot].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[cols_to_plot]
                            
                            # Create heatmap
                            fig = px.imshow(
                                sample_df.isna(),
                                labels=dict(x="Column", y="Row", color="Missing"),
                                color_continuous_scale=["blue", "red"],
                                title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as markdown
            st.download_button(
                label="Download as Markdown",
                data=report,
                file_name=f"{report_title.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            try:
                import markdown
                html = markdown.markdown(report)
                
                st.download_button(
                    label="Download as HTML",
                    data=html,
                    file_name=f"{report_title.replace(' ', '_').lower()}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_export_options():
    """Render export options tab"""
    st.subheader("Export Options")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Data")
        
        # Create exporter instance
        exporter = DataExporter(st.session_state.df)
        
        # Render export options
        exporter.render_export_options()
    
    with col2:
        st.markdown("### Export Charts")
        
        # Select chart type to export
        chart_type = st.selectbox(
            "Select chart type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Configure chart based on type
        if chart_type == "Bar Chart":
            # Column selections
            x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Create chart
            fig = px.bar(
                st.session_state.df,
                x=x_col,
                y=y_col,
                title=f"Bar Chart: {x_col} vs {y_col}"
            )
        
        elif chart_type == "Line Chart":
            # Column selections
            x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist())
            y_col = st.selectbox("Y-axis:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Create chart
            fig = px.line(
                st.session_state.df,
                x=x_col,
                y=y_col,
                title=f"Line Chart: {x_col} vs {y_col}"
            )
        
        elif chart_type == "Scatter Plot":
            # Column selections
            x_col = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist())
            y_col = st.selectbox("Y-axis:", [col for col in num_cols if col != x_col] if len(num_cols) > 1 else num_cols)
            
            # Create chart
            fig = px.scatter(
                st.session_state.df,
                x=x_col,
                y=y_col,
                title=f"Scatter Plot: {x_col} vs {y_col}"
            )
        
        elif chart_type == "Pie Chart":
            # Column selections
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            name_col = st.selectbox("Name column:", cat_cols if cat_cols else st.session_state.df.columns.tolist())
            
            # Limit categories for better visualization
            top_n = st.slider("Top N categories:", 3, 20, 10)
            
            # Calculate values for pie chart
            value_counts = st.session_state.df.groupby(name_col)[value_col].sum().nlargest(top_n)
            
            # Create chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Pie Chart: {value_col} by {name_col}"
            )
        
        elif chart_type == "Histogram":
            # Column selection
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Number of bins
            bins = st.slider("Number of bins:", 5, 100, 20)
            
            # Create chart
            fig = px.histogram(
                st.session_state.df,
                x=value_col,
                nbins=bins,
                title=f"Histogram of {value_col}"
            )
        
        elif chart_type == "Box Plot":
            # Column selection
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist())
            
            # Optional grouping
            use_groups = st.checkbox("Group by category")
            if use_groups and cat_cols:
                group_col = st.selectbox("Group by:", cat_cols)
                
                # Create chart with grouping
                fig = px.box(
                    st.session_state.df,
                    x=group_col,
                    y=value_col,
                    title=f"Box Plot of {value_col} by {group_col}"
                )
            else:
                # Create chart without grouping
                fig = px.box(
                    st.session_state.df,
                    y=value_col,
                    title=f"Box Plot of {value_col}"
                )
        
        elif chart_type == "Heatmap":
            if len(num_cols) < 1:
                st.warning("Need numeric columns for heatmap.")
                return
            
            # Create correlation matrix for heatmap
            corr_cols = st.multiselect(
                "Select columns for correlation heatmap:",
                num_cols,
                default=num_cols[:min(8, len(num_cols))]
            )
            
            if len(corr_cols) < 2:
                st.warning("Please select at least two columns for correlation heatmap.")
            else:
                # Create chart
                corr_matrix = st.session_state.df[corr_cols].corr().round(2)
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("### Export Chart")
        
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["PNG", "JPEG", "SVG", "HTML"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Chart", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

def render_dashboard():
    """Render dashboard section"""
    st.header("Interactive Dashboard")
    
    # Dashboard generator
    dashboard_generator = DashboardGenerator(st.session_state.df)
    
    # Create or load dashboard
    with st.expander("Dashboard Settings", expanded=True):
        # Dashboard title
        dashboard_title = st.text_input("Dashboard Title:", "Data Insights Dashboard")
        
        # Dashboard layout
        layout_type = st.radio(
            "Layout Type:",
            ["2 columns", "3 columns", "Custom"],
            horizontal=True
        )
        
        # Components selection
        st.subheader("Add Dashboard Components")
        
        # Create tabs for component types
        component_tabs = st.tabs(["Charts", "Metrics", "Tables", "Filters"])
        
        with component_tabs[0]:
            # Charts section
            st.markdown("### Chart Components")
            
            # Get numeric and categorical columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Number of charts to add
            n_charts = st.number_input("Number of charts to add:", 1, 8, 3)
            
            # List to store chart configurations
            chart_configs = []
            
            for i in range(n_charts):
                with st.container():
                    st.markdown(f"#### Chart {i+1}")
                    
                    # Chart type selection
                    chart_type = st.selectbox(
                        "Chart type:",
                        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"],
                        key=f"chart_type_{i}"
                    )
                    
                    # Chart title
                    chart_title = st.text_input("Chart title:", value=f"Chart {i+1}", key=f"chart_title_{i}")
                    
                    # Configure chart based on type
                    if chart_type == "Bar Chart":
                        # Column selections
                        x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "bar",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Line Chart":
                        # Column selections
                        x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "line",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Scatter Plot":
                        # Column selections
                        x_col = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis:", [col for col in num_cols if col != x_col] if len(num_cols) > 1 else num_cols, key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "scatter",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Pie Chart":
                        # Column selections
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        name_col = st.selectbox("Name column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key=f"name_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "pie",
                            "title": chart_title,
                            "value": value_col,
                            "name": name_col
                        })
                    
                    elif chart_type == "Histogram":
                        # Column selection
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "histogram",
                            "title": chart_title,
                            "value": value_col
                        })
                    
                    elif chart_type == "Box Plot":
                        # Column selection
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        
                        # Optional grouping
                        use_groups = st.checkbox("Group by category", key=f"use_groups_{i}")
                        if use_groups and cat_cols:
                            group_col = st.selectbox("Group by:", cat_cols, key=f"group_col_{i}")
                            
                            # Add to configurations with grouping
                            chart_configs.append({
                                "type": "box",
                                "title": chart_title,
                                "value": value_col,
                                "group": group_col
                            })
                        else:
                            # Add to configurations without grouping
                            chart_configs.append({
                                "type": "box",
                                "title": chart_title,
                                "value": value_col
                            })
                    
                    st.markdown("---")
        
        with component_tabs[1]:
            # Metrics section
            st.markdown("### Metric Components")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.info("No numeric columns found for metrics.")
            else:
                # Number of metrics to add
                n_metrics = st.number_input("Number of metrics to add:", 1, 8, 4)
                
                # List to store metric configurations
                metric_configs = []
                
                for i in range(n_metrics):
                    with st.container():
                        st.markdown(f"#### Metric {i+1}")
                        
                        # Metric label
                        metric_label = st.text_input("Metric label:", value=f"Metric {i+1}", key=f"metric_label_{i}")
                        
                        # Metric column
                        metric_col = st.selectbox("Column:", num_cols, key=f"metric_col_{i}")
                        
                        # Aggregation function
                        agg_func = st.selectbox(
                            "Aggregation:",
                            ["Mean", "Median", "Sum", "Min", "Max", "Count"],
                            key=f"agg_func_{i}"
                        )
                        
                        # Formatting
                        format_type = st.selectbox(
                            "Format as:",
                            ["Number", "Percentage", "Currency"],
                            key=f"format_type_{i}"
                        )
                        
                        # Decimal places
                        decimal_places = st.slider("Decimal places:", 0, 4, 2, key=f"decimal_places_{i}")
                        
                        # Add to configurations
                        metric_configs.append({
                            "label": metric_label,
                            "column": metric_col,
                            "aggregation": agg_func.lower(),
                            "format": format_type.lower(),
                            "decimals": decimal_places
                        })
                        
                        st.markdown("---")
        
        with component_tabs[2]:
            # Tables section
            st.markdown("### Table Components")
            
            # Number of tables to add
            n_tables = st.number_input("Number of tables to add:", 0, 4, 1)
            
            # List to store table configurations
            table_configs = []
            
            for i in range(n_tables):
                with st.container():
                    st.markdown(f"#### Table {i+1}")
                    
                    # Table title
                    table_title = st.text_input("Table title:", value=f"Table {i+1}", key=f"table_title_{i}")
                    
                    # Columns to include
                    include_cols = st.multiselect(
                        "Columns to include:",
                        st.session_state.df.columns.tolist(),
                        default=st.session_state.df.columns[:min(5, len(st.session_state.df.columns))].tolist(),
                        key=f"include_cols_{i}"
                    )
                    
                    # Max rows
                    max_rows = st.slider("Maximum rows:", 5, 100, 10, key=f"max_rows_{i}")
                    
                    # Include index
                    include_index = st.checkbox("Include row index", value=False, key=f"include_index_{i}")
                    
                    # Add to configurations
                    table_configs.append({
                        "title": table_title,
                        "columns": include_cols,
                        "max_rows": max_rows,
                        "include_index": include_index
                    })
                    
                    st.markdown("---")
        
        with component_tabs[3]:
            # Filters section
            st.markdown("### Filter Components")
            
            # Number of filters to add
            n_filters = st.number_input("Number of filters to add:", 0, 8, 2)
            
            # List to store filter configurations
            filter_configs = []
            
            for i in range(n_filters):
                with st.container():
                    st.markdown(f"#### Filter {i+1}")
                    
                    # Filter label
                    filter_label = st.text_input("Filter label:", value=f"Filter {i+1}", key=f"filter_label_{i}")
                    
                    # Filter column
                    filter_col = st.selectbox("Column to filter:", st.session_state.df.columns.tolist(), key=f"filter_col_{i}")
                    
                    # Filter type (based on column data type)
                    if st.session_state.df[filter_col].dtype.kind in 'bifc':  # Numeric
                        filter_type = "range"
                    elif st.session_state.df[filter_col].dtype.kind == 'O' and st.session_state.df[filter_col].nunique() <= 20:  # Categorical with few values
                        filter_type = "multiselect"
                    else:  # Categorical with many values or other types
                        filter_type = "text"
                    
                    # Add to configurations
                    filter_configs.append({
                        "label": filter_label,
                        "column": filter_col,
                        "type": filter_type
                    })
                    
                    st.markdown("---")
    
    # Generate the dashboard
    st.subheader(dashboard_title)
    
    # Create a filtered dataframe based on any active filters
    if 'filter_configs' in locals() and filter_configs:
        filtered_df = dashboard_generator.apply_filters(filter_configs)
    else:
        filtered_df = st.session_state.df.copy()
    
    # Display filters
    if 'filter_configs' in locals() and filter_configs:
        st.markdown("### Filters")
        
        # Create columns for filters
        filter_cols = st.columns(min(3, len(filter_configs)))
        
        # Display each filter
        for i, filter_config in enumerate(filter_configs):
            col_idx = i % len(filter_cols)
            with filter_cols[col_idx]:
                dashboard_generator.render_filter(filter_config, key_suffix=f"dash_{i}")
        
        # Apply filters button
        if st.button("Apply Filters", use_container_width=True):
            filtered_df = dashboard_generator.apply_filters(filter_configs)
    
    # Display metrics
    if 'metric_configs' in locals() and metric_configs:
        st.markdown("### Key Metrics")
        
        # Create columns for metrics
        metric_cols = st.columns(min(4, len(metric_configs)))
        
        # Display each metric
        for i, metric_config in enumerate(metric_configs):
            col_idx = i % len(metric_cols)
            with metric_cols[col_idx]:
                dashboard_generator.render_metric(filtered_df, metric_config)
    
    # Display charts
    if 'chart_configs' in locals() and chart_configs:
        st.markdown("### Charts")
        
        # Determine layout
        if layout_type == "2 columns":
            n_cols = 2
        elif layout_type == "3 columns":
            n_cols = 3
        else:  # Custom
            n_cols = st.slider("Number of columns for charts:", 1, 4, 2)
        
        # Create columns for charts
        chart_cols = []
        for i in range(0, len(chart_configs), n_cols):
            # For each row, create n_cols columns (unless we're at the end)
            row_cols = st.columns(min(n_cols, len(chart_configs) - i))
            chart_cols.extend(row_cols)
        
        # Display each chart
        for i, chart_config in enumerate(chart_configs):
            with chart_cols[i]:
                dashboard_generator.render_chart(filtered_df, chart_config)
    
    # Display tables
    if 'table_configs' in locals() and table_configs:
        st.markdown("### Data Tables")
        
        # Display each table
        for table_config in table_configs:
            st.subheader(table_config["title"])
            dashboard_generator.render_table(filtered_df, table_config)

# Main execution
if __name__ == "__main__":
    main()

def render_exploratory_report():
    """Render exploratory analysis report tab"""
    st.subheader("Exploratory Data Analysis Report")
    
    # Report configuration
    st.write("Generate an exploratory data analysis (EDA) report with visualizations and insights.")
    
    with st.expander("Report Configuration", expanded=True):
        # Report title and description
        report_title = st.text_input("Report Title:", "Exploratory Data Analysis Report")
        report_description = st.text_area("Report Description:", "This report provides an exploratory analysis of the dataset.")
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Comprehensive"],
            value="Standard"
        )
        
        # Visualization settings
        st.write("Visualization Settings:")
        max_categorical_values = st.slider("Max categorical values to display:", 5, 30, 10)
        correlation_threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.5)
    
    # Generate report
    if st.button("Generate EDA Report", use_container_width=True):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        report += f"{report_description}\n\n"
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Analysis Depth:** {analysis_depth}\n\n"
        
        # Dataset Overview
        report += "## 1. Dataset Overview\n\n"
        
        # Basic stats
        report += f"* **Rows:** {len(st.session_state.df):,}\n"
        report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
        
        # Data types
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        other_cols = [col for col in st.session_state.df.columns if col not in num_cols + cat_cols + date_cols]
        
        report += f"* **Numeric Columns:** {len(num_cols)}\n"
        report += f"* **Categorical Columns:** {len(cat_cols)}\n"
        report += f"* **DateTime Columns:** {len(date_cols)}\n"
        if other_cols:
            report += f"* **Other Columns:** {len(other_cols)}\n"
        
        # Missing data overview
        missing_values = st.session_state.df.isna().sum().sum()
        missing_pct = (missing_values / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100).round(2)
        report += f"* **Missing Values:** {missing_values:,} ({missing_pct}% of all values)\n\n"
        
        progress_bar.progress(10)
        
        # Column Analysis
        report += "## 2. Column Analysis\n\n"
        
        # Sort columns by type for better organization
        all_cols = num_cols + cat_cols + date_cols + other_cols
        
        # For standard and comprehensive analysis, analyze all columns
        # For basic analysis, limit to a subset
        if analysis_depth == "Basic":
            max_cols_per_type = 3
            analyzed_num_cols = num_cols[:min(max_cols_per_type, len(num_cols))]
            analyzed_cat_cols = cat_cols[:min(max_cols_per_type, len(cat_cols))]
            analyzed_date_cols = date_cols[:min(max_cols_per_type, len(date_cols))]
            analyzed_cols = analyzed_num_cols + analyzed_cat_cols + analyzed_date_cols + other_cols
        else:
            analyzed_num_cols = num_cols
            analyzed_cat_cols = cat_cols
            analyzed_date_cols = date_cols
            analyzed_cols = all_cols
        
        # Initialize charts counter for progress tracking
        total_charts = len(analyzed_cols)
        charts_done = 0
        
        # Analysis for each column
        for col in analyzed_cols:
            # Determine column type
            if col in num_cols:
                col_type = "Numeric"
            elif col in cat_cols:
                col_type = "Categorical"
            elif col in date_cols:
                col_type = "DateTime"
            else:
                col_type = "Other"
            
            report += f"### 2.{analyzed_cols.index(col) + 1}. {col} ({col_type})\n\n"
            
            # Basic column statistics
            non_null_count = st.session_state.df[col].count()
            null_count = st.session_state.df[col].isna().sum()
            null_pct = (null_count / len(st.session_state.df) * 100).round(2)
            
            report += f"* **Non-null Count:** {non_null_count:,} ({100 - null_pct:.2f}%)\n"
            report += f"* **Null Count:** {null_count:,} ({null_pct}%)\n"
            
            if col_type == "Numeric":
                # Numeric column analysis
                numeric_stats = st.session_state.df[col].describe().to_dict()
                
                report += f"* **Mean:** {numeric_stats['mean']:.3f}\n"
                report += f"* **Median:** {st.session_state.df[col].median():.3f}\n"
                report += f"* **Std Dev:** {numeric_stats['std']:.3f}\n"
                report += f"* **Min:** {numeric_stats['min']:.3f}\n"
                report += f"* **Max:** {numeric_stats['max']:.3f}\n"
                
                # Additional statistics for comprehensive analysis
                if analysis_depth == "Comprehensive":
                    try:
                        skewness = st.session_state.df[col].skew()
                        kurtosis = st.session_state.df[col].kurtosis()
                        
                        report += f"* **Skewness:** {skewness:.3f}\n"
                        report += f"* **Kurtosis:** {kurtosis:.3f}\n"
                        
                        # Interpret skewness
                        if abs(skewness) < 0.5:
                            report += f"* **Distribution:** Approximately symmetric\n"
                        elif skewness < 0:
                            report += f"* **Distribution:** Negatively skewed (left-tailed)\n"
                        else:
                            report += f"* **Distribution:** Positively skewed (right-tailed)\n"
                    except:
                        pass
                
                report += f"\n**Histogram will be included in the final report.**\n\n"
                
            elif col_type == "Categorical":
                # Categorical column analysis
                unique_count = st.session_state.df[col].nunique()
                report += f"* **Unique Values:** {unique_count}\n"
                
                # Show top categories
                value_counts = st.session_state.df[col].value_counts().nlargest(max_categorical_values)
                value_pcts = (value_counts / len(st.session_state.df) * 100).round(2)
                
                report += f"\n**Top {min(max_categorical_values, unique_count)} Values:**\n\n"
                
                # Create value counts dataframe
                vc_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': value_pcts.values
                })
                
                # Convert to markdown table
                report += vc_df.to_markdown(index=False)
                report += "\n\n**Bar chart will be included in the final report.**\n\n"
                
            elif col_type == "DateTime":
                # DateTime column analysis
                try:
                    min_date = st.session_state.df[col].min()
                    max_date = st.session_state.df[col].max()
                    date_range = max_date - min_date
                    
                    report += f"* **Minimum Date:** {min_date}\n"
                    report += f"* **Maximum Date:** {max_date}\n"
                    report += f"* **Date Range:** {date_range}\n"
                    
                    # For comprehensive analysis, include time-based statistics
                    if analysis_depth in ["Standard", "Comprehensive"]:
                        # Extract year, month, day
                        years = pd.DatetimeIndex(st.session_state.df[col].dropna()).year
                        months = pd.DatetimeIndex(st.session_state.df[col].dropna()).month
                        days = pd.DatetimeIndex(st.session_state.df[col].dropna()).day
                        
                        # Most common year
                        if len(years) > 0:
                            common_year = pd.Series(years).value_counts().nlargest(1).index[0]
                            common_year_count = pd.Series(years).value_counts().nlargest(1).values[0]
                            common_year_pct = (common_year_count / len(years) * 100).round(2)
                            
                            report += f"* **Most Common Year:** {common_year} ({common_year_count:,} occurrences, {common_year_pct}%)\n"
                        
                        # For comprehensive analysis, include month and day statistics
                        if analysis_depth == "Comprehensive":
                            # Most common month
                            if len(months) > 0:
                                common_month = pd.Series(months).value_counts().nlargest(1).index[0]
                                common_month_count = pd.Series(months).value_counts().nlargest(1).values[0]
                                common_month_pct = (common_month_count / len(months) * 100).round(2)
                                
                                # Convert month number to name
                                import calendar
                                month_name = calendar.month_name[common_month]
                                
                                report += f"* **Most Common Month:** {month_name} ({common_month_count:,} occurrences, {common_month_pct}%)\n"
                            
                            # Most common day of week
                            try:
                                weekdays = pd.DatetimeIndex(st.session_state.df[col].dropna()).dayofweek
                                
                                if len(weekdays) > 0:
                                    common_weekday = pd.Series(weekdays).value_counts().nlargest(1).index[0]
                                    common_weekday_count = pd.Series(weekdays).value_counts().nlargest(1).values[0]
                                    common_weekday_pct = (common_weekday_count / len(weekdays) * 100).round(2)
                                    
                                    # Convert weekday number to name
                                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                    weekday_name = weekday_names[common_weekday]
                                    
                                    report += f"* **Most Common Day of Week:** {weekday_name} ({common_weekday_count:,} occurrences, {common_weekday_pct}%)\n"
                            except:
                                pass
                                
                    report += f"\n**Time series plot will be included in the final report.**\n\n"
                except Exception as e:
                    report += f"*Error analyzing datetime column: {str(e)}*\n\n"
            
            # Update progress
            charts_done += 1
            progress_bar.progress(10 + int(40 * charts_done / total_charts))
        
        # Correlation Analysis (for numeric columns)
        if len(analyzed_num_cols) >= 2:
            report += "## 3. Correlation Analysis\n\n"
            
            # Calculate correlation matrix
            corr_matrix = st.session_state.df[analyzed_num_cols].corr().round(3)
            
            # For basic analysis, limit to top correlations
            if analysis_depth == "Basic":
                report += "### Top Correlations\n\n"
                
                # Convert correlation matrix to a list of pairs
                pairs = []
                for i, col1 in enumerate(analyzed_num_cols):
                    for j, col2 in enumerate(analyzed_num_cols):
                        if i < j:  # Only include each pair once
                            pairs.append({
                                'Column 1': col1,
                                'Column 2': col2,
                                'Correlation': corr_matrix.loc[col1, col2]
                            })
                
                # Convert to dataframe
                pairs_df = pd.DataFrame(pairs)
                
                # Sort by absolute correlation
                pairs_df['Abs Correlation'] = pairs_df['Correlation'].abs()
                pairs_df = pairs_df.sort_values('Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
                
                # Display top correlations above threshold
                strong_pairs = pairs_df[pairs_df['Correlation'].abs() >= correlation_threshold]
                
                if len(strong_pairs) > 0:
                    report += f"**Strong correlations (|r| ≥ {correlation_threshold}):**\n\n"
                    report += strong_pairs.to_markdown(index=False)
                    report += "\n\n"
                else:
                    report += f"*No strong correlations found (using threshold |r| ≥ {correlation_threshold}).*\n\n"
            else:
                # For standard and comprehensive analysis, include full correlation matrix
                report += "### Correlation Matrix\n\n"
                report += "*Correlation heatmap will be included in the final report.*\n\n"
                
                if analysis_depth == "Comprehensive":
                    report += "### Full Correlation Matrix\n\n"
                    report += corr_matrix.to_markdown()
                    report += "\n\n"
        
        progress_bar.progress(60)
        
        # Missing Value Analysis
        report += "## 4. Missing Value Analysis\n\n"
        
        # Calculate missing values for each column
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isna().sum().values,
            'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
        })
        
        # Sort by missing count (descending)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        # Get columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
        
        if len(cols_with_missing) > 0:
            report += f"**Total Columns with Missing Values:** {len(cols_with_missing)} out of {len(st.session_state.df.columns)}\n\n"
            
            # Convert to markdown table
            report += cols_with_missing.to_markdown(index=False)
            report += "\n\n*Missing values bar chart will be included in the final report.*\n\n"
            
            # For comprehensive analysis, include detailed pattern analysis
            if analysis_depth == "Comprehensive" and len(cols_with_missing) > 1:
                report += "### Missing Value Patterns\n\n"
                report += "*Missing values heatmap will be included in the final report.*\n\n"
                
                # Calculate correlations between missing values
                # This indicates if missingness in one column correlates with missingness in another
                missing_pattern = st.session_state.df[cols_with_missing['Column']].isna()
                missing_corr = missing_pattern.corr().round(3)
                
                report += "**Missing Value Correlation Matrix** (correlation between missing patterns)\n\n"
                report += missing_corr.to_markdown()
                report += "\n\n*High correlation suggests missing values occur together in the same rows.*\n\n"
        else:
            report += "*No missing values found in the dataset.*\n\n"
        
        progress_bar.progress(70)
        
        # Outlier Analysis (for numeric columns)
        if len(analyzed_num_cols) > 0:
            report += "## 5. Outlier Analysis\n\n"
            
            outlier_summary = []
            
            for col in analyzed_num_cols:
                # Calculate IQR
                Q1 = st.session_state.df[col].quantile(0.25)
                Q3 = st.session_state.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = st.session_state.df[(st.session_state.df[col] < lower_bound) | 
                                     (st.session_state.df[col] > upper_bound)]
                
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(st.session_state.df) * 100).round(2)
                
                # Add to summary
                outlier_summary.append({
                    'Column': col,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Outlier Count': outlier_count,
                    'Outlier %': outlier_pct
                })
            
            # Convert to dataframe
            outlier_df = pd.DataFrame(outlier_summary)
            
            # Sort by outlier count (descending)
            outlier_df = outlier_df.sort_values('Outlier Count', ascending=False)
            
            # For standard and comprehensive analysis, include detailed outlier information
            if analysis_depth in ["Standard", "Comprehensive"]:
                report += "### Outlier Summary\n\n"
                
                # Convert to markdown table
                summary_cols = ['Column', 'Outlier Count', 'Outlier %', 'Lower Bound', 'Upper Bound']
                report += outlier_df[summary_cols].to_markdown(index=False)
                report += "\n\n"
                
                report += "*Box plots for columns with outliers will be included in the final report.*\n\n"
                
                # For comprehensive analysis, include detailed statistics
                if analysis_depth == "Comprehensive":
                    report += "### Detailed Outlier Statistics\n\n"
                    report += outlier_df.to_markdown(index=False)
                    report += "\n\n"
            else:
                # For basic analysis, just include a summary
                report += f"**Total columns with outliers:** {len(outlier_df[outlier_df['Outlier Count'] > 0])}\n\n"
                
                # Show top columns with outliers
                top_outliers = outlier_df[outlier_df['Outlier Count'] > 0].head(5)
                
                if len(top_outliers) > 0:
                    report += "**Top columns with outliers:**\n\n"
                    report += top_outliers[['Column', 'Outlier Count', 'Outlier %']].to_markdown(index=False)
                    report += "\n\n"
                else:
                    report += "*No outliers found in the dataset using the IQR method.*\n\n"
        
        progress_bar.progress(80)
        
        # Distribution Analysis (for categorical columns)
        if len(analyzed_cat_cols) > 0:
            report += "## 6. Categorical Data Analysis\n\n"
            
            # For each categorical column, analyze distribution
            for i, col in enumerate(analyzed_cat_cols[:5]):  # Limit to top 5 columns
                unique_count = st.session_state.df[col].nunique()
                
                report += f"### 6.{i+1}. {col}\n\n"
                report += f"* **Unique Values:** {unique_count}\n"
                
                # For columns with many unique values, show value counts
                if unique_count <= 30 or analysis_depth == "Comprehensive":
                    # Get value counts
                    value_counts = st.session_state.df[col].value_counts().nlargest(max_categorical_values)
                    value_pcts = (value_counts / len(st.session_state.df) * 100).round(2)
                    
                    report += f"\n**Top {min(max_categorical_values, unique_count)} Values:**\n\n"
                    
                    # Create value counts dataframe
                    vc_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': value_pcts.values
                    })
                    
                    # Convert to markdown table
                    report += vc_df.to_markdown(index=False)
                    report += "\n\n"
                
                report += "*Bar chart will be included in the final report.*\n\n"
            
            # If too many columns, note that only top ones were shown
            if len(analyzed_cat_cols) > 5:
                report += f"*Note: Only the first 5 out of {len(analyzed_cat_cols)} categorical columns are shown in detail.*\n\n"
        
        progress_bar.progress(90)
        
        # Insights and Recommendations
        report += "## 7. Insights and Recommendations\n\n"
        
        # Basic insights
        report += "### Key Insights\n\n"
        
        # Dataset size
        report += f"* **Dataset Size:** The dataset contains {len(st.session_state.df):,} rows and {len(st.session_state.df.columns)} columns.\n"
        
        # Data types
        report += f"* **Data Composition:** The dataset includes {len(num_cols)} numeric columns, {len(cat_cols)} categorical columns, and {len(date_cols)} datetime columns.\n"
        
        # Missing values
        if missing_values > 0:
            report += f"* **Missing Data:** {missing_values:,} values ({missing_pct}% of all values) are missing in the dataset.\n"
            
            # Top columns with missing values
            top_missing = cols_with_missing.head(3)
            if len(top_missing) > 0:
                missing_cols = ", ".join([f"'{col}' ({pct}%)" for col, pct in zip(top_missing['Column'], top_missing['Missing %'])])
                report += f"  * Columns with most missing values: {missing_cols}\n"
        else:
            report += f"* **Missing Data:** No missing values found in the dataset.\n"
        
        # Outliers (if analyzed)
        if len(analyzed_num_cols) > 0:
            outlier_cols = outlier_df[outlier_df['Outlier %'] > 5]
            if len(outlier_cols) > 0:
                report += f"* **Outliers:** {len(outlier_cols)} columns have more than 5% outliers.\n"
                
                # Top columns with outliers
                top_outlier_cols = outlier_cols.head(3)
                outlier_col_list = ", ".join([f"'{col}' ({pct}%)" for col, pct in zip(top_outlier_cols['Column'], top_outlier_cols['Outlier %'])])
                report += f"  * Columns with most outliers: {outlier_col_list}\n"
        
        # Correlations (if analyzed)
        if len(analyzed_num_cols) >= 2:
            strong_corrs = pairs_df[pairs_df['Correlation'].abs() >= 0.7]
            if len(strong_corrs) > 0:
                report += f"* **Strong Correlations:** {len(strong_corrs)} pairs of numeric columns have strong correlations (|r| ≥ 0.7).\n"
                
                # Top correlations
                top_corr = strong_corrs.head(2)
                if len(top_corr) > 0:
                    corr_list = ", ".join([f"'{col1}' and '{col2}' (r={corr:.2f})" for col1, col2, corr in zip(
                        top_corr['Column 1'], top_corr['Column 2'], top_corr['Correlation'])])
                    report += f"  * Strongest correlations: {corr_list}\n"
        
        # Recommendations
        report += "\n### Recommendations\n\n"
        
        # Missing data recommendations
        if missing_values > 0:
            report += "* **Missing Data Handling:**\n"
            report += "  * Consider imputation strategies for columns with missing values, such as mean/median imputation for numeric columns or mode imputation for categorical columns.\n"
            report += "  * For columns with high missing percentages, evaluate whether they should be kept or dropped.\n"
            report += "  * Investigate if missing data follows a pattern that could introduce bias.\n"
        
        # Outlier recommendations
        if len(analyzed_num_cols) > 0 and len(outlier_df[outlier_df['Outlier Count'] > 0]) > 0:
            report += "* **Outlier Treatment:**\n"
            report += "  * Inspect outliers to determine if they are valid data points or errors.\n"
            report += "  * Consider appropriate treatment such as capping, transforming, or removing outliers depending on your analysis goals.\n"
        
        # Correlation recommendations
        if len(analyzed_num_cols) >= 2:
            report += "* **Feature Selection/Engineering:**\n"
            if len(strong_corrs) > 0:
                report += "  * Highly correlated features may contain redundant information. Consider removing some of them or creating composite features.\n"
            report += "  * Explore feature engineering opportunities to create more predictive variables.\n"
        
        # Categorical data recommendations
        if len(analyzed_cat_cols) > 0:
            high_cardinality_cols = [col for col in analyzed_cat_cols if st.session_state.df[col].nunique() > 20]
            if high_cardinality_cols:
                report += "* **Categorical Variable Treatment:**\n"
                report += "  * Some categorical variables have high cardinality (many unique values). Consider grouping less frequent categories.\n"
                report += "  * For machine learning, use appropriate encoding techniques (one-hot encoding for low cardinality, target or frequency encoding for high cardinality).\n"
        
        # General recommendations
        report += "* **Further Analysis:**\n"
        report += "  * Consider more advanced analytics like clustering, dimensionality reduction, or predictive modeling.\n"
        report += "  * Validate any patterns or insights with domain experts.\n"
        
        progress_bar.progress(100)
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate charts for visualizations mentioned in the report
        st.subheader("Report Charts")
        
        # Create a list to store chart figures
        charts = []
        
        # Numeric column histograms
        for col in analyzed_num_cols[:3]:  # Limit to first 3 columns
            fig = px.histogram(
                st.session_state.df,
                x=col,
                title=f"Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Categorical column bar charts
        for col in analyzed_cat_cols[:3]:  # Limit to first 3 columns
            # Limit to top categories
            top_cats = st.session_state.df[col].value_counts().nlargest(max_categorical_values).index.tolist()
            filtered_df = st.session_state.df[st.session_state.df[col].isin(top_cats)]
            
            fig = px.bar(
                filtered_df[col].value_counts().reset_index(),
                x="index",
                y=col,
                title=f"Frequency of {col} Categories (Top {len(top_cats)})",
                labels={"index": col, col: "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Correlation heatmap
        if len(analyzed_num_cols) >= 2:
            corr_matrix = st.session_state.df[analyzed_num_cols].corr().round(3)
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Missing values bar chart
        if len(cols_with_missing) > 0:
            fig = px.bar(
                cols_with_missing,
                x='Column',
                y='Missing %',
                title='Missing Values by Column (%)',
                color='Missing %',
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Export options
        st.subheader("Export Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as markdown
            st.download_button(
                label="Download as Markdown",
                data=report,
                file_name=f"{report_title.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            try:
                import markdown
                html = markdown.markdown(report)
                
                st.download_button(
                    label="Download as HTML",
                    data=html,
                    file_name=f"{report_title.replace(' ', '_').lower()}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_trend_analysis():
    """Render trend analysis tab"""
    st.subheader("Trend Analysis")
    
    # Check if we have date columns
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Also look for potential date columns
    potential_date_cols = []
    for col in st.session_state.df.columns:
        if col in date_cols:
            continue
        
        if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
            potential_date_cols.append(col)
    
    all_date_cols = date_cols + potential_date_cols
    
    if not all_date_cols:
        st.info("No date/time columns found for trend analysis. Convert a column to datetime first.")
        return
    
    # Column selections
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Date/Time column:", all_date_cols)
    
    with col2:
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns found for trend analysis.")
            return
        
        y_col = st.selectbox("Value column:", num_cols)
    
    # Convert date column to datetime if it's not already
    if x_col in potential_date_cols:
        try:
            # Try to convert to datetime
            st.session_state.df[x_col] = pd.to_datetime(st.session_state.df[x_col])
            st.success(f"Converted '{x_col}' to datetime format.")
        except Exception as e:
            st.error(f"Could not convert '{x_col}' to datetime: {str(e)}")
            st.info("Try selecting a different column or convert it to datetime in the Data Processing tab.")
            return
    
    # Trend analysis type
    analysis_type = st.radio(
        "Analysis type:",
        ["Time Series Plot", "Moving Averages", "Seasonality Analysis", "Trend Decomposition"],
        horizontal=True
    )
    
    if analysis_type == "Time Series Plot":
        # Optional grouping
        use_grouping = st.checkbox("Group by a categorical column")
        group_col = None
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                group_col = st.selectbox("Group by:", cat_cols)
        
        # Create time series plot
        if use_grouping and group_col:
            # Get top groups for better visualization
            top_n = st.slider("Show top N groups:", 1, 10, 5)
            top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
            
            # Filter to top groups
            filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
            
            # Create grouped time series
            fig = px.line(
                filtered_df,
                x=x_col,
                y=y_col,
                color=group_col,
                title=f"Time Series Plot of {y_col} over {x_col}, grouped by {group_col}"
            )
        else:
            # Simple time series
            fig = px.line(
                st.session_state.df,
                x=x_col,
                y=y_col,
                title=f"Time Series Plot of {y_col} over {x_col}"
            )
        
        # Add trend line
        add_trend = st.checkbox("Add trend line")
        if add_trend:
            try:
                from scipy import stats as scipy_stats
                
                # Convert dates to ordinal values for regression
                x_values = pd.to_numeric(pd.to_datetime(st.session_state.df[x_col])).values
                y_values = st.session_state.df[y_col].values
                
                # Remove NaN values
                mask = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[mask]
                y_values = y_values[mask]
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_values, y_values)
                
                # Create line
                x_range = np.array([x_values.min(), x_values.max()])
                y_range = intercept + slope * x_range
                
                # Convert back to datetime for plotting
                x_dates = pd.to_datetime(x_range)
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=x_dates,
                        y=y_range,
                        mode="lines",
                        name=f"Trend (r²={r_value**2:.3f})",
                        line=dict(color="red", dash="dash")
                    )
                )
                
            except Exception as e:
                st.warning(f"Could not add trend line: {str(e)}")
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader("Time Series Statistics")
        
        # Calculate statistics
        try:
            # First and last values
            first_value = st.session_state.df.sort_values(x_col).iloc[0][y_col]
            last_value = st.session_state.df.sort_values(x_col).iloc[-1][y_col]
            
            # Change and percent change
            change = last_value - first_value
            pct_change = (change / first_value) * 100 if first_value != 0 else float('inf')
            
            # Min and max values
            min_value = st.session_state.df[y_col].min()
            max_value = st.session_state.df[y_col].max()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("First Value", f"{first_value:.2f}")
            with col2:
                st.metric("Last Value", f"{last_value:.2f}")
            with col3:
                st.metric("Change", f"{change:.2f}")
            with col4:
                st.metric("% Change", f"{pct_change:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Minimum", f"{min_value:.2f}")
            with col2:
                st.metric("Maximum", f"{max_value:.2f}")
            with col3:
                st.metric("Range", f"{max_value - min_value:.2f}")
            with col4:
                st.metric("Mean", f"{st.session_state.df[y_col].mean():.2f}")
            
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
    
    elif analysis_type == "Moving Averages":
        # Window sizes
        window_sizes = st.multiselect(
            "Select moving average window sizes:",
            [7, 14, 30, 60, 90, 180, 365],
            default=[30]
        )
        
        if not window_sizes:
            st.warning("Please select at least one window size.")
            return
        
        # Create base time series plot
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_col],
                y=st.session_state.df[y_col],
                mode="lines",
                name=f"Original {y_col}",
                line=dict(color="blue")
            )
        )
        
        # Add moving averages
        colors = ["red", "green", "purple", "orange", "brown", "pink", "grey"]
        
        for i, window in enumerate(window_sizes):
            # Calculate moving average
            ma = st.session_state.df[y_col].rolling(window=window).mean()
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=ma,
                    mode="lines",
                    name=f"{window}-period MA",
                    line=dict(color=colors[i % len(colors)])
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Moving Averages of {y_col} over {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis options
        st.subheader("Additional Analysis")
        
        # Volatility analysis
        show_volatility = st.checkbox("Show volatility (rolling standard deviation)")
        if show_volatility:
            # Create volatility plot
            fig_vol = go.Figure()
            
            # Calculate rolling standard deviation
            vol_window = st.slider("Volatility window size:", 5, 100, 30)
            volatility = st.session_state.df[y_col].rolling(window=vol_window).std()
            
            # Add to plot
            fig_vol.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=volatility,
                    mode="lines",
                    name=f"{vol_window}-period Volatility",
                    line=dict(color="red")
                )
            )
            
            # Update layout
            fig_vol.update_layout(
                title=f"Volatility (Rolling Std Dev) of {y_col} over {x_col}",
                xaxis_title=x_col,
                yaxis_title=f"Standard Deviation of {y_col}"
            )
            
            # Show plot
            st.plotly_chart(fig_vol, use_container_width=True)
    
    elif analysis_type == "Seasonality Analysis":
        try:
            # Make sure we have enough data
            if len(st.session_state.df) < 10:
                st.warning("Need more data points for seasonality analysis.")
                return
            
            # Check if the data is regularly spaced in time
            df_sorted = st.session_state.df.sort_values(x_col)
            date_diffs = df_sorted[x_col].diff().dropna()
            
            if date_diffs.nunique() > 5:
                st.warning("Data points are not regularly spaced in time. Seasonality analysis may not be accurate.")
            
            # Time period selection
            period_type = st.selectbox(
                "Analyze seasonality by:",
                ["Day of Week", "Month", "Quarter", "Year"]
            )
            
            # Extract the relevant period component
            if period_type == "Day of Week":
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.day_name()
                period_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            elif period_type == "Month":
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                period_order = ["January", "February", "March", "April", "May", "June", 
                               "July", "August", "September", "October", "November", "December"]
            elif period_type == "Quarter":
                df_sorted['period'] = "Q" + pd.to_datetime(df_sorted[x_col]).dt.quarter.astype(str)
                period_order = ["Q1", "Q2", "Q3", "Q4"]
            else:  # Year
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.year
                years = sorted(df_sorted['period'].unique())
                period_order = years
            
            # Calculate statistics by period
            period_stats = df_sorted.groupby('period')[y_col].agg(['mean', 'median', 'min', 'max', 'std', 'count'])
            
            # Reorder based on natural order
            if period_type != "Year":
                # For named periods, use predefined order
                period_stats = period_stats.reindex(period_order)
            else:
                # For years, sort numerically
                period_stats = period_stats.sort_index()
            
            # Fill NaN with 0
            period_stats = period_stats.fillna(0)
            
            # Create bar chart
            fig = go.Figure()
            
            # Add mean bars
            fig.add_trace(
                go.Bar(
                    x=period_stats.index,
                    y=period_stats['mean'],
                    name="Mean",
                    error_y=dict(
                        type='data',
                        array=period_stats['std'],
                        visible=True
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Seasonality Analysis of {y_col} by {period_type}",
                xaxis_title=period_type,
                yaxis_title=f"Mean {y_col} (with Std Dev)"
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics table
            st.subheader(f"Statistics by {period_type}")
            st.dataframe(period_stats.round(2), use_container_width=True)
            
            # Heatmap visualization
            st.subheader("Seasonality Heatmap")
            
            # Create heatmap based on period type
            if period_type in ["Month", "Day of Week"]:
                # For month or day, we can also show year-on-year patterns
                
                if period_type == "Month":
                    # Extract month and year
                    df_sorted['month'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                    df_sorted['year'] = pd.to_datetime(df_sorted[x_col]).dt.year
                    
                    # Calculate monthly averages by year
                    heatmap_data = df_sorted.groupby(['year', 'month'])[y_col].mean().unstack()
                    
                    # Reorder months
                    heatmap_data = heatmap_data[period_order]
                    
                else:  # Day of Week
                    # Extract day of week and week of year
                    df_sorted['day'] = pd.to_datetime(df_sorted[x_col]).dt.day_name()
                    
                    # Use month or week number as the other dimension
                    use_month = st.radio(
                        "Second dimension for day of week heatmap:",
                        ["Month", "Week of Year"],
                        horizontal=True
                    )
                    
                    if use_month:
                        df_sorted['second_dim'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                        dim_order = ["January", "February", "March", "April", "May", "June", 
                                   "July", "August", "September", "October", "November", "December"]
                        dim_name = "Month"
                    else:
                        df_sorted['second_dim'] = pd.to_datetime(df_sorted[x_col]).dt.isocalendar().week
                        weeks = sorted(df_sorted['second_dim'].unique())
                        dim_order = weeks
                        dim_name = "Week of Year"
                    
                    # Calculate averages
                    heatmap_data = df_sorted.groupby(['second_dim', 'day'])[y_col].mean().unstack()
                    
                    # Reorder days
                    heatmap_data = heatmap_data[period_order]
                
                # Create heatmap
                try:
                    # Convert to numpy for Plotly
                    z_data = heatmap_data.values
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='Blues',
                        colorbar=dict(title=f"Mean {y_col}")
                    ))
                    
                    # Update layout
                    if period_type == "Month":
                        fig.update_layout(
                            title=f"Monthly {y_col} Heatmap by Year",
                            xaxis_title="Month",
                            yaxis_title="Year"
                        )
                    else:
                        fig.update_layout(
                            title=f"Day of Week {y_col} Heatmap by {dim_name}",
                            xaxis_title="Day of Week",
                            yaxis_title=dim_name
                        )
                    
                    # Show heatmap
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in seasonality analysis: {str(e)}")
    
    elif analysis_type == "Trend Decomposition":
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Need to make sure data is sorted and has a regular time index
            st.info("Trend decomposition requires regularly spaced time series data.")
            
            # Sort data by date
            df_sorted = st.session_state.df.sort_values(x_col)
            
            # Check if we need to resample
            resample = st.checkbox("Resample data to regular intervals")
            
            if resample:
                # Frequency selection
                freq = st.selectbox(
                    "Resampling frequency:",
                    ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
                )
                
                # Map to pandas frequency string
                freq_map = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "MS",
                    "Quarterly": "QS",
                    "Yearly": "YS"
                }
                
                # Set date as index
                df_sorted = df_sorted.set_index(x_col)
                
                # Resample
                df_resampled = df_sorted[y_col].resample(freq_map[freq]).mean()
                
                # Convert back to dataframe
                data_ts = pd.DataFrame({y_col: df_resampled})
                
                # Reset index
                data_ts = data_ts.reset_index()
                
                # Rename index column
                data_ts = data_ts.rename(columns={"index": x_col})
                
            else:
                # Use original data
                data_ts = df_sorted[[x_col, y_col]]
            
            # Make sure we have enough data
            if len(data_ts) < 10:
                st.warning("Need at least 10 data points for decomposition. Try resampling to a lower frequency.")
                return
            
            # Set date as index
            data_ts = data_ts.set_index(x_col)
            
            # Decomposition model
            model = st.selectbox(
                "Decomposition model:",
                ["Additive", "Multiplicative"]
            )
            
            # Period selection
            period = st.slider("Period (number of time units in a seasonal cycle):", 2, 52, 12)
            
            # Perform decomposition
            if model == "Additive":
                result = seasonal_decompose(data_ts[y_col], model='additive', period=period)
            else:
                result = seasonal_decompose(data_ts[y_col], model='multiplicative', period=period)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                shared_xaxes=True,
                vertical_spacing=0.05
            )
            
            # Add observed data
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.observed,
                    mode="lines",
                    name="Observed"
                ),
                row=1, col=1
            )
            
            # Add trend
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.trend,
                    mode="lines",
                    name="Trend",
                    line=dict(color="red")
                ),
                row=2, col=1
            )
            
            # Add seasonal
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.seasonal,
                    mode="lines",
                    name="Seasonal",
                    line=dict(color="green")
                ),
                row=3, col=1
            )
            
            # Add residual
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.resid,
                    mode="lines",
                    name="Residual",
                    line=dict(color="purple")
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title=f"{model} Decomposition of {y_col} (Period={period})",
                showlegend=False
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Component Statistics")
            
            # Calculate statistics for each component
            stats = pd.DataFrame({
                "Component": ["Observed", "Trend", "Seasonal", "Residual"],
                "Mean": [
                    result.observed.mean(),
                    result.trend.mean(),
                    result.seasonal.mean(),
                    result.resid.dropna().mean()
                ],
                "Std Dev": [
                    result.observed.std(),
                    result.trend.std(),
                    result.seasonal.std(),
                    result.resid.dropna().std()
                ],
                "Min": [
                    result.observed.min(),
                    result.trend.min(),
                    result.seasonal.min(),
                    result.resid.dropna().min()
                ],
                "Max": [
                    result.observed.max(),
                    result.trend.max(),
                    result.seasonal.max(),
                    result.resid.dropna().max()
                ]
            })
            
            # Display statistics
            st.dataframe(stats.round(2), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in trend decomposition: {str(e)}")
            st.info("Try resampling the data or adjusting the period parameter.")
    
    # Export options
    if 'fig' in locals():
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
        
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Chart",
                data=html_bytes,
                file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            # Export as image
            img_bytes = fig.to_image(format=export_format.lower())
            
            st.download_button(
                label=f"Download Chart",
                data=img_bytes,
                file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True
            )

def render_distribution_analysis():
    """Render distribution analysis tab"""
    st.subheader("Distribution Analysis")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns found for distribution analysis.")
        return
    
    # Column selection
    selected_col = st.selectbox(
        "Select column for distribution analysis:",
        num_cols
    )
    
    # Analysis type selection
    analysis_type = st.radio(
        "Analysis type:",
        ["Histogram", "Density Plot", "Box Plot", "Q-Q Plot", "Distribution Fitting"],
        horizontal=True
    )
    
    if analysis_type == "Histogram":
        # Histogram options
        n_bins = st.slider("Number of bins:", 5, 100, 20)
        
        # Create histogram
        fig = px.histogram(
            st.session_state.df,
            x=selected_col,
            nbins=n_bins,
            title=f"Histogram of {selected_col}",
            marginal="box"
        )
        
        # Show histogram
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate basic statistics
        stats = st.session_state.df[selected_col].describe().to_dict()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{stats['mean']:.3f}")
        with col2:
            st.metric("Median", f"{stats['50%']:.3f}")
        with col3:
            st.metric("Std Dev", f"{stats['std']:.3f}")
        with col4:
            st.metric("Count", f"{stats['count']}")
        
    elif analysis_type == "Density Plot":
        # Create KDE plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Calculate KDE
            kde_x = np.linspace(data.min(), data.max(), 1000)
            kde = scipy_stats.gaussian_kde(data)
            kde_y = kde(kde_x)
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            show_hist = st.checkbox("Show histogram with density plot", value=True)
            if show_hist:
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name="Histogram",
                        opacity=0.7,
                        histnorm="probability density"
                    )
                )
            
            # Add KDE
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name="KDE",
                    line=dict(color="red", width=2)
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Density Plot of {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Density"
            )
            
            # Show density plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate skewness and kurtosis
            skewness = scipy_stats.skew(data)
            kurtosis = scipy_stats.kurtosis(data)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{data.mean():.3f}")
            with col2:
                st.metric("Median", f"{data.median():.3f}")
            with col3:
                st.metric("Skewness", f"{skewness:.3f}")
            with col4:
                st.metric("Kurtosis", f"{kurtosis:.3f}")
            
            # Interpretation of skewness and kurtosis
            st.subheader("Distribution Interpretation")
            
            # Skewness interpretation
            if abs(skewness) < 0.5:
                skew_text = "approximately symmetric"
            elif skewness < 0:
                skew_text = "negatively skewed (left-tailed)"
            else:
                skew_text = "positively skewed (right-tailed)"
            
            # Kurtosis interpretation
            if abs(kurtosis) < 0.5:
                kurt_text = "approximately mesokurtic (similar to normal distribution)"
            elif kurtosis < 0:
                kurt_text = "platykurtic (flatter than normal distribution)"
            else:
                kurt_text = "leptokurtic (more peaked than normal distribution)"
            
            st.write(f"The distribution is {skew_text} and {kurt_text}.")
            
        except Exception as e:
            st.error(f"Error creating density plot: {str(e)}")
            return
        
    elif analysis_type == "Box Plot":
        # Optional grouping
        use_grouping = st.checkbox("Group by a categorical column")
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                group_col = st.selectbox("Group by:", cat_cols)
                
                # Limit number of groups
                top_n = st.slider("Show top N groups:", 2, 20, 10)
                
                # Get top groups
                top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
                
                # Filter to top groups
                filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
                
                # Create grouped box plot
                fig = px.box(
                    filtered_df,
                    x=group_col,
                    y=selected_col,
                    title=f"Box Plot of {selected_col} by {group_col}",
                    points="all"
                )
        
        if not use_grouping or not cat_cols:
            # Simple box plot
            fig = px.box(
                st.session_state.df,
                y=selected_col,
                title=f"Box Plot of {selected_col}",
                points="all"
            )
        
        # Show box plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate quartiles and IQR
        q1 = st.session_state.df[selected_col].quantile(0.25)
        q3 = st.session_state.df[selected_col].quantile(0.75)
        iqr = q3 - q1
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{st.session_state.df[selected_col].min():.3f}")
        with col2:
            st.metric("Q1 (25%)", f"{q1:.3f}")
        with col3:
            st.metric("Median", f"{st.session_state.df[selected_col].median():.3f}")
        with col4:
            st.metric("Q3 (75%)", f"{q3:.3f}")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Max", f"{st.session_state.df[selected_col].max():.3f}")
        with col2:
            st.metric("IQR", f"{iqr:.3f}")
        with col3:
            # Calculate outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = st.session_state.df[(st.session_state.df[selected_col] < lower_bound) | 
                                  (st.session_state.df[selected_col] > upper_bound)]
            
            st.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(st.session_state.df)*100:.1f}%)")
        
    elif analysis_type == "Q-Q Plot":
        # Distribution selection
        dist = st.selectbox(
            "Reference distribution:",
            ["Normal", "Uniform", "Exponential", "Log-normal"]
        )
        
        # Create Q-Q plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Calculate theoretical quantiles
            if dist == "Normal":
                theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Normal"
            elif dist == "Uniform":
                theoretical_quantiles = scipy_stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Uniform"
            elif dist == "Exponential":
                theoretical_quantiles = scipy_stats.expon.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Exponential"
            elif dist == "Log-normal":
                theoretical_quantiles = scipy_stats.lognorm.ppf(np.linspace(0.01, 0.99, len(data)), s=1)
                theoretical_name = "Log-normal"
            
            # Sort the data
            sample_quantiles = np.sort(data)
            
            # Create Q-Q plot
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Data'
                )
            )
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Q-Q Plot: {selected_col} vs {theoretical_name} Distribution",
                xaxis_title=f"Theoretical Quantiles ({theoretical_name})",
                yaxis_title=f"Sample Quantiles ({selected_col})"
            )
            
            # Show Q-Q plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Perform Shapiro-Wilk test for normality if normal distribution selected
            if dist == "Normal":
                shapiro_test = scipy_stats.shapiro(data)
                
                st.subheader("Normality Test (Shapiro-Wilk)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{shapiro_test.statistic:.4f}")
                with col2:
                    st.metric("p-value", f"{shapiro_test.pvalue:.4f}")
                
                alpha = 0.05
                if shapiro_test.pvalue < alpha:
                    st.write(f"With p-value < {alpha}, we can reject the null hypothesis that the data is normally distributed.")
                else:
                    st.write(f"With p-value >= {alpha}, we cannot reject the null hypothesis that the data is normally distributed.")
            
        except Exception as e:
            st.error(f"Error creating Q-Q plot: {str(e)}")
            return
        
    elif analysis_type == "Distribution Fitting":
        # Try to fit distributions to the data
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Distributions to try
            distributions = [
                ("Normal", scipy_stats.norm),
                ("Log-normal", scipy_stats.lognorm),
                ("Exponential", scipy_stats.expon),
                ("Gamma", scipy_stats.gamma),
                ("Beta", scipy_stats.beta),
                ("Weibull", scipy_stats.weibull_min)
            ]
            
            # Select distributions to try
            selected_dists = st.multiselect(
                "Select distributions to fit:",
                [d[0] for d in distributions],
                default=["Normal", "Log-normal", "Gamma"]
            )
            
            if not selected_dists:
                st.warning("Please select at least one distribution to fit.")
                return
            
            # Filter to selected distributions
            distributions = [d for d in distributions if d[0] in selected_dists]
            
            # Fit distributions and calculate goodness of fit
            results = []
            
            for dist_name, distribution in distributions:
                try:
                    # Handle special case for lognorm
                    if dist_name == "Log-normal":
                        # For lognormal, use only positive values
                        pos_data = data[data > 0]
                        if len(pos_data) < 10:
                            results.append({
                                "Distribution": dist_name,
                                "Parameters": "N/A",
                                "AIC": float('inf'),
                                "BIC": float('inf'),
                                "Error": "Not enough positive values for Log-normal"
                            })
                            continue
                            
                        # Fit using positive data
                        params = distribution.fit(pos_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(pos_data, *params))
                    elif dist_name == "Beta":
                        # For beta, scale data to [0, 1]
                        min_val = data.min()
                        max_val = data.max()
                        scaled_data = (data - min_val) / (max_val - min_val)
                        # Handle edge cases
                        scaled_data = np.clip(scaled_data, 0.001, 0.999)
                        # Fit using scaled data
                        params = distribution.fit(scaled_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(scaled_data, *params))
                    elif dist_name == "Weibull":
                        # For Weibull, use only positive values and shift
                        pos_data = data[data > 0]
                        if len(pos_data) < 10:
                            results.append({
                                "Distribution": dist_name,
                                "Parameters": "N/A",
                                "AIC": float('inf'),
                                "BIC": float('inf'),
                                "Error": "Not enough positive values for Weibull"
                            })
                            continue
                        # Fit using positive data
                        params = distribution.fit(pos_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(pos_data, *params))
                    else:
                        # Fit distribution
                        params = distribution.fit(data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(data, *params))
                    
                    # Calculate AIC and BIC
                    k = len(params)
                    n = len(data)
                    aic = 2 * k - 2 * loglik
                    bic = k * np.log(n) - 2 * loglik
                    
                    results.append({
                        "Distribution": dist_name,
                        "Parameters": params,
                        "AIC": aic,
                        "BIC": bic,
                        "Error": None
                    })
                    
                except Exception as e:
                    results.append({
                        "Distribution": dist_name,
                        "Parameters": "N/A",
                        "AIC": float('inf'),
                        "BIC": float('inf'),
                        "Error": str(e)
                    })
            
            # Filter out errors
            valid_results = [r for r in results if r["Error"] is None]
            
            if not valid_results:
                st.error("Could not fit any of the selected distributions to the data.")
                return
            
            # Find best fit by AIC
            best_fit = min(valid_results, key=lambda x: x["AIC"])
            
            # Create visualization
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name="Data",
                    opacity=0.7,
                    histnorm="probability density",
                    nbinsx=30
                )
            )
            
            # Add fitted distributions
            x = np.linspace(data.min(), data.max(), 1000)
            
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
            
            for i, result in enumerate(valid_results):
                dist_name = result["Distribution"]
                params = result["Parameters"]
                dist_obj = [d[1] for d in distributions if d[0] == dist_name][0]
                
                # Calculate PDF
                try:
                    if dist_name == "Log-normal":
                        # Handle special case for lognormal
                        pdf = dist_obj.pdf(x, *params)
                    elif dist_name == "Beta":
                        # For beta, need to scale x to [0, 1]
                        min_val = data.min()
                        max_val = data.max()
                        scaled_x = (x - min_val) / (max_val - min_val)
                        # Handle edge cases
                        scaled_x = np.clip(scaled_x, 0.001, 0.999)
                        # Calculate PDF
                        pdf = dist_obj.pdf(scaled_x, *params) / (max_val - min_val)
                    elif dist_name == "Weibull":
                        # Handle special case for Weibull
                        pdf = dist_obj.pdf(x, *params)
                    else:
                        pdf = dist_obj.pdf(x, *params)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pdf,
                            mode="lines",
                            name=f"{dist_name} (AIC: {result['AIC']:.2f})",
                            line=dict(color=colors[i % len(colors)])
                        )
                    )
                except Exception as e:
                    st.warning(f"Could not plot {dist_name} distribution: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title=f"Distribution Fitting for {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Probability Density"
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display fitting results
            st.subheader("Goodness of Fit")
            
            # Create results table
            fit_results = pd.DataFrame([
                {
                    "Distribution": r["Distribution"],
                    "AIC": r["AIC"] if r["Error"] is None else "Error",
                    "BIC": r["BIC"] if r["Error"] is None else "Error",
                    "Status": "Success" if r["Error"] is None else f"Error: {r['Error']}"
                }
                for r in results
            ])
            
            # Sort by AIC (if available)
            fit_results = fit_results.sort_values(
                by="AIC", 
                key=lambda x: pd.to_numeric(x, errors='coerce'),
                ascending=True
            )
            
            st.dataframe(fit_results, use_container_width=True)
            
            # Display best fit
            st.subheader("Best Fit Distribution")
            st.info(f"The best fit distribution is **{best_fit['Distribution']}** with AIC = {best_fit['AIC']:.2f}.")
            
        except Exception as e:
            st.error(f"Error in distribution fitting: {str(e)}")
            return
    
    # Export options
    export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
    
    if 'fig' in locals():
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Chart",
                data=html_bytes,
                file_name=f"distribution_{analysis_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            # Export as image
            img_bytes = fig.to_image(format=export_format.lower())
            
            st.download_button(
                label=f"Download Chart",
                data=img_bytes,
                file_name=f"distribution_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True
            )