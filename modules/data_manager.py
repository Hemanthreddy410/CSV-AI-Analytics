import streamlit as st
import pandas as pd
import datetime
from modules.file_handler import handle_uploaded_file, load_sample_data

def initialize_session_state():
    """Initialize the session state variables if they don't exist"""
    if 'df' not in st.session_state:
        st.session_state.df = None
        
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
        
    if 'file_details' not in st.session_state:
        st.session_state.file_details = None
        
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
        
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
        
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
        
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"

def load_file(uploaded_file):
    """
    Process an uploaded file and update session state
    
    Parameters:
    - uploaded_file: The file object from st.file_uploader
    
    Returns:
    - Boolean indicating success
    """
    if uploaded_file is None:
        return False
    
    # Process the file
    df, file_details = handle_uploaded_file(uploaded_file)
    
    if df is not None and file_details is not None:
        # Update session state
        st.session_state.df = df
        st.session_state.original_df = df.copy()
        st.session_state.file_details = file_details
        st.session_state.processing_history = []
        
        return True
    
    return False

def load_sample_dataset(sample_name):
    """
    Load a sample dataset and update session state
    
    Parameters:
    - sample_name: Name of the sample dataset
    
    Returns:
    - Boolean indicating success
    """
    # Load the sample data
    df, file_details = load_sample_data(sample_name)
    
    if df is not None and file_details is not None:
        # Update session state
        st.session_state.df = df
        st.session_state.original_df = df.copy()
        st.session_state.file_details = file_details
        st.session_state.processing_history = []
        
        return True
    
    return False

def save_to_project(project_name):
    """
    Save current data to a project
    
    Parameters:
    - project_name: Name of the project to save to
    
    Returns:
    - Boolean indicating success
    """
    if st.session_state.df is None:
        return False
    
    if project_name not in st.session_state.projects:
        # Create new project
        st.session_state.projects[project_name] = {
            'created': datetime.datetime.now(),
            'last_modified': datetime.datetime.now(),
            'data': st.session_state.df.copy(),
            'file_details': st.session_state.file_details,
            'processing_history': st.session_state.processing_history.copy() if hasattr(st.session_state, 'processing_history') else []
        }
    else:
        # Update existing project
        st.session_state.projects[project_name]['data'] = st.session_state.df.copy()
        st.session_state.projects[project_name]['file_details'] = st.session_state.file_details
        st.session_state.projects[project_name]['last_modified'] = datetime.datetime.now()
        st.session_state.projects[project_name]['processing_history'] = st.session_state.processing_history.copy() if hasattr(st.session_state, 'processing_history') else []
    
    st.session_state.current_project = project_name
    
    return True

def load_from_project(project_name):
    """
    Load data from a project
    
    Parameters:
    - project_name: Name of the project to load from
    
    Returns:
    - Boolean indicating success
    """
    if project_name not in st.session_state.projects:
        return False
    
    # Load project data
    project = st.session_state.projects[project_name]
    
    st.session_state.df = project['data'].copy()
    st.session_state.original_df = project['data'].copy()
    st.session_state.file_details = project['file_details']
    st.session_state.processing_history = project.get('processing_history', []).copy()
    st.session_state.current_project = project_name
    
    return True

def get_dataframe_info():
    """
    Get formatted information about the current dataframe
    
    Returns:
    - Dictionary of dataframe info
    """
    if st.session_state.df is None:
        return None
    
    df = st.session_state.df
    
    info = {
        'rows': format(df.shape[0], ','),
        'columns': df.shape[1],
        'memory_usage': format_memory_size(df.memory_usage(deep=True).sum()),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
        'missing_values': format(df.isna().sum().sum(), ','),
        'missing_pct': f"{df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%"
    }
    
    return info

def format_memory_size(size_bytes):
    """Format memory size from bytes to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.2f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"