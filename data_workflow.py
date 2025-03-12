import streamlit as st
import pandas as pd
import os
import datetime
import uuid
import tempfile

class DataWorkflowManager:
    """
    Manages data workflow from upload to processing to usage by other modules.
    Ensures that modules only use processed data after processing is complete.
    """
    
    def __init__(self):
        """Initialize the workflow manager"""
        # Ensure workflow state exists
        if 'data_workflow' not in st.session_state:
            st.session_state.data_workflow = {
                'original_file': None,
                'processed_file': None,
                'processing_done': False,
                'temp_dir': tempfile.mkdtemp(),
                'processing_timestamp': None
            }
    
    def handle_upload(self, uploaded_file):
        """
        Handle the initial file upload
        Returns DataFrame of the uploaded file
        """
        if uploaded_file is None:
            return None
            
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Store original data info
            st.session_state.data_workflow['original_file'] = uploaded_file.name
            st.session_state.data_workflow['processing_done'] = False
            st.session_state.data_workflow['processed_file'] = None
            
            # Store a copy of the original data
            st.session_state.original_df = df.copy()
            
            # Generate a temporary file path for the original data
            orig_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                   f"original_{uuid.uuid4().hex}.csv")
            
            # Save the original file
            df.to_csv(orig_path, index=False)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def save_processed_data(self, df):
        """
        Save processed data and update workflow state
        """
        if df is None or df.empty:
            return False
            
        try:
            # Generate a unique filename for the processed data
            processed_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                       f"processed_{uuid.uuid4().hex}.csv")
            
            # Save the processed DataFrame to CSV
            df.to_csv(processed_path, index=False)
            
            # Update workflow state
            st.session_state.data_workflow['processed_file'] = processed_path
            st.session_state.data_workflow['processing_done'] = True
            st.session_state.data_workflow['processing_timestamp'] = datetime.datetime.now()
            
            return True
            
        except Exception as e:
            st.error(f"Error saving processed data: {str(e)}")
            return False
    
    def get_data(self, require_processed=True):
        """
        Get the appropriate data based on workflow state
        If require_processed is True, will only return data if processing is complete
        Otherwise, returns original data with a warning
        """
        # Check if processing is complete
        if st.session_state.data_workflow['processing_done']:
            # Return the processed data
            try:
                processed_path = st.session_state.data_workflow['processed_file']
                return pd.read_csv(processed_path)
            except Exception as e:
                st.error(f"Error loading processed data: {str(e)}")
                return None
        else:
            # Processing not complete
            if require_processed:
                st.warning("⚠️ Data processing has not been completed. Please process the data first.")
                return None
            else:
                # Return original data with a warning
                st.warning("⚠️ Using original data - processing has not been completed")
                return st.session_state.original_df.copy() if 'original_df' in st.session_state else None
    
    def is_processing_done(self):
        """Check if data processing is complete"""
        return st.session_state.data_workflow['processing_done']
    
    def get_workflow_status(self):
        """Get current workflow status details"""
        workflow = st.session_state.data_workflow
        
        return {
            'has_original_data': workflow['original_file'] is not None,
            'original_file': workflow['original_file'],
            'processing_done': workflow['processing_done'],
            'processing_timestamp': workflow['processing_timestamp'],
            'has_processed_data': workflow['processed_file'] is not None
        }
    
    def reset_workflow(self):
        """Reset the workflow state"""
        st.session_state.data_workflow = {
            'original_file': None,
            'processed_file': None,
            'processing_done': False,
            'temp_dir': st.session_state.data_workflow['temp_dir'],
            'processing_timestamp': None
        }
        
        # Also clear the dataframes from session state
        if 'original_df' in st.session_state:
            del st.session_state.original_df
        if 'df' in st.session_state:
            del st.session_state.df