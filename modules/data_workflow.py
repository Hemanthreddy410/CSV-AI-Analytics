import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import uuid
import tempfile
import gc
import psutil
import time
from pathlib import Path

class DataWorkflowManager:
    """
    Manages data workflow from upload to processing to usage by other modules.
    Includes optimizations for large datasets and memory management.
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
                'processing_timestamp': None,
                'full_data_loaded': False,
                'sample_size': 10000,
                'is_large_dataset': False,
                'performance_metrics': {}
            }
        
        # Create directories for data storage
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        # Create temp directory if it doesn't exist
        os.makedirs(st.session_state.data_workflow['temp_dir'], exist_ok=True)
        
        # Create cache directory
        cache_dir = os.path.join(st.session_state.data_workflow['temp_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
    
    def handle_upload(self, uploaded_file, sample_size=None):
        """
        Handle the initial file upload with optimizations for large files
        Returns DataFrame of the uploaded file
        
        Parameters:
        - uploaded_file: The file object from st.file_uploader
        - sample_size: Number of rows to sample for large files
        """
        if uploaded_file is None:
            return None
        
        try:
            # Display memory usage before loading
            mem_before = self.get_memory_usage()
            start_time = time.time()
            
            # Check if this might be a large file
            file_size_mb = uploaded_file.size / (1024 * 1024)
            is_large = file_size_mb > 100  # Consider files over 100MB as large
            
            if is_large:
                st.session_state.data_workflow['is_large_dataset'] = True
                # For large files, ask user if they want to load a sample first
                if sample_size is None:
                    st.info(f"This is a large file ({file_size_mb:.1f} MB). Loading a sample is recommended for faster processing.")
                    use_sample = st.checkbox("Load a sample of the data first", value=True)
                    if use_sample:
                        sample_size = st.slider("Sample size (rows):", 1000, 100000, 10000)
                        st.session_state.data_workflow['sample_size'] = sample_size
            else:
                st.session_state.data_workflow['is_large_dataset'] = False
            
            # Show a progress bar
            progress = st.progress(0)
            st.info("Reading file... This may take a moment.")
            
            # Determine file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Initialize dataframe
            df = None
            
            # Chunked reading for CSV files
            if file_extension == '.csv':
                # Try to count lines first for progress reporting
                try:
                    # Save to a temporary file for line counting
                    temp_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                         f"temp_{uuid.uuid4().hex}{file_extension}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Count lines
                    with open(temp_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    
                    # Reset uploaded_file
                    uploaded_file.seek(0)
                    
                    # Update progress
                    progress.progress(0.1)
                    
                    # If file is large, use chunked reading with sampling
                    if is_large:
                        chunk_size = 10000
                        chunks = []
                        
                        # Get CSV reading options from user
                        sep = st.selectbox("CSV Separator:", [",", ";", "\t", "|"], index=0)
                        encoding = st.selectbox("File Encoding:", ["utf-8", "latin1", "iso-8859-1"], index=0)
                        
                        # Read in chunks with progress updates
                        for i, chunk in enumerate(pd.read_csv(temp_path, chunksize=chunk_size, 
                                                           sep=sep, encoding=encoding, 
                                                           low_memory=True)):
                            # Update progress
                            progress_val = min(0.1 + 0.8 * ((i * chunk_size) / line_count), 0.9)
                            progress.progress(progress_val)
                            
                            # If sampling, only store a portion of each chunk
                            if sample_size is not None:
                                sample_frac = min(1.0, sample_size / line_count)
                                if np.random.random() < sample_frac:
                                    chunks.append(chunk)
                            else:
                                chunks.append(chunk)
                        
                        # Combine chunks
                        if chunks:
                            df = pd.concat(chunks, ignore_index=True)
                            
                            # Apply final sampling if needed
                            if sample_size is not None and len(df) > sample_size:
                                df = df.sample(sample_size, random_state=42)
                    else:
                        # For smaller files, read directly
                        df = pd.read_csv(temp_path, low_memory=True)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.warning(f"Error in optimized loading: {str(e)}. Falling back to standard loading.")
                    # Fallback to standard loading
                    df = pd.read_csv(uploaded_file, low_memory=True)
                    
                    if sample_size is not None and len(df) > sample_size:
                        df = df.sample(sample_size, random_state=42)
            
            # Reading Excel files
            elif file_extension in ['.xlsx', '.xls']:
                if is_large:
                    st.warning("Large Excel files may be slow to process. Consider converting to CSV for better performance.")
                
                # Save to temp file for processing
                temp_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                     f"temp_{uuid.uuid4().hex}{file_extension}")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Get sheet names
                import openpyxl
                wb = openpyxl.load_workbook(temp_path, read_only=True)
                sheet_names = wb.sheetnames
                
                # Let user select sheet
                sheet_name = st.selectbox("Select Excel Sheet:", sheet_names)
                
                # Read selected sheet
                progress.progress(0.3)
                if sample_size is not None:
                    # Read with nrows for sampling
                    df = pd.read_excel(temp_path, sheet_name=sheet_name, nrows=sample_size)
                else:
                    df = pd.read_excel(temp_path, sheet_name=sheet_name)
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # Reading other file types
            else:
                # For other file types, use standard pandas methods
                if file_extension == '.json':
                    df = pd.read_json(uploaded_file)
                else:
                    # Try as CSV with different delimiters
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                
                # Apply sampling if needed
                if sample_size is not None and len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
            
            # Optimize memory usage
            progress.progress(0.95)
            df = self.optimize_memory_usage(df)
            
            # Store original data info
            st.session_state.data_workflow['original_file'] = uploaded_file.name
            st.session_state.data_workflow['full_data_loaded'] = sample_size is None
            st.session_state.data_workflow['processing_done'] = False
            st.session_state.data_workflow['processed_file'] = None
            
            # Store a copy of the data
            st.session_state.original_df = df.copy()
            
            # Generate a temporary file path for the original data
            orig_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                   f"original_{uuid.uuid4().hex}.csv")
            
            # Save the original file efficiently using chunks
            self.save_dataframe_to_csv(df, orig_path)
            
            # Important: Set this file as processed by default to unblock other tabs
            processed_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                       f"processed_{uuid.uuid4().hex}.csv")
            self.save_dataframe_to_csv(df, processed_path)
            
            st.session_state.data_workflow['processed_file'] = processed_path
            st.session_state.data_workflow['processing_done'] = True
            st.session_state.data_workflow['processing_timestamp'] = datetime.datetime.now()
            
            # Calculate memory usage after loading
            mem_after = self.get_memory_usage()
            elapsed_time = time.time() - start_time
            
            # Store performance metrics
            st.session_state.data_workflow['performance_metrics']['load_time'] = elapsed_time
            st.session_state.data_workflow['performance_metrics']['memory_before'] = mem_before
            st.session_state.data_workflow['performance_metrics']['memory_after'] = mem_after
            st.session_state.data_workflow['performance_metrics']['memory_increase'] = mem_after - mem_before
            
            # Make sure df is set in session state
            if 'df' not in st.session_state or st.session_state.df is None:
                st.session_state.df = df
            
            # Complete progress bar
            progress.progress(1.0)
            
            # Show performance information
            if is_large:
                rows_loaded = len(df)
                if sample_size is not None:
                    st.success(f"Loaded a sample of {rows_loaded:,} rows in {elapsed_time:.2f} seconds. Memory usage: {(mem_after - mem_before):.1f} MB")
                else:
                    st.success(f"Loaded {rows_loaded:,} rows in {elapsed_time:.2f} seconds. Memory usage: {(mem_after - mem_before):.1f} MB")
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def save_processed_data(self, df):
        """
        Save processed data and update workflow state
        Uses efficient chunked CSV writing for large dataframes
        """
        if df is None or df.empty:
            return False
            
        try:
            # Start timing for performance tracking
            start_time = time.time()
            
            # Generate a unique filename for the processed data
            processed_path = os.path.join(st.session_state.data_workflow['temp_dir'], 
                                       f"processed_{uuid.uuid4().hex}.csv")
            
            # Save the DataFrame efficiently using chunks
            self.save_dataframe_to_csv(df, processed_path)
            
            # Update workflow state
            st.session_state.data_workflow['processed_file'] = processed_path
            st.session_state.data_workflow['processing_done'] = True
            st.session_state.data_workflow['processing_timestamp'] = datetime.datetime.now()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store performance metrics
            st.session_state.data_workflow['performance_metrics']['save_time'] = processing_time
            
            # Update the session state DataFrame as well
            st.session_state.df = df
            
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
                if processed_path and os.path.exists(processed_path):
                    # Check if the data is already in memory
                    if 'df' in st.session_state and st.session_state.df is not None:
                        return st.session_state.df
                    
                    # Otherwise, load from file
                    return self.load_dataframe_from_csv(processed_path)
                else:
                    # Fall back to session state DataFrame if processed file doesn't exist
                    if 'df' in st.session_state and st.session_state.df is not None:
                        return st.session_state.df
                    # Final fallback to original data
                    if 'original_df' in st.session_state:
                        return st.session_state.original_df.copy()
                    return None
            except Exception as e:
                st.error(f"Error loading processed data: {str(e)}")
                # Fallback to session state DataFrame
                if 'df' in st.session_state and st.session_state.df is not None:
                    return st.session_state.df
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
    
    def load_full_dataset(self):
        """
        Load the full dataset if a sample was loaded initially
        """
        if st.session_state.data_workflow['full_data_loaded']:
            st.info("Full dataset is already loaded.")
            return True
            
        # Check if original file exists
        original_file = st.session_state.data_workflow['original_file']
        if not original_file:
            st.error("No original file information found.")
            return False
            
        try:
            st.info(f"Loading full dataset from {original_file}...")
            
            # Ask user to confirm
            if not st.button("Confirm Loading Full Dataset", key="confirm_full_load"):
                st.warning("Loading the full dataset may cause performance issues with large files.")
                return False
                
            # TODO: Implement full dataset loading logic
            # This would require storing the original file path or the file itself
            
            # For now, just return false
            st.error("Full dataset loading is not implemented in this version.")
            return False
            
        except Exception as e:
            st.error(f"Error loading full dataset: {str(e)}")
            return False
    
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
            'has_processed_data': workflow['processed_file'] is not None,
            'is_sample': not workflow['full_data_loaded'],
            'sample_size': workflow['sample_size'] if not workflow['full_data_loaded'] else None,
            'is_large_dataset': workflow['is_large_dataset'],
            'performance_metrics': workflow['performance_metrics']
        }
    
    def reset_workflow(self):
        """Reset the workflow state"""
        temp_dir = st.session_state.data_workflow['temp_dir']
        st.session_state.data_workflow = {
            'original_file': None,
            'processed_file': None,
            'processing_done': False,
            'temp_dir': temp_dir,
            'processing_timestamp': None,
            'full_data_loaded': False,
            'sample_size': 10000,
            'is_large_dataset': False,
            'performance_metrics': {}
        }
        
        # Also clear the dataframes from session state
        if 'original_df' in st.session_state:
            del st.session_state.original_df
        if 'df' in st.session_state:
            del st.session_state.df
            
        # Force garbage collection
        gc.collect()
        
        return True
    
    def optimize_memory_usage(self, df):
        """
        Optimize memory usage of dataframe by downcasting datatypes
        """
        if df is None or df.empty:
            return df
            
        try:
            # Show info message
            st.info("Optimizing memory usage...")
            
            # Integer optimization
            for col in df.select_dtypes(include=['int']):
                col_min = df[col].min()
                col_max = df[col].max()
                
                # Find the smallest integer type that can represent this range
                if col_min >= 0:  # Unsigned int
                    if col_max < 2**8:
                        df[col] = df[col].astype(np.uint8)
                    elif col_max < 2**16:
                        df[col] = df[col].astype(np.uint16)
                    elif col_max < 2**32:
                        df[col] = df[col].astype(np.uint32)
                else:  # Signed int
                    if col_min >= -2**7 and col_max < 2**7:
                        df[col] = df[col].astype(np.int8)
                    elif col_min >= -2**15 and col_max < 2**15:
                        df[col] = df[col].astype(np.int16)
                    elif col_min >= -2**31 and col_max < 2**31:
                        df[col] = df[col].astype(np.int32)
            
            # Float optimization
            for col in df.select_dtypes(include=['float']):
                df[col] = df[col].astype(np.float32)
            
            # Object (string) optimization
            for col in df.select_dtypes(include=['object']):
                # If a column has a limited number of unique values, convert to categorical
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            # Success message
            mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # in MB
            st.success(f"Memory optimization complete. Current usage: {mem_usage:.2f} MB")
            
            return df
            
        except Exception as e:
            st.warning(f"Error in memory optimization: {str(e)}")
            return df
    
    def save_dataframe_to_csv(self, df, filepath, chunk_size=10000):
        """
        Save a DataFrame to CSV efficiently using chunks
        """
        if df.empty:
            # If DataFrame is empty, save an empty file
            df.to_csv(filepath, index=False)
            return True
            
        try:
            # For small DataFrames, use standard to_csv
            if len(df) < chunk_size:
                df.to_csv(filepath, index=False)
                return True
                
            # For larger DataFrames, use chunks
            chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            # Write first chunk with header
            chunks[0].to_csv(filepath, index=False)
            
            # Append remaining chunks without header
            for i, chunk in enumerate(chunks[1:], 1):
                chunk.to_csv(filepath, mode='a', header=False, index=False)
                
            return True
            
        except Exception as e:
            st.error(f"Error saving DataFrame to CSV: {str(e)}")
            return False
    
    def load_dataframe_from_csv(self, filepath, chunk_size=10000, sample_size=None):
        """
        Load a DataFrame from CSV efficiently using chunks
        with option to load only a sample
        """
        try:
            # Get file size to decide on chunking
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            
            # For small files, load directly
            if file_size < 10:  # Less than 10MB
                return pd.read_csv(filepath)
                
            # For sample loading, use nrows
            if sample_size is not None:
                return pd.read_csv(filepath, nrows=sample_size)
                
            # For larger files, use chunking
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                chunks.append(chunk)
                
            # Combine chunks
            return pd.concat(chunks, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error loading DataFrame from CSV: {str(e)}")
            return None
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return memory_mb
        except:
            return 0