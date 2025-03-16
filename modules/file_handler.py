import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import tempfile
import io
import time
from typing import Tuple, Dict, Any, Optional

def handle_uploaded_file(uploaded_file, file_type=None, sample_size=None, chunk_size=10000) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Process the uploaded file efficiently with chunking and sampling options
    
    Parameters:
    - uploaded_file: The file object from st.file_uploader
    - file_type: The type of file to process (can be inferred from filename if None)
    - sample_size: Number of rows to sample (None loads all data)
    - chunk_size: Number of rows to process at a time for large files
    
    Returns:
    - DataFrame, file_details dictionary
    """
    if uploaded_file is None:
        return None, None
    
    try:
        # Start timing for performance tracking
        start_time = time.time()
        
        # Determine file type if not provided
        if file_type is None:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in ['.csv']:
                file_type = 'csv'
            elif file_extension in ['.xlsx', '.xls']:
                file_type = 'excel'
            elif file_extension in ['.json']:
                file_type = 'json'
            elif file_extension in ['.txt', '.tsv']:
                file_type = 'text'
            else:
                file_type = 'unknown'
        
        # Create a progress bar
        progress_bar = st.progress(0)
        st.info("Loading file... This may take a moment for large files.")
        
        # Get import options from the user
        if file_type == 'csv':
            with st.expander("CSV Import Options", expanded=False):
                sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
                encoding = st.selectbox("Encoding", ["utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
                decimal = st.selectbox("Decimal Separator", [".", ","], index=0)
                thousands = st.selectbox("Thousands Separator", ["", ",", ".", " "], index=0)
                use_chunking = st.checkbox("Use chunked loading (for very large files)", value=True)
                preview_only = st.checkbox("Load preview only (faster)", value=True)
                
                if preview_only:
                    # Only load first n rows for preview
                    nrows = st.slider("Number of rows to preview:", 1000, 100000, 10000)
                else:
                    nrows = None
                
                # Option to sample data for faster processing
                sample_data = st.checkbox("Sample data for faster processing", value=sample_size is not None)
                if sample_data:
                    sample_size = st.slider("Sample size (rows):", 1000, 100000, 10000)
                else:
                    sample_size = None
            
            # Process CSV file - with chunking for large files
            if use_chunking and not preview_only:
                # Count total rows first to set up progress bar
                total_rows = 0
                for chunk in pd.read_csv(
                    uploaded_file, 
                    sep=sep, 
                    encoding=encoding,
                    decimal=decimal,
                    thousands=thousands if thousands else None,
                    chunksize=chunk_size,
                    low_memory=True
                ):
                    total_rows += len(chunk)
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Now process file in chunks
                chunks = []
                rows_processed = 0
                
                for i, chunk in enumerate(pd.read_csv(
                    uploaded_file, 
                    sep=sep, 
                    encoding=encoding,
                    decimal=decimal,
                    thousands=thousands if thousands else None,
                    chunksize=chunk_size,
                    low_memory=True
                )):
                    # Sample from each chunk if sample_size is specified
                    if sample_size is not None:
                        chunk_sample_size = max(1, int(sample_size * len(chunk) / total_rows))
                        chunk = chunk.sample(min(chunk_sample_size, len(chunk)))
                    
                    chunks.append(chunk)
                    rows_processed += len(chunk)
                    
                    # Update progress
                    progress = min(0.95, rows_processed / total_rows)
                    progress_bar.progress(progress)
                    
                # Combine chunks
                df = pd.concat(chunks, ignore_index=True)
                
                # Sample from final dataset if sample_size is specified and not done at chunk level
                if sample_size is not None and sample_size < len(df):
                    df = df.sample(sample_size)
            else:
                # Simpler loading for smaller files or preview
                df = pd.read_csv(
                    uploaded_file, 
                    sep=sep, 
                    encoding=encoding,
                    decimal=decimal,
                    thousands=thousands if thousands else None,
                    nrows=nrows,
                    low_memory=True
                )
                
                # Sample if requested
                if sample_size is not None and sample_size < len(df):
                    df = df.sample(sample_size)
            
        elif file_type == 'excel':
            with st.expander("Excel Import Options", expanded=False):
                # Get sheet names
                xlsx = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Select Sheet", xlsx.sheet_names)
                header_row = st.number_input("Header Row", min_value=0, value=0, help="Row with column names (0-based)")
                preview_only = st.checkbox("Load preview only (faster)", value=True)
                
                if preview_only:
                    # Only load first n rows for preview
                    nrows = st.slider("Number of rows to preview:", 1000, 50000, 5000)
                else:
                    nrows = None
                
                # Option to sample data
                sample_data = st.checkbox("Sample data for faster processing", value=sample_size is not None)
                if sample_data:
                    sample_size = st.slider("Sample size (rows):", 1000, 50000, 5000)
                else:
                    sample_size = None
            
            # Read the Excel file
            df = pd.read_excel(
                uploaded_file, 
                sheet_name=sheet_name, 
                header=header_row,
                nrows=nrows
            )
            
            # Sample if requested
            if sample_size is not None and sample_size < len(df):
                df = df.sample(sample_size)
            
        elif file_type == 'json':
            with st.expander("JSON Import Options", expanded=False):
                orient = st.selectbox(
                    "JSON Orientation", 
                    ["records", "columns", "index", "split", "table"],
                    help="The structure of the JSON file"
                )
                lines = st.checkbox("Lines format", value=False, help="Read JSON document as one JSON object per line")
                
                # Option to sample data
                sample_data = st.checkbox("Sample data for faster processing", value=sample_size is not None)
                if sample_data:
                    sample_size = st.slider("Sample size (rows):", 1000, 50000, 5000)
                else:
                    sample_size = None
            
            # Read the JSON file
            if lines:
                df = pd.read_json(uploaded_file, lines=True)
            else:
                df = pd.read_json(uploaded_file, orient=orient)
            
            # Sample if requested
            if sample_size is not None and sample_size < len(df):
                df = df.sample(sample_size)
        
        elif file_type == 'text':
            with st.expander("Text Import Options", expanded=False):
                sep = st.selectbox("Separator", ["\t", ",", ";", "|", " "], index=0)
                header = st.checkbox("First row is header", value=True)
                header_row = 0 if header else None
                preview_only = st.checkbox("Load preview only (faster)", value=True)
                
                if preview_only:
                    # Only load first n rows for preview
                    nrows = st.slider("Number of rows to preview:", 1000, 100000, 10000)
                else:
                    nrows = None
                
                # Option to sample data
                sample_data = st.checkbox("Sample data for faster processing", value=sample_size is not None)
                if sample_data:
                    sample_size = st.slider("Sample size (rows):", 1000, 100000, 10000)
                else:
                    sample_size = None
            
            # Read the text file
            df = pd.read_csv(
                uploaded_file, 
                sep=sep, 
                header=header_row,
                nrows=nrows,
                low_memory=True
            )
            
            # Sample if requested
            if sample_size is not None and sample_size < len(df):
                df = df.sample(sample_size)
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None, None
        
        # Apply memory optimizations
        df = optimize_dataframe_memory(df)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        # Create file details dictionary
        file_details = {
            'filename': uploaded_file.name,
            'type': file_type.upper(),
            'size': uploaded_file.size,
            'last_modified': datetime.datetime.now(),
            'rows': len(df),
            'rows_in_original': None if preview_only else len(df),
            'is_sample': sample_size is not None,
            'sample_size': sample_size,
            'columns': len(df.columns),
            'processing_time': processing_time,
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        st.success(f"File loaded successfully in {processing_time:.2f} seconds!")
        
        return df, file_details
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def optimize_dataframe_memory(df):
    """
    Optimize memory usage of dataframe by using appropriate data types
    """
    if df is None or len(df) == 0:
        return df
        
    # Create a progress bar
    progress_bar = st.progress(0)
    st.info("Optimizing memory usage...")
    
    try:
        # Process columns in batches to avoid hanging the UI
        total_cols = len(df.columns)
        for i, col in enumerate(df.columns):
            # Update progress
            progress = (i + 1) / total_cols
            progress_bar.progress(progress)
            
            # Optimize integers
            if df[col].dtype == 'int64':
                if df[col].min() >= 0:  # No negative values
                    if df[col].max() < 2**8:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 2**16:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() < 2**32:
                        df[col] = df[col].astype('uint32')
                else:  # Has negative values
                    if df[col].min() > -2**7 and df[col].max() < 2**7-1:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() > -2**15 and df[col].max() < 2**15-1:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() > -2**31 and df[col].max() < 2**31-1:
                        df[col] = df[col].astype('int32')
            
            # Optimize floats
            elif df[col].dtype == 'float64':
                if df[col].isnull().sum() > 0:  # Has null values
                    df[col] = df[col].astype('float32')  # Can't use smaller type with nulls
                else:
                    # Check if all values are integers
                    if np.all(df[col] == df[col].astype(int)):
                        # Convert to appropriate integer type
                        if df[col].min() >= 0:  # No negative values
                            if df[col].max() < 2**8:
                                df[col] = df[col].astype('uint8')
                            elif df[col].max() < 2**16:
                                df[col] = df[col].astype('uint16')
                            elif df[col].max() < 2**32:
                                df[col] = df[col].astype('uint32')
                            else:
                                df[col] = df[col].astype('uint64')
                        else:  # Has negative values
                            if df[col].min() > -2**7 and df[col].max() < 2**7-1:
                                df[col] = df[col].astype('int8')
                            elif df[col].min() > -2**15 and df[col].max() < 2**15-1:
                                df[col] = df[col].astype('int16')
                            elif df[col].min() > -2**31 and df[col].max() < 2**31-1:
                                df[col] = df[col].astype('int32')
                            else:
                                df[col] = df[col].astype('int64')
                    else:
                        df[col] = df[col].astype('float32')
            
            # Optimize strings (objects)
            elif df[col].dtype == 'object':
                # Check if column can be converted to categorical
                num_unique = df[col].nunique()
                if num_unique < len(df) * 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        # Display memory savings
        original_mem = df.memory_usage(deep=True).sum()
        optimized_mem = df.memory_usage(deep=True).sum()
        savings = 1 - (optimized_mem / original_mem) if original_mem > 0 else 0
        
        st.success(f"Memory usage optimized! Saved {savings:.1%} of memory.")
        
        return df
    
    except Exception as e:
        st.warning(f"Memory optimization partially failed: {str(e)}")
        return df
        
def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to a temporary location efficiently
    using streaming to avoid memory issues
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create a temporary file with the same extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        file_size = uploaded_file.size
        
        # Write the uploaded file to the temporary file in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        bytes_written = 0
        
        # Get file data
        file_data = uploaded_file.getvalue()
        
        # Write in chunks and update progress
        for i in range(0, len(file_data), chunk_size):
            chunk = file_data[i:i+chunk_size]
            temp_file.write(chunk)
            bytes_written += len(chunk)
            progress = min(0.99, bytes_written / file_size)
            progress_bar.progress(progress)
        
        temp_file.close()
        progress_bar.progress(1.0)
        
        return temp_file.name
    
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def load_sample_data(sample_name, sample_size=None):
    """
    Load a sample dataset with option to limit size
    """
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        st.info(f"Loading {sample_name} sample dataset...")
        
        if sample_name == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['species'] = pd.Series(data.target).map({
                0: 'setosa', 
                1: 'versicolor', 
                2: 'virginica'
            })
            progress_bar.progress(0.5)
            
            dataset_info = "Iris flower dataset with measurements of 150 iris flowers from three species."
        
        elif sample_name == "Titanic":
            # Load Titanic dataset
            progress_bar.progress(0.2)
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            progress_bar.progress(0.8)
            
            dataset_info = "Titanic passenger data with survival information from the 1912 disaster."
        
        elif sample_name == "Boston Housing":
            progress_bar.progress(0.3)
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['PRICE'] = data.target
            progress_bar.progress(0.7)
            
            dataset_info = "Boston housing dataset with house prices and neighborhood attributes."
        
        elif sample_name == "Wine Quality":
            progress_bar.progress(0.3)
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=';')
            progress_bar.progress(0.8)
            
            dataset_info = "Wine quality dataset with physicochemical properties and quality ratings."
        
        elif sample_name == "Diabetes":
            progress_bar.progress(0.3)
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['progression'] = data.target
            progress_bar.progress(0.7)
            
            dataset_info = "Diabetes progression dataset with patient measurements and disease progression indicator."
        
        else:
            progress_bar.progress(1.0)
            return None, None
        
        # Sample if requested
        original_size = len(df)
        if sample_size is not None and sample_size < len(df):
            df = df.sample(sample_size)
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        progress_bar.progress(1.0)
        
        # Create file details dictionary
        file_details = {
            'filename': f"{sample_name} Sample Dataset",
            'type': "Sample Dataset",
            'size': None,
            'last_modified': datetime.datetime.now(),
            'rows': len(df),
            'rows_in_original': original_size,
            'is_sample': sample_size is not None and sample_size < original_size,
            'sample_size': sample_size if sample_size is not None and sample_size < original_size else None,
            'columns': len(df.columns),
            'info': dataset_info
        }
        
        return df, file_details
    
    except Exception as e:
        progress_bar.progress(1.0)
        st.error(f"Error loading sample data: {str(e)}")
        return None, None