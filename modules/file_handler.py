import streamlit as st
import pandas as pd
import io
import os
import tempfile
import datetime

def handle_uploaded_file(uploaded_file, file_type=None):
    """
    Process the uploaded file and return a pandas DataFrame
    
    Parameters:
    - uploaded_file: The file object from st.file_uploader
    - file_type: The type of file to process (can be inferred from filename if None)
    
    Returns:
    - DataFrame, file_details dictionary
    """
    if uploaded_file is None:
        return None, None
    
    try:
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
        
        # Process file based on type
        if file_type == 'csv':
            # Let user specify parameters for CSV
            with st.expander("CSV Import Options", expanded=False):
                sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
                encoding = st.selectbox("Encoding", ["utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
                decimal = st.selectbox("Decimal Separator", [".", ","], index=0)
                thousands = st.selectbox("Thousands Separator", ["", ",", ".", " "], index=0)
            
            # Read the CSV
            df = pd.read_csv(
                uploaded_file, 
                sep=sep, 
                encoding=encoding,
                decimal=decimal,
                thousands=thousands if thousands else None
            )
            
        elif file_type == 'excel':
            # Let user specify sheet name for Excel
            with st.expander("Excel Import Options", expanded=False):
                # Get sheet names
                xlsx = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Select Sheet", xlsx.sheet_names)
                header_row = st.number_input("Header Row", min_value=0, value=0, help="Row with column names (0-based)")
            
            # Read the Excel file
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
            
        elif file_type == 'json':
            # Let user specify JSON format
            with st.expander("JSON Import Options", expanded=False):
                orient = st.selectbox(
                    "JSON Orientation", 
                    ["records", "columns", "index", "split", "table"],
                    help="The structure of the JSON file"
                )
                lines = st.checkbox("Lines format", value=False, help="Read JSON document as one JSON object per line")
            
            # Read the JSON file
            if lines:
                df = pd.read_json(uploaded_file, lines=True)
            else:
                df = pd.read_json(uploaded_file, orient=orient)
        
        elif file_type == 'text':
            # Let user specify text file format
            with st.expander("Text Import Options", expanded=False):
                sep = st.selectbox("Separator", ["\t", ",", ";", "|", " "], index=0)
                header = st.checkbox("First row is header", value=True)
                header_row = 0 if header else None
            
            # Read the text file
            df = pd.read_csv(
                uploaded_file, 
                sep=sep, 
                header=header_row
            )
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None, None
        
        # Create file details dictionary
        file_details = {
            'filename': uploaded_file.name,
            'type': file_type.upper(),
            'size': uploaded_file.size,
            'last_modified': datetime.datetime.now(),
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        return df, file_details
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to a temporary location
    
    Parameters:
    - uploaded_file: The file object from st.file_uploader
    
    Returns:
    - Path to the saved file
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create a temporary file with the same extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Write the uploaded file to the temporary file
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        return temp_file.name
    
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def load_sample_data(sample_name):
    """
    Load a sample dataset
    
    Parameters:
    - sample_name: Name of the sample dataset to load
    
    Returns:
    - DataFrame, file_details dictionary
    """
    try:
        if sample_name == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['species'] = pd.Series(data.target).map({
                0: 'setosa', 
                1: 'versicolor', 
                2: 'virginica'
            })
            
            dataset_info = "Iris flower dataset with measurements of 150 iris flowers from three species."
        
        elif sample_name == "Titanic":
            # Load Titanic dataset
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            
            dataset_info = "Titanic passenger data with survival information from the 1912 disaster."
        
        elif sample_name == "Boston Housing":
            from sklearn.datasets import load_boston
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['PRICE'] = data.target
            
            dataset_info = "Boston housing dataset with house prices and neighborhood attributes."
        
        elif sample_name == "Wine Quality":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=';')
            
            dataset_info = "Wine quality dataset with physicochemical properties and quality ratings."
        
        elif sample_name == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['progression'] = data.target
            
            dataset_info = "Diabetes progression dataset with patient measurements and disease progression indicator."
        
        else:
            return None, None
        
        # Create file details dictionary
        file_details = {
            'filename': f"{sample_name} Sample Dataset",
            'type': "Sample Dataset",
            'size': None,
            'last_modified': datetime.datetime.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'info': dataset_info
        }
        
        return df, file_details
    
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None