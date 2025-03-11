import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import datetime
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

class DataProcessor:
    """Class for processing and transforming data"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.original_df = df.copy() if df is not None else None
        
        # Store processing history
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []

    def _update_session_state(self):
        """
        Update session state with current dataframe state.
        Call this after any operation that modifies the dataframe.
        """
        # If we're in a project context, update the project data
        if 'current_project' in st.session_state and st.session_state.current_project is not None:
            if st.session_state.current_project in st.session_state.projects:
                # The dataframe is already modified by reference
                # But we should make sure the project data is updated too
                st.session_state.projects[st.session_state.current_project]['data'] = self.df
                
                # Update last_modified timestamp
                st.session_state.projects[st.session_state.current_project]['last_modified'] = datetime.datetime.now()
                
        # Update the main dataframe reference
        st.session_state.df = self.df
    
    def render_interface(self):
        """Render the data processing interface"""
        st.header("Data Processing")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin data processing.")
            return
        
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
            self._render_data_cleaning()
        
        # Data Transformation Tab
        with processing_tabs[1]:
            self._render_data_transformation()
        
        # Feature Engineering Tab
        with processing_tabs[2]:
            self._render_feature_engineering()
        
        # Data Filtering Tab
        with processing_tabs[3]:
            self._render_data_filtering()
        
        # Column Management Tab
        with processing_tabs[4]:
            self._render_column_management()

        # Processing History and Export Options
        if st.session_state.processing_history:
            st.header("Processing History")
    
    # Create collapsible section for history
            with st.expander("View Processing Steps", expanded=False):
                for i, step in enumerate(st.session_state.processing_history):
                    st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Reset and Export buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Reset to Original Data", key="reset_data", use_container_width=True):
                    self.df = self.original_df.copy()
                    st.session_state.df = self.original_df.copy()
                    st.session_state.processing_history = []
                    st.success("Data reset to original state!")
                    st.rerun()
            with col2:
                if st.button("Download Processed Data", key="export_data", use_container_width=True):
                    self._export_processed_data()
            with col3:
                # Quick export as CSV
                if self.df is not None and len(self.df) > 0:
                    csv = self.df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv" class="download-button" style="text-decoration:none;">Quick Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        # # Processing History
        # if st.session_state.processing_history:
        #     st.header("Processing History")
            
        #     # Create collapsible section for history
        #     with st.expander("View Processing Steps", expanded=False):
        #         for i, step in enumerate(st.session_state.processing_history):
        #             st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
        #     # Reset button
        #     col1, col2 = st.columns([1, 1])
        #     with col1:
        #         if st.button("Reset to Original Data", key="reset_data", use_container_width=True):
        #             self.df = self.original_df.copy()
        #             st.session_state.df = self.original_df.copy()
        #             st.session_state.processing_history = []
        #             st.success("Data reset to original state!")
        #             st.rerun()
        #     with col2:
        #         if st.button("Export Processed Data", key="export_data", use_container_width=True):
        #             st.markdown("---")
        #             self._export_processed_data()
    
    def _render_data_cleaning(self):
        """Render data cleaning interface"""
        st.subheader("Data Cleaning")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Missing Values")
            
            # Show missing value statistics
            missing_vals = self.df.isna().sum()
            total_missing = missing_vals.sum()
            
            if total_missing == 0:
                st.success("No missing values found in the dataset!")
            else:
                st.warning(f"Found {total_missing} missing values across {(missing_vals > 0).sum()} columns")
                
                # Display columns with missing values
                cols_with_missing = self.df.columns[missing_vals > 0].tolist()
                
                # Create a dataframe to show missing value statistics
                missing_df = pd.DataFrame({
                    'Column': cols_with_missing,
                    'Missing Values': [missing_vals[col] for col in cols_with_missing],
                    'Percentage': [missing_vals[col] / len(self.df) * 100 for col in cols_with_missing]
                })
                
                st.dataframe(missing_df, use_container_width=True)
                
                # Options for handling missing values
                st.markdown("#### Handle Missing Values")
                
                # Add option to handle all columns at once
                handle_all = st.checkbox("Apply to all columns with missing values", value=False)
                
                if handle_all:
                    col_to_handle = "All columns with missing values"
                else:
                    col_to_handle = st.selectbox(
                        "Select column to handle:",
                        cols_with_missing
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
                    ]
                )
                
                # Additional input for constant value if selected
                if handling_method == "Fill with constant value":
                    constant_value = st.text_input("Enter constant value:")
                
                # Apply button
                if st.button("Apply Missing Value Treatment", key="apply_missing", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Determine columns to process
                        columns_to_process = cols_with_missing if handle_all else [col_to_handle]
                        
                        # Apply the selected method
                        if handling_method == "Drop rows":
                            self.df = self.df.dropna(subset=columns_to_process)
                            
                            # Add to processing history
                            rows_removed = orig_shape[0] - self.df.shape[0]
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
                                if self.df[col].dtype.kind in 'bifc':  # Check if numeric
                                    mean_val = self.df[col].mean()
                                    self.df[col] = self.df[col].fillna(mean_val)
                                    
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
                                if self.df[col].dtype.kind in 'bifc':  # Check if numeric
                                    median_val = self.df[col].median()
                                    self.df[col] = self.df[col].fillna(median_val)
                                    
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
                                mode_val = self.df[col].mode()[0]
                                self.df[col] = self.df[col].fillna(mode_val)
                                
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
                                        if self.df[col].dtype.kind in 'bifc':  # Numeric
                                            constant_val = float(constant_value)
                                        elif self.df[col].dtype.kind == 'b':  # Boolean
                                            constant_val = constant_value.lower() in ['true', 'yes', '1', 't', 'y']
                                        else:
                                            constant_val = constant_value
                                            
                                        self.df[col] = self.df[col].fillna(constant_val)
                                        
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
                                self.df[col] = self.df[col].fillna(method='ffill')
                                
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
                                self.df[col] = self.df[col].fillna(method='bfill')
                                
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
                                if self.df[col].dtype.kind in 'bifc':  # Check if numeric
                                    self.df[col] = self.df[col].interpolate(method='linear')
                                    
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
                    
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error handling missing values: {str(e)}")
        
        with col2:
            st.markdown("### Duplicate Rows")
            
            # Check for duplicates
            dup_count = self.df.duplicated().sum()
            
            if dup_count == 0:
                st.success("No duplicate rows found in the dataset!")
            else:
                st.warning(f"Found {dup_count} duplicate rows in the dataset")
                
                # Display sample of duplicates
                if st.checkbox("Show sample of duplicates"):
                    duplicates = self.df[self.df.duplicated(keep='first')]
                    st.dataframe(duplicates.head(5), use_container_width=True)
                
                # Button to remove duplicates
                if st.button("Remove Duplicate Rows", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Remove duplicates
                        self.df = self.df.drop_duplicates()
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        
                        # Add to processing history
                        rows_removed = orig_shape[0] - self.df.shape[0]
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
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.info("No numeric columns found for outlier detection.")
            else:
                col_for_outliers = st.selectbox(
                    "Select column for outlier detection:",
                    num_cols
                )
                
                # Calculate outlier bounds using IQR method
                Q1 = self.df[col_for_outliers].quantile(0.25)
                Q3 = self.df[col_for_outliers].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = self.df[(self.df[col_for_outliers] < lower_bound) | (self.df[col_for_outliers] > upper_bound)]
                outlier_count = len(outliers)
                
                # Visualize the distribution with outlier bounds
                fig = px.box(self.df, y=col_for_outliers, title=f"Distribution of {col_for_outliers} with Outlier Bounds")
                fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower bound")
                fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper bound")
                st.plotly_chart(fig, use_container_width=True)
                
                if outlier_count == 0:
                    st.success(f"No outliers found in column '{col_for_outliers}'")
                else:
                    st.warning(f"Found {outlier_count} outliers in column '{col_for_outliers}'")
                    st.write(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                    # Display sample of outliers
                    if st.checkbox("Show sample of outliers"):
                        st.dataframe(outliers.head(5), use_container_width=True)
                    
                    # Options for handling outliers
                    outlier_method = st.selectbox(
                        "Select handling method:",
                        [
                            "Remove outliers",
                            "Cap outliers (winsorize)",
                            "Replace with NaN",
                            "Replace with median"
                        ]
                    )
                    
                    # Button to handle outliers
                    if st.button("Handle Outliers", use_container_width=True):
                        try:
                            # Store original shape for reporting
                            orig_shape = self.df.shape
                            
                            if outlier_method == "Remove outliers":
                                # Remove rows with outliers
                                self.df = self.df[
                                    (self.df[col_for_outliers] >= lower_bound) & 
                                    (self.df[col_for_outliers] <= upper_bound)
                                ]
                                
                                # Add to processing history
                                rows_removed = orig_shape[0] - self.df.shape[0]
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
                                self.df[col_for_outliers] = self.df[col_for_outliers].clip(lower=lower_bound, upper=upper_bound)
                                
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
                                self.df.loc[
                                    (self.df[col_for_outliers] < lower_bound) | 
                                    (self.df[col_for_outliers] > upper_bound),
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
                                median_val = self.df[col_for_outliers].median()
                                
                                # Replace outliers with median
                                self.df.loc[
                                    (self.df[col_for_outliers] < lower_bound) | 
                                    (self.df[col_for_outliers] > upper_bound),
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
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error handling outliers: {str(e)}")
            
            # Data Type Conversion
            st.markdown("### Data Type Conversion")
            
            # Show current data types
            st.write("Current Data Types:")
            dtypes_df = pd.DataFrame({
                'Column': self.df.columns,
                'Type': [str(self.df[col].dtype) for col in self.df.columns]  # Already using str() here, which is good
            })
            st.dataframe(dtypes_df, use_container_width=True)
            
            # Column selection for type conversion
            col_to_convert = st.selectbox(
                "Select column to convert:",
                self.df.columns
            )
            
            # Target data type selection
            target_type = st.selectbox(
                "Convert to type:",
                ["int", "float", "string", "category", "datetime", "boolean"]
            )
            
            # DateTime format if target type is datetime
            date_format = None
            if target_type == "datetime":
                date_format = st.text_input(
                    "Enter date format (e.g., '%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):",
                    placeholder="Leave blank to auto-detect"
                )
            
            # Apply conversion button
            if st.button("Convert Data Type", use_container_width=True):
                try:
                    # Current type for reporting
                    current_type = str(self.df[col_to_convert].dtype)
                    
                    # Apply conversion
                    if target_type == "int":
                        self.df[col_to_convert] = pd.to_numeric(self.df[col_to_convert], errors='coerce').astype('Int64')
                    elif target_type == "float":
                        self.df[col_to_convert] = pd.to_numeric(self.df[col_to_convert], errors='coerce')
                    elif target_type == "string":
                        self.df[col_to_convert] = self.df[col_to_convert].astype(str)
                    elif target_type == "category":
                        self.df[col_to_convert] = self.df[col_to_convert].astype('category')
                    elif target_type == "datetime":
                        if date_format and date_format.strip():
                            self.df[col_to_convert] = pd.to_datetime(self.df[col_to_convert], format=date_format, errors='coerce')
                        else:
                            self.df[col_to_convert] = pd.to_datetime(self.df[col_to_convert], errors='coerce')
                    elif target_type == "boolean":
                        if self.df[col_to_convert].dtype == 'object':
                            # Convert strings to boolean
                            self.df[col_to_convert] = self.df[col_to_convert].map({
                                'true': True, 'True': True, 'TRUE': True, '1': True, 'yes': True, 'Yes': True, 'YES': True,
                                'false': False, 'False': False, 'FALSE': False, '0': False, 'no': False, 'No': False, 'NO': False
                            }).astype('boolean')
                        else:
                            # Convert numbers to boolean
                            self.df[col_to_convert] = self.df[col_to_convert].astype('boolean')
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Converted column '{col_to_convert}' from {current_type} to {target_type}",
                        "timestamp": datetime.datetime.now(),
                        "type": "data_type_conversion",
                        "details": {
                            "column": col_to_convert,
                            "from_type": current_type,
                            "to_type": target_type
                        }
                    })
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    
                    st.success(f"Converted column '{col_to_convert}' to {target_type}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
                    st.info("Try a different target type or check the data format")
    
    def _render_data_transformation(self):
        """Render data transformation interface"""
        st.subheader("Data Transformation")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Numeric Transformations")
            
            # Get numeric columns
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
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
                            data = self.df[col_to_transform].values.reshape(-1, 1)
                            scaler = StandardScaler()
                            transformed_data = scaler.fit_transform(data).flatten()
                            
                            # Handle potential NaN values
                            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
                            
                            # Store transformation parameters
                            mean = scaler.mean_[0]
                            std = scaler.scale_[0]
                            
                            self.df[output_col] = transformed_data
                            
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
                            data = self.df[col_to_transform].values.reshape(-1, 1)
                            scaler = MinMaxScaler()
                            transformed_data = scaler.fit_transform(data).flatten()
                            
                            # Handle potential NaN values
                            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
                            
                            # Store transformation parameters
                            min_val = scaler.data_min_[0]
                            max_val = scaler.data_max_[0]
                            
                            self.df[output_col] = transformed_data
                            
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
                            data = self.df[col_to_transform].values.reshape(-1, 1)
                            scaler = RobustScaler()
                            transformed_data = scaler.fit_transform(data).flatten()
                            
                            # Handle potential NaN values
                            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmedian(transformed_data))
                            
                            self.df[output_col] = transformed_data
                            
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
                            min_val = self.df[col_to_transform].min()
                            
                            if min_val <= 0:
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1
                                self.df[output_col] = np.log(self.df[col_to_transform] + const)
                                
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
                                self.df[output_col] = np.log(self.df[col_to_transform])
                                
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
                            min_val = self.df[col_to_transform].min()
                            
                            if min_val < 0:
                                # Add a constant to make all values non-negative
                                const = abs(min_val) + 1
                                self.df[output_col] = np.sqrt(self.df[col_to_transform] + const)
                                
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
                                self.df[output_col] = np.sqrt(self.df[col_to_transform])
                                
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
                            min_val = self.df[col_to_transform].min()
                            
                            if min_val <= 0:
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1
                                
                                from scipy import stats
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform] + const)
                                self.df[output_col] = transformed_data
                                
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
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform])
                                self.df[output_col] = transformed_data
                                
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
                                    self.df[col_to_transform], 
                                    bins=num_bins, 
                                    labels=bin_labels.split(',') if bin_labels else None
                                )
                            elif bin_strategy == "quantile":
                                # Quantile-based binning
                                bins = pd.qcut(
                                    self.df[col_to_transform], 
                                    q=num_bins, 
                                    labels=bin_labels.split(',') if bin_labels else None
                                )
                            elif bin_strategy == "kmeans":
                                # K-means clustering binning
                                from sklearn.cluster import KMeans
                                
                                # Reshape data for KMeans
                                data = self.df[col_to_transform].values.reshape(-1, 1)
                                
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
                            
                            self.df[output_col] = bins
                            
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
                            self.df[output_col] = np.power(self.df[col_to_transform], power)
                            
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
                            total = self.df[col_to_transform].sum()
                            self.df[output_col] = self.df[col_to_transform] / total
                            
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
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
        
        with col2:
            st.markdown("### Categorical Transformations")
            
            # Get categorical columns
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
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
                    target_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                    if target_cols:
                        target_col = st.selectbox("Select target column:", target_cols)
                    else:
                        st.warning("No numeric columns available for target encoding.")
                
                elif transform_method == "Ordinal Encoding":
                    # Get unique values
                    unique_vals = self.df[col_to_transform].dropna().unique().tolist()
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
                            dummies = pd.get_dummies(self.df[col_to_transform], prefix=prefix)
                            
                            # Add to dataframe
                            if create_new_col:
                                self.df = pd.concat([self.df, dummies], axis=1)
                            else:
                                # Drop original column and add dummies
                                self.df = self.df.drop(columns=[col_to_transform])
                                self.df = pd.concat([self.df, dummies], axis=1)
                            
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
                            unique_values = self.df[col_to_transform].dropna().unique()
                            mapping = {val: i for i, val in enumerate(unique_values)}
                            
                            self.df[output_col] = self.df[col_to_transform].map(mapping)
                            
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
                            freq = self.df[col_to_transform].value_counts(normalize=True)
                            self.df[output_col] = self.df[col_to_transform].map(freq)
                            
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
                            target_means = self.df.groupby(col_to_transform)[target_col].mean()
                            self.df[output_col] = self.df[col_to_transform].map(target_means)
                            
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
                                self.df[col_to_transform].astype('category').cat.codes, 
                                index=self.df.index
                            )
                            
                            # Convert to binary and create columns
                            for i in range(label_encoded.max().bit_length()):
                                bit_val = label_encoded.apply(lambda x: (x >> i) & 1)
                                self.df[f"{prefix}_{i}"] = bit_val
                            
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
                            counts = self.df[col_to_transform].value_counts()
                            self.df[output_col] = self.df[col_to_transform].map(counts)
                            
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
                            self.df[output_col] = self.df[col_to_transform].map(mapping)
                            
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
                            cleaned_series = self.df[col_to_transform].astype(str)
                            
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
                            
                            self.df[output_col] = cleaned_series
                            
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
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
            
            # Date Transformations
            st.markdown("### Date/Time Transformations")
            
            # Get datetime columns and potential date columns
            datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
            potential_date_cols = []
            
            # Look for columns that might contain dates but aren't datetime type
            for col in self.df.columns:
                if col in datetime_cols:
                    continue
                    
                # Check column name for date-related terms
                if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                    potential_date_cols.append(col)
                    
                # Check sample values if it's a string column
                elif self.df[col].dtype == 'object':
                    # Check first non-null value
                    sample_val = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
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
                                self.df[date_col] = pd.to_datetime(self.df[date_col], format=date_format, errors='coerce')
                            else:
                                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                            
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
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
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
                        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                        
                        # Extract each selected feature
                        for feature in date_features:
                            if feature == "Year":
                                self.df[f"{date_col}_year"] = self.df[date_col].dt.year
                            elif feature == "Month":
                                self.df[f"{date_col}_month"] = self.df[date_col].dt.month
                            elif feature == "Day":
                                self.df[f"{date_col}_day"] = self.df[date_col].dt.day
                            elif feature == "Hour":
                                self.df[f"{date_col}_hour"] = self.df[date_col].dt.hour
                            elif feature == "Minute":
                                self.df[f"{date_col}_minute"] = self.df[date_col].dt.minute
                            elif feature == "Second":
                                self.df[f"{date_col}_second"] = self.df[date_col].dt.second
                            elif feature == "Day of Week":
                                self.df[f"{date_col}_dayofweek"] = self.df[date_col].dt.dayofweek
                            elif feature == "Quarter":
                                self.df[f"{date_col}_quarter"] = self.df[date_col].dt.quarter
                            elif feature == "Week of Year":
                                self.df[f"{date_col}_weekofyear"] = self.df[date_col].dt.isocalendar().week
                            elif feature == "Is Weekend":
                                self.df[f"{date_col}_is_weekend"] = self.df[date_col].dt.dayofweek.isin([5, 6])
                            elif feature == "Is Month End":
                                self.df[f"{date_col}_is_month_end"] = self.df[date_col].dt.is_month_end
                            elif feature == "Is Month Start":
                                self.df[f"{date_col}_is_month_start"] = self.df[date_col].dt.is_month_start
                        
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
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
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
                                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                            
                            # Calculate time difference
                            time_diff = self.df[date_col] - self.df[date_col2]
                            
                            # Convert to selected unit
                            if time_units == "Days":
                                self.df[result_name] = time_diff.dt.total_seconds() / (24 * 3600)
                            elif time_units == "Hours":
                                self.df[result_name] = time_diff.dt.total_seconds() / 3600
                            elif time_units == "Minutes":
                                self.df[result_name] = time_diff.dt.total_seconds() / 60
                            else:  # Seconds
                                self.df[result_name] = time_diff.dt.total_seconds()
                            
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
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.success(f"Calculated time difference in {time_units.lower()} as '{result_name}'")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error calculating time difference: {str(e)}")
                else:
                    st.info("Need at least two datetime columns to calculate time difference.")
    
    def _render_feature_engineering(self):
        """Render feature engineering interface"""
        st.subheader("Feature Engineering")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Create New Features")
            
            # Get all columns
            all_cols = self.df.columns.tolist()
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            # Feature creation methods
            feature_method = st.selectbox(
                "Feature creation method:",
                [
                    "Arithmetic Operation",
                    "Mathematical Function",
                    "String Operation",
                    "Conditional Logic",
                    "Aggregation by Group",
                    "Rolling Window",
                    "Polynomial Features",
                    "Interaction Terms"
                ],
                key="feature_method"
            )
            
            # Create new feature options based on method
            if feature_method == "Arithmetic Operation":
                # Implementation details here...
                pass  # This is a placeholder - the full implementation would go here
            
            if num_cols and len(num_cols) > 1:
                st.markdown("#### Feature Correlation")
                
                # Let user select columns or use all
                use_all_numeric = st.checkbox("Use all numeric columns", value=True)
                
                if use_all_numeric:
                    corr_cols = num_cols
                else:
                    corr_cols = st.multiselect("Select columns for correlation:", num_cols, default=num_cols[:min(5, len(num_cols))])
                
                if corr_cols and len(corr_cols) > 1:
                    # Calculate correlation
                    corr = self.df[corr_cols].corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr, 
                        text_auto=".2f", 
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strongest correlations
                    st.markdown("#### Strongest Correlations")
                    
                    # Create a list of correlations with feature pairs
                    corr_pairs = []
                    for i in range(len(corr_cols)):
                        for j in range(i+1, len(corr_cols)):
                            corr_val = corr.iloc[i, j]
                            corr_pairs.append({
                                'Feature 1': corr.index[i],
                                'Feature 2': corr.columns[j],
                                'Correlation': corr_val,
                                'Abs Correlation': abs(corr_val)
                            })
                    
                    # Convert to dataframe and sort
                    corr_df = pd.DataFrame(corr_pairs)
                    top_corr = corr_df.sort_values('Abs Correlation', ascending=False).head(10)
                    
                    st.dataframe(
                        top_corr[['Feature 1', 'Feature 2', 'Correlation']].reset_index(drop=True),
                        use_container_width=True
                    )
            
            # Feature importance estimation
            if num_cols and len(num_cols) > 1:
                st.markdown("#### Feature Importance Estimation")
                st.info("Estimate feature importance for classification or regression tasks.")
                
                # Select target variable
                target_col = st.selectbox("Select target variable:", num_cols)
                
                # Get feature columns (excluding target)
                feature_cols = [col for col in num_cols if col != target_col]
                
                # Select task type
                task_type = st.radio("Task type:", ["Classification", "Regression"], horizontal=True)
                
                if st.button("Estimate Feature Importance", use_container_width=True):
                    try:
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data
                        X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
                        y = self.df[target_col].fillna(self.df[target_col].mean())
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Train a model to get feature importance
                        if task_type == "Classification":
                            model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42, min_samples_leaf=3)
                        else:  # Regression
                            model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42, min_samples_leaf=3)
                        
                        with st.spinner("Estimating feature importance..."):
                            model.fit(X_scaled, y)
                            
                            # Get feature importance
                            importances = model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': importances
                            })
                            
                            # Sort by importance
                            importance_df = importance_df.sort_values('Importance', ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Estimated Feature Importance",
                                labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
                                color='Importance',
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display table
                            st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error estimating feature importance: {str(e)}")

    def _render_data_filtering(self):
        """Render data filtering interface"""
        st.subheader("Data Filtering")
        
        # Make sure we have data
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin data filtering.")
            return
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Filter by Conditions")
            
            # Get all columns
            all_cols = self.df.columns.tolist()
            
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
                    if self.df[filter_col].dtype.kind in 'bifc':
                        # Numeric
                        operators = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                    elif self.df[filter_col].dtype.kind in 'M':
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
                        if self.df[filter_col].dtype.kind in 'bifc':
                            # Numeric range
                            min_val, max_val = st.slider(
                                "Range", 
                                float(self.df[filter_col].min()), 
                                float(self.df[filter_col].max()),
                                (float(self.df[filter_col].min()), float(self.df[filter_col].max())),
                                key="filter_range_1"
                            )
                            filter_val = (min_val, max_val)
                        elif self.df[filter_col].dtype.kind in 'M':
                            # Date range
                            min_date = pd.to_datetime(self.df[filter_col].min()).date()
                            max_date = pd.to_datetime(self.df[filter_col].max()).date()
                            
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
                        if self.df[filter_col].dtype.kind in 'bifc':
                            # Numeric input
                            filter_val = st.number_input("Value", value=0, key="filter_val_1")
                        elif self.df[filter_col].dtype.kind in 'M':
                            # Date input
                            min_date = pd.to_datetime(self.df[filter_col].min()).date()
                            max_date = pd.to_datetime(self.df[filter_col].max()).date()
                            filter_val = st.date_input("Value", max_date, min_value=min_date, max_value=max_date, key="filter_date_1")
                            filter_val = pd.Timestamp(filter_val)
                        else:
                            # Get unique values if not too many
                            unique_vals = self.df[filter_col].dropna().unique()
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
                    if self.df[filter_col2].dtype.kind in 'bifc':
                        # Numeric
                        operators2 = ["==", "!=", ">", ">=", "<", "<=", "between", "is null", "is not null"]
                    elif self.df[filter_col2].dtype.kind in 'M':
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
                        if self.df[filter_col2].dtype.kind in 'bifc':
                            # Numeric range
                            min_val2, max_val2 = st.slider(
                                "Range", 
                                float(self.df[filter_col2].min()), 
                                float(self.df[filter_col2].max()),
                                (float(self.df[filter_col2].min()), float(self.df[filter_col2].max())),
                                key="filter_range_2"
                            )
                            filter_val2 = (min_val2, max_val2)
                        elif self.df[filter_col2].dtype.kind in 'M':
                            # Date range
                            min_date2 = pd.to_datetime(self.df[filter_col2].min()).date()
                            max_date2 = pd.to_datetime(self.df[filter_col2].max()).date()
                            
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
                        if self.df[filter_col2].dtype.kind in 'bifc':
                            # Numeric input
                            filter_val2 = st.number_input("Value", value=0, key="filter_val_2")
                        elif self.df[filter_col2].dtype.kind in 'M':
                            # Date input
                            min_date2 = pd.to_datetime(self.df[filter_col2].min()).date()
                            max_date2 = pd.to_datetime(self.df[filter_col2].max()).date()
                            filter_val2 = st.date_input("Value", max_date2, min_value=min_date2, max_value=max_date2, key="filter_date_2")
                            filter_val2 = pd.Timestamp(filter_val2)
                        else:
                            # Get unique values if not too many
                            unique_vals2 = self.df[filter_col2].dropna().unique()
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
                    orig_shape = self.df.shape
                    
                    # Create filter mask for each condition
                    masks = []
                    for condition in conditions:
                        col = condition["column"]
                        op = condition["operator"]
                        val = condition["value"]
                        
                        # Create the mask based on the operator
                        if op == "==":
                            mask = self.df[col] == val
                        elif op == "!=":
                            mask = self.df[col] != val
                        elif op == ">":
                            mask = self.df[col] > val
                        elif op == ">=":
                            mask = self.df[col] >= val
                        elif op == "<":
                            mask = self.df[col] < val
                        elif op == "<=":
                            mask = self.df[col] <= val
                        elif op == "between":
                            min_val, max_val = val
                            mask = (self.df[col] >= min_val) & (self.df[col] <= max_val)
                        elif op == "is null":
                            mask = self.df[col].isna()
                        elif op == "is not null":
                            mask = ~self.df[col].isna()
                        elif op == "contains":
                            mask = self.df[col].astype(str).str.contains(str(val), na=False)
                        elif op == "starts with":
                            mask = self.df[col].astype(str).str.startswith(str(val), na=False)
                        elif op == "ends with":
                            mask = self.df[col].astype(str).str.endswith(str(val), na=False)
                        elif op == "in":
                            # Split comma-separated values
                            in_vals = [v.strip() for v in str(val).split(",")]
                            mask = self.df[col].isin(in_vals)
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
                    self.df = self.df[final_mask]
                    
                    # Calculate how many rows were filtered out
                    rows_filtered = orig_shape[0] - self.df.shape[0]
                    
                    # Add to processing history
                    filter_details = {
                        "conditions": conditions,
                        "combine_operator": combine_op if len(conditions) > 1 else None,
                        "rows_filtered": rows_filtered,
                        "rows_remaining": self.df.shape[0]
                    }
                    
                    st.session_state.processing_history.append({
                        "description": f"Filtered data: {rows_filtered} rows removed, {self.df.shape[0]} remaining",
                        "timestamp": datetime.datetime.now(),
                        "type": "filter",
                        "details": filter_details
                    })
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    
                    st.success(f"Filter applied: {rows_filtered} rows removed, {self.df.shape[0]} remaining")
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
                    sample_size = int(len(self.df) * sample_pct / 100)
                else:
                    sample_size = st.number_input(
                        "Number of rows to sample:",
                        min_value=1,
                        max_value=len(self.df),
                        value=min(100, len(self.df))
                    )
                
                random_state = st.number_input("Random seed (for reproducibility):", value=42)
                
                if st.button("Apply Random Sampling", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply random sampling
                        self.df = self.df.sample(n=sample_size, random_state=random_state)
                        
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
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.success(f"Applied random sampling: {sample_size} rows selected")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying sampling: {str(e)}")
            
            elif sample_method == "Stratified Sample":
                # Implementation details here...
                pass
            
            elif sample_method == "Systematic Sample":
                # Implementation details here...
                pass
            
            elif sample_method == "First/Last N Rows":
                # Implementation details here...
                pass
            
            # Data preview after filtering/sampling
            st.markdown("### Data Preview")
            st.dataframe(self.df.head(5), use_container_width=True)

    def _render_column_management(self):
        """Render column management interface"""
        st.subheader("Column Management")
        
        # Make sure we have data
        if self.df is None or self.df.empty:
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
                'Current Name': self.df.columns,
                'Type': [str(dtype) for dtype in self.df.dtypes.values],  # Convert dtype to string
                'Non-Null Values': self.df.count().values,
                'Null Values': self.df.isna().sum().values
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
                    for i in range(min(5, len(self.df.columns))):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            old_col = st.selectbox(
                                f"Column {i+1}:",
                                [col for col in self.df.columns if col not in rename_dict.keys()],
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
                        self.df = self.df.rename(columns=rename_dict)
                        
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
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
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
                    preview = {col: f"{prefix}{col}" for col in self.df.columns[:3]}
                    
                elif pattern_type == "Add Suffix":
                    suffix = st.text_input("Suffix to add:")
                    preview = {col: f"{col}{suffix}" for col in self.df.columns[:3]}
                    
                elif pattern_type == "Remove Text":
                    text_to_remove = st.text_input("Text to remove:")
                    preview = {col: col.replace(text_to_remove, "") for col in self.df.columns[:3]}
                    
                elif pattern_type == "Replace Text":
                    find_text = st.text_input("Find text:")
                    replace_text = st.text_input("Replace with:")
                    preview = {col: col.replace(find_text, replace_text) for col in self.df.columns[:3]}
                    
                elif pattern_type == "Change Case":
                    case_option = st.selectbox(
                        "Convert to:",
                        ["lowercase", "UPPERCASE", "Title Case", "snake_case", "camelCase"]
                    )
                    
                    # Show preview
                    if case_option == "lowercase":
                        preview = {col: col.lower() for col in self.df.columns[:3]}
                    elif case_option == "UPPERCASE":
                        preview = {col: col.upper() for col in self.df.columns[:3]}
                    elif case_option == "Title Case":
                        preview = {col: col.title() for col in self.df.columns[:3]}
                    elif case_option == "snake_case":
                        preview = {col: col.lower().replace(" ", "_") for col in self.df.columns[:3]}
                    elif case_option == "camelCase":
                        preview = {col: ''.join([s.capitalize() if i > 0 else s.lower() 
                                                for i, s in enumerate(col.split())]) 
                                  for col in self.df.columns[:3]}
                
                elif pattern_type == "Strip Whitespace":
                    preview = {col: col.strip() for col in self.df.columns[:3]}
                
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
                        self.df.columns.tolist()
                    )
                
                # Apply button
                if st.button("Apply Bulk Rename", use_container_width=True):
                    try:
                        # Determine which columns to rename
                        cols_to_rename = selected_cols if apply_to == "Selected Columns" else self.df.columns
                        
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
                            self.df = self.df.rename(columns=rename_dict)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Bulk renamed {len(rename_dict)} columns using {pattern_type}",
                                "timestamp": datetime.datetime.now(),
                                "type": "column_management",
                                "details": {
                                    "operation": "bulk_rename",
                                    "pattern_type": pattern_type,
                                    "columns_affected": len(rename_dict),
                                    "rename_map": rename_dict
                                }
                            })
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.success(f"Renamed {len(rename_dict)} columns")
                            st.rerun()
                        else:
                            st.info("No changes were made to column names.")
                        
                    except Exception as e:
                        st.error(f"Error renaming columns: {str(e)}")
        
        # Select/Drop Columns Tab
        with col_tabs[1]:
            st.markdown("### Select or Drop Columns")
            
            # Column operation
            operation = st.radio(
                "Operation:",
                ["Keep Selected Columns", "Drop Selected Columns"],
                horizontal=True
            )
            
            # Column selection
            selected_cols = st.multiselect(
                "Select columns:",
                self.df.columns.tolist(),
                default=[]
            )
            
            # Preview
            if selected_cols:
                if operation == "Keep Selected Columns":
                    preview_df = self.df[selected_cols].head(5)
                    st.markdown(f"Preview with {len(selected_cols)} columns:")
                else:
                    remaining_cols = [col for col in self.df.columns if col not in selected_cols]
                    preview_df = self.df[remaining_cols].head(5)
                    st.markdown(f"Preview with {len(remaining_cols)} columns:")
                
                st.dataframe(preview_df, use_container_width=True)
            
                # Apply button
                if st.button("Apply Column Selection", use_container_width=True):
                    try:
                        # Apply column selection
                        if operation == "Keep Selected Columns":
                            self.df = self.df[selected_cols].copy()
                            op_description = f"Kept {len(selected_cols)} columns, dropped {len(self.df.columns) - len(selected_cols)}"
                        else:
                            self.df = self.df.drop(columns=selected_cols)
                            op_description = f"Dropped {len(selected_cols)} columns, kept {len(self.df.columns)}"
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": op_description,
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "keep" if operation == "Keep Selected Columns" else "drop",
                                "columns": selected_cols
                            }
                        })
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.success(op_description)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error selecting columns: {str(e)}")
        
        # Reorder Columns Tab
        with col_tabs[2]:
            st.markdown("### Reorder Columns")
            
            # Column reordering
            st.markdown("Drag and reorder columns:")
            
            # Get current columns
            current_cols = self.df.columns.tolist()
            
            # Allow user to reorder
            reordered_cols = st.multiselect(
                "New column order (select in desired order):",
                current_cols,
                default=current_cols
            )
            
            # Check if all columns are selected
            if reordered_cols and len(reordered_cols) != len(current_cols):
                missing_cols = [col for col in current_cols if col not in reordered_cols]
                st.warning(f"Not all columns selected. Missing: {', '.join(missing_cols)}")
            
            # Preview of reordered columns
            if reordered_cols:
                # Only show columns that exist
                preview_cols = [col for col in reordered_cols if col in current_cols]
                st.markdown("Preview with reordered columns:")
                st.dataframe(self.df[preview_cols].head(5), use_container_width=True)
            
                # Apply button
                if st.button("Apply Column Reordering", use_container_width=True) and len(reordered_cols) == len(current_cols):
                    try:
                        # Reorder columns
                        self.df = self.df[reordered_cols]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": "Reordered columns",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "reorder",
                                "new_order": reordered_cols
                            }
                        })
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.success("Columns reordered successfully")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error reordering columns: {str(e)}")
        
        # Split & Merge Columns Tab
        with col_tabs[3]:
            st.markdown("### Split & Merge Columns")
            
            # Operation selection
            split_merge_op = st.radio(
                "Operation:",
                ["Split Column", "Merge Columns"],
                horizontal=True
            )
            
            if split_merge_op == "Split Column":
                # Column to split
                str_cols = self.df.select_dtypes(include=['object']).columns.tolist()
                
                if not str_cols:
                    st.warning("No string columns found for splitting.")
                else:
                    # Select column to split
                    split_col = st.selectbox("Column to split:", str_cols)
                    
                    # Show sample values
                    st.markdown("Sample values:")
                    sample_values = self.df[split_col].dropna().head(3).tolist()
                    for val in sample_values:
                        st.code(val)
                    
                    # Split options
                    split_by = st.selectbox(
                        "Split by:",
                        ["Delimiter", "Character Position", "Regular Expression"]
                    )
                    
                    if split_by == "Delimiter":
                        # Delimiter options
                        delimiter = st.text_input("Delimiter:", ",")
                        max_splits = st.number_input("Maximum number of splits (0 for all possible):", min_value=0, value=0)
                        
                    elif split_by == "Character Position":
                        # Position options
                        positions = st.text_input("Split positions (comma-separated):", "3,7")
                        
                    elif split_by == "Regular Expression":
                        # Regex options
                        pattern = st.text_input("Regular expression pattern:", "(\\d+)-(\\d+)")
                    
                    # Output column naming
                    naming_method = st.radio(
                        "New column naming:",
                        ["Automatic", "Custom"],
                        horizontal=True
                    )
                    
                    custom_names = None
                    if naming_method == "Custom":
                        # Get custom names
                        custom_names = st.text_input("New column names (comma-separated):")
                    
                    # Apply button
                    if st.button("Split Column", use_container_width=True):
                        try:
                            # Apply split based on method
                            if split_by == "Delimiter":
                                # Determine number of splits
                                num_splits = max_splits if max_splits > 0 else None
                                
                                # Split the column
                                split_df = self.df[split_col].str.split(delimiter, n=num_splits, expand=True)
                                
                                # Generate column names
                                if naming_method == "Custom" and custom_names:
                                    new_cols = custom_names.split(",")
                                    # Truncate or extend as needed
                                    if len(new_cols) < split_df.shape[1]:
                                        new_cols.extend([f"{split_col}_{i}" for i in range(len(new_cols), split_df.shape[1])])
                                    elif len(new_cols) > split_df.shape[1]:
                                        new_cols = new_cols[:split_df.shape[1]]
                                else:
                                    new_cols = [f"{split_col}_{i}" for i in range(split_df.shape[1])]
                                
                                # Rename split columns
                                split_df.columns = new_cols
                                
                                # Add to the original dataframe
                                for col in new_cols:
                                    self.df[col] = split_df[col]
                                
                                num_new_cols = len(new_cols)
                                
                            elif split_by == "Character Position":
                                # Parse positions
                                try:
                                    pos_list = [int(p.strip()) for p in positions.split(",")]
                                    pos_list.sort()  # Ensure positions are in ascending order
                                except:
                                    st.error("Invalid positions. Please enter comma-separated integers.")
                                    return
                                
                                # Create position ranges
                                ranges = [(0, pos_list[0])]
                                for i in range(1, len(pos_list)):
                                    ranges.append((pos_list[i-1], pos_list[i]))
                                ranges.append((pos_list[-1], None))  # Last position to end
                                
                                # Generate column names
                                if naming_method == "Custom" and custom_names:
                                    new_cols = custom_names.split(",")
                                    # Truncate or extend as needed
                                    if len(new_cols) < len(ranges):
                                        new_cols.extend([f"{split_col}_{i}" for i in range(len(new_cols), len(ranges))])
                                    elif len(new_cols) > len(ranges):
                                        new_cols = new_cols[:len(ranges)]
                                else:
                                    new_cols = [f"{split_col}_{i}" for i in range(len(ranges))]
                                
                                # Split the column by positions
                                for i, (start, end) in enumerate(ranges):
                                    if end is None:
                                        self.df[new_cols[i]] = self.df[split_col].str[start:]
                                    else:
                                        self.df[new_cols[i]] = self.df[split_col].str[start:end]
                                
                                num_new_cols = len(new_cols)
                                
                            elif split_by == "Regular Expression":
                                # Extract groups using regex
                                extracted = self.df[split_col].str.extract(pattern)
                                
                                # Generate column names
                                if naming_method == "Custom" and custom_names:
                                    new_cols = custom_names.split(",")
                                    # Truncate or extend as needed
                                    if len(new_cols) < extracted.shape[1]:
                                        new_cols.extend([f"{split_col}_group{i}" for i in range(len(new_cols), extracted.shape[1])])
                                    elif len(new_cols) > extracted.shape[1]:
                                        new_cols = new_cols[:extracted.shape[1]]
                                else:
                                    new_cols = [f"{split_col}_group{i}" for i in range(extracted.shape[1])]
                                
                                # Rename extracted columns
                                extracted.columns = new_cols
                                
                                # Add to the original dataframe
                                for col in new_cols:
                                    self.df[col] = extracted[col]
                                
                                num_new_cols = len(new_cols)
                            
                            # Add to processing history
                            split_details = {
                                "operation": "split",
                                "column": split_col,
                                "method": split_by,
                                "new_columns": new_cols
                            }
                            
                            if split_by == "Delimiter":
                                split_details["delimiter"] = delimiter
                                split_details["max_splits"] = max_splits
                            elif split_by == "Character Position":
                                split_details["positions"] = pos_list
                            elif split_by == "Regular Expression":
                                split_details["pattern"] = pattern
                            
                            st.session_state.processing_history.append({
                                "description": f"Split column '{split_col}' into {num_new_cols} new columns",
                                "timestamp": datetime.datetime.now(),
                                "type": "column_management",
                                "details": split_details
                            })
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.success(f"Split column '{split_col}' into {num_new_cols} new columns")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error splitting column: {str(e)}")
            
            else:  # Merge Columns
                # Columns to merge
                st.markdown("Select columns to merge:")
                merge_cols = st.multiselect("Columns to merge:", self.df.columns.tolist())
                
                if len(merge_cols) < 2:
                    st.info("Please select at least two columns to merge.")
                else:
                    # Merge options
                    separator = st.text_input("Separator:", " ")
                    new_col_name = st.text_input("New column name:", "_".join(merge_cols))
                    
                    # Keep original columns
                    keep_original = st.checkbox("Keep original columns", value=False)
                    
                    # Apply button
                    if st.button("Merge Columns", use_container_width=True):
                        try:
                            # Convert all selected columns to string
                            string_cols = [self.df[col].astype(str) for col in merge_cols]
                            
                            # Merge columns with separator
                            self.df[new_col_name] = string_cols[0]
                            for col in string_cols[1:]:
                                self.df[new_col_name] = self.df[new_col_name] + separator + col
                            
                            # Drop original columns if not keeping them
                            if not keep_original:
                                self.df = self.df.drop(columns=merge_cols)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Merged {len(merge_cols)} columns into new column '{new_col_name}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "column_management",
                                "details": {
                                    "operation": "merge",
                                    "columns": merge_cols,
                                    "separator": separator,
                                    "new_column": new_col_name,
                                    "keep_original": keep_original
                                }
                            })
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.success(f"Merged columns into new column '{new_col_name}'")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error merging columns: {str(e)}")

    def _export_processed_data(self):
        """Export processed data to various formats"""
        if self.df is None or self.df.empty:
            st.error("No data available to export.")
            return
        
        # Create a modal dialog for export options
        st.subheader("Export Data")
        
        # Format selection
        export_format = st.selectbox(
            "Export format:",
            ["CSV", "Excel", "JSON", "HTML", "Pickle"]
        )
        
        # Additional options per format
        if export_format == "CSV":
            # CSV options
            csv_sep = st.selectbox("Separator:", [",", ";", "\t", "|"])
            csv_encoding = st.selectbox("Encoding:", ["utf-8", "latin1", "utf-16"])
            csv_index = st.checkbox("Include index", value=False)
            
            # Generate file
            if st.button("Generate CSV for Download", use_container_width=True):
                try:
                    csv_buffer = io.StringIO()
                    self.df.to_csv(csv_buffer, sep=csv_sep, encoding=csv_encoding, index=csv_index)
                    csv_str = csv_buffer.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV File",
                        data=csv_str,
                        file_name="processed_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": "Exported data to CSV format",
                        "timestamp": datetime.datetime.now(),
                        "type": "export",
                        "details": {
                            "format": "CSV",
                            "rows": len(self.df),
                            "columns": len(self.df.columns)
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to CSV: {str(e)}")
        
        elif export_format == "Excel":
            # Excel options
            excel_sheet = st.text_input("Sheet name:", "Sheet1")
            excel_index = st.checkbox("Include index", value=False, key="excel_idx")
            
            # Generate file
            if st.button("Generate Excel for Download", use_container_width=True):
                try:
                    excel_buffer = io.BytesIO()
                    self.df.to_excel(excel_buffer, sheet_name=excel_sheet, index=excel_index, engine="openpyxl")
                    excel_data = excel_buffer.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name="processed_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": "Exported data to Excel format",
                        "timestamp": datetime.datetime.now(),
                        "type": "export",
                        "details": {
                            "format": "Excel",
                            "rows": len(self.df),
                            "columns": len(self.df.columns),
                            "sheet_name": excel_sheet
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to Excel: {str(e)}")
        
        elif export_format == "JSON":
            # JSON options
            json_orient = st.selectbox(
                "JSON orientation:",
                ["records", "columns", "index", "split", "table"]
            )
            json_lines = st.checkbox("Lines format (records only)", value=False)
            
            # Generate file
            if st.button("Generate JSON for Download", use_container_width=True):
                try:
                    if json_lines and json_orient == "records":
                        json_str = self.df.to_json(orient="records", lines=True)
                    else:
                        json_str = self.df.to_json(orient=json_orient)
                    
                    # Create download button
                    st.download_button(
                        label="Download JSON File",
                        data=json_str,
                        file_name="processed_data.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": "Exported data to JSON format",
                        "timestamp": datetime.datetime.now(),
                        "type": "export",
                        "details": {
                            "format": "JSON",
                            "rows": len(self.df),
                            "columns": len(self.df.columns),
                            "orient": json_orient,
                            "lines": json_lines and json_orient == "records"
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to JSON: {str(e)}")
        
        elif export_format == "HTML":
            # HTML options
            html_index = st.checkbox("Include index", value=True, key="html_idx")
            html_classes = st.text_input("CSS classes:", "dataframe table table-striped")
            
            # Generate file
            if st.button("Generate HTML for Download", use_container_width=True):
                try:
                    html_str = self.df.to_html(index=html_index, classes=html_classes.split())
                    
                    # Add basic styling
                    styled_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <title>Processed Data</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .dataframe {{ border-collapse: collapse; width: 100%; }}
                            .dataframe th, .dataframe td {{ 
                                border: 1px solid #ddd; 
                                padding: 8px; 
                                text-align: left;
                            }}
                            .dataframe th {{ 
                                background-color: #f2f2f2; 
                                color: #333;
                            }}
                            .dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
                            .dataframe tr:hover {{ background-color: #eef; }}
                        </style>
                    </head>
                    <body>
                        <h1>Processed Data</h1>
                        <p>Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        {html_str}
                    </body>
                    </html>
                    """
                    
                    # Create download button
                    st.download_button(
                        label="Download HTML File",
                        data=styled_html,
                        file_name="processed_data.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": "Exported data to HTML format",
                        "timestamp": datetime.datetime.now(),
                        "type": "export",
                        "details": {
                            "format": "HTML",
                            "rows": len(self.df),
                            "columns": len(self.df.columns)
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to HTML: {str(e)}")
        
        elif export_format == "Pickle":
            # Pickle options
            pickle_compression = st.selectbox(
                "Compression:",
                ["None", "gzip", "bz2", "zip", "xz"]
            )
            
            # Generate file
            if st.button("Generate Pickle for Download", use_container_width=True):
                try:
                    pickle_buffer = io.BytesIO()
                    compression = None if pickle_compression == "None" else pickle_compression
                    self.df.to_pickle(pickle_buffer, compression=compression)
                    pickle_data = pickle_buffer.getvalue()
                    
                    # File extension
                    ext = ".pkl"
                    if compression:
                        if compression == "gzip":
                            ext = ".pkl.gz"
                        elif compression == "bz2":
                            ext = ".pkl.bz2"
                        elif compression == "zip":
                            ext = ".pkl.zip"
                        elif compression == "xz":
                            ext = ".pkl.xz"
                    
                    # Create download button
                    st.download_button(
                        label="Download Pickle File",
                        data=pickle_data,
                        file_name=f"processed_data{ext}",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": "Exported data to Pickle format",
                        "timestamp": datetime.datetime.now(),
                        "type": "export",
                        "details": {
                            "format": "Pickle",
                            "rows": len(self.df),
                            "columns": len(self.df.columns),
                            "compression": pickle_compression
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to Pickle: {str(e)}")