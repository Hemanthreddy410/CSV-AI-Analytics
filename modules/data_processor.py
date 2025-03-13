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
    
    def __init__(self, df, workflow_manager=None):
        """Initialize with dataframe and workflow manager"""
        self.df = df
        self.original_df = df.copy() if df is not None else None
        self.workflow_manager = workflow_manager
        
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
                    
                    # Reset workflow state
                    if self.workflow_manager is not None:
                        self.workflow_manager.reset_workflow()
                        
                    st.success("Data reset to original state!")
                    st.rerun()
            with col2:
                if st.button("Save Processed Data", key="save_data", use_container_width=True):
                    # Save processed data through workflow manager
                    if self.workflow_manager is not None:
                        if self.workflow_manager.save_processed_data(self.df):
                            st.success("✅ Processed data saved successfully!")
                        else:
                            st.error("Failed to save processed data")
                    else:
                        self._export_processed_data()
            with col3:
                # Quick export as CSV
                if self.df is not None and len(self.df) > 0:
                    csv = self.df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv" class="download-button" style="text-decoration:none;">Quick Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        # Display workflow status if workflow manager exists
        if self.workflow_manager is not None:
            status = self.workflow_manager.get_workflow_status()
            if status['processing_done']:
                st.success(f"✅ Data processing complete. All modules will use the processed data.")
                if status['processing_timestamp']:
                    st.info(f"Last processed: {status['processing_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("⚠️ Data processing not completed. Please process your data before using other modules.")
    
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
                        self._update_session_state()
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
                        self._update_session_state()
                        
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
                            self._update_session_state()
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
                    self._update_session_state()
                    
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
                        self._update_session_state()
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
                        self._update_session_state()
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
                    
                # Look for columns with "date", "time", "year", "month", "day" in the name
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
                            self._update_session_state()
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
                        self._update_session_state()
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
                            self._update_session_state()
                            st.success(f"Calculated time difference in {time_units.lower()} as '{result_name}'")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error calculating time difference: {str(e)}")
                else:
                    st.info("Need at least two datetime columns to calculate time difference.")
    
    def _render_feature_engineering(self):
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
            self._render_mathematical_operations()
            
        # Text Features tab
        with fe_tabs[1]:
            self._render_text_features()
            
        # Interaction Features tab
        with fe_tabs[2]:
            self._render_interaction_features()
            
        # Advanced Features tab
        with fe_tabs[3]:
            self._render_advanced_features()
            
        # Custom Formula tab
        with fe_tabs[4]:
            self._render_custom_formula()
    
    def _render_mathematical_operations(self):
        """Render interface for creating new columns from mathematical operations"""
        st.markdown("### Mathematical Operations")
        st.write("Create new columns by applying mathematical operations to existing numeric columns.")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
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
                second_operand = self.df[col2]
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
                    result = self.df[col1] + second_operand
                elif operation == "Subtraction (-)":
                    result = self.df[col1] - second_operand
                elif operation == "Multiplication (*)":
                    result = self.df[col1] * second_operand
                elif operation == "Division (/)":
                    # Avoid division by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot divide by zero!")
                        return
                    result = self.df[col1] / second_operand
                elif operation == "Modulo (%)":
                    # Avoid modulo by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot perform modulo by zero!")
                        return
                    result = self.df[col1] % second_operand
                elif operation == "Power (^)":
                    result = self.df[col1] ** second_operand
                elif operation == "Absolute Value (|x|)":
                    result = self.df[col1].abs()
                elif operation == "Logarithm (log)":
                    # Check for non-positive values
                    if (self.df[col1] <= 0).any():
                        st.warning("Column contains non-positive values. Will apply log to positive values only and return NaN for others.")
                    result = np.log(self.df[col1])
                elif operation == "Exponential (exp)":
                    result = np.exp(self.df[col1])
                elif operation == "Round":
                    result = self.df[col1].round()
                elif operation == "Floor":
                    result = np.floor(self.df[col1])
                elif operation == "Ceiling":
                    result = np.ceil(self.df[col1])
                
                # Preview the first few rows
                preview_df = pd.DataFrame({
                    col1: self.df[col1].head(5),
                    "Result": result.head(5)
                })
                
                # Add second column to preview if applicable
                if operation in binary_ops and not use_constant:
                    preview_df.insert(1, col2, self.df[col2].head(5))
                
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
                    self.df[new_col_name] = self.df[col1] + second_operand
                    op_desc = f"Added {col1} and {operand_desc}"
                elif operation == "Subtraction (-)":
                    self.df[new_col_name] = self.df[col1] - second_operand
                    op_desc = f"Subtracted {operand_desc} from {col1}"
                elif operation == "Multiplication (*)":
                    self.df[new_col_name] = self.df[col1] * second_operand
                    op_desc = f"Multiplied {col1} by {operand_desc}"
                elif operation == "Division (/)":
                    # Avoid division by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot divide by zero!")
                        return
                    self.df[new_col_name] = self.df[col1] / second_operand
                    op_desc = f"Divided {col1} by {operand_desc}"
                elif operation == "Modulo (%)":
                    # Avoid modulo by zero
                    if isinstance(second_operand, (int, float)) and second_operand == 0:
                        st.error("Cannot perform modulo by zero!")
                        return
                    self.df[new_col_name] = self.df[col1] % second_operand
                    op_desc = f"Calculated {col1} modulo {operand_desc}"
                elif operation == "Power (^)":
                    self.df[new_col_name] = self.df[col1] ** second_operand
                    op_desc = f"Raised {col1} to the power of {operand_desc}"
                elif operation == "Absolute Value (|x|)":
                    self.df[new_col_name] = self.df[col1].abs()
                    op_desc = f"Calculated absolute value of {col1}"
                elif operation == "Logarithm (log)":
                    self.df[new_col_name] = np.log(self.df[col1])
                    op_desc = f"Calculated natural logarithm of {col1}"
                elif operation == "Exponential (exp)":
                    self.df[new_col_name] = np.exp(self.df[col1])
                    op_desc = f"Calculated exponential of {col1}"
                elif operation == "Round":
                    self.df[new_col_name] = self.df[col1].round()
                    op_desc = f"Rounded {col1} to nearest integer"
                elif operation == "Floor":
                    self.df[new_col_name] = np.floor(self.df[col1])
                    op_desc = f"Applied floor function to {col1}"
                elif operation == "Ceiling":
                    self.df[new_col_name] = np.ceil(self.df[col1])
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
                
                # Update the dataframe in session state
                self._update_session_state()
                
                st.success(f"Created new column '{new_col_name}'")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating new column: {str(e)}")
    
    def _render_text_features(self):
        """Render interface for creating text-based features"""
        st.markdown("### Text Features")
        st.write("Extract features from text columns.")
        
        # Get text columns
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
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
                    self.df[new_col_name] = self.df[text_col].astype(str).apply(len)
                    desc = f"Extracted text length from '{text_col}'"
                
                elif feature_type == "Word Count":
                    self.df[new_col_name] = self.df[text_col].astype(str).apply(
                        lambda x: len(str(x).split())
                    )
                    desc = f"Counted words in '{text_col}'"
                
                elif feature_type == "Character Count":
                    # Count only alphabetic characters
                    self.df[new_col_name] = self.df[text_col].astype(str).apply(
                        lambda x: sum(c.isalpha() for c in str(x))
                    )
                    desc = f"Counted alphabetic characters in '{text_col}'"
                
                elif feature_type == "Contains Specific Text":
                    if not search_text:
                        st.error("Please enter text to search for")
                        return
                    
                    if case_sensitive:
                        self.df[new_col_name] = self.df[text_col].astype(str).apply(
                            lambda x: search_text in str(x)
                        )
                    else:
                        self.df[new_col_name] = self.df[text_col].astype(str).apply(
                            lambda x: search_text.lower() in str(x).lower()
                        )
                    
                    desc = f"Checked if '{text_col}' contains '{search_text}'"
                
                elif feature_type == "Extract Pattern":
                    if not pattern:
                        st.error("Please enter a regex pattern")
                        return
                    
                    # Extract first match of the pattern
                    self.df[new_col_name] = self.df[text_col].astype(str).apply(
                        lambda x: re.search(pattern, str(x)).group(0) if re.search(pattern, str(x)) else None
                    )
                    
                    desc = f"Extracted pattern '{pattern}' from '{text_col}'"
                
                elif feature_type == "Count Specific Pattern":
                    if not pattern:
                        st.error("Please enter a regex pattern")
                        return
                    
                    # Count occurrences of the pattern
                    self.df[new_col_name] = self.df[text_col].astype(str).apply(
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
                    
                    self.df[new_col_name] = self.df[text_col].apply(simple_sentiment)
                    
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
                
                # Update the dataframe in session state
                self._update_session_state()
                
                st.success(f"Created new column '{new_col_name}'")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating text feature: {str(e)}")
    
    def _render_interaction_features(self):
        """Render interface for creating interaction features between columns"""
        st.markdown("### Interaction Features")
        st.write("Create features that capture interactions between multiple columns.")
        
        # Get columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
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
                        result = self.df[cols_for_interaction[0]].copy()
                        for col in cols_for_interaction[1:]:
                            result *= self.df[col]
                        self.df[new_col_name] = result
                        desc = f"Multiplication of {', '.join(cols_for_interaction)}"
                    elif interaction_op == "Addition":
                        self.df[new_col_name] = sum(self.df[col] for col in cols_for_interaction)
                        desc = f"Sum of {', '.join(cols_for_interaction)}"
                    elif interaction_op == "Subtraction":
                        if len(cols_for_interaction) != 2:
                            st.warning("Subtraction works best with exactly 2 columns. Using first two selected columns.")
                        result = self.df[cols_for_interaction[0]] - self.df[cols_for_interaction[1]]
                        self.df[new_col_name] = result
                        desc = f"Subtraction: {cols_for_interaction[0]} - {cols_for_interaction[1]}"
                    else:  # Division
                        if len(cols_for_interaction) != 2:
                            st.warning("Division works best with exactly 2 columns. Using first two selected columns.")
                        # Avoid division by zero by replacing zeros with NaN
                        denominator = self.df[cols_for_interaction[1]].replace(0, np.nan)
                        self.df[new_col_name] = self.df[cols_for_interaction[0]] / denominator
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                    
                    # Calculate aggregations
                    agg_df = self.df.groupby(groupby_cols)[value_col].agg(selected_aggs).reset_index()
                    
                    # Create feature columns
                    group_key = "_".join(groupby_cols)
                    new_columns = []
                    
                    # Merge aggregated features back to original dataframe
                    for agg_func in selected_aggs:
                        # Create column name
                        col_name = f"{group_key}_{value_col}_{agg_func}"
                        
                        # Add to new columns list
                        new_columns.append(col_name)
                        
                        # Rename the column in the aggregated dataframe
                        if len(selected_aggs) > 1:
                            # Handle case where pandas creates a MultiIndex for multiple aggregations
                            agg_df = agg_df.rename(columns={(value_col, agg_func): col_name})
                        else:
                            # Single aggregation case
                            agg_df = agg_df.rename(columns={value_col: col_name})
                    
                    # Merge back to the original dataframe
                    self.df = self.df.merge(agg_df, on=groupby_cols, how='left')
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                    X = self.df[poly_cols].fillna(0)  # Replace NaNs with 0 for polynomial generation
                    
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
                        index=self.df.index
                    )
                    
                    # Drop the original columns from the polynomial features (first len(poly_cols) columns)
                    poly_df = poly_df.iloc[:, len(poly_cols):]
                    
                    # Add new features to original dataframe
                    for col in poly_df.columns:
                        col_name = col.replace(' ', '_').replace('^', '_pow_')
                        self.df[col_name] = poly_df[col]
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                        denominator = self.df[denominator_col].replace(0, np.nan)
                    elif handle_zeros == "Replace with small value (1e-6)":
                        denominator = self.df[denominator_col].replace(0, 1e-6)
                    else:  # Add small value
                        denominator = self.df[denominator_col] + 1e-6
                    
                    # Calculate ratio
                    self.df[new_col_name] = self.df[numerator_col] / denominator
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
                    st.success(f"Created new ratio column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating ratio feature: {str(e)}")
    
    def _render_advanced_features(self):
        """Render interface for creating advanced features"""
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
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
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
                    X = self.df[pca_cols].fillna(0)
                    
                    # Standardize data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply PCA
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(X_scaled)
                    
                    # Create new columns with PCA results
                    for i in range(n_components):
                        self.df[f"pca_component_{i+1}"] = pca_result[:, i]
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                    X = self.df[cluster_cols].fillna(0)
                    
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
                    self.df[new_col_name] = cluster_labels
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                    X = self.df[anomaly_cols].fillna(0)
                    
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
                    self.df[new_col_name] = anomaly_scores
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
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
                    col_min = float(self.df[col].min())
                    col_max = float(self.df[col].max())
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
                    X = self.df[distance_cols].fillna(0)
                    
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
                    self.df[new_col_name] = distances
                    
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
                    st.success(f"Created distance feature column '{new_col_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating distance feature: {str(e)}")
        
        elif feature_type == "Time-window Statistics":
            st.write("Calculate statistics over time windows (requires date/time column).")
            
            # Get datetime columns
            datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Also check for other columns that might be dates
            potential_date_cols = []
            for col in self.df.columns:
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
                if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
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
                    if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                    
                    # Sort by date
                    df_sorted = self.df.sort_values(date_col)
                    
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
                        self.df[col_name] = stat_result
                        
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
                    
                    # Update the dataframe in session state
                    self._update_session_state()
                    
                    st.success(f"Created {len(stats_to_calc)} time-window statistic columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating time-window features: {str(e)}")
    
    def _render_custom_formula(self):
        """Render interface for creating features using custom formulas"""
        st.markdown("### Custom Formula")
        st.write("Create new features using custom formulas or expressions.")
        
        # Available columns
        available_cols = self.df.columns.tolist()
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
                local_dict = {col: self.df[col] for col in self.df.columns}
                
                # Add NumPy for advanced functions
                local_dict['np'] = np
                
                # Evaluate the formula
                result = eval(formula, {"__builtins__": {}}, local_dict)
                
                # Add result to dataframe
                self.df[new_col_name] = result
                
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
                
                # Update the dataframe in session state
                self._update_session_state()
                
                st.success(f"Created new column '{new_col_name}' using custom formula")
                
                # Show statistics about the result
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Mean", f"{self.df[new_col_name].mean():.2f}")
                with stat_cols[1]:
                    st.metric("Min", f"{self.df[new_col_name].min():.2f}")
                with stat_cols[2]:
                    st.metric("Max", f"{self.df[new_col_name].max():.2f}")
                with stat_cols[3]:
                    st.metric("Missing Values", f"{self.df[new_col_name].isna().sum()}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error evaluating formula: {str(e)}")
                st.info("Check that your formula uses valid column names and operations.")
    
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
                    self._update_session_state()
                    
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
                        self._update_session_state()
                        st.success(f"Applied random sampling: {sample_size} rows selected")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying sampling: {str(e)}")
            
            elif sample_method == "Stratified Sample":
                # Stratified sampling options
                strat_col = st.selectbox(
                    "Select column to stratify by:",
                    self.df.select_dtypes(include=['object', 'category']).columns.tolist()
                )
                
                sample_pct = st.slider("Percentage to sample from each stratum:", 1, 100, 10)
                
                if st.button("Apply Stratified Sampling", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply stratified sampling
                        sampled_dfs = []
                        for value, group in self.df.groupby(strat_col):
                            sample_size = int(len(group) * sample_pct / 100)
                            if sample_size > 0:
                                sampled_dfs.append(group.sample(n=min(sample_size, len(group))))
                        
                        # Combine samples
                        self.df = pd.concat(sampled_dfs)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied stratified sampling by '{strat_col}': {self.df.shape[0]} rows selected",
                            "timestamp": datetime.datetime.now(),
                            "type": "sampling",
                            "details": {
                                "method": "stratified",
                                "stratify_column": strat_col,
                                "percentage": sample_pct,
                                "original_rows": orig_shape[0],
                                "sampled_rows": self.df.shape[0]
                            }
                        })
                        
                        # Update the dataframe in session state
                        self._update_session_state()
                        st.success(f"Applied stratified sampling: {self.df.shape[0]} rows selected")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying stratified sampling: {str(e)}")
            
            elif sample_method == "Systematic Sample":
                # Systematic sampling options
                k = st.number_input(
                    "Select every kth row:",
                    min_value=2,
                    max_value=len(self.df),
                    value=min(10, len(self.df))
                )
                
                if st.button("Apply Systematic Sampling", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply systematic sampling
                        indices = range(0, len(self.df), k)
                        self.df = self.df.iloc[indices]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied systematic sampling (every {k}th row): {len(self.df)} rows selected",
                            "timestamp": datetime.datetime.now(),
                            "type": "sampling",
                            "details": {
                                "method": "systematic",
                                "k": k,
                                "original_rows": orig_shape[0],
                                "sampled_rows": self.df.shape[0]
                            }
                        })
                        
                        # Update the dataframe in session state
                        self._update_session_state()
                        st.success(f"Applied systematic sampling: {len(self.df)} rows selected")
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
                    max_value=len(self.df),
                    value=min(100, len(self.df))
                )
                
                if st.button("Apply Selection", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply selection
                        if select_type == "First N Rows":
                            self.df = self.df.head(n_rows)
                            selection_type = "first"
                        else:
                            self.df = self.df.tail(n_rows)
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
                                "sampled_rows": self.df.shape[0]
                            }
                        })
                        
                        # Update the dataframe in session state
                        self._update_session_state()
                        st.success(f"Selected {selection_type} {n_rows} rows")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error selecting rows: {str(e)}")
            
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
                        self._update_session_state()
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
                            self._update_session_state()
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
                        self._update_session_state()
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
                        self._update_session_state()
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
                            self._update_session_state()
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
                            self._update_session_state()
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
            pickle_protocol = st.slider("Pickle protocol version:", 0, 5, 4)
            pickle_compression = st.selectbox("Compression:", ["None", "gzip", "bz2", "xz"])
            
            # Generate file
            if st.button("Generate Pickle for Download", use_container_width=True):
                try:
                    pickle_buffer = io.BytesIO()
                    
                    if pickle_compression == "None":
                        self.df.to_pickle(pickle_buffer, protocol=pickle_protocol)
                    else:
                        self.df.to_pickle(
                            pickle_buffer, 
                            protocol=pickle_protocol,
                            compression=pickle_compression
                        )
                    
                    pickle_data = pickle_buffer.getvalue()
                    
                    # Create download button
                    file_extension = ".pkl" if pickle_compression == "None" else f".{pickle_compression}.pkl"
                    st.download_button(
                        label="Download Pickle File",
                        data=pickle_data,
                        file_name=f"processed_data{file_extension}",
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
                            "protocol": pickle_protocol,
                            "compression": pickle_compression
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to Pickle: {str(e)}")

class DataVisualization:
    """Class for creating data visualizations"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
    
    def render_interface(self):
        """Render the data visualization interface"""
        st.header("Data Visualization")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin data visualization.")
            return
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs([
            "Basic Charts",
            "Statistical Plots",
            "Categorical Analysis",
            "Time Series",
            "Multivariate Analysis"
        ])
        
        # Basic Charts Tab
        with viz_tabs[0]:
            self._render_basic_charts()
        
        # Statistical Plots Tab
        with viz_tabs[1]:
            self._render_statistical_plots()
        
        # Categorical Analysis Tab
        with viz_tabs[2]:
            self._render_categorical_analysis()
        
        # Time Series Tab
        with viz_tabs[3]:
            self._render_time_series()
        
        # Multivariate Analysis Tab
        with viz_tabs[4]:
            self._render_multivariate_analysis()
    
    def _render_basic_charts(self):
        """Render basic charts visualizations"""
        st.subheader("Basic Charts")
        
        # Get columns by type
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Area Chart", "Histogram"]
        )
        
        # Common settings for all chart types
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            # X-axis selection (dependent on chart type)
            if chart_type == "Histogram":
                # For histogram, only need one numeric column
                if not num_cols:
                    st.warning("No numeric columns available for histogram.")
                    return
                
                x_axis = st.selectbox("Column for histogram:", num_cols)
                y_axis = None
            
            elif chart_type == "Pie Chart":
                # For pie chart, need one categorical and one numeric column
                if not cat_cols:
                    st.warning("No categorical columns available for pie chart labels.")
                    return
                
                if not num_cols:
                    st.warning("No numeric columns available for pie chart values.")
                    return
                
                x_axis = st.selectbox("Labels (categorical):", cat_cols)
                y_axis = st.selectbox("Values (numeric):", num_cols)
            
            else:
                # For other charts, x-axis can be numeric or categorical
                x_axis = st.selectbox("X-axis:", num_cols + cat_cols)
                
                # Y-axis is required for all except pie chart and histogram
                # Only show numeric columns for y-axis
                if not num_cols:
                    st.warning("No numeric columns available for y-axis.")
                    return
                
                y_axis = st.selectbox("Y-axis:", num_cols)
        
        with settings_col2:
            # Color by column (optional)
            color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
            color_by = None if color_by == "None" else color_by
            
            # Title
            title = st.text_input("Chart title:", f"{chart_type} of {y_axis or x_axis}")
        
        # Chart specific options
        if chart_type == "Bar Chart":
            orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
            is_horizontal = orientation == "Horizontal"
            
            # Sort bars
            sort_bars = st.checkbox("Sort bars", value=False)
            if sort_bars:
                sort_order = st.radio("Sort order:", ["Ascending", "Descending"], horizontal=True)
            
            # Create bar chart
            if st.button("Generate Bar Chart", use_container_width=True):
                try:
                    # Create figure
                    if sort_bars:
                        if sort_order == "Ascending":
                            sort_dir = True
                        else:
                            sort_dir = False
                        
                        # Sort the dataframe
                        df_sorted = self.df.sort_values(by=y_axis, ascending=sort_dir)
                    else:
                        df_sorted = self.df
                    
                    if is_horizontal:
                        fig = px.bar(
                            df_sorted, 
                            y=x_axis, 
                            x=y_axis, 
                            title=title,
                            color=color_by,
                            orientation='h'
                        )
                    else:
                        fig = px.bar(
                            df_sorted, 
                            x=x_axis, 
                            y=y_axis, 
                            title=title,
                            color=color_by
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=y_axis if is_horizontal else x_axis,
                        yaxis_title=x_axis if is_horizontal else y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating bar chart: {str(e)}")
        
        elif chart_type == "Line Chart":
            # Line chart options
            line_mode = st.radio("Line mode:", ["Lines", "Lines + Markers", "Markers"], horizontal=True)
            
            # Create line chart
            if st.button("Generate Line Chart", use_container_width=True):
                try:
                    # Determine mode
                    if line_mode == "Lines":
                        mode = "lines"
                    elif line_mode == "Lines + Markers":
                        mode = "lines+markers"
                    else:  # Markers only
                        mode = "markers"
                    
                    # Create figure
                    fig = px.line(
                        self.df, 
                        x=x_axis, 
                        y=y_axis, 
                        title=title,
                        color=color_by
                    )
                    
                    # Update mode for all traces
                    fig.update_traces(mode=mode)
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating line chart: {str(e)}")
        
        elif chart_type == "Scatter Plot":
            # Scatter plot options
            size_by = st.selectbox("Size by (optional):", ["None"] + num_cols)
            size_by = None if size_by == "None" else size_by
            
            # Create scatter plot
            if st.button("Generate Scatter Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.scatter(
                        self.df, 
                        x=x_axis, 
                        y=y_axis, 
                        title=title,
                        color=color_by,
                        size=size_by,
                        opacity=0.7
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating scatter plot: {str(e)}")
        
        elif chart_type == "Pie Chart":
            # Pie chart options
            hole_size = st.slider("Hole size (donut chart):", 0.0, 0.8, 0.0, 0.1)
            
            # Create pie chart
            if st.button("Generate Pie Chart", use_container_width=True):
                try:
                    # Aggregate data for pie chart
                    df_agg = self.df.groupby(x_axis)[y_axis].sum().reset_index()
                    
                    # Create figure
                    fig = px.pie(
                        df_agg, 
                        names=x_axis, 
                        values=y_axis, 
                        title=title,
                        hole=hole_size
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Update traces
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hoverinfo='label+percent+value'
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating pie chart: {str(e)}")
        
        elif chart_type == "Area Chart":
            # Area chart options
            group_mode = st.radio("Mode:", ["Normal", "Stacked", "Filled"], horizontal=True)
            
            # Create area chart
            if st.button("Generate Area Chart", use_container_width=True):
                try:
                    # Create base figure
                    if color_by:
                        # Pivot data for grouped area chart
                        df_pivot = self.df.pivot_table(
                            index=x_axis, 
                            columns=color_by, 
                            values=y_axis, 
                            aggfunc='sum'
                        ).fillna(0)
                        
                        # Create figure based on mode
                        if group_mode == "Normal":
                            fig = px.area(df_pivot, title=title)
                        elif group_mode == "Stacked":
                            fig = px.area(df_pivot, title=title, groupnorm='')
                        else:  # Filled
                            fig = px.area(df_pivot, title=title, groupnorm='fraction')
                    else:
                        # Create simple area chart
                        fig = px.area(
                            self.df, 
                            x=x_axis, 
                            y=y_axis, 
                            title=title
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating area chart: {str(e)}")
        
        elif chart_type == "Histogram":
            # Histogram options
            num_bins = st.slider("Number of bins:", 5, 100, 20)
            show_kde = st.checkbox("Show kernel density estimate", value=True)
            
            # Create histogram
            if st.button("Generate Histogram", use_container_width=True):
                try:
                    # Create figure
                    fig = px.histogram(
                        self.df, 
                        x=x_axis, 
                        color=color_by,
                        nbins=num_bins,
                        title=title,
                        marginal="box" if show_kde else None
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title="Count",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("Statistics for " + x_axis)
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Mean", f"{self.df[x_axis].mean():.2f}")
                    with stats_cols[1]:
                        st.metric("Median", f"{self.df[x_axis].median():.2f}")
                    with stats_cols[2]:
                        st.metric("Standard Deviation", f"{self.df[x_axis].std():.2f}")
                    with stats_cols[3]:
                        st.metric("Count", f"{self.df[x_axis].count()}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating histogram: {str(e)}")
    
    def _render_statistical_plots(self):
        """Render statistical plots visualizations"""
        st.subheader("Statistical Plots")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns available for statistical plots.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot type:",
            ["Box Plot", "Violin Plot", "Distribution Plot", "Q-Q Plot", "Correlation Heatmap", "Radar Chart"]
        )
        
        if plot_type == "Box Plot":
            # Box plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Y-axis (numeric values)
                y_axis = st.selectbox("Y-axis (values):", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Box Plot of {y_axis}")
            
            with col2:
                # X-axis (categories, optional)
                x_axis = st.selectbox("X-axis (categories, optional):", ["None"] + cat_cols)
                x_axis = None if x_axis == "None" else x_axis
                
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
            
            # Box plot options
            show_points = st.checkbox("Show all points", value=False)
            notched = st.checkbox("Notched boxes", value=False)
            
            # Create box plot
            if st.button("Generate Box Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.box(
                        self.df, 
                        y=y_axis, 
                        x=x_axis, 
                        color=color_by,
                        title=title,
                        points="all" if show_points else "outliers",
                        notched=notched
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis if x_axis else "",
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader(f"Statistics for {y_axis}")
                    stats_df = self.df[y_axis].describe().reset_index()
                    stats_df.columns = ["Statistic", "Value"]
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating box plot: {str(e)}")
        
        elif plot_type == "Violin Plot":
            # Violin plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Y-axis (numeric values)
                y_axis = st.selectbox("Y-axis (values):", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Violin Plot of {y_axis}")
            
            with col2:
                # X-axis (categories, optional)
                x_axis = st.selectbox("X-axis (categories, optional):", ["None"] + cat_cols)
                x_axis = None if x_axis == "None" else x_axis
                
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
            
            # Violin plot options
            show_box = st.checkbox("Show box plot inside", value=True)
            show_points = st.checkbox("Show all points", value=False, key="violin_points")
            
            # Create violin plot
            if st.button("Generate Violin Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.violin(
                        self.df, 
                        y=y_axis, 
                        x=x_axis, 
                        color=color_by,
                        title=title,
                        box=show_box,
                        points="all" if show_points else False
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis if x_axis else "",
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating violin plot: {str(e)}")
        
        elif plot_type == "Distribution Plot":
            # Distribution plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Variable to plot
                variable = st.selectbox("Variable:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Distribution of {variable}")
            
            with col2:
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
                
                # Distribution type
                dist_type = st.selectbox("Distribution type:", ["Histogram + KDE", "KDE", "ECDF"])
            
            # Distribution options
            if dist_type in ["Histogram + KDE", "KDE"]:
                num_bins = st.slider("Number of bins (histogram):", 5, 100, 20, key="dist_bins")
                
            # Create distribution plot
            if st.button("Generate Distribution Plot", use_container_width=True):
                try:
                    if dist_type == "Histogram + KDE":
                        # Create histogram with kernel density
                        fig = px.histogram(
                            self.df, 
                            x=variable, 
                            color=color_by,
                            nbins=num_bins,
                            title=title,
                            marginal="rug",
                            histnorm="probability density"
                        )
                        
                        # Add KDE traces
                        if color_by:
                            for color_val in self.df[color_by].unique():
                                subset = self.df[self.df[color_by] == color_val]
                                kde_x, kde_y = self._calculate_kde(subset[variable].dropna())
                                fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name=f"KDE {color_val}")
                        else:
                            kde_x, kde_y = self._calculate_kde(self.df[variable].dropna())
                            fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name="KDE")
                    
                    elif dist_type == "KDE":
                        # Create a figure for KDE
                        fig = go.Figure()
                        
                        # Add KDE traces
                        if color_by:
                            for color_val in self.df[color_by].unique():
                                subset = self.df[self.df[color_by] == color_val]
                                kde_x, kde_y = self._calculate_kde(subset[variable].dropna())
                                fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name=f"{color_val}")
                        else:
                            kde_x, kde_y = self._calculate_kde(self.df[variable].dropna())
                            fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name="KDE")
                        
                        # Update layout
                        fig.update_layout(title=title)
                    
                    else:  # ECDF
                        # Create a figure for ECDF
                        fig = px.ecdf(
                            self.df,
                            x=variable,
                            color=color_by,
                            title=title
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=variable,
                        yaxis_title="Density" if dist_type != "ECDF" else "Cumulative Probability",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader(f"Statistics for {variable}")
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Mean", f"{self.df[variable].mean():.2f}")
                    with stats_cols[1]:
                        st.metric("Median", f"{self.df[variable].median():.2f}")
                    with stats_cols[2]:
                        st.metric("Standard Deviation", f"{self.df[variable].std():.2f}")
                    with stats_cols[3]:
                        st.metric("Skewness", f"{self.df[variable].skew():.2f}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating distribution plot: {str(e)}")
        
        elif plot_type == "Q-Q Plot":
            # Q-Q plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Variable to plot
                variable = st.selectbox("Variable:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Q-Q Plot of {variable}")
            
            with col2:
                # Distribution to compare against
                dist = st.selectbox("Theoretical distribution:", ["Normal", "Exponential", "Uniform"])
                
                # Line color
                line_color = st.color_picker("Line color:", "#FF4B4B")
            
            # Create Q-Q plot
            if st.button("Generate Q-Q Plot", use_container_width=True):
                try:
                    from scipy import stats
                    
                    # Get data without NaN values
                    data = self.df[variable].dropna()
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Calculate probabilities for the specified distribution
                    if dist == "Normal":
                        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Normal Distribution"
                    elif dist == "Exponential":
                        theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Exponential Distribution"
                    else:  # Uniform
                        theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Uniform Distribution"
                    
                    # Add scatter points
                    fig.add_scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Data Points'
                    )
                    
                    # Add reference line
                    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
                    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
                    ref_line = np.linspace(min_val, max_val, 100)
                    
                    fig.add_scatter(
                        x=ref_line,
                        y=ref_line,
                        mode='lines',
                        name='Reference Line',
                        line=dict(color=line_color, dash='dash')
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=f"Theoretical Quantiles ({dist_name})",
                        yaxis_title=f"Sample Quantiles ({variable})",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and show the correlation coefficient
                    correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                    st.info(f"Correlation between sample and theoretical quantiles: {correlation:.4f}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating Q-Q plot: {str(e)}")
        
        elif plot_type == "Correlation Heatmap":
            # Correlation heatmap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select columns for correlation
                corr_cols = st.multiselect(
                    "Columns to include:",
                    num_cols,
                    default=num_cols[:min(8, len(num_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Correlation Heatmap")
            
            with col2:
                # Correlation method
                corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", ["RdBu_r", "Viridis", "Plasma", "Cividis", "Spectral"])
            
            # Show correlation values
            show_values = st.checkbox("Show correlation values", value=True)
            
            # Create correlation heatmap
            if st.button("Generate Correlation Heatmap", use_container_width=True):
                if not corr_cols:
                    st.warning("Please select at least one column for correlation analysis.")
                else:
                    try:
                        # Calculate correlation
                        corr_df = self.df[corr_cols].corr(method=corr_method)
                        
                        # Create figure
                        fig = px.imshow(
                            corr_df,
                            text_auto=show_values,
                            color_continuous_scale=color_scheme,
                            title=title,
                            zmin=-1, zmax=1
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation table
                        st.subheader("Correlation Table")
                        st.dataframe(corr_df.round(2), use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating correlation heatmap: {str(e)}")
        
        elif plot_type == "Radar Chart":
            # Radar chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select numeric columns for radar chart
                radar_cols = st.multiselect(
                    "Variables to include:",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Radar Chart")
            
            with col2:
                # Select categorical column for grouping
                if not cat_cols:
                    st.warning("No categorical columns available for grouping. Will use the entire dataset.")
                    group_col = None
                else:
                    group_col = st.selectbox("Group by (categorical):", ["None"] + cat_cols)
                    group_col = None if group_col == "None" else group_col
                
                # Fill type
                fill_type = st.selectbox("Fill type:", ["toself", "tonext", "none"])
            
            # Scale values
            scale_values = st.checkbox("Scale values (0-1)", value=True)
            
            # Create radar chart
            if st.button("Generate Radar Chart", use_container_width=True):
                if not radar_cols:
                    st.warning("Please select at least three variables for the radar chart.")
                elif len(radar_cols) < 3:
                    st.warning("Please select at least three variables for a meaningful radar chart.")
                else:
                    try:
                        # Create radar chart data
                        fig = go.Figure()
                        
                        # Scale the data if requested
                        if scale_values:
                            # Create a copy of the dataframe with scaled values
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            scaled_df = pd.DataFrame(
                                scaler.fit_transform(self.df[radar_cols]),
                                columns=radar_cols,
                                index=self.df.index
                            )
                            
                            for col in radar_cols:
                                scaled_df[col] = self.df[col].copy()
                        else:
                            scaled_df = self.df.copy()
                        
                        # If grouping by a column
                        if group_col:
                            # Get group values
                            groups = self.df[group_col].unique()
                            
                            for group in groups:
                                # Filter data for this group
                                group_data = scaled_df[self.df[group_col] == group]
                                
                                # Calculate mean for each variable
                                group_means = group_data[radar_cols].mean()
                                
                                # Add to chart
                                fig.add_trace(go.Scatterpolar(
                                    r=group_means.values,
                                    theta=radar_cols,
                                    fill=fill_type,
                                    name=str(group)
                                ))
                        else:
                            # Use the entire dataset
                            means = scaled_df[radar_cols].mean()
                            
                            # Add to chart
                            fig.add_trace(go.Scatterpolar(
                                r=means.values,
                                theta=radar_cols,
                                fill=fill_type,
                                name="Overall"
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=title,
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1] if scale_values else None
                                )
                            ),
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating radar chart: {str(e)}")
    
    def _render_categorical_analysis(self):
        """Render categorical analysis visualizations"""
        st.subheader("Categorical Analysis")
        
        # Get columns by type
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns available for categorical analysis.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Analysis type:",
            ["Category Counts", "Category Proportions", "Grouped Bar Chart", "Heatmap", "Treemap", "Sunburst", "Parallel Categories"]
        )
        
        if plot_type == "Category Counts":
            # Category counts settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical column
                cat_col = st.selectbox("Categorical column:", cat_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Counts of {cat_col}")
            
            with col2:
                # Sort values
                sort_values = st.checkbox("Sort values", value=True)
                
                # Sort order if sorting
                sort_order = None
                if sort_values:
                    sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
                
                # Limit categories
                limit_cats = st.checkbox("Limit top categories", value=False)
                n_cats = 10
                if limit_cats:
                    n_cats = st.slider("Number of categories to show:", 3, 50, 10)
            
            # Chart type
            chart_type = st.radio("Chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Category Counts", use_container_width=True):
                try:
                    # Calculate category counts
                    cat_counts = self.df[cat_col].value_counts()
                    
                    # Apply sorting
                    if sort_values:
                        ascending = sort_order == "Ascending"
                        cat_counts = cat_counts.sort_values(ascending=ascending)
                    
                    # Apply category limit
                    if limit_cats:
                        if len(cat_counts) > n_cats:
                            # Keep top N categories, group others
                            top_cats = cat_counts.head(n_cats)
                            others_sum = cat_counts.iloc[n_cats:].sum()
                            
                            # Add "Others" category if it would be non-empty
                            if others_sum > 0:
                                top_cats = pd.concat([top_cats, pd.Series({"Others": others_sum})])
                            
                            cat_counts = top_cats
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Category': cat_counts.index,
                        'Count': cat_counts.values
                    })
                    
                    # Create the chart
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            plot_df, 
                            x='Category', 
                            y='Count', 
                            title=title,
                            labels={'Category': cat_col, 'Count': 'Count'},
                            color='Category'
                        )
                    else:  # Pie chart
                        fig = px.pie(
                            plot_df, 
                            names='Category', 
                            values='Count', 
                            title=title,
                            labels={'Category': cat_col, 'Count': 'Count'}
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Category Counts")
                    st.dataframe(plot_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating category counts: {str(e)}")
        
        elif plot_type == "Category Proportions":
            # Category proportions settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical column
                cat_col = st.selectbox("Categorical column:", cat_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Proportions of {cat_col}")
            
            with col2:
                # Sort values
                sort_values = st.checkbox("Sort values", value=True)
                
                # Sort order if sorting
                sort_order = None
                if sort_values:
                    sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
                
                # Limit categories
                limit_cats = st.checkbox("Limit top categories", value=False)
                n_cats = 10
                if limit_cats:
                    n_cats = st.slider("Number of categories to show:", 3, 50, 10)
            
            # Chart type
            chart_type = st.radio("Chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Category Proportions", use_container_width=True):
                try:
                    # Calculate category proportions
                    cat_proportions = self.df[cat_col].value_counts(normalize=True)
                    
                    # Apply sorting
                    if sort_values:
                        ascending = sort_order == "Ascending"
                        cat_proportions = cat_proportions.sort_values(ascending=ascending)
                    
                    # Apply category limit
                    if limit_cats:
                        if len(cat_proportions) > n_cats:
                            # Keep top N categories, group others
                            top_cats = cat_proportions.head(n_cats)
                            others_sum = cat_proportions.iloc[n_cats:].sum()
                            
                            # Add "Others" category if it would be non-empty
                            if others_sum > 0:
                                top_cats = pd.concat([top_cats, pd.Series({"Others": others_sum})])
                            
                            cat_proportions = top_cats
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Category': cat_proportions.index,
                        'Proportion': cat_proportions.values,
                        'Percentage': cat_proportions.values * 100
                    })
                    
                    # Create the chart
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            plot_df, 
                            x='Category', 
                            y='Proportion', 
                            title=title,
                            labels={'Category': cat_col, 'Proportion': 'Proportion'},
                            color='Category',
                            text='Percentage'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    else:  # Pie chart
                        fig = px.pie(
                            plot_df, 
                            names='Category', 
                            values='Proportion', 
                            title=title,
                            labels={'Category': cat_col, 'Proportion': 'Proportion'}
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Category Proportions")
                    display_df = plot_df.copy()
                    display_df['Percentage'] = display_df['Percentage'].round(2).astype(str) + '%'
                    display_df['Proportion'] = display_df['Proportion'].round(4)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating category proportions: {str(e)}")
        
        elif plot_type == "Grouped Bar Chart":
            # Grouped bar chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select primary categorical column
                primary_cat = st.selectbox("Primary categorical column:", cat_cols)
                
                # Select secondary categorical column
                secondary_cat = st.selectbox("Secondary categorical column:", 
                                            [c for c in cat_cols if c != primary_cat])
                
                # Title
                title = st.text_input("Chart title:", f"{primary_cat} by {secondary_cat}")
            
            with col2:
                # Orientation
                orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
                
                # Count or proportion
                count_type = st.radio("Show as:", ["Counts", "Proportions"], horizontal=True)
                
                # Stacked or grouped
                bar_mode = st.radio("Bar mode:", ["Grouped", "Stacked"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Grouped Bar Chart", use_container_width=True):
                try:
                    # Create grouped counts or proportions
                    if count_type == "Counts":
                        grouped_data = pd.crosstab(self.df[primary_cat], self.df[secondary_cat])
                        y_title = "Count"
                    else:  # Proportions
                        # Normalize by primary category
                        grouped_data = pd.crosstab(
                            self.df[primary_cat], 
                            self.df[secondary_cat], 
                            normalize='index'
                        )
                        y_title = "Proportion"
                    
                    # Create figure
                    if orientation == "Vertical":
                        # Reset index to make primary_cat a column
                        plot_df = grouped_data.reset_index().melt(
                            id_vars=primary_cat,
                            var_name=secondary_cat,
                            value_name=y_title
                        )
                        
                        fig = px.bar(
                            plot_df,
                            x=primary_cat,
                            y=y_title,
                            color=secondary_cat,
                            title=title,
                            barmode='group' if bar_mode == 'Grouped' else 'stack'
                        )
                    else:  # Horizontal
                        # Reset index to make primary_cat a column
                        plot_df = grouped_data.reset_index().melt(
                            id_vars=primary_cat,
                            var_name=secondary_cat,
                            value_name=y_title
                        )
                        
                        fig = px.bar(
                            plot_df,
                            y=primary_cat,
                            x=y_title,
                            color=secondary_cat,
                            title=title,
                            barmode='group' if bar_mode == 'Grouped' else 'stack',
                            orientation='h'
                        )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Grouped Data")
                    st.dataframe(grouped_data, use_container_width=True)
                    
                    # Calculate chi-square test for independence
                    from scipy.stats import chi2_contingency
                    
                    chi2, p, dof, expected = chi2_contingency(grouped_data)
                    
                    st.subheader("Chi-Square Test for Independence")
                    st.write(f"Chi-square value: {chi2:.2f}")
                    st.write(f"p-value: {p:.4f}")
                    st.write(f"Degrees of freedom: {dof}")
                    
                    if p < 0.05:
                        st.info("The two categorical variables appear to be dependent (significant association).")
                    else:
                        st.info("No significant association detected between the variables.")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating grouped bar chart: {str(e)}")
        
        elif plot_type == "Heatmap":
            # Heatmap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select rows (first categorical column)
                row_cat = st.selectbox("Row variable:", cat_cols)
                
                # Select columns (second categorical column)
                col_cat = st.selectbox("Column variable:", 
                                      [c for c in cat_cols if c != row_cat])
                
                # Title
                title = st.text_input("Chart title:", f"Heatmap of {row_cat} vs {col_cat}")
            
            with col2:
                # Value type
                value_type = st.radio("Values to show:", ["Counts", "Proportions"], horizontal=True)
                
                # Normalization (for proportions)
                norm_by = None
                if value_type == "Proportions":
                    norm_by = st.radio("Normalize by:", ["Row", "Column", "All"], horizontal=True)
                    norm_by = norm_by.lower()
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"])
            
            # Generate heatmap
            if st.button("Generate Heatmap", use_container_width=True):
                try:
                    # Create cross-tabulation
                    if value_type == "Counts":
                        heatmap_data = pd.crosstab(self.df[row_cat], self.df[col_cat])
                        z_title = "Count"
                    else:  # Proportions
                        heatmap_data = pd.crosstab(
                            self.df[row_cat], 
                            self.df[col_cat], 
                            normalize=norm_by
                        )
                        z_title = "Proportion"
                    
                    # Create figure
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x=col_cat, y=row_cat, color=z_title),
                        title=title,
                        color_continuous_scale=color_scheme,
                        text_auto=True
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Cross-tabulation")
                    st.dataframe(heatmap_data, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
        
        elif plot_type == "Treemap":
            # Treemap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select path columns (hierarchical categories)
                path_cols = st.multiselect(
                    "Hierarchy levels (order matters):",
                    cat_cols,
                    default=[cat_cols[0]] if cat_cols else []
                )
                
                # Title
                title = st.text_input("Chart title:", "Treemap")
            
            with col2:
                # Value column (numeric, optional)
                if num_cols:
                    value_col = st.selectbox("Value column (optional):", ["None"] + num_cols)
                    value_col = None if value_col == "None" else value_col
                else:
                    value_col = None
                    st.info("No numeric columns available for values. Will use counts.")
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="treemap_colors")
            
            # Generate treemap
            if st.button("Generate Treemap", use_container_width=True):
                if not path_cols:
                    st.warning("Please select at least one column for the hierarchy.")
                else:
                    try:
                        # Create treemap
                        fig = px.treemap(
                            self.df,
                            path=path_cols,
                            values=value_col,
                            title=title,
                            color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme)
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating treemap: {str(e)}")
        
        elif plot_type == "Sunburst":
            # Sunburst settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select path columns (hierarchical categories)
                path_cols = st.multiselect(
                    "Hierarchy levels (order matters):",
                    cat_cols,
                    default=[cat_cols[0]] if cat_cols else []
                )
                
                # Title
                title = st.text_input("Chart title:", "Sunburst Chart")
            
            with col2:
                # Value column (numeric, optional)
                if num_cols:
                    value_col = st.selectbox("Value column (optional):", ["None"] + num_cols)
                    value_col = None if value_col == "None" else value_col
                else:
                    value_col = None
                    st.info("No numeric columns available for values. Will use counts.")
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="sunburst_colors")
            
            # Generate sunburst
            if st.button("Generate Sunburst", use_container_width=True):
                if not path_cols:
                    st.warning("Please select at least one column for the hierarchy.")
                else:
                    try:
                        # Create sunburst
                        fig = px.sunburst(
                            self.df,
                            path=path_cols,
                            values=value_col,
                            title=title,
                            color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme)
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating sunburst chart: {str(e)}")
        
        elif plot_type == "Parallel Categories":
            # Parallel categories settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical columns
                parallel_cats = st.multiselect(
                    "Categorical columns to include:",
                    cat_cols,
                    default=cat_cols[:min(4, len(cat_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Parallel Categories Diagram")
            
            with col2:
                # Color column
                color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols)
                color_col = None if color_col == "None" else color_col
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="parcats_colors")
            
            # Generate parallel categories
            if st.button("Generate Parallel Categories", use_container_width=True):
                if len(parallel_cats) < 2:
                    st.warning("Please select at least two categorical columns.")
                else:
                    try:
                        # Create parallel categories
                        fig = px.parallel_categories(
                            self.df,
                            dimensions=parallel_cats,
                            color=color_col,
                            title=title,
                            color_continuous_scale=color_scheme
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating parallel categories diagram: {str(e)}")
    
    def _render_time_series(self):
        """Render time series visualizations"""
        st.subheader("Time Series Analysis")
        
        # Get columns by type
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Look for potential date columns if no datetime columns
        potential_date_cols = []
        if not datetime_cols:
            for col in self.df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day']):
                    # Check first value to see if it looks like a date
                    try:
                        sample = self.df[col].dropna().iloc[0]
                        pd.to_datetime(sample)
                        potential_date_cols.append(col)
                    except:
                        pass
        
        # Combined date columns
        all_date_cols = datetime_cols + potential_date_cols
        
        if not all_date_cols:
            st.warning("No datetime columns detected. Please convert a column to datetime format first.")
            return
        
        if not num_cols:
            st.warning("No numeric columns available for time series analysis.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot type:",
            ["Line Chart", "Area Chart", "Heatmap Calendar", "Seasonal Decomposition", "Time Series Components", "Resampling & Rolling Windows"]
        )
        
        if plot_type == "Line Chart":
            # Line chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value columns
                value_cols = st.multiselect("Value column(s):", num_cols, default=[num_cols[0]])
                
                # Title
                title = st.text_input("Chart title:", "Time Series Analysis")
            
            with col2:
                # Select grouping (categorical, optional)
                group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols)
                group_by = None if group_by == "None" else group_by
                
                # Line style
                line_shape = st.selectbox("Line shape:", ["linear", "spline", "hv", "vh", "hvh", "vhv"])
                
                # Include markers
                show_markers = st.checkbox("Show markers", value=False)
            
            # Generate line chart
            if st.button("Generate Time Series Chart", use_container_width=True):
                if not value_cols:
                    st.warning("Please select at least one value column.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Sort by date
                        df_copy = df_copy.sort_values(by=date_col)
                        
                        # Create figure
                        if len(value_cols) == 1 and group_by:
                            # One value column with grouping
                            fig = px.line(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                color=group_by,
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        elif len(value_cols) > 1 and not group_by:
                            # Multiple value columns, no grouping
                            # Need to melt the dataframe
                            df_melted = df_copy.melt(
                                id_vars=[date_col],
                                value_vars=value_cols,
                                var_name='Variable',
                                value_name='Value'
                            )
                            
                            fig = px.line(
                                df_melted, 
                                x=date_col, 
                                y='Value', 
                                color='Variable',
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        else:
                            # Basic case: one value column, no grouping
                            # or multiple value columns with grouping (show warning)
                            if len(value_cols) > 1 and group_by:
                                st.warning("Using only the first value column when grouping is applied.")
                            
                            fig = px.line(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title=date_col,
                            yaxis_title=value_cols[0] if len(value_cols) == 1 else "Value",
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show basic time series stats
                        st.subheader("Time Series Statistics")
                        
                        # For each value column
                        for col in value_cols:
                            st.write(f"**{col}**")
                            stats_cols = st.columns(4)
                            with stats_cols[0]:
                                st.metric("Mean", f"{df_copy[col].mean():.2f}")
                            with stats_cols[1]:
                                st.metric("Min", f"{df_copy[col].min():.2f}")
                            with stats_cols[2]:
                                st.metric("Max", f"{df_copy[col].max():.2f}")
                            with stats_cols[3]:
                                # Calculate trend direction
                                first_valid = df_copy[col].first_valid_index()
                                last_valid = df_copy[col].last_valid_index()
                                if first_valid is not None and last_valid is not None:
                                    start_val = df_copy.loc[first_valid, col]
                                    end_val = df_copy.loc[last_valid, col]
                                    change = end_val - start_val
                                    pct_change = (change / start_val) * 100 if start_val != 0 else float('inf')
                                    
                                    if change > 0:
                                        trend = "↑ Increasing"
                                        delta_color = "normal"
                                    elif change < 0:
                                        trend = "↓ Decreasing"
                                        delta_color = "inverse"
                                    else:
                                        trend = "→ Stable"
                                        delta_color = "off"
                                    
                                    st.metric("Trend", trend, f"{pct_change:.1f}%" if abs(pct_change) != float('inf') else "N/A", delta_color=delta_color)
                                else:
                                    st.metric("Trend", "N/A")
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating time series chart: {str(e)}")
        
        elif plot_type == "Area Chart":
            # Area chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value columns
                value_cols = st.multiselect("Value column(s):", num_cols, default=[num_cols[0]])
                
                # Title
                title = st.text_input("Chart title:", "Time Series Area Chart")
            
            with col2:
                # Select grouping (categorical, optional)
                group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols)
                group_by = None if group_by == "None" else group_by
                
                # Stacking mode
                stack_mode = st.radio("Stacking mode:", ["Overlay", "Stack", "Stack 100%"], horizontal=True)
                
                # Line shape
                line_shape = st.selectbox("Line shape:", ["spline", "linear", "hv", "vh"], key="area_shape")
            
            # Generate area chart
            if st.button("Generate Area Chart", use_container_width=True):
                if not value_cols:
                    st.warning("Please select at least one value column.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Sort by date
                        df_copy = df_copy.sort_values(by=date_col)
                        
                        # Determine stacking mode
                        if stack_mode == "Overlay":
                            groupnorm = None
                        elif stack_mode == "Stack":
                            groupnorm = ''
                        else:  # "Stack 100%"
                            groupnorm = 'percent'
                        
                        # Create figure
                        if len(value_cols) == 1 and group_by:
                            # One value column with grouping
                            fig = px.area(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                color=group_by,
                                title=title,
                                line_shape=line_shape,
                                groupnorm=groupnorm
                            )
                        elif len(value_cols) > 1 and not group_by:
                            # Multiple value columns, no grouping
                            # Need to melt the dataframe
                            df_melted = df_copy.melt(
                                id_vars=[date_col],
                                value_vars=value_cols,
                                var_name='Variable',
                                value_name='Value'
                            )
                            
                            fig = px.area(
                                df_melted, 
                                x=date_col, 
                                y='Value', 
                                color='Variable',
                                title=title,
                                line_shape=line_shape,
                                groupnorm=groupnorm
                            )
                        else:
                            # Basic case: one value column, no grouping
                            # or multiple value columns with grouping (show warning)
                            if len(value_cols) > 1 and group_by:
                                st.warning("Using only the first value column when grouping is applied.")
                            
                            fig = px.area(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                title=title,
                                line_shape=line_shape
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title=date_col,
                            yaxis_title="Value",
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating area chart: {str(e)}")
        
        elif plot_type == "Heatmap Calendar":
            # Heatmap calendar settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Calendar Heatmap")
            
            with col2:
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Sum", "Count", "Min", "Max"])
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="calendar_colors")
            
            # Generate calendar heatmap
            if st.button("Generate Calendar Heatmap", use_container_width=True):
                try:
                    import plotly.graph_objects as go
                    import calendar
                    
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Extract date components
                    df_copy['year'] = df_copy[date_col].dt.year
                    df_copy['month'] = df_copy[date_col].dt.month
                    df_copy['day'] = df_copy[date_col].dt.day
                    df_copy['weekday'] = df_copy[date_col].dt.weekday
                    
                    # Determine aggregation function
                    agg_func = agg_method.lower()
                    
                    # Group by date and aggregate
                    df_agg = df_copy.groupby([date_col])[value_col].agg(agg_func).reset_index()
                    
                    # Extract date components for aggregated data
                    df_agg['year'] = df_agg[date_col].dt.year
                    df_agg['month'] = df_agg[date_col].dt.month
                    df_agg['day'] = df_agg[date_col].dt.day
                    df_agg['weekday'] = df_agg[date_col].dt.weekday
                    
                    # Get unique years and months
                    years = sorted(df_agg['year'].unique())
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Create a calendar heatmap for each year
                    for year in years:
                        year_data = df_agg[df_agg['year'] == year]
                        
                        # For each month in the year
                        for month in range(1, 13):
                            month_data = year_data[year_data['month'] == month]
                            
                            # Create data for this month
                            month_days = calendar.monthrange(year, month)[1]
                            
                            # Create the calendar grid
                            month_name = calendar.month_name[month]
                            
                            # Add a trace for this month
                            hovertext = []
                            values = []
                            
                            for day in range(1, month_days + 1):
                                day_data = month_data[month_data['day'] == day]
                                if not day_data.empty:
                                    value = day_data[value_col].iloc[0]
                                    hovertext.append(f"{month_name} {day}, {year}<br>{value_col}: {value:.2f}")
                                    values.append(value)
                                else:
                                    hovertext.append(f"{month_name} {day}, {year}<br>No data")
                                    values.append(None)
                            
                            # Skip months with no data
                            if all(v is None for v in values):
                                continue
                            
                            # Create a trace for this month
                            fig.add_trace(go.Heatmap(
                                z=[values],
                                x=[f"{month_name} {day}" for day in range(1, month_days + 1)],
                                y=[f"{year} - {month_name}"],
                                hoverinfo="text",
                                text=[hovertext],
                                colorscale=color_scheme,
                                showscale=True,
                                name=f"{month_name} {year}"
                            ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        height=max(600, 200 * len(years)),
                        xaxis_title="Day",
                        yaxis_title="Month-Year",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(0, 31, 5)),
                            ticktext=[str(i) for i in range(1, 32, 5)]
                        )
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating calendar heatmap: {str(e)}")
        
        elif plot_type == "Seasonal Decomposition":
            # Seasonal decomposition settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Seasonal Decomposition")
            
            with col2:
                # Decomposition method
                decomp_method = st.radio("Decomposition method:", ["Additive", "Multiplicative"], horizontal=True)
                
                # Period selection
                freq_options = ["Day", "Week", "Month", "Quarter", "Year", "Custom"]
                freq_select = st.selectbox("Seasonality period:", freq_options)
                
                if freq_select == "Custom":
                    period = st.number_input("Number of periods:", min_value=2, value=12)
                else:
                    # Map selection to period
                    freq_map = {
                        "Day": 1,
                        "Week": 7,
                        "Month": 30,
                        "Quarter": 90,
                        "Year": 365
                    }
                    period = freq_map.get(freq_select, 12)
            
            # Generate seasonal decomposition
            if st.button("Generate Seasonal Decomposition", use_container_width=True):
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Sort by date
                    df_copy = df_copy.sort_values(by=date_col)
                    
                    # Set date as index
                    df_copy = df_copy.set_index(date_col)
                    
                    # Decompose the time series
                    decomposition = seasonal_decompose(
                        df_copy[value_col].dropna(), 
                        model=decomp_method.lower(),
                        period=period
                    )
                    
                    # Create plots
                    fig = go.Figure()
                    
                    # Add original data
                    fig.add_trace(go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        mode='lines',
                        name='Original'
                    ))
                    
                    # Add trend
                    fig.add_trace(go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red')
                    ))
                    
                    # Add seasonal
                    fig.add_trace(go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        mode='lines',
                        name='Seasonal',
                        line=dict(color='green')
                    ))
                    
                    # Add residual
                    fig.add_trace(go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        mode='lines',
                        name='Residual',
                        line=dict(color='purple')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=date_col,
                        yaxis_title=value_col,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create individual components charts
                    st.subheader("Individual Components")
                    
                    component_tabs = st.tabs(["Trend", "Seasonal", "Residual"])
                    
                    with component_tabs[0]:
                        trend_fig = px.line(
                            x=decomposition.trend.index, 
                            y=decomposition.trend,
                            title="Trend Component",
                            labels={"x": date_col, "y": f"Trend of {value_col}"}
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)
                    
                    with component_tabs[1]:
                        seasonal_fig = px.line(
                            x=decomposition.seasonal.index, 
                            y=decomposition.seasonal,
                            title="Seasonal Component",
                            labels={"x": date_col, "y": f"Seasonality of {value_col}"}
                        )
                        st.plotly_chart(seasonal_fig, use_container_width=True)
                    
                    with component_tabs[2]:
                        resid_fig = px.line(
                            x=decomposition.resid.index, 
                            y=decomposition.resid,
                            title="Residual Component",
                            labels={"x": date_col, "y": f"Residual of {value_col}"}
                        )
                        st.plotly_chart(resid_fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating seasonal decomposition: {str(e)}")
        
        elif plot_type == "Time Series Components":
            # Time series components settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Time Series Components")
            
            with col2:
                # Components to show
                components = st.multiselect(
                    "Components to show:",
                    ["Day of Week", "Month of Year", "Hour of Day", "Quarter", "Year", "Week of Year"],
                    default=["Day of Week", "Month of Year"]
                )
                
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Median", "Sum", "Count", "Min", "Max"], key="comp_agg")
            
            # Generate time series components
            if st.button("Generate Time Series Components", use_container_width=True):
                if not components:
                    st.warning("Please select at least one component.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Create tabs for each component
                        component_tabs = st.tabs(components)
                        
                        for i, component in enumerate(components):
                            with component_tabs[i]:
                                # Extract the relevant component
                                if component == "Day of Week":
                                    df_copy['component'] = df_copy[date_col].dt.day_name()
                                    x_title = "Day of Week"
                                    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                elif component == "Month of Year":
                                    df_copy['component'] = df_copy[date_col].dt.month_name()
                                    x_title = "Month"
                                    order = ["January", "February", "March", "April", "May", "June", 
                                            "July", "August", "September", "October", "November", "December"]
                                elif component == "Hour of Day":
                                    df_copy['component'] = df_copy[date_col].dt.hour
                                    x_title = "Hour"
                                    order = list(range(24))
                                elif component == "Quarter":
                                    df_copy['component'] = "Q" + df_copy[date_col].dt.quarter.astype(str)
                                    x_title = "Quarter"
                                    order = ["Q1", "Q2", "Q3", "Q4"]
                                elif component == "Year":
                                    df_copy['component'] = df_copy[date_col].dt.year
                                    x_title = "Year"
                                    order = sorted(df_copy['component'].unique())
                                elif component == "Week of Year":
                                    df_copy['component'] = df_copy[date_col].dt.isocalendar().week
                                    x_title = "Week of Year"
                                    order = list(range(1, 54))
                                
                                # Aggregate by component
                                agg_func = agg_method.lower()
                                component_agg = df_copy.groupby('component')[value_col].agg(agg_func).reset_index()
                                
                                # Create ordered categorical if applicable
                                if order:
                                    # Filter order to only include values that exist in the data
                                    order = [o for o in order if o in component_agg['component'].values]
                                    component_agg['component'] = pd.Categorical(
                                        component_agg['component'], 
                                        categories=order,
                                        ordered=True
                                    )
                                    # Sort by the ordered categorical
                                    component_agg = component_agg.sort_values('component')
                                
                                # Create figure
                                component_fig = px.bar(
                                    component_agg,
                                    x='component',
                                    y=value_col,
                                    title=f"{component} Analysis of {value_col} ({agg_method})",
                                    labels={"component": x_title, "value": value_col},
                                    color='component'
                                )
                                
                                # Add average line
                                avg_value = df_copy[value_col].mean()
                                component_fig.add_hline(
                                    y=avg_value,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Overall {agg_method}: {avg_value:.2f}"
                                )
                                
                                # Update layout
                                component_fig.update_layout(
                                    showlegend=False,
                                    height=500
                                )
                                
                                # Show the chart
                                st.plotly_chart(component_fig, use_container_width=True)
                                
                                # Show statistics
                                st.subheader(f"Statistics by {component}")
                                st.dataframe(component_agg, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating time series components: {str(e)}")
        
        elif plot_type == "Resampling & Rolling Windows":
            # Resampling settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Resampled Time Series")
            
            with col2:
                # Analysis type
                analysis_type = st.radio("Analysis type:", ["Resampling", "Rolling Window"], horizontal=True)
                
                # Settings based on analysis type
                if analysis_type == "Resampling":
                    # Resampling period
                    resample_options = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y",
                        "Hourly": "H"
                    }
                    resample_period = st.selectbox("Resample to:", list(resample_options.keys()))
                    resample_rule = resample_options[resample_period]
                    
                    window_size = None
                else:  # Rolling Window
                    # Window size
                    window_size = st.slider("Window size:", 2, 100, 7)
                    resample_rule = None
                
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Median", "Sum", "Min", "Max", "Count", "Std"], key="resample_agg")
            
            # Generate resampled/rolling window chart
            if st.button("Generate Chart", use_container_width=True):
                try:
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Sort by date
                    df_copy = df_copy.sort_values(by=date_col)
                    
                    # Set date as index
                    df_copy = df_copy.set_index(date_col)
                    
                    # Determine aggregation function
                    agg_func = agg_method.lower()
                    
                    # Apply resampling or rolling window
                    if analysis_type == "Resampling":
                        # Resample data
                        resampled = getattr(df_copy[value_col].resample(resample_rule), agg_func)()
                        result_df = resampled.reset_index()
                        result_df.columns = [date_col, f"{value_col}_{agg_func}"]
                        
                        # Create figure
                        fig = px.line(
                            result_df, 
                            x=date_col, 
                            y=f"{value_col}_{agg_func}", 
                            title=f"{title} ({resample_period} {agg_method})",
                            markers=True
                        )
                    else:  # Rolling Window
                        # Apply rolling window
                        rolling = getattr(df_copy[value_col].rolling(window=window_size), agg_func)()
                        result_df = rolling.reset_index()
                        result_df.columns = [date_col, f"{value_col}_{agg_func}"]
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Add original data
                        fig.add_trace(go.Scatter(
                            x=df_copy.index,
                            y=df_copy[value_col],
                            mode='lines',
                            name='Original',
                            line=dict(color='blue', width=1, dash='dash')
                        ))
                        
                        # Add rolling window result
                        fig.add_trace(go.Scatter(
                            x=result_df[date_col],
                            y=result_df[f"{value_col}_{agg_func}"],
                            mode='lines',
                            name=f'{window_size}-period Rolling {agg_method}',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{title} ({window_size}-period Rolling {agg_method})",
                        )
                    
                    # Update common layout settings
                    fig.update_layout(
                        xaxis_title=date_col,
                        yaxis_title=value_col,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
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
            pickle_protocol = st.slider("Pickle protocol version:", 0, 5, 4)
            pickle_compression = st.selectbox("Compression:", ["None", "gzip", "bz2", "xz"])
            
            # Generate file
            if st.button("Generate Pickle for Download", use_container_width=True):
                try:
                    pickle_buffer = io.BytesIO()
                    
                    if pickle_compression == "None":
                        self.df.to_pickle(pickle_buffer, protocol=pickle_protocol)
                    else:
                        self.df.to_pickle(
                            pickle_buffer, 
                            protocol=pickle_protocol,
                            compression=pickle_compression
                        )
                    
                    pickle_data = pickle_buffer.getvalue()
                    
                    # Create download button
                    file_extension = ".pkl" if pickle_compression == "None" else f".{pickle_compression}.pkl"
                    st.download_button(
                        label="Download Pickle File",
                        data=pickle_data,
                        file_name=f"processed_data{file_extension}",
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
                            "protocol": pickle_protocol,
                            "compression": pickle_compression
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error exporting to Pickle: {str(e)}")

class DataVisualization:
    """Class for creating data visualizations"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
    
    def render_interface(self):
        """Render the data visualization interface"""
        st.header("Data Visualization")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin data visualization.")
            return
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs([
            "Basic Charts",
            "Statistical Plots",
            "Categorical Analysis",
            "Time Series",
            "Multivariate Analysis"
        ])
        
        # Basic Charts Tab
        with viz_tabs[0]:
            self._render_basic_charts()
        
        # Statistical Plots Tab
        with viz_tabs[1]:
            self._render_statistical_plots()
        
        # Categorical Analysis Tab
        with viz_tabs[2]:
            self._render_categorical_analysis()
        
        # Time Series Tab
        with viz_tabs[3]:
            self._render_time_series()
        
        # Multivariate Analysis Tab
        with viz_tabs[4]:
            self._render_multivariate_analysis()
    
    def _render_basic_charts(self):
        """Render basic charts visualizations"""
        st.subheader("Basic Charts")
        
        # Get columns by type
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Area Chart", "Histogram"]
        )
        
        # Common settings for all chart types
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            # X-axis selection (dependent on chart type)
            if chart_type == "Histogram":
                # For histogram, only need one numeric column
                if not num_cols:
                    st.warning("No numeric columns available for histogram.")
                    return
                
                x_axis = st.selectbox("Column for histogram:", num_cols)
                y_axis = None
            
            elif chart_type == "Pie Chart":
                # For pie chart, need one categorical and one numeric column
                if not cat_cols:
                    st.warning("No categorical columns available for pie chart labels.")
                    return
                
                if not num_cols:
                    st.warning("No numeric columns available for pie chart values.")
                    return
                
                x_axis = st.selectbox("Labels (categorical):", cat_cols)
                y_axis = st.selectbox("Values (numeric):", num_cols)
            
            else:
                # For other charts, x-axis can be numeric or categorical
                x_axis = st.selectbox("X-axis:", num_cols + cat_cols)
                
                # Y-axis is required for all except pie chart and histogram
                # Only show numeric columns for y-axis
                if not num_cols:
                    st.warning("No numeric columns available for y-axis.")
                    return
                
                y_axis = st.selectbox("Y-axis:", num_cols)
        
        with settings_col2:
            # Color by column (optional)
            color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
            color_by = None if color_by == "None" else color_by
            
            # Title
            title = st.text_input("Chart title:", f"{chart_type} of {y_axis or x_axis}")
        
        # Chart specific options
        if chart_type == "Bar Chart":
            orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
            is_horizontal = orientation == "Horizontal"
            
            # Sort bars
            sort_bars = st.checkbox("Sort bars", value=False)
            if sort_bars:
                sort_order = st.radio("Sort order:", ["Ascending", "Descending"], horizontal=True)
            
            # Create bar chart
            if st.button("Generate Bar Chart", use_container_width=True):
                try:
                    # Create figure
                    if sort_bars:
                        if sort_order == "Ascending":
                            sort_dir = True
                        else:
                            sort_dir = False
                        
                        # Sort the dataframe
                        df_sorted = self.df.sort_values(by=y_axis, ascending=sort_dir)
                    else:
                        df_sorted = self.df
                    
                    if is_horizontal:
                        fig = px.bar(
                            df_sorted, 
                            y=x_axis, 
                            x=y_axis, 
                            title=title,
                            color=color_by,
                            orientation='h'
                        )
                    else:
                        fig = px.bar(
                            df_sorted, 
                            x=x_axis, 
                            y=y_axis, 
                            title=title,
                            color=color_by
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=y_axis if is_horizontal else x_axis,
                        yaxis_title=x_axis if is_horizontal else y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating bar chart: {str(e)}")
        
        elif chart_type == "Line Chart":
            # Line chart options
            line_mode = st.radio("Line mode:", ["Lines", "Lines + Markers", "Markers"], horizontal=True)
            
            # Create line chart
            if st.button("Generate Line Chart", use_container_width=True):
                try:
                    # Determine mode
                    if line_mode == "Lines":
                        mode = "lines"
                    elif line_mode == "Lines + Markers":
                        mode = "lines+markers"
                    else:  # Markers only
                        mode = "markers"
                    
                    # Create figure
                    fig = px.line(
                        self.df, 
                        x=x_axis, 
                        y=y_axis, 
                        title=title,
                        color=color_by
                    )
                    
                    # Update mode for all traces
                    fig.update_traces(mode=mode)
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating line chart: {str(e)}")
        
        elif chart_type == "Scatter Plot":
            # Scatter plot options
            size_by = st.selectbox("Size by (optional):", ["None"] + num_cols)
            size_by = None if size_by == "None" else size_by
            
            # Create scatter plot
            if st.button("Generate Scatter Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.scatter(
                        self.df, 
                        x=x_axis, 
                        y=y_axis, 
                        title=title,
                        color=color_by,
                        size=size_by,
                        opacity=0.7
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating scatter plot: {str(e)}")
        
        elif chart_type == "Pie Chart":
            # Pie chart options
            hole_size = st.slider("Hole size (donut chart):", 0.0, 0.8, 0.0, 0.1)
            
            # Create pie chart
            if st.button("Generate Pie Chart", use_container_width=True):
                try:
                    # Aggregate data for pie chart
                    df_agg = self.df.groupby(x_axis)[y_axis].sum().reset_index()
                    
                    # Create figure
                    fig = px.pie(
                        df_agg, 
                        names=x_axis, 
                        values=y_axis, 
                        title=title,
                        hole=hole_size
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Update traces
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hoverinfo='label+percent+value'
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating pie chart: {str(e)}")
        
        elif chart_type == "Area Chart":
            # Area chart options
            group_mode = st.radio("Mode:", ["Normal", "Stacked", "Filled"], horizontal=True)
            
            # Create area chart
            if st.button("Generate Area Chart", use_container_width=True):
                try:
                    # Create base figure
                    if color_by:
                        # Pivot data for grouped area chart
                        df_pivot = self.df.pivot_table(
                            index=x_axis, 
                            columns=color_by, 
                            values=y_axis, 
                            aggfunc='sum'
                        ).fillna(0)
                        
                        # Create figure based on mode
                        if group_mode == "Normal":
                            fig = px.area(df_pivot, title=title)
                        elif group_mode == "Stacked":
                            fig = px.area(df_pivot, title=title, groupnorm='')
                        else:  # Filled
                            fig = px.area(df_pivot, title=title, groupnorm='fraction')
                    else:
                        # Create simple area chart
                        fig = px.area(
                            self.df, 
                            x=x_axis, 
                            y=y_axis, 
                            title=title
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating area chart: {str(e)}")
        
        elif chart_type == "Histogram":
            # Histogram options
            num_bins = st.slider("Number of bins:", 5, 100, 20)
            show_kde = st.checkbox("Show kernel density estimate", value=True)
            
            # Create histogram
            if st.button("Generate Histogram", use_container_width=True):
                try:
                    # Create figure
                    fig = px.histogram(
                        self.df, 
                        x=x_axis, 
                        color=color_by,
                        nbins=num_bins,
                        title=title,
                        marginal="box" if show_kde else None
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title="Count",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("Statistics for " + x_axis)
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Mean", f"{self.df[x_axis].mean():.2f}")
                    with stats_cols[1]:
                        st.metric("Median", f"{self.df[x_axis].median():.2f}")
                    with stats_cols[2]:
                        st.metric("Standard Deviation", f"{self.df[x_axis].std():.2f}")
                    with stats_cols[3]:
                        st.metric("Count", f"{self.df[x_axis].count()}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating histogram: {str(e)}")
    
    def _render_statistical_plots(self):
        """Render statistical plots visualizations"""
        st.subheader("Statistical Plots")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns available for statistical plots.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot type:",
            ["Box Plot", "Violin Plot", "Distribution Plot", "Q-Q Plot", "Correlation Heatmap", "Radar Chart"]
        )
        
        if plot_type == "Box Plot":
            # Box plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Y-axis (numeric values)
                y_axis = st.selectbox("Y-axis (values):", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Box Plot of {y_axis}")
            
            with col2:
                # X-axis (categories, optional)
                x_axis = st.selectbox("X-axis (categories, optional):", ["None"] + cat_cols)
                x_axis = None if x_axis == "None" else x_axis
                
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
            
            # Box plot options
            show_points = st.checkbox("Show all points", value=False)
            notched = st.checkbox("Notched boxes", value=False)
            
            # Create box plot
            if st.button("Generate Box Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.box(
                        self.df, 
                        y=y_axis, 
                        x=x_axis, 
                        color=color_by,
                        title=title,
                        points="all" if show_points else "outliers",
                        notched=notched
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis if x_axis else "",
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader(f"Statistics for {y_axis}")
                    stats_df = self.df[y_axis].describe().reset_index()
                    stats_df.columns = ["Statistic", "Value"]
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating box plot: {str(e)}")
        
        elif plot_type == "Violin Plot":
            # Violin plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Y-axis (numeric values)
                y_axis = st.selectbox("Y-axis (values):", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Violin Plot of {y_axis}")
            
            with col2:
                # X-axis (categories, optional)
                x_axis = st.selectbox("X-axis (categories, optional):", ["None"] + cat_cols)
                x_axis = None if x_axis == "None" else x_axis
                
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
            
            # Violin plot options
            show_box = st.checkbox("Show box plot inside", value=True)
            show_points = st.checkbox("Show all points", value=False, key="violin_points")
            
            # Create violin plot
            if st.button("Generate Violin Plot", use_container_width=True):
                try:
                    # Create figure
                    fig = px.violin(
                        self.df, 
                        y=y_axis, 
                        x=x_axis, 
                        color=color_by,
                        title=title,
                        box=show_box,
                        points="all" if show_points else False
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_axis if x_axis else "",
                        yaxis_title=y_axis,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating violin plot: {str(e)}")
        
        elif plot_type == "Distribution Plot":
            # Distribution plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Variable to plot
                variable = st.selectbox("Variable:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Distribution of {variable}")
            
            with col2:
                # Color by (optional)
                color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                color_by = None if color_by == "None" else color_by
                
                # Distribution type
                dist_type = st.selectbox("Distribution type:", ["Histogram + KDE", "KDE", "ECDF"])
            
            # Distribution options
            if dist_type in ["Histogram + KDE", "KDE"]:
                num_bins = st.slider("Number of bins (histogram):", 5, 100, 20, key="dist_bins")
                
            # Create distribution plot
            if st.button("Generate Distribution Plot", use_container_width=True):
                try:
                    if dist_type == "Histogram + KDE":
                        # Create histogram with kernel density
                        fig = px.histogram(
                            self.df, 
                            x=variable, 
                            color=color_by,
                            nbins=num_bins,
                            title=title,
                            marginal="rug",
                            histnorm="probability density"
                        )
                        
                        # Add KDE traces
                        if color_by:
                            for color_val in self.df[color_by].unique():
                                subset = self.df[self.df[color_by] == color_val]
                                kde_x, kde_y = self._calculate_kde(subset[variable].dropna())
                                fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name=f"KDE {color_val}")
                        else:
                            kde_x, kde_y = self._calculate_kde(self.df[variable].dropna())
                            fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name="KDE")
                    
                    elif dist_type == "KDE":
                        # Create a figure for KDE
                        fig = go.Figure()
                        
                        # Add KDE traces
                        if color_by:
                            for color_val in self.df[color_by].unique():
                                subset = self.df[self.df[color_by] == color_val]
                                kde_x, kde_y = self._calculate_kde(subset[variable].dropna())
                                fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name=f"{color_val}")
                        else:
                            kde_x, kde_y = self._calculate_kde(self.df[variable].dropna())
                            fig.add_scatter(x=kde_x, y=kde_y, mode='lines', name="KDE")
                        
                        # Update layout
                        fig.update_layout(title=title)
                    
                    else:  # ECDF
                        # Create a figure for ECDF
                        fig = px.ecdf(
                            self.df,
                            x=variable,
                            color=color_by,
                            title=title
                        )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=variable,
                        yaxis_title="Density" if dist_type != "ECDF" else "Cumulative Probability",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader(f"Statistics for {variable}")
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Mean", f"{self.df[variable].mean():.2f}")
                    with stats_cols[1]:
                        st.metric("Median", f"{self.df[variable].median():.2f}")
                    with stats_cols[2]:
                        st.metric("Standard Deviation", f"{self.df[variable].std():.2f}")
                    with stats_cols[3]:
                        st.metric("Skewness", f"{self.df[variable].skew():.2f}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating distribution plot: {str(e)}")
        
        elif plot_type == "Q-Q Plot":
            # Q-Q plot settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Variable to plot
                variable = st.selectbox("Variable:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Q-Q Plot of {variable}")
            
            with col2:
                # Distribution to compare against
                dist = st.selectbox("Theoretical distribution:", ["Normal", "Exponential", "Uniform"])
                
                # Line color
                line_color = st.color_picker("Line color:", "#FF4B4B")
            
            # Create Q-Q plot
            if st.button("Generate Q-Q Plot", use_container_width=True):
                try:
                    from scipy import stats
                    
                    # Get data without NaN values
                    data = self.df[variable].dropna()
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Calculate probabilities for the specified distribution
                    if dist == "Normal":
                        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Normal Distribution"
                    elif dist == "Exponential":
                        theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Exponential Distribution"
                    else:  # Uniform
                        theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, 100))
                        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                        dist_name = "Uniform Distribution"
                    
                    # Add scatter points
                    fig.add_scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Data Points'
                    )
                    
                    # Add reference line
                    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
                    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
                    ref_line = np.linspace(min_val, max_val, 100)
                    
                    fig.add_scatter(
                        x=ref_line,
                        y=ref_line,
                        mode='lines',
                        name='Reference Line',
                        line=dict(color=line_color, dash='dash')
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=f"Theoretical Quantiles ({dist_name})",
                        yaxis_title=f"Sample Quantiles ({variable})",
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and show the correlation coefficient
                    correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                    st.info(f"Correlation between sample and theoretical quantiles: {correlation:.4f}")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating Q-Q plot: {str(e)}")
        
        elif plot_type == "Correlation Heatmap":
            # Correlation heatmap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select columns for correlation
                corr_cols = st.multiselect(
                    "Columns to include:",
                    num_cols,
                    default=num_cols[:min(8, len(num_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Correlation Heatmap")
            
            with col2:
                # Correlation method
                corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", ["RdBu_r", "Viridis", "Plasma", "Cividis", "Spectral"])
            
            # Show correlation values
            show_values = st.checkbox("Show correlation values", value=True)
            
            # Create correlation heatmap
            if st.button("Generate Correlation Heatmap", use_container_width=True):
                if not corr_cols:
                    st.warning("Please select at least one column for correlation analysis.")
                else:
                    try:
                        # Calculate correlation
                        corr_df = self.df[corr_cols].corr(method=corr_method)
                        
                        # Create figure
                        fig = px.imshow(
                            corr_df,
                            text_auto=show_values,
                            color_continuous_scale=color_scheme,
                            title=title,
                            zmin=-1, zmax=1
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation table
                        st.subheader("Correlation Table")
                        st.dataframe(corr_df.round(2), use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating correlation heatmap: {str(e)}")
        
        elif plot_type == "Radar Chart":
            # Radar chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select numeric columns for radar chart
                radar_cols = st.multiselect(
                    "Variables to include:",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Radar Chart")
            
            with col2:
                # Select categorical column for grouping
                if not cat_cols:
                    st.warning("No categorical columns available for grouping. Will use the entire dataset.")
                    group_col = None
                else:
                    group_col = st.selectbox("Group by (categorical):", ["None"] + cat_cols)
                    group_col = None if group_col == "None" else group_col
                
                # Fill type
                fill_type = st.selectbox("Fill type:", ["toself", "tonext", "none"])
            
            # Scale values
            scale_values = st.checkbox("Scale values (0-1)", value=True)
            
            # Create radar chart
            if st.button("Generate Radar Chart", use_container_width=True):
                if not radar_cols:
                    st.warning("Please select at least three variables for the radar chart.")
                elif len(radar_cols) < 3:
                    st.warning("Please select at least three variables for a meaningful radar chart.")
                else:
                    try:
                        # Create radar chart data
                        fig = go.Figure()
                        
                        # Scale the data if requested
                        if scale_values:
                            # Create a copy of the dataframe with scaled values
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            scaled_df = pd.DataFrame(
                                scaler.fit_transform(self.df[radar_cols]),
                                columns=radar_cols,
                                index=self.df.index
                            )
                            
                            for col in radar_cols:
                                scaled_df[col] = self.df[col].copy()
                        else:
                            scaled_df = self.df.copy()
                        
                        # If grouping by a column
                        if group_col:
                            # Get group values
                            groups = self.df[group_col].unique()
                            
                            for group in groups:
                                # Filter data for this group
                                group_data = scaled_df[self.df[group_col] == group]
                                
                                # Calculate mean for each variable
                                group_means = group_data[radar_cols].mean()
                                
                                # Add to chart
                                fig.add_trace(go.Scatterpolar(
                                    r=group_means.values,
                                    theta=radar_cols,
                                    fill=fill_type,
                                    name=str(group)
                                ))
                        else:
                            # Use the entire dataset
                            means = scaled_df[radar_cols].mean()
                            
                            # Add to chart
                            fig.add_trace(go.Scatterpolar(
                                r=means.values,
                                theta=radar_cols,
                                fill=fill_type,
                                name="Overall"
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=title,
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1] if scale_values else None
                                )
                            ),
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating radar chart: {str(e)}")
    
    def _render_categorical_analysis(self):
        """Render categorical analysis visualizations"""
        st.subheader("Categorical Analysis")
        
        # Get columns by type
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns available for categorical analysis.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Analysis type:",
            ["Category Counts", "Category Proportions", "Grouped Bar Chart", "Heatmap", "Treemap", "Sunburst", "Parallel Categories"]
        )
        
        if plot_type == "Category Counts":
            # Category counts settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical column
                cat_col = st.selectbox("Categorical column:", cat_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Counts of {cat_col}")
            
            with col2:
                # Sort values
                sort_values = st.checkbox("Sort values", value=True)
                
                # Sort order if sorting
                sort_order = None
                if sort_values:
                    sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
                
                # Limit categories
                limit_cats = st.checkbox("Limit top categories", value=False)
                n_cats = 10
                if limit_cats:
                    n_cats = st.slider("Number of categories to show:", 3, 50, 10)
            
            # Chart type
            chart_type = st.radio("Chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Category Counts", use_container_width=True):
                try:
                    # Calculate category counts
                    cat_counts = self.df[cat_col].value_counts()
                    
                    # Apply sorting
                    if sort_values:
                        ascending = sort_order == "Ascending"
                        cat_counts = cat_counts.sort_values(ascending=ascending)
                    
                    # Apply category limit
                    if limit_cats:
                        if len(cat_counts) > n_cats:
                            # Keep top N categories, group others
                            top_cats = cat_counts.head(n_cats)
                            others_sum = cat_counts.iloc[n_cats:].sum()
                            
                            # Add "Others" category if it would be non-empty
                            if others_sum > 0:
                                top_cats = pd.concat([top_cats, pd.Series({"Others": others_sum})])
                            
                            cat_counts = top_cats
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Category': cat_counts.index,
                        'Count': cat_counts.values
                    })
                    
                    # Create the chart
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            plot_df, 
                            x='Category', 
                            y='Count', 
                            title=title,
                            labels={'Category': cat_col, 'Count': 'Count'},
                            color='Category'
                        )
                    else:  # Pie chart
                        fig = px.pie(
                            plot_df, 
                            names='Category', 
                            values='Count', 
                            title=title,
                            labels={'Category': cat_col, 'Count': 'Count'}
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Category Counts")
                    st.dataframe(plot_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating category counts: {str(e)}")
        
        elif plot_type == "Category Proportions":
            # Category proportions settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical column
                cat_col = st.selectbox("Categorical column:", cat_cols)
                
                # Title
                title = st.text_input("Chart title:", f"Proportions of {cat_col}")
            
            with col2:
                # Sort values
                sort_values = st.checkbox("Sort values", value=True)
                
                # Sort order if sorting
                sort_order = None
                if sort_values:
                    sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
                
                # Limit categories
                limit_cats = st.checkbox("Limit top categories", value=False)
                n_cats = 10
                if limit_cats:
                    n_cats = st.slider("Number of categories to show:", 3, 50, 10)
            
            # Chart type
            chart_type = st.radio("Chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Category Proportions", use_container_width=True):
                try:
                    # Calculate category proportions
                    cat_proportions = self.df[cat_col].value_counts(normalize=True)
                    
                    # Apply sorting
                    if sort_values:
                        ascending = sort_order == "Ascending"
                        cat_proportions = cat_proportions.sort_values(ascending=ascending)
                    
                    # Apply category limit
                    if limit_cats:
                        if len(cat_proportions) > n_cats:
                            # Keep top N categories, group others
                            top_cats = cat_proportions.head(n_cats)
                            others_sum = cat_proportions.iloc[n_cats:].sum()
                            
                            # Add "Others" category if it would be non-empty
                            if others_sum > 0:
                                top_cats = pd.concat([top_cats, pd.Series({"Others": others_sum})])
                            
                            cat_proportions = top_cats
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Category': cat_proportions.index,
                        'Proportion': cat_proportions.values,
                        'Percentage': cat_proportions.values * 100
                    })
                    
                    # Create the chart
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            plot_df, 
                            x='Category', 
                            y='Proportion', 
                            title=title,
                            labels={'Category': cat_col, 'Proportion': 'Proportion'},
                            color='Category',
                            text='Percentage'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    else:  # Pie chart
                        fig = px.pie(
                            plot_df, 
                            names='Category', 
                            values='Proportion', 
                            title=title,
                            labels={'Category': cat_col, 'Proportion': 'Proportion'}
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Category Proportions")
                    display_df = plot_df.copy()
                    display_df['Percentage'] = display_df['Percentage'].round(2).astype(str) + '%'
                    display_df['Proportion'] = display_df['Proportion'].round(4)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating category proportions: {str(e)}")
        
        elif plot_type == "Grouped Bar Chart":
            # Grouped bar chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select primary categorical column
                primary_cat = st.selectbox("Primary categorical column:", cat_cols)
                
                # Select secondary categorical column
                secondary_cat = st.selectbox("Secondary categorical column:", 
                                            [c for c in cat_cols if c != primary_cat])
                
                # Title
                title = st.text_input("Chart title:", f"{primary_cat} by {secondary_cat}")
            
            with col2:
                # Orientation
                orientation = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
                
                # Count or proportion
                count_type = st.radio("Show as:", ["Counts", "Proportions"], horizontal=True)
                
                # Stacked or grouped
                bar_mode = st.radio("Bar mode:", ["Grouped", "Stacked"], horizontal=True)
            
            # Generate chart
            if st.button("Generate Grouped Bar Chart", use_container_width=True):
                try:
                    # Create grouped counts or proportions
                    if count_type == "Counts":
                        grouped_data = pd.crosstab(self.df[primary_cat], self.df[secondary_cat])
                        y_title = "Count"
                    else:  # Proportions
                        # Normalize by primary category
                        grouped_data = pd.crosstab(
                            self.df[primary_cat], 
                            self.df[secondary_cat], 
                            normalize='index'
                        )
                        y_title = "Proportion"
                    
                    # Create figure
                    if orientation == "Vertical":
                        # Reset index to make primary_cat a column
                        plot_df = grouped_data.reset_index().melt(
                            id_vars=primary_cat,
                            var_name=secondary_cat,
                            value_name=y_title
                        )
                        
                        fig = px.bar(
                            plot_df,
                            x=primary_cat,
                            y=y_title,
                            color=secondary_cat,
                            title=title,
                            barmode='group' if bar_mode == 'Grouped' else 'stack'
                        )
                    else:  # Horizontal
                        # Reset index to make primary_cat a column
                        plot_df = grouped_data.reset_index().melt(
                            id_vars=primary_cat,
                            var_name=secondary_cat,
                            value_name=y_title
                        )
                        
                        fig = px.bar(
                            plot_df,
                            y=primary_cat,
                            x=y_title,
                            color=secondary_cat,
                            title=title,
                            barmode='group' if bar_mode == 'Grouped' else 'stack',
                            orientation='h'
                        )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Grouped Data")
                    st.dataframe(grouped_data, use_container_width=True)
                    
                    # Calculate chi-square test for independence
                    from scipy.stats import chi2_contingency
                    
                    chi2, p, dof, expected = chi2_contingency(grouped_data)
                    
                    st.subheader("Chi-Square Test for Independence")
                    st.write(f"Chi-square value: {chi2:.2f}")
                    st.write(f"p-value: {p:.4f}")
                    st.write(f"Degrees of freedom: {dof}")
                    
                    if p < 0.05:
                        st.info("The two categorical variables appear to be dependent (significant association).")
                    else:
                        st.info("No significant association detected between the variables.")
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating grouped bar chart: {str(e)}")
        
        elif plot_type == "Heatmap":
            # Heatmap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select rows (first categorical column)
                row_cat = st.selectbox("Row variable:", cat_cols)
                
                # Select columns (second categorical column)
                col_cat = st.selectbox("Column variable:", 
                                      [c for c in cat_cols if c != row_cat])
                
                # Title
                title = st.text_input("Chart title:", f"Heatmap of {row_cat} vs {col_cat}")
            
            with col2:
                # Value type
                value_type = st.radio("Values to show:", ["Counts", "Proportions"], horizontal=True)
                
                # Normalization (for proportions)
                norm_by = None
                if value_type == "Proportions":
                    norm_by = st.radio("Normalize by:", ["Row", "Column", "All"], horizontal=True)
                    norm_by = norm_by.lower()
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"])
            
            # Generate heatmap
            if st.button("Generate Heatmap", use_container_width=True):
                try:
                    # Create cross-tabulation
                    if value_type == "Counts":
                        heatmap_data = pd.crosstab(self.df[row_cat], self.df[col_cat])
                        z_title = "Count"
                    else:  # Proportions
                        heatmap_data = pd.crosstab(
                            self.df[row_cat], 
                            self.df[col_cat], 
                            normalize=norm_by
                        )
                        z_title = "Proportion"
                    
                    # Create figure
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x=col_cat, y=row_cat, color=z_title),
                        title=title,
                        color_continuous_scale=color_scheme,
                        text_auto=True
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Cross-tabulation")
                    st.dataframe(heatmap_data, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
        
        elif plot_type == "Treemap":
            # Treemap settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select path columns (hierarchical categories)
                path_cols = st.multiselect(
                    "Hierarchy levels (order matters):",
                    cat_cols,
                    default=[cat_cols[0]] if cat_cols else []
                )
                
                # Title
                title = st.text_input("Chart title:", "Treemap")
            
            with col2:
                # Value column (numeric, optional)
                if num_cols:
                    value_col = st.selectbox("Value column (optional):", ["None"] + num_cols)
                    value_col = None if value_col == "None" else value_col
                else:
                    value_col = None
                    st.info("No numeric columns available for values. Will use counts.")
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="treemap_colors")
            
            # Generate treemap
            if st.button("Generate Treemap", use_container_width=True):
                if not path_cols:
                    st.warning("Please select at least one column for the hierarchy.")
                else:
                    try:
                        # Create treemap
                        fig = px.treemap(
                            self.df,
                            path=path_cols,
                            values=value_col,
                            title=title,
                            color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme)
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating treemap: {str(e)}")
        
        elif plot_type == "Sunburst":
            # Sunburst settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select path columns (hierarchical categories)
                path_cols = st.multiselect(
                    "Hierarchy levels (order matters):",
                    cat_cols,
                    default=[cat_cols[0]] if cat_cols else []
                )
                
                # Title
                title = st.text_input("Chart title:", "Sunburst Chart")
            
            with col2:
                # Value column (numeric, optional)
                if num_cols:
                    value_col = st.selectbox("Value column (optional):", ["None"] + num_cols)
                    value_col = None if value_col == "None" else value_col
                else:
                    value_col = None
                    st.info("No numeric columns available for values. Will use counts.")
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="sunburst_colors")
            
            # Generate sunburst
            if st.button("Generate Sunburst", use_container_width=True):
                if not path_cols:
                    st.warning("Please select at least one column for the hierarchy.")
                else:
                    try:
                        # Create sunburst
                        fig = px.sunburst(
                            self.df,
                            path=path_cols,
                            values=value_col,
                            title=title,
                            color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme)
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating sunburst chart: {str(e)}")
        
        elif plot_type == "Parallel Categories":
            # Parallel categories settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select categorical columns
                parallel_cats = st.multiselect(
                    "Categorical columns to include:",
                    cat_cols,
                    default=cat_cols[:min(4, len(cat_cols))]
                )
                
                # Title
                title = st.text_input("Chart title:", "Parallel Categories Diagram")
            
            with col2:
                # Color column
                color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols)
                color_col = None if color_col == "None" else color_col
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="parcats_colors")
            
            # Generate parallel categories
            if st.button("Generate Parallel Categories", use_container_width=True):
                if len(parallel_cats) < 2:
                    st.warning("Please select at least two categorical columns.")
                else:
                    try:
                        # Create parallel categories
                        fig = px.parallel_categories(
                            self.df,
                            dimensions=parallel_cats,
                            color=color_col,
                            title=title,
                            color_continuous_scale=color_scheme
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating parallel categories diagram: {str(e)}")
    
    def _render_time_series(self):
        """Render time series visualizations"""
        st.subheader("Time Series Analysis")
        
        # Get columns by type
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Look for potential date columns if no datetime columns
        potential_date_cols = []
        if not datetime_cols:
            for col in self.df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day']):
                    # Check first value to see if it looks like a date
                    try:
                        sample = self.df[col].dropna().iloc[0]
                        pd.to_datetime(sample)
                        potential_date_cols.append(col)
                    except:
                        pass
        
        # Combined date columns
        all_date_cols = datetime_cols + potential_date_cols
        
        if not all_date_cols:
            st.warning("No datetime columns detected. Please convert a column to datetime format first.")
            return
        
        if not num_cols:
            st.warning("No numeric columns available for time series analysis.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot type:",
            ["Line Chart", "Area Chart", "Heatmap Calendar", "Seasonal Decomposition", "Time Series Components", "Resampling & Rolling Windows"]
        )
        
        if plot_type == "Line Chart":
            # Line chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value columns
                value_cols = st.multiselect("Value column(s):", num_cols, default=[num_cols[0]])
                
                # Title
                title = st.text_input("Chart title:", "Time Series Analysis")
            
            with col2:
                # Select grouping (categorical, optional)
                group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols)
                group_by = None if group_by == "None" else group_by
                
                # Line style
                line_shape = st.selectbox("Line shape:", ["linear", "spline", "hv", "vh", "hvh", "vhv"])
                
                # Include markers
                show_markers = st.checkbox("Show markers", value=False)
            
            # Generate line chart
            if st.button("Generate Time Series Chart", use_container_width=True):
                if not value_cols:
                    st.warning("Please select at least one value column.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Sort by date
                        df_copy = df_copy.sort_values(by=date_col)
                        
                        # Create figure
                        if len(value_cols) == 1 and group_by:
                            # One value column with grouping
                            fig = px.line(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                color=group_by,
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        elif len(value_cols) > 1 and not group_by:
                            # Multiple value columns, no grouping
                            # Need to melt the dataframe
                            df_melted = df_copy.melt(
                                id_vars=[date_col],
                                value_vars=value_cols,
                                var_name='Variable',
                                value_name='Value'
                            )
                            
                            fig = px.line(
                                df_melted, 
                                x=date_col, 
                                y='Value', 
                                color='Variable',
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        else:
                            # Basic case: one value column, no grouping
                            # or multiple value columns with grouping (show warning)
                            if len(value_cols) > 1 and group_by:
                                st.warning("Using only the first value column when grouping is applied.")
                            
                            fig = px.line(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                title=title,
                                line_shape=line_shape,
                                markers=show_markers
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title=date_col,
                            yaxis_title=value_cols[0] if len(value_cols) == 1 else "Value",
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show basic time series stats
                        st.subheader("Time Series Statistics")
                        
                        # For each value column
                        for col in value_cols:
                            st.write(f"**{col}**")
                            stats_cols = st.columns(4)
                            with stats_cols[0]:
                                st.metric("Mean", f"{df_copy[col].mean():.2f}")
                            with stats_cols[1]:
                                st.metric("Min", f"{df_copy[col].min():.2f}")
                            with stats_cols[2]:
                                st.metric("Max", f"{df_copy[col].max():.2f}")
                            with stats_cols[3]:
                                # Calculate trend direction
                                first_valid = df_copy[col].first_valid_index()
                                last_valid = df_copy[col].last_valid_index()
                                if first_valid is not None and last_valid is not None:
                                    start_val = df_copy.loc[first_valid, col]
                                    end_val = df_copy.loc[last_valid, col]
                                    change = end_val - start_val
                                    pct_change = (change / start_val) * 100 if start_val != 0 else float('inf')
                                    
                                    if change > 0:
                                        trend = "↑ Increasing"
                                        delta_color = "normal"
                                    elif change < 0:
                                        trend = "↓ Decreasing"
                                        delta_color = "inverse"
                                    else:
                                        trend = "→ Stable"
                                        delta_color = "off"
                                    
                                    st.metric("Trend", trend, f"{pct_change:.1f}%" if abs(pct_change) != float('inf') else "N/A", delta_color=delta_color)
                                else:
                                    st.metric("Trend", "N/A")
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating time series chart: {str(e)}")
        
        elif plot_type == "Area Chart":
            # Area chart settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value columns
                value_cols = st.multiselect("Value column(s):", num_cols, default=[num_cols[0]])
                
                # Title
                title = st.text_input("Chart title:", "Time Series Area Chart")
            
            with col2:
                # Select grouping (categorical, optional)
                group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols)
                group_by = None if group_by == "None" else group_by
                
                # Stacking mode
                stack_mode = st.radio("Stacking mode:", ["Overlay", "Stack", "Stack 100%"], horizontal=True)
                
                # Line shape
                line_shape = st.selectbox("Line shape:", ["spline", "linear", "hv", "vh"], key="area_shape")
            
            # Generate area chart
            if st.button("Generate Area Chart", use_container_width=True):
                if not value_cols:
                    st.warning("Please select at least one value column.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Sort by date
                        df_copy = df_copy.sort_values(by=date_col)
                        
                        # Determine stacking mode
                        if stack_mode == "Overlay":
                            groupnorm = None
                        elif stack_mode == "Stack":
                            groupnorm = ''
                        else:  # "Stack 100%"
                            groupnorm = 'percent'
                        
                        # Create figure
                        if len(value_cols) == 1 and group_by:
                            # One value column with grouping
                            fig = px.area(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                color=group_by,
                                title=title,
                                line_shape=line_shape,
                                groupnorm=groupnorm
                            )
                        elif len(value_cols) > 1 and not group_by:
                            # Multiple value columns, no grouping
                            # Need to melt the dataframe
                            df_melted = df_copy.melt(
                                id_vars=[date_col],
                                value_vars=value_cols,
                                var_name='Variable',
                                value_name='Value'
                            )
                            
                            fig = px.area(
                                df_melted, 
                                x=date_col, 
                                y='Value', 
                                color='Variable',
                                title=title,
                                line_shape=line_shape,
                                groupnorm=groupnorm
                            )
                        else:
                            # Basic case: one value column, no grouping
                            # or multiple value columns with grouping (show warning)
                            if len(value_cols) > 1 and group_by:
                                st.warning("Using only the first value column when grouping is applied.")
                            
                            fig = px.area(
                                df_copy, 
                                x=date_col, 
                                y=value_cols[0], 
                                title=title,
                                line_shape=line_shape
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title=date_col,
                            yaxis_title="Value",
                            height=600
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating area chart: {str(e)}")
        
        elif plot_type == "Heatmap Calendar":
            # Heatmap calendar settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Calendar Heatmap")
            
            with col2:
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Sum", "Count", "Min", "Max"])
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"],
                                          key="calendar_colors")
            
            # Generate calendar heatmap
            if st.button("Generate Calendar Heatmap", use_container_width=True):
                try:
                    import plotly.graph_objects as go
                    import calendar
                    
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Extract date components
                    df_copy['year'] = df_copy[date_col].dt.year
                    df_copy['month'] = df_copy[date_col].dt.month
                    df_copy['day'] = df_copy[date_col].dt.day
                    df_copy['weekday'] = df_copy[date_col].dt.weekday
                    
                    # Determine aggregation function
                    agg_func = agg_method.lower()
                    
                    # Group by date and aggregate
                    df_agg = df_copy.groupby([date_col])[value_col].agg(agg_func).reset_index()
                    
                    # Extract date components for aggregated data
                    df_agg['year'] = df_agg[date_col].dt.year
                    df_agg['month'] = df_agg[date_col].dt.month
                    df_agg['day'] = df_agg[date_col].dt.day
                    df_agg['weekday'] = df_agg[date_col].dt.weekday
                    
                    # Get unique years and months
                    years = sorted(df_agg['year'].unique())
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Create a calendar heatmap for each year
                    for year in years:
                        year_data = df_agg[df_agg['year'] == year]
                        
                        # For each month in the year
                        for month in range(1, 13):
                            month_data = year_data[year_data['month'] == month]
                            
                            # Create data for this month
                            month_days = calendar.monthrange(year, month)[1]
                            
                            # Create the calendar grid
                            month_name = calendar.month_name[month]
                            
                            # Add a trace for this month
                            hovertext = []
                            values = []
                            
                            for day in range(1, month_days + 1):
                                day_data = month_data[month_data['day'] == day]
                                if not day_data.empty:
                                    value = day_data[value_col].iloc[0]
                                    hovertext.append(f"{month_name} {day}, {year}<br>{value_col}: {value:.2f}")
                                    values.append(value)
                                else:
                                    hovertext.append(f"{month_name} {day}, {year}<br>No data")
                                    values.append(None)
                            
                            # Skip months with no data
                            if all(v is None for v in values):
                                continue
                            
                            # Create a trace for this month
                            fig.add_trace(go.Heatmap(
                                z=[values],
                                x=[f"{month_name} {day}" for day in range(1, month_days + 1)],
                                y=[f"{year} - {month_name}"],
                                hoverinfo="text",
                                text=[hovertext],
                                colorscale=color_scheme,
                                showscale=True,
                                name=f"{month_name} {year}"
                            ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        height=max(600, 200 * len(years)),
                        xaxis_title="Day",
                        yaxis_title="Month-Year",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(0, 31, 5)),
                            ticktext=[str(i) for i in range(1, 32, 5)]
                        )
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating calendar heatmap: {str(e)}")
        
        elif plot_type == "Seasonal Decomposition":
            # Seasonal decomposition settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Seasonal Decomposition")
            
            with col2:
                # Decomposition method
                decomp_method = st.radio("Decomposition method:", ["Additive", "Multiplicative"], horizontal=True)
                
                # Period selection
                freq_options = ["Day", "Week", "Month", "Quarter", "Year", "Custom"]
                freq_select = st.selectbox("Seasonality period:", freq_options)
                
                if freq_select == "Custom":
                    period = st.number_input("Number of periods:", min_value=2, value=12)
                else:
                    # Map selection to period
                    freq_map = {
                        "Day": 1,
                        "Week": 7,
                        "Month": 30,
                        "Quarter": 90,
                        "Year": 365
                    }
                    period = freq_map.get(freq_select, 12)
            
            # Generate seasonal decomposition
            if st.button("Generate Seasonal Decomposition", use_container_width=True):
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Sort by date
                    df_copy = df_copy.sort_values(by=date_col)
                    
                    # Set date as index
                    df_copy = df_copy.set_index(date_col)
                    
                    # Decompose the time series
                    decomposition = seasonal_decompose(
                        df_copy[value_col].dropna(), 
                        model=decomp_method.lower(),
                        period=period
                    )
                    
                    # Create plots
                    fig = go.Figure()
                    
                    # Add original data
                    fig.add_trace(go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        mode='lines',
                        name='Original'
                    ))
                    
                    # Add trend
                    fig.add_trace(go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red')
                    ))
                    
                    # Add seasonal
                    fig.add_trace(go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        mode='lines',
                        name='Seasonal',
                        line=dict(color='green')
                    ))
                    
                    # Add residual
                    fig.add_trace(go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        mode='lines',
                        name='Residual',
                        line=dict(color='purple')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=date_col,
                        yaxis_title=value_col,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create individual components charts
                    st.subheader("Individual Components")
                    
                    component_tabs = st.tabs(["Trend", "Seasonal", "Residual"])
                    
                    with component_tabs[0]:
                        trend_fig = px.line(
                            x=decomposition.trend.index, 
                            y=decomposition.trend,
                            title="Trend Component",
                            labels={"x": date_col, "y": f"Trend of {value_col}"}
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)
                    
                    with component_tabs[1]:
                        seasonal_fig = px.line(
                            x=decomposition.seasonal.index, 
                            y=decomposition.seasonal,
                            title="Seasonal Component",
                            labels={"x": date_col, "y": f"Seasonality of {value_col}"}
                        )
                        st.plotly_chart(seasonal_fig, use_container_width=True)
                    
                    with component_tabs[2]:
                        resid_fig = px.line(
                            x=decomposition.resid.index, 
                            y=decomposition.resid,
                            title="Residual Component",
                            labels={"x": date_col, "y": f"Residual of {value_col}"}
                        )
                        st.plotly_chart(resid_fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating seasonal decomposition: {str(e)}")
        
        elif plot_type == "Time Series Components":
            # Time series components settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Time Series Components")
            
            with col2:
                # Components to show
                components = st.multiselect(
                    "Components to show:",
                    ["Day of Week", "Month of Year", "Hour of Day", "Quarter", "Year", "Week of Year"],
                    default=["Day of Week", "Month of Year"]
                )
                
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Median", "Sum", "Count", "Min", "Max"], key="comp_agg")
            
            # Generate time series components
            if st.button("Generate Time Series Components", use_container_width=True):
                if not components:
                    st.warning("Please select at least one component.")
                else:
                    try:
                        # Ensure date column is datetime
                        df_copy = self.df.copy()
                        if date_col in potential_date_cols:
                            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        # Create tabs for each component
                        component_tabs = st.tabs(components)
                        
                        for i, component in enumerate(components):
                            with component_tabs[i]:
                                # Extract the relevant component
                                if component == "Day of Week":
                                    df_copy['component'] = df_copy[date_col].dt.day_name()
                                    x_title = "Day of Week"
                                    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                elif component == "Month of Year":
                                    df_copy['component'] = df_copy[date_col].dt.month_name()
                                    x_title = "Month"
                                    order = ["January", "February", "March", "April", "May", "June", 
                                            "July", "August", "September", "October", "November", "December"]
                                elif component == "Hour of Day":
                                    df_copy['component'] = df_copy[date_col].dt.hour
                                    x_title = "Hour"
                                    order = list(range(24))
                                elif component == "Quarter":
                                    df_copy['component'] = "Q" + df_copy[date_col].dt.quarter.astype(str)
                                    x_title = "Quarter"
                                    order = ["Q1", "Q2", "Q3", "Q4"]
                                elif component == "Year":
                                    df_copy['component'] = df_copy[date_col].dt.year
                                    x_title = "Year"
                                    order = sorted(df_copy['component'].unique())
                                elif component == "Week of Year":
                                    df_copy['component'] = df_copy[date_col].dt.isocalendar().week
                                    x_title = "Week of Year"
                                    order = list(range(1, 54))
                                
                                # Aggregate by component
                                agg_func = agg_method.lower()
                                component_agg = df_copy.groupby('component')[value_col].agg(agg_func).reset_index()
                                
                                # Create ordered categorical if applicable
                                if order:
                                    # Filter order to only include values that exist in the data
                                    order = [o for o in order if o in component_agg['component'].values]
                                    component_agg['component'] = pd.Categorical(
                                        component_agg['component'], 
                                        categories=order,
                                        ordered=True
                                    )
                                    # Sort by the ordered categorical
                                    component_agg = component_agg.sort_values('component')
                                
                                # Create figure
                                component_fig = px.bar(
                                    component_agg,
                                    x='component',
                                    y=value_col,
                                    title=f"{component} Analysis of {value_col} ({agg_method})",
                                    labels={"component": x_title, "value": value_col},
                                    color='component'
                                )
                                
                                # Add average line
                                avg_value = df_copy[value_col].mean()
                                component_fig.add_hline(
                                    y=avg_value,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Overall {agg_method}: {avg_value:.2f}"
                                )
                                
                                # Update layout
                                component_fig.update_layout(
                                    showlegend=False,
                                    height=500
                                )
                                
                                # Show the chart
                                st.plotly_chart(component_fig, use_container_width=True)
                                
                                # Show statistics
                                st.subheader(f"Statistics by {component}")
                                st.dataframe(component_agg, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating time series components: {str(e)}")
        
        elif plot_type == "Resampling & Rolling Windows":
            # Resampling settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Select date column
                date_col = st.selectbox("Date column:", all_date_cols)
                
                # Convert to datetime if not already
                if date_col in potential_date_cols:
                    st.info(f"Column '{date_col}' will be converted to datetime. Make sure it contains valid dates.")
                
                # Select value column
                value_col = st.selectbox("Value column:", num_cols)
                
                # Title
                title = st.text_input("Chart title:", "Resampled Time Series")
            
            with col2:
                # Analysis type
                analysis_type = st.radio("Analysis type:", ["Resampling", "Rolling Window"], horizontal=True)
                
                # Settings based on analysis type
                if analysis_type == "Resampling":
                    # Resampling period
                    resample_options = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y",
                        "Hourly": "H"
                    }
                    resample_period = st.selectbox("Resample to:", list(resample_options.keys()))
                    resample_rule = resample_options[resample_period]
                    
                    window_size = None
                else:  # Rolling Window
                    # Window size
                    window_size = st.slider("Window size:", 2, 100, 7)
                    resample_rule = None
                
                # Aggregation method
                agg_method = st.selectbox("Aggregation method:", ["Mean", "Median", "Sum", "Min", "Max", "Count", "Std"], key="resample_agg")
            
            # Generate resampled/rolling window chart
            if st.button("Generate Chart", use_container_width=True):
                try:
                    # Ensure date column is datetime
                    df_copy = self.df.copy()
                    if date_col in potential_date_cols:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    
                    # Sort by date
                    df_copy = df_copy.sort_values(by=date_col)
                    
                    # Set date as index
                    df_copy = df_copy.set_index(date_col)
                    
                    # Determine aggregation function
                    agg_func = agg_method.lower()
                    
                    # Apply resampling or rolling window
                    if analysis_type == "Resampling":
                        # Resample data
                        resampled = getattr(df_copy[value_col].resample(resample_rule), agg_func)()
                        result_df = resampled.reset_index()
                        result_df.columns = [date_col, f"{value_col}_{agg_func}"]
                        
                        # Create figure
                        fig = px.line(
                            result_df, 
                            x=date_col, 
                            y=f"{value_col}_{agg_func}", 
                            title=f"{title} ({resample_period} {agg_method})",
                            markers=True
                        )
                    else:  # Rolling Window
                        # Apply rolling window
                        rolling = getattr(df_copy[value_col].rolling(window=window_size), agg_func)()
                        result_df = rolling.reset_index()
                        result_df.columns = [date_col, f"{value_col}_{agg_func}"]
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Add original data
                        fig.add_trace(go.Scatter(
                            x=df_copy.index,
                            y=df_copy[value_col],
                            mode='lines',
                            name='Original',
                            line=dict(color='blue', width=1, dash='dash')
                        ))
                        
                        # Add rolling window result
                        fig.add_trace(go.Scatter(
                            x=result_df[date_col],
                            y=result_df[f"{value_col}_{agg_func}"],
                            mode='lines',
                            name=f'{window_size}-period Rolling {agg_method}',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{title} ({window_size}-period Rolling {agg_method})",
                        )
                    
                    # Update common layout settings
                    fig.update_layout(
                        xaxis_title=date_col,
                        yaxis_title=value_col,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data
                    with st.expander("Show Data"):
                        st.dataframe(result_df, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
    
    def _render_multivariate_analysis(self):
        """Render multivariate analysis visualizations"""
        st.subheader("Multivariate Analysis")
        
        # Get columns by type
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for multivariate analysis.")
            return
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot type:",
            ["Scatter Matrix", "Parallel Coordinates", "3D Scatter Plot", "Bubble Chart", "Heatmap", "Contour Plot"]
        )
        
        if plot_type == "Scatter Matrix":
            # Scatter matrix settings
            st.write("Create a matrix of scatter plots to visualize relationships between multiple variables.")
            
            # Select variables
            vars_for_matrix = st.multiselect(
                "Variables to include:",
                num_cols,
                default=num_cols[:min(4, len(num_cols))]
            )
            
            # Color by column
            color_by = st.selectbox("Color by (optional):", ["None"] + cat_cols)
            color_by = None if color_by == "None" else color_by
            
            # Generate scatter matrix
            if st.button("Generate Scatter Matrix", use_container_width=True):
                if len(vars_for_matrix) < 2:
                    st.warning("Please select at least two variables.")
                else:
                    try:
                        # Create scatter matrix
                        fig = px.scatter_matrix(
                            self.df,
                            dimensions=vars_for_matrix,
                            color=color_by,
                            title="Scatter Matrix",
                            opacity=0.7
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=max(600, 200 * len(vars_for_matrix))
                        )
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation matrix
                        st.subheader("Correlation Matrix")
                        corr_matrix = self.df[vars_for_matrix].corr().round(2)
                        st.dataframe(corr_matrix, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating scatter matrix: {str(e)}")
        
        elif plot_type == "Parallel Coordinates":
            # Parallel coordinates settings
            st.write("Visualize high-dimensional data by plotting each variable on a separate vertical axis.")
            
            # Select variables
            vars_for_parallel = st.multiselect(
                "Variables to include:",
                num_cols,
                default=num_cols[:min(5, len(num_cols))]
            )
            
            # Color by column
            color_by = st.selectbox("Color by:", ["None"] + num_cols + cat_cols)
            color_by = None if color_by == "None" else color_by
            
            # Color scale
            color_scale = st.selectbox("Color scale:", 
                                     ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds"])
            
            # Generate parallel coordinates
            if st.button("Generate Parallel Coordinates", use_container_width=True):
                if len(vars_for_parallel) < 2:
                    st.warning("Please select at least two variables.")
                else:
                    try:
                        # Create parallel coordinates
                        fig = px.parallel_coordinates(
                            self.df,
                            dimensions=vars_for_parallel,
                            color=color_by,
                            color_continuous_scale=color_scale.lower(),
                            title="Parallel Coordinates Plot"
                        )
                        
                        # Update layout
                        fig.update_layout(height=600)
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Offer download options
                        self._offer_download_options(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating parallel coordinates: {str(e)}")
        
        elif plot_type == "3D Scatter Plot":
            # 3D scatter plot settings
            st.write("Create a 3D scatter plot to visualize relationships between three variables.")
            
            # Select variables
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X-axis:", num_cols, index=0)
                y_var = st.selectbox("Y-axis:", [c for c in num_cols if c != x_var], index=0)
                z_var = st.selectbox("Z-axis:", [c for c in num_cols if c not in [x_var, y_var]], index=0)
            
            with col2:
                # Color by column
                color_by = st.selectbox("Color by:", ["None"] + num_cols + cat_cols)
                color_by = None if color_by == "None" else color_by
                
                # Size by column
                size_by = st.selectbox("Size by:", ["None"] + num_cols)
                size_by = None if size_by == "None" else size_by
                
                # Opacity
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.7, 0.1)
            
            # Generate 3D scatter plot
            if st.button("Generate 3D Scatter Plot", use_container_width=True):
                try:
                    # Create 3D scatter plot
                    fig = px.scatter_3d(
                        self.df,
                        x=x_var,
                        y=y_var,
                        z=z_var,
                        color=color_by,
                        size=size_by,
                        opacity=opacity,
                        title="3D Scatter Plot"
                    )
                    
                    # Update layout
                    fig.update_layout(height=700)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating 3D scatter plot: {str(e)}")
        
        elif plot_type == "Bubble Chart":
            # Bubble chart settings
            st.write("Create a bubble chart to visualize relationships between three or four variables.")
            
            # Select variables
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X-axis:", num_cols, index=0)
                y_var = st.selectbox("Y-axis:", [c for c in num_cols if c != x_var], index=0)
                size_var = st.selectbox("Bubble size:", [c for c in num_cols if c not in [x_var, y_var]], index=0)
            
            with col2:
                # Color by column
                color_by = st.selectbox("Color by:", ["None"] + num_cols + cat_cols)
                color_by = None if color_by == "None" else color_by
                
                # Hover name
                hover_name = st.selectbox("Hover name:", ["None"] + cat_cols)
                hover_name = None if hover_name == "None" else hover_name
                
                # Opacity
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.7, 0.1, key="bubble_opacity")
            
            # Generate bubble chart
            if st.button("Generate Bubble Chart", use_container_width=True):
                try:
                    # Create bubble chart
                    fig = px.scatter(
                        self.df,
                        x=x_var,
                        y=y_var,
                        size=size_var,
                        color=color_by,
                        hover_name=hover_name,
                        opacity=opacity,
                        title="Bubble Chart",
                        size_max=50
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating bubble chart: {str(e)}")
        
        elif plot_type == "Heatmap":
            # Heatmap settings
            st.write("Create a heatmap to visualize patterns in a 2D grid of data.")
            
            # Select variables for x, y, and value
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X-axis:", num_cols + cat_cols, index=0)
                y_var = st.selectbox("Y-axis:", [c for c in num_cols + cat_cols if c != x_var], index=0)
                
                # Check if both are categorical
                if x_var in cat_cols and y_var in cat_cols:
                    # Need a value to aggregate
                    if num_cols:
                        value_var = st.selectbox("Aggregate value:", num_cols)
                    else:
                        st.warning("Need at least one numeric column for value.")
                        return
                else:
                    value_var = None
            
            with col2:
                # Aggregation function if needed
                if value_var:
                    agg_func = st.selectbox("Aggregation function:", ["Mean", "Sum", "Count", "Min", "Max"])
                    agg_func = agg_func.lower()
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                         ["Viridis", "Plasma", "Inferno", "Magma", "Blues", "Reds", "Greens", "YlOrRd"],
                                         key="heatmap_colors")
                
                # Show values
                show_values = st.checkbox("Show values", value=True)
            
            # Generate heatmap
            if st.button("Generate Heatmap", use_container_width=True):
                try:
                    # Create heatmap data
                    if value_var:
                        # Categorical x categorical case
                        pivot_data = self.df.pivot_table(
                            index=y_var,
                            columns=x_var,
                            values=value_var,
                            aggfunc=agg_func,
                            fill_value=0
                        )
                        
                        title = f"Heatmap of {value_var} ({agg_func}) by {x_var} and {y_var}"
                    else:
                        # Handle numeric x numeric or numeric x categorical
                        # Create bins for numeric columns
                        df_binned = self.df.copy()
                        
                        if x_var in num_cols:
                            # Create bins for x
                            x_min, x_max = df_binned[x_var].min(), df_binned[x_var].max()
                            x_bins = 10
                            x_bin_width = (x_max - x_min) / x_bins
                            df_binned[f"{x_var}_bin"] = pd.cut(
                                df_binned[x_var], 
                                bins=x_bins,
                                labels=[f"{x_min + i*x_bin_width:.2f}-{x_min + (i+1)*x_bin_width:.2f}" 
                                        for i in range(x_bins)]
                            )
                            x_col = f"{x_var}_bin"
                        else:
                            x_col = x_var
                        
                        if y_var in num_cols:
                            # Create bins for y
                            y_min, y_max = df_binned[y_var].min(), df_binned[y_var].max()
                            y_bins = 10
                            y_bin_width = (y_max - y_min) / y_bins
                            df_binned[f"{y_var}_bin"] = pd.cut(
                                df_binned[y_var], 
                                bins=y_bins,
                                labels=[f"{y_min + i*y_bin_width:.2f}-{y_min + (i+1)*y_bin_width:.2f}" 
                                        for i in range(y_bins)]
                            )
                            y_col = f"{y_var}_bin"
                        else:
                            y_col = y_var
                        
                        # Create pivot table
                        pivot_data = pd.crosstab(df_binned[y_col], df_binned[x_col])
                        
                        title = f"Heatmap of Frequency by {x_var} and {y_var}"
                    
                    # Create figure
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x=x_var, y=y_var, color="Value"),
                        color_continuous_scale=color_scheme.lower(),
                        title=title,
                        text_auto=show_values
                    )
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data
                    st.subheader("Heatmap Data")
                    st.dataframe(pivot_data, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
        
        elif plot_type == "Contour Plot":
            # Contour plot settings
            st.write("Create a contour plot to visualize the 3D relationship between variables.")
            
            # Select variables
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X-axis:", num_cols, index=0)
                y_var = st.selectbox("Y-axis:", [c for c in num_cols if c != x_var], index=0)
                z_var = st.selectbox("Z-axis (contour values):", [c for c in num_cols if c not in [x_var, y_var]], index=0)
            
            with col2:
                # Contour type
                contour_type = st.radio("Contour type:", ["Filled", "Lines", "Both"], horizontal=True)
                
                # Color scheme
                color_scheme = st.selectbox("Color scheme:", 
                                          ["Viridis", "Plasma", "Inferno", "Magma", "Blues", "Reds", "Greens", "YlOrRd"],
                                          key="contour_colors")
                
                # Number of contours
                n_contours = st.slider("Number of contours:", 5, 50, 20)
            
            # Generate contour plot
            if st.button("Generate Contour Plot", use_container_width=True):
                try:
                    # Determine contour type
                    if contour_type == "Filled":
                        contour_mode = "fill"
                    elif contour_type == "Lines":
                        contour_mode = "lines"
                    else:  # Both
                        contour_mode = "fill+lines"
                    
                    # Create contour plot
                    fig = px.density_contour(
                        self.df,
                        x=x_var,
                        y=y_var,
                        z=z_var,
                        title=f"Contour Plot of {z_var} by {x_var} and {y_var}",
                        color_scale=color_scheme.lower(),
                        nbinsx=n_contours,
                        nbinsy=n_contours
                    )
                    
                    # Update traces for contour type
                    fig.update_traces(contours_coloring=contour_mode)
                    
                    # Update layout
                    fig.update_layout(height=600)
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Offer download options
                    self._offer_download_options(fig)
                    
                except Exception as e:
                    st.error(f"Error generating contour plot: {str(e)}")
    
    def _calculate_kde(self, data, bw_method=None):
        """Calculate kernel density estimate for a given dataset"""
        from scipy.stats import gaussian_kde
        import numpy as np
        
        # Check if we have enough data
        if len(data) < 2:
            # Return empty data if not enough points
            return [], []
        
        # Fit KDE
        kde = gaussian_kde(data, bw_method=bw_method)
        
        # Generate points to evaluate KDE
        x_min, x_max = data.min(), data.max()
        margin = 0.1 * (x_max - x_min)
        x_grid = np.linspace(x_min - margin, x_max + margin, 1000)
        
        # Evaluate KDE on grid
        y_grid = kde(x_grid)
        
        return x_grid, y_grid
    
    def _offer_download_options(self, fig):
        """Offer options to download the visualization"""
        # Create columns for download options
        download_cols = st.columns(4)
        
        with download_cols[0]:
            # Download as PNG
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="Download as PNG",
                data=img_bytes,
                file_name="visualization.png",
                mime="image/png",
                use_container_width=True
            )
        
        with download_cols[1]:
            # Download as SVG
            svg_bytes = fig.to_image(format="svg", width=1200, height=800)
            st.download_button(
                label="Download as SVG",
                data=svg_bytes,
                file_name="visualization.svg",
                mime="image/svg+xml",
                use_container_width=True
            )
        
        with download_cols[2]:
            # Download as HTML
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode()
            st.download_button(
                label="Download as HTML",
                data=html_bytes,
                file_name="visualization.html",
                mime="text/html",
                use_container_width=True
            )
        
        with download_cols[3]:
            # Download as JSON (for plotly)
            json_bytes = fig.to_json().encode()
            st.download_button(
                label="Download as JSON",
                data=json_bytes,
                file_name="visualization.json",
                mime="application/json",
                use_container_width=True
            )

class DataStatistics:
    """Class for calculating and displaying data statistics"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
    
    def render_interface(self):
        """Render the data statistics interface"""
        st.header("Data Statistics")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to view statistics.")
            return
        
        # Create tabs for different statistical analyses
        stats_tabs = st.tabs([
            "Basic Statistics",
            "Group Statistics",
            "Correlation Analysis",
            "Distribution Analysis",
            "Outlier Detection"
        ])
        
        # Basic Statistics Tab
        with stats_tabs[0]:
            self._render_basic_statistics()
        
        # Group Statistics Tab
        with stats_tabs[1]:
            self._render_group_statistics()
        
        # Correlation Analysis Tab
        with stats_tabs[2]:
            self._render_correlation_analysis()
        
        # Distribution Analysis Tab
        with stats_tabs[3]:
            self._render_distribution_analysis()
        
        # Outlier Detection Tab
        with stats_tabs[4]:
            self._render_outlier_detection()
    
    def _render_basic_statistics(self):
        """Render basic statistics"""
        st.subheader("Basic Statistics")
        
        # Get columns by type
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Overview")
            
            # Show basic dataset info
            st.write(f"**Number of rows:** {len(self.df)}")
            st.write(f"**Number of columns:** {len(self.df.columns)}")
            st.write(f"**Number of numeric columns:** {len(num_cols)}")
            st.write(f"**Number of categorical columns:** {len(cat_cols)}")
            
            # Missing values
            missing_vals = self.df.isna().sum()
            total_missing = missing_vals.sum()
            st.write(f"**Total missing values:** {total_missing}")
            st.write(f"**Columns with missing values:** {sum(missing_vals > 0)}")
            
            # Memory usage
            mem_usage = self.df.memory_usage(deep=True).sum()
            if mem_usage < 1024:
                mem_str = f"{mem_usage} bytes"
            elif mem_usage < 1024 ** 2:
                mem_str = f"{mem_usage / 1024:.2f} KB"
            elif mem_usage < 1024 ** 3:
                mem_str = f"{mem_usage / (1024 ** 2):.2f} MB"
            else:
                mem_str = f"{mem_usage / (1024 ** 3):.2f} GB"
            
            st.write(f"**Memory usage:** {mem_str}")
        
        with col2:
            st.markdown("### Column Types")
            
            # Create a dataframe with column type information
            dtype_counts = self.df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            
            # Show the type counts
            st.dataframe(dtype_counts, use_container_width=True)
            
            # Create a pie chart of column types
            fig = px.pie(
                dtype_counts, 
                values='Count', 
                names='Data Type', 
                title='Column Data Types'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create tabs for numeric and categorical statistics
        if num_cols or cat_cols:
            num_cat_tabs = st.tabs(["Numeric Columns", "Categorical Columns"])
            
            # Numeric Columns Tab
            with num_cat_tabs[0]:
                if num_cols:
                    # Calculate statistics for numeric columns
                    num_stats = self.df[num_cols].describe().T
                    
                    # Add more statistics
                    num_stats['missing'] = self.df[num_cols].isna().sum()
                    num_stats['missing_pct'] = (self.df[num_cols].isna().sum() / len(self.df) * 100).round(2)
                    num_stats['unique'] = self.df[num_cols].nunique()
                    
                    # Round to 3 decimal places
                    num_stats = num_stats.round(3)
                    
                    # Show the statistics
                    st.markdown("### Numeric Columns Statistics")
                    st.dataframe(num_stats, use_container_width=True)
                    
                    # Add download button for CSV
                    csv = num_stats.to_csv()
                    st.download_button(
                        label="Download Statistics as CSV",
                        data=csv,
                        file_name="numeric_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No numeric columns in the dataset.")
            
            # Categorical Columns Tab
            with num_cat_tabs[1]:
                if cat_cols:
                    # Create a dataframe for categorical statistics
                    cat_stats = pd.DataFrame(index=cat_cols)
                    
                    # Add statistics
                    cat_stats['unique'] = self.df[cat_cols].nunique()
                    cat_stats['missing'] = self.df[cat_cols].isna().sum()
                    cat_stats['missing_pct'] = (self.df[cat_cols].isna().sum() / len(self.df) * 100).round(2)
                    cat_stats['most_common'] = [self.df[col].value_counts().index[0] if not self.df[col].isna().all() else None for col in cat_cols]
                    cat_stats['most_common_count'] = [self.df[col].value_counts().iloc[0] if not self.df[col].isna().all() else 0 for col in cat_cols]
                    cat_stats['most_common_pct'] = [(cat_stats.loc[col, 'most_common_count'] / len(self.df) * 100).round(2) for col in cat_cols]
                    
                    # Show the statistics
                    st.markdown("### Categorical Columns Statistics")
                    st.dataframe(cat_stats, use_container_width=True)
                    
                    # Add download button for CSV
                    csv = cat_stats.to_csv()
                    st.download_button(
                        label="Download Statistics as CSV",
                        data=csv,
                        file_name="categorical_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Add option to show top values for a column
                    st.markdown("### Top Values for Categorical Column")
                    selected_col = st.selectbox("Select column:", cat_cols)
                    top_n = st.slider("Number of top values:", 3, 20, 5)
                    
                    # Calculate and show top values
                    value_counts = self.df[selected_col].value_counts().reset_index()
                    value_counts.columns = ['Value', 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / len(self.df) * 100).round(2)
                    
                    st.dataframe(value_counts.head(top_n), use_container_width=True)
                    
                    # Create a bar chart of top values
                    fig = px.bar(
                        value_counts.head(top_n), 
                        x='Value', 
                        y='Count', 
                        title=f'Top {top_n} Values for {selected_col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical columns in the dataset.")
        
        # Show sample data
        st.markdown("### Sample Data")
        n_rows = st.slider("Number of rows to display:", 5, 100, 10)
        st.dataframe(self.df.head(n_rows), use_container_width=True)
    
    def _render_group_statistics(self):
        """Render group statistics"""
        st.subheader("Group Statistics")
        
        # Get columns by type
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns available for grouping.")
            return
            
        if not num_cols:
            st.warning("No numeric columns available for analysis.")
            return
        
        # Create form for group statistics
        with st.form(key="group_stats_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Select grouping columns
                group_cols = st.multiselect(
                    "Group by:",
                    cat_cols,
                    default=[cat_cols[0]] if cat_cols else []
                )
                
                # Select numeric columns to analyze
                analyze_cols = st.multiselect(
                    "Analyze columns:",
                    num_cols,
                    default=[num_cols[0]] if num_cols else []
                )
            
            with col2:
                # Select aggregation functions
                agg_funcs = st.multiselect(
                    "Aggregation functions:",
                    ["Mean", "Sum", "Count", "Min", "Max", "Median", "Std", "Var"],
                    default=["Mean", "Sum", "Count"]
                )
                
                # Additional options
                show_pct = st.checkbox("Show percentages (for count)", value=True)
                sort_by = st.selectbox(
                    "Sort results by:",
                    ["Group", "Mean", "Sum", "Count", "Min", "Max", "Median", "Std", "Var"],
                    index=0
                )
                
                # Sort order
                ascending = st.radio("Sort order:", ["Ascending", "Descending"], horizontal=True) == "Ascending"
            
            # Submit button
            submit_button = st.form_submit_button("Calculate Group Statistics", use_container_width=True)
        
        # Calculate group statistics
        if submit_button:
            if not group_cols:
                st.warning("Please select at least one column for grouping.")
            elif not analyze_cols:
                st.warning("Please select at least one column to analyze.")
            elif not agg_funcs:
                st.warning("Please select at least one aggregation function.")
            else:
                try:
                    # Translate selected functions to pandas agg functions
                    agg_dict = {
                        "Mean": "mean",
                        "Sum": "sum",
                        "Count": "count",
                        "Min": "min",
                        "Max": "max",
                        "Median": "median",
                        "Std": "std",
                        "Var": "var"
                    }
                    
                    selected_aggs = [agg_dict[func] for func in agg_funcs]
                    
                    # Create aggregation dictionary
                    agg_map = {col: selected_aggs for col in analyze_cols}
                    
                    # Calculate grouped statistics
                    grouped_stats = self.df.groupby(group_cols)[analyze_cols].agg(agg_map)
                    
                    # Flatten MultiIndex if needed
                    if isinstance(grouped_stats.columns, pd.MultiIndex):
                        grouped_stats.columns = [f"{col}_{agg}" for col, agg in grouped_stats.columns]
                    
                    # Reset index for better display
                    grouped_stats = grouped_stats.reset_index()
                    
                    # Add percentage calculations for count if requested
                    if show_pct and "count" in selected_aggs:
                        for col in analyze_cols:
                            count_col = f"{col}_count"
                            if count_col in grouped_stats.columns:
                                total_count = grouped_stats[count_col].sum()
                                grouped_stats[f"{col}_count_pct"] = (grouped_stats[count_col] / total_count * 100).round(2)
                    
                    # Sort if needed
                    if sort_by != "Group":
                        # Find the column that matches the sort criteria
                        sort_cols = [col for col in grouped_stats.columns if sort_by.lower() in col.lower()]
                        if sort_cols:
                            grouped_stats = grouped_stats.sort_values(sort_cols[0], ascending=ascending)
                    
                    # Display results
                    st.markdown(f"### Group Statistics for {', '.join(group_cols)}")
                    st.dataframe(grouped_stats, use_container_width=True)
                    
                    # Add download button
                    csv = grouped_stats.to_csv(index=False)
                    st.download_button(
                        label="Download Group Statistics",
                        data=csv,
                        file_name="group_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Create visualization
                    st.markdown("### Visualization")
                    
                    # Select column to visualize
                    viz_metric = st.selectbox(
                        "Select metric to visualize:",
                        grouped_stats.columns[len(group_cols):]
                    )
                    
                    # Visualization type
                    viz_type = st.radio(
                        "Visualization type:",
                        ["Bar Chart", "Pie Chart", "Line Chart"],
                        horizontal=True
                    )
                    
                    # Create visualization
                    if viz_type == "Bar Chart":
                        if len(group_cols) == 1:
                            # Simple bar chart for single grouping
                            fig = px.bar(
                                grouped_stats,
                                x=group_cols[0],
                                y=viz_metric,
                                title=f"{viz_metric} by {group_cols[0]}",
                                color=group_cols[0] if len(grouped_stats) <= 15 else None
                            )
                        else:
                            # Grouped bar chart for multiple groupings
                            group_col1, group_col2 = group_cols[0], group_cols[1]
                            
                            pivot_data = grouped_stats.pivot(
                                index=group_col1,
                                columns=group_col2,
                                values=viz_metric
                            ).reset_index()
                            
                            fig = px.bar(
                                pivot_data,
                                x=group_col1,
                                y=pivot_data.columns[1:],
                                title=f"{viz_metric} by {group_col1} and {group_col2}",
                                barmode="group"
                            )
                    
                    elif viz_type == "Pie Chart":
                        # Limit to top categories if too many
                        if len(grouped_stats) > 10:
                            st.warning("Too many categories for pie chart. Showing top 10 by value.")
                            sorted_data = grouped_stats.sort_values(viz_metric, ascending=False).head(10)
                        else:
                            sorted_data = grouped_stats
                        
                        fig = px.pie(
                            sorted_data,
                            names=group_cols[0],
                            values=viz_metric,
                            title=f"{viz_metric} by {group_cols[0]}"
                        )
                    
                    else:  # Line Chart
                        if len(group_cols) == 1:
                            # Simple line chart for single grouping
                            fig = px.line(
                                grouped_stats,
                                x=group_cols[0],
                                y=viz_metric,
                                title=f"{viz_metric} by {group_cols[0]}",
                                markers=True
                            )
                        else:
                            # Line chart with multiple lines for multiple groupings
                            group_col1, group_col2 = group_cols[0], group_cols[1]
                            
                            pivot_data = grouped_stats.pivot(
                                index=group_col1,
                                columns=group_col2,
                                values=viz_metric
                            ).reset_index()
                            
                            fig = px.line(
                                pivot_data,
                                x=group_col1,
                                y=pivot_data.columns[1:],
                                title=f"{viz_metric} by {group_col1} and {group_col2}",
                                markers=True
                            )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating group statistics: {str(e)}")
    
    def _render_correlation_analysis(self):
        """Render correlation analysis"""
        st.subheader("Correlation Analysis")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for correlation analysis.")
            return
        
        # Create form for correlation analysis
        with st.form(key="correlation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Select columns for correlation
                corr_cols = st.multiselect(
                    "Select columns for correlation:",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))]
                )
                
                # Select correlation method
                corr_method = st.selectbox(
                    "Correlation method:",
                    ["Pearson", "Spearman", "Kendall"],
                    index=0
                )
            
            with col2:
                # Color scheme
                color_scheme = st.selectbox(
                    "Color scheme:",
                    ["RdBu_r", "Viridis", "Plasma", "Blues", "Reds"]
                )
                
                # Display options
                show_values = st.checkbox("Show correlation values", value=True)
                show_colorbar = st.checkbox("Show color scale", value=True)
            
            # Submit button
            submit_button = st.form_submit_button("Calculate Correlation", use_container_width=True)
        
        # Calculate correlation
        if submit_button:
            if len(corr_cols) < 2:
                st.warning("Please select at least two columns for correlation analysis.")
            else:
                try:
                    # Calculate correlation matrix
                    corr_matrix = self.df[corr_cols].corr(method=corr_method.lower())
                    
                    # Display correlation matrix
                    st.markdown(f"### {corr_method} Correlation Matrix")
                    st.dataframe(corr_matrix.round(3), use_container_width=True)
                    
                    # Create a heatmap visualization
                    fig = px.imshow(
                        corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        text_auto=show_values,
                        color_continuous_scale=color_scheme,
                        zmin=-1, zmax=1,
                        title=f"{corr_method} Correlation Heatmap"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        coloraxis_colorbar_x=1.05 if show_colorbar else None,
                        coloraxis_showscale=show_colorbar,
                        height=600
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the strongest correlations
                    st.markdown("### Strongest Correlations")
                    
                    # Get correlations in tabular format
                    corrs = corr_matrix.unstack().reset_index()
                    corrs.columns = ['Variable 1', 'Variable 2', 'Correlation']
                    
                    # Remove self-correlations
                    corrs = corrs[corrs['Variable 1'] != corrs['Variable 2']]
                    
                    # Sort by absolute correlation and take top 10
                    corrs['Abs Correlation'] = corrs['Correlation'].abs()
                    corrs = corrs.sort_values('Abs Correlation', ascending=False).head(10)
                    corrs = corrs.drop('Abs Correlation', axis=1)
                    
                    # Show the results
                    st.dataframe(corrs.round(3), use_container_width=True)
                    
                    # Add download buttons
                    csv = corr_matrix.to_csv()
                    st.download_button(
                        label="Download Correlation Matrix",
                        data=csv,
                        file_name="correlation_matrix.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error calculating correlation: {str(e)}")
    
    def _render_distribution_analysis(self):
        """Render distribution analysis"""
        st.subheader("Distribution Analysis")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns available for distribution analysis.")
            return
        
        # Create distribution analysis form
        col1, col2 = st.columns(2)
        
        with col1:
            # Select variable
            variable = st.selectbox("Select variable for distribution analysis:", num_cols)
            
            # Distribution plot type
            plot_type = st.selectbox(
                "Plot type:",
                ["Histogram", "Box Plot", "Violin Plot", "Combined (Histogram + Box)"]
            )
        
        with col2:
            # Bin options for histogram
            if plot_type in ["Histogram", "Combined (Histogram + Box)"]:
                n_bins = st.slider("Number of bins:", 5, 100, 20)
                
                # KDE option
                show_kde = st.checkbox("Show KDE curve", value=True)
                
                # Distribution fit
                fit_dist = st.checkbox("Fit distribution", value=False)
                if fit_dist:
                    dist_type = st.selectbox(
                        "Distribution type:",
                        ["Normal", "Exponential", "Gamma", "Weibull", "Lognormal"]
                    )
            
            # Split by categorical (optional)
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                split_by = st.selectbox("Split by category (optional):", ["None"] + cat_cols)
                split_by = None if split_by == "None" else split_by
            else:
                split_by = None
        
        # Generate distribution analysis
        if st.button("Analyze Distribution", use_container_width=True):
            try:
                # Basic statistics
                st.markdown(f"### Distribution Statistics for {variable}")
                
                # Get data without NaN values
                data = self.df[variable].dropna()
                
                # Create a statistics table
                stats_df = pd.DataFrame({
                    "Statistic": [
                        "Count", "Mean", "Median", "Standard Deviation", "Variance",
                        "Minimum", "Maximum", "Range", "Skewness", "Kurtosis"
                    ],
                    "Value": [
                        len(data),
                        data.mean(),
                        data.median(),
                        data.std(),
                        data.var(),
                        data.min(),
                        data.max(),
                        data.max() - data.min(),
                        data.skew(),
                        data.kurtosis()
                    ]
                })
                
                # Round numeric values
                stats_df["Value"] = stats_df["Value"].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
                
                # Display statistics
                st.dataframe(stats_df, use_container_width=True)
                
                # Display percentiles
                st.markdown("#### Percentiles")
                percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
                perc_values = [np.percentile(data, p) for p in percentiles]
                perc_df = pd.DataFrame({
                    "Percentile": [f"{p}%" for p in percentiles],
                    "Value": [round(v, 4) for v in perc_values]
                })
                st.dataframe(perc_df, use_container_width=True)
                
                # Create the distribution plot
                st.markdown(f"### Distribution Plot for {variable}")
                
                if plot_type == "Histogram":
                    if split_by:
                        fig = px.histogram(
                            self.df,
                            x=variable,
                            color=split_by,
                            nbins=n_bins,
                            marginal="box" if show_kde else None,
                            title=f"Histogram of {variable} by {split_by}"
                        )
                    else:
                        fig = px.histogram(
                            self.df,
                            x=variable,
                            nbins=n_bins,
                            title=f"Histogram of {variable}"
                        )
                        
                        # Add KDE curve if requested
                        if show_kde:
                            import scipy.stats as stats
                            
                            # Calculate KDE
                            kde_x, kde_y = self._calculate_kde(data)
                            
                            # Add KDE curve
                            fig.add_scatter(
                                x=kde_x,
                                y=kde_y * len(data) * (data.max() - data.min()) / n_bins,
                                mode='lines',
                                name='KDE',
                                line=dict(color='red', width=2)
                            )
                        
                        # Fit distribution if requested
                        if fit_dist:
                            import scipy.stats as stats
                            
                            # Define distribution mapping
                            dist_map = {
                                "Normal": stats.norm,
                                "Exponential": stats.expon,
                                "Gamma": stats.gamma,
                                "Weibull": stats.weibull_min,
                                "Lognormal": stats.lognorm
                            }
                            
                            # Get distribution
                            distribution = dist_map[dist_type]
                            
                            # Fit distribution
                            params = distribution.fit(data)
                            
                            # Generate x values for PDF
                            x = np.linspace(data.min(), data.max(), 1000)
                            
                            # Calculate PDF
                            if dist_type == "Normal":
                                pdf = distribution.pdf(x, params[0], params[1])
                            elif dist_type == "Exponential":
                                pdf = distribution.pdf(x, params[0], params[1])
                            elif dist_type == "Gamma":
                                pdf = distribution.pdf(x, params[0], params[1], params[2])
                            elif dist_type == "Weibull":
                                pdf = distribution.pdf(x, params[0], params[1], params[2])
                            elif dist_type == "Lognormal":
                                pdf = distribution.pdf(x, params[0], params[1], params[2])
                            
                            # Scale PDF to match histogram
                            scale_factor = len(data) * (data.max() - data.min()) / n_bins
                            pdf_scaled = pdf * scale_factor
                            
                            # Add PDF curve
                            fig.add_scatter(
                                x=x,
                                y=pdf_scaled,
                                mode='lines',
                                name=f'{dist_type} Fit',
                                line=dict(color='green', width=2, dash='dash')
                            )
                            
                            # Add fit parameters
                            param_names = {
                                "Normal": ["mean", "std"],
                                "Exponential": ["loc", "scale"],
                                "Gamma": ["a", "loc", "scale"],
                                "Weibull": ["c", "loc", "scale"],
                                "Lognormal": ["s", "loc", "scale"]
                            }
                            
                            param_text = f"{dist_type} Parameters:<br>"
                            for i, name in enumerate(param_names[dist_type]):
                                param_text += f"{name}: {params[i]:.4f}<br>"
                            
                            fig.add_annotation(
                                x=0.95,
                                y=0.95,
                                xref="paper",
                                yref="paper",
                                text=param_text,
                                showarrow=False,
                                align="right",
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="black",
                                borderwidth=1
                            )
                
                elif plot_type == "Box Plot":
                    if split_by:
                        fig = px.box(
                            self.df,
                            x=split_by,
                            y=variable,
                            color=split_by,
                            title=f"Box Plot of {variable} by {split_by}"
                        )
                    else:
                        fig = px.box(
                            self.df,
                            y=variable,
                            title=f"Box Plot of {variable}"
                        )
                
                elif plot_type == "Violin Plot":
                    if split_by:
                        fig = px.violin(
                            self.df,
                            x=split_by,
                            y=variable,
                            color=split_by,
                            box=True,
                            title=f"Violin Plot of {variable} by {split_by}"
                        )
                    else:
                        fig = px.violin(
                            self.df,
                            y=variable,
                            box=True,
                            title=f"Violin Plot of {variable}"
                        )
                
                else:  # Combined (Histogram + Box)
                    if split_by:
                        fig = px.histogram(
                            self.df,
                            x=variable,
                            color=split_by,
                            nbins=n_bins,
                            marginal="box",
                            title=f"Distribution of {variable} by {split_by}"
                        )
                    else:
                        fig = px.histogram(
                            self.df,
                            x=variable,
                            nbins=n_bins,
                            marginal="box",
                            title=f"Distribution of {variable}"
                        )
                        
                        # Add KDE curve if requested
                        if show_kde:
                            import scipy.stats as stats
                            
                            # Calculate KDE
                            kde_x, kde_y = self._calculate_kde(data)
                            
                            # Add KDE curve
                            fig.add_scatter(
                                x=kde_x,
                                y=kde_y * len(data) * (data.max() - data.min()) / n_bins,
                                mode='lines',
                                name='KDE',
                                line=dict(color='red', width=2)
                            )
                
                # Update layout
                fig.update_layout(height=500)
                
                # Show the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add QQ plot
                st.markdown("### Quantile-Quantile (Q-Q) Plot")
                st.write("Q-Q plot compares the distribution of the data to a theoretical distribution.")
                
                # Create QQ plot
                from scipy import stats
                
                # Calculate theoretical quantiles (normal distribution)
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
                
                # Create QQ plot
                qq_fig = px.scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                    title=f"Q-Q Plot for {variable} (vs. Normal Distribution)"
                )
                
                # Add reference line
                min_val = min(min(theoretical_quantiles), min(sample_quantiles))
                max_val = max(max(theoretical_quantiles), max(sample_quantiles))
                ref_line = np.linspace(min_val, max_val, 100)
                
                qq_fig.add_scatter(
                    x=ref_line,
                    y=ref_line,
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                )
                
                # Show QQ plot
                st.plotly_chart(qq_fig, use_container_width=True)
                
                # Test for normality
                st.markdown("### Normality Tests")
                st.write("These tests evaluate whether the data follows a normal distribution.")
                
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                
                # D'Agostino-Pearson test
                dagostino_stat, dagostino_p = stats.normaltest(data)
                
                # Create a dataframe with test results
                normality_df = pd.DataFrame({
                    "Test": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "D'Agostino-Pearson"],
                    "Statistic": [shapiro_stat, ks_stat, dagostino_stat],
                    "p-value": [shapiro_p, ks_p, dagostino_p],
                    "Interpretation": [
                        "Normal" if shapiro_p > 0.05 else "Not Normal",
                        "Normal" if ks_p > 0.05 else "Not Normal",
                        "Normal" if dagostino_p > 0.05 else "Not Normal"
                    ]
                })
                
                # Display test results
                st.dataframe(normality_df.round(4), use_container_width=True)
                st.info("Interpretation: If p-value > 0.05, we cannot reject the hypothesis that the data follows a normal distribution.")
                
            except Exception as e:
                st.error(f"Error analyzing distribution: {str(e)}")
    
    def _render_outlier_detection(self):
        """Render outlier detection"""
        st.subheader("Outlier Detection")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns available for outlier detection.")
            return
        
        # Create outlier detection form
        col1, col2 = st.columns(2)
        
        with col1:
            # Select variable
            variable = st.selectbox("Select variable for outlier detection:", num_cols)
            
            # Detection method
            detection_method = st.selectbox(
                "Detection method:",
                ["IQR (Tukey)", "Z-Score", "Modified Z-Score", "Isolation Forest"]
            )
        
        with col2:
            # Method specific parameters
            if detection_method == "IQR (Tukey)":
                # IQR multiplier
                k = st.slider("IQR multiplier (k):", 1.0, 3.0, 1.5, 0.1)
                threshold = None
                contamination = None
            elif detection_method == "Z-Score":
                # Z-score threshold
                threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1)
                k = None
                contamination = None
            elif detection_method == "Modified Z-Score":
                # Modified Z-score threshold
                threshold = st.slider("Modified Z-score threshold:", 1.0, 5.0, 3.5, 0.1)
                k = None
                contamination = None
            else:  # Isolation Forest
                # Contamination parameter
                contamination = st.slider("Contamination (expected proportion of outliers):", 0.01, 0.5, 0.05, 0.01)
                k = None
                threshold = None
        
        # Generate outlier detection
        if st.button("Detect Outliers", use_container_width=True):
            try:
                # Get clean data (remove NaN values)
                clean_data = self.df[variable].dropna()
                
                # Detect outliers based on the selected method
                if detection_method == "IQR (Tukey)":
                    # Calculate IQR
                    Q1 = clean_data.quantile(0.25)
                    Q3 = clean_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Calculate bounds
                    lower_bound = Q1 - k * IQR
                    upper_bound = Q3 + k * IQR
                    
                    # Identify outliers
                    outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                    non_outliers = clean_data[(clean_data >= lower_bound) & (clean_data <= upper_bound)]
                    
                    # Create a dataframe with outlier flags
                    outlier_df = pd.DataFrame({
                        "Value": clean_data,
                        "Is Outlier": clean_data.apply(lambda x: x < lower_bound or x > upper_bound)
                    })
                    
                    # Description
                    method_desc = f"IQR method with k={k}. Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
                    
                elif detection_method == "Z-Score":
                    # Calculate Z-scores
                    mean = clean_data.mean()
                    std = clean_data.std()
                    z_scores = (clean_data - mean) / std
                    
                    # Identify outliers
                    outliers = clean_data[abs(z_scores) > threshold]
                    non_outliers = clean_data[abs(z_scores) <= threshold]
                    
                    # Create a dataframe with outlier flags and Z-scores
                    outlier_df = pd.DataFrame({
                        "Value": clean_data,
                        "Z-Score": z_scores,
                        "Is Outlier": abs(z_scores) > threshold
                    })
                    
                    # Description
                    method_desc = f"Z-score method with threshold={threshold}. Mean={mean:.2f}, Std={std:.2f}"
                    
                elif detection_method == "Modified Z-Score":
                    # Calculate Modified Z-scores
                    median = clean_data.median()
                    MAD = np.median(abs(clean_data - median))
                    
                    # Avoid division by zero
                    if MAD == 0:
                        st.warning("Median Absolute Deviation (MAD) is zero. Cannot calculate Modified Z-scores.")
                        return
                    
                    # Modified Z-score formula: 0.6745 * (x - median) / MAD
                    # The constant 0.6745 makes the MAD comparable to the standard deviation for normal distributions
                    mod_z_scores = 0.6745 * (clean_data - median) / MAD
                    
                    # Identify outliers
                    outliers = clean_data[abs(mod_z_scores) > threshold]
                    non_outliers = clean_data[abs(mod_z_scores) <= threshold]
                    
                    # Create a dataframe with outlier flags and Modified Z-scores
                    outlier_df = pd.DataFrame({
                        "Value": clean_data,
                        "Modified Z-Score": mod_z_scores,
                        "Is Outlier": abs(mod_z_scores) > threshold
                    })
                    
                    # Description
                    method_desc = f"Modified Z-score method with threshold={threshold}. Median={median:.2f}, MAD={MAD:.2f}"
                    
                else:  # Isolation Forest
                    from sklearn.ensemble import IsolationForest
                    
                    # Reshape data for scikit-learn
                    X = clean_data.values.reshape(-1, 1)
                    
                    # Create and fit the model
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    outlier_labels = iso_forest.fit_predict(X)
                    
                    # Convert to boolean (True for outliers)
                    is_outlier = outlier_labels == -1
                    
                    # Identify outliers
                    outliers = clean_data[is_outlier]
                    non_outliers = clean_data[~is_outlier]
                    
                    # Create a dataframe with outlier flags
                    outlier_df = pd.DataFrame({
                        "Value": clean_data,
                        "Is Outlier": is_outlier
                    })
                    
                    # Description
                    method_desc = f"Isolation Forest method with contamination={contamination}"
                
                # Display results
                st.markdown(f"### Outlier Detection Results for {variable}")
                st.write(f"Method: {detection_method}")
                st.write(method_desc)
                
                # Number of outliers
                n_outliers = len(outliers)
                outlier_pct = n_outliers / len(clean_data) * 100
                
                st.write(f"**Number of outliers detected:** {n_outliers} ({outlier_pct:.2f}% of non-missing values)")
                
                # Display statistics
                st.markdown("#### Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Min (All Data)", f"{clean_data.min():.2f}")
                    st.metric("Min (Without Outliers)", f"{non_outliers.min():.2f}")
                
                with col2:
                    st.metric("Mean (All Data)", f"{clean_data.mean():.2f}")
                    st.metric("Mean (Without Outliers)", f"{non_outliers.mean():.2f}")
                
                with col3:
                    st.metric("Max (All Data)", f"{clean_data.max():.2f}")
                    st.metric("Max (Without Outliers)", f"{non_outliers.max():.2f}")
                
                # Visualize outliers
                st.markdown("#### Visualization")
                
                # Create a box plot
                box_fig = px.box(
                    outlier_df,
                    y="Value",
                    title=f"Box Plot of {variable} with Outliers Highlighted",
                    points="all"
                )
                
                # Color outliers red
                box_fig.update_traces(
                    marker=dict(
                        color=outlier_df["Is Outlier"].map({True: "red", False: "blue"}),
                        size=outlier_df["Is Outlier"].map({True: 8, False: 5}),
                        opacity=outlier_df["Is Outlier"].map({True: 1.0, False: 0.6})
                    )
                )
                
                # Show the box plot
                st.plotly_chart(box_fig, use_container_width=True)
                
                # Create a histogram
                hist_fig = px.histogram(
                    outlier_df,
                    x="Value",
                    color="Is Outlier",
                    title=f"Histogram of {variable} with Outliers Highlighted",
                    color_discrete_map={True: "red", False: "blue"},
                    barmode="overlay",
                    nbins=30
                )
                
                # Show the histogram
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Show outliers
                if n_outliers > 0:
                    st.markdown("#### Outlier Values")
                    
                    # Sort outliers by value
                    sorted_outliers = outliers.sort_values(ascending=False)
                    
                    # Create a dataframe with outlier values and ranks
                    outlier_vals_df = pd.DataFrame({
                        "Value": sorted_outliers,
                        "Rank": range(1, len(sorted_outliers) + 1)
                    })
                    
                    # Display outliers
                    st.dataframe(outlier_vals_df, use_container_width=True)
                    
                    # Download outliers
                    csv = outlier_vals_df.to_csv(index=False)
                    st.download_button(
                        label="Download Outlier List",
                        data=csv,
                        file_name=f"{variable}_outliers.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Show all data with outlier flags
                with st.expander("View All Data with Outlier Flags"):
                    # If we have scores, sort by that
                    if "Z-Score" in outlier_df.columns:
                        outlier_df = outlier_df.sort_values("Z-Score", ascending=False)
                    elif "Modified Z-Score" in outlier_df.columns:
                        outlier_df = outlier_df.sort_values("Modified Z-Score", ascending=False)
                    else:
                        # Otherwise sort by value
                        outlier_df = outlier_df.sort_values("Value", ascending=False)
                    
                    # Reset index for display
                    outlier_df = outlier_df.reset_index(drop=True)
                    
                    # Display dataframe
                    st.dataframe(outlier_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error detecting outliers: {str(e)}")
    
    def _calculate_kde(self, data, bw_method=None):
        """Calculate kernel density estimate for a given dataset"""
        from scipy.stats import gaussian_kde
        import numpy as np
        
        # Check if we have enough data
        if len(data) < 2:
            # Return empty data if not enough points
            return [], []
        
        # Fit KDE
        kde = gaussian_kde(data, bw_method=bw_method)
        
        # Generate points to evaluate KDE
        x_min, x_max = data.min(), data.max()
        margin = 0.1 * (x_max - x_min)
        x_grid = np.linspace(x_min - margin, x_max + margin, 1000)
        
        # Evaluate KDE on grid
        y_grid = kde(x_grid)
        
        return x_grid, y_grid