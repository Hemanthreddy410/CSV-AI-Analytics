import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import datetime
import re

class DataProcessor:
    """Class for processing and transforming data"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.original_df = df.copy() if df is not None else None
        
        # Store processing history
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
    
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
        
        # Processing History
        if st.session_state.processing_history:
            st.header("Processing History")
            
            # Create collapsible section for history
            with st.expander("View Processing Steps", expanded=False):
                for i, step in enumerate(st.session_state.processing_history):
                    st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Reset button
            if st.button("Reset to Original Data", key="reset_data"):
                self.df = self.original_df.copy()
                st.session_state.df = self.original_df.copy()
                st.session_state.processing_history = []
                st.success("Data reset to original state!")
                st.rerun()
    
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
                        "Fill with backward fill"
                    ]
                )
                
                # Additional input for constant value if selected
                if handling_method == "Fill with constant value":
                    constant_value = st.text_input("Enter constant value:")
                
                # Apply button
                if st.button("Apply Missing Value Treatment", key="apply_missing"):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply the selected method
                        if handling_method == "Drop rows":
                            self.df = self.df.dropna(subset=[col_to_handle])
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            rows_removed = orig_shape[0] - self.df.shape[0]
                            st.session_state.processing_history.append({
                                "description": f"Dropped {rows_removed} rows with missing values in column '{col_to_handle}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "drop",
                                    "rows_affected": rows_removed
                                }
                            })
                            
                            st.success(f"Dropped {rows_removed} rows with missing values in '{col_to_handle}'")
                            
                        elif handling_method == "Fill with mean":
                            if self.df[col_to_handle].dtype.kind in 'bifc':  # Check if numeric
                                mean_val = self.df[col_to_handle].mean()
                                self.df[col_to_handle] = self.df[col_to_handle].fillna(mean_val)
                                st.session_state.df = self.df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col_to_handle}' with mean ({mean_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col_to_handle,
                                        "method": "mean",
                                        "value": mean_val
                                    }
                                })
                                
                                st.success(f"Filled missing values in '{col_to_handle}' with mean: {mean_val:.2f}")
                            else:
                                st.error(f"Column '{col_to_handle}' is not numeric. Cannot use mean imputation.")
                                
                        elif handling_method == "Fill with median":
                            if self.df[col_to_handle].dtype.kind in 'bifc':  # Check if numeric
                                median_val = self.df[col_to_handle].median()
                                self.df[col_to_handle] = self.df[col_to_handle].fillna(median_val)
                                st.session_state.df = self.df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col_to_handle}' with median ({median_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col_to_handle,
                                        "method": "median",
                                        "value": median_val
                                    }
                                })
                                
                                st.success(f"Filled missing values in '{col_to_handle}' with median: {median_val:.2f}")
                            else:
                                st.error(f"Column '{col_to_handle}' is not numeric. Cannot use median imputation.")
                                
                        elif handling_method == "Fill with mode":
                            mode_val = self.df[col_to_handle].mode()[0]
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(mode_val)
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with mode ({mode_val})",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "mode",
                                    "value": mode_val
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with mode: {mode_val}")
                            
                        elif handling_method == "Fill with constant value":
                            if constant_value:
                                # Try to convert to appropriate type
                                try:
                                    if self.df[col_to_handle].dtype.kind in 'bifc':  # Numeric
                                        constant_val = float(constant_value)
                                    elif self.df[col_to_handle].dtype.kind == 'b':  # Boolean
                                        constant_val = constant_value.lower() in ['true', 'yes', '1', 't', 'y']
                                    else:
                                        constant_val = constant_value
                                        
                                    self.df[col_to_handle] = self.df[col_to_handle].fillna(constant_val)
                                    st.session_state.df = self.df
                                    
                                    # Add to processing history
                                    st.session_state.processing_history.append({
                                        "description": f"Filled missing values in '{col_to_handle}' with constant ({constant_val})",
                                        "timestamp": datetime.datetime.now(),
                                        "type": "missing_values",
                                        "details": {
                                            "column": col_to_handle,
                                            "method": "constant",
                                            "value": constant_val
                                        }
                                    })
                                    
                                    st.success(f"Filled missing values in '{col_to_handle}' with: {constant_val}")
                                except ValueError:
                                    st.error(f"Could not convert '{constant_value}' to appropriate type for column '{col_to_handle}'")
                            else:
                                st.error("Please enter a constant value")
                                
                        elif handling_method == "Fill with forward fill":
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(method='ffill')
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with forward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "ffill"
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with forward fill")
                            
                        elif handling_method == "Fill with backward fill":
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(method='bfill')
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with backward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "bfill"
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with backward fill")
                    
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
                if st.button("Remove Duplicate Rows"):
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
                            "Replace with NaN"
                        ]
                    )
                    
                    # Button to handle outliers
                    if st.button("Handle Outliers"):
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
                                
                                st.success(f"Removed {rows_removed} outliers from column '{col_for_outliers}'")
                                
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
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error handling outliers: {str(e)}")
    
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
                col_to_transform = st.selectbox(
                    "Select column to transform:",
                    num_cols,
                    key="num_transform_col"
                )
                
                transform_method = st.selectbox(
                    "Select transformation method:",
                    [
                        "Standardization (Z-score)",
                        "Min-Max Scaling",
                        "Log Transform",
                        "Square Root Transform",
                        "Box-Cox Transform",
                        "Binning/Discretization"
                    ],
                    key="transform_method"
                )
                
                # Additional parameters for specific transforms
                if transform_method == "Binning/Discretization":
                    num_bins = st.slider("Number of bins:", 2, 20, 5)
                    bin_labels = st.text_input("Bin labels (comma-separated, leave empty for default):")
                
                # Apply button
                if st.button("Apply Transformation", key="apply_transform"):
                    try:
                        if transform_method == "Standardization (Z-score)":
                            # Z-score standardization: (x - mean) / std
                            mean = self.df[col_to_transform].mean()
                            std = self.df[col_to_transform].std()
                            
                            if std == 0:
                                st.error(f"Standard deviation is zero for column '{col_to_transform}'. Cannot standardize.")
                            else:
                                self.df[f"{col_to_transform}_standardized"] = (self.df[col_to_transform] - mean) / std
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied standardization to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "standardization",
                                        "new_column": f"{col_to_transform}_standardized"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_standardized' with standardized values")
                        
                        elif transform_method == "Min-Max Scaling":
                            # Min-Max scaling: (x - min) / (max - min)
                            min_val = self.df[col_to_transform].min()
                            max_val = self.df[col_to_transform].max()
                            
                            if max_val == min_val:
                                st.error(f"Maximum and minimum values are the same for column '{col_to_transform}'. Cannot scale.")
                            else:
                                self.df[f"{col_to_transform}_scaled"] = (self.df[col_to_transform] - min_val) / (max_val - min_val)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied min-max scaling to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "min_max_scaling",
                                        "new_column": f"{col_to_transform}_scaled"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_scaled' with scaled values (0-1)")
                        
                        elif transform_method == "Log Transform":
                            # Check for non-positive values
                            if (self.df[col_to_transform] <= 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1 if min_val <= 0 else 0
                                self.df[f"{col_to_transform}_log"] = np.log(self.df[col_to_transform] + const)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied log transform to '{col_to_transform}' with constant {const}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "log_transform",
                                        "constant": const,
                                        "new_column": f"{col_to_transform}_log"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_log' with log transform (added constant {const})")
                            else:
                                self.df[f"{col_to_transform}_log"] = np.log(self.df[col_to_transform])
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied log transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "log_transform",
                                        "new_column": f"{col_to_transform}_log"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_log' with log transform")
                        
                        elif transform_method == "Square Root Transform":
                            # Check for negative values
                            if (self.df[col_to_transform] < 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values non-negative
                                const = abs(min_val) + 1 if min_val < 0 else 0
                                self.df[f"{col_to_transform}_sqrt"] = np.sqrt(self.df[col_to_transform] + const)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied square root transform to '{col_to_transform}' with constant {const}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "sqrt_transform",
                                        "constant": const,
                                        "new_column": f"{col_to_transform}_sqrt"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_sqrt' with square root transform (added constant {const})")
                            else:
                                self.df[f"{col_to_transform}_sqrt"] = np.sqrt(self.df[col_to_transform])
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied square root transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "sqrt_transform",
                                        "new_column": f"{col_to_transform}_sqrt"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_sqrt' with square root transform")
                        
                        elif transform_method == "Box-Cox Transform":
                            # Check if all values are positive
                            if (self.df[col_to_transform] <= 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1 if min_val <= 0 else 0
                                
                                from scipy import stats
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform] + const)
                                self.df[f"{col_to_transform}_boxcox"] = transformed_data
                                
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
                                        "new_column": f"{col_to_transform}_boxcox"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_boxcox' with Box-Cox transform (lambda={lambda_val:.4f}, added constant {const})")
                            else:
                                from scipy import stats
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform])
                                self.df[f"{col_to_transform}_boxcox"] = transformed_data
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied Box-Cox transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "boxcox_transform",
                                        "lambda": lambda_val,
                                        "new_column": f"{col_to_transform}_boxcox"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_boxcox' with Box-Cox transform (lambda={lambda_val:.4f})")
                        
                        elif transform_method == "Binning/Discretization":
                            # Create bins
                            bins = pd.cut(
                                self.df[col_to_transform], 
                                bins=num_bins, 
                                labels=bin_labels.split(',') if bin_labels else None
                            )
                            
                            self.df[f"{col_to_transform}_binned"] = bins
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied binning to '{col_to_transform}' with {num_bins} bins",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "binning",
                                    "num_bins": num_bins,
                                    "new_column": f"{col_to_transform}_binned"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_binned' with {num_bins} bins")
                        
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
                col_to_transform = st.selectbox(
                    "Select column to transform:",
                    cat_cols,
                    key="cat_transform_col"
                )
                
                transform_method = st.selectbox(
                    "Select transformation method:",
                    [
                        "One-Hot Encoding",
                        "Label Encoding",
                        "Frequency Encoding",
                        "Target Encoding (requires target column)"
                    ],
                    key="cat_transform_method"
                )
                
                # Additional parameters for specific transforms
                if transform_method == "Target Encoding (requires target column)":
                    target_col = st.selectbox(
                        "Select target column:",
                        self.df.columns
                    )
                
                # Apply button
                if st.button("Apply Transformation", key="apply_cat_transform"):
                    try:
                        if transform_method == "One-Hot Encoding":
                            # Get dummies (one-hot encoding)
                            dummies = pd.get_dummies(self.df[col_to_transform], prefix=col_to_transform)
                            
                            # Add to dataframe
                            self.df = pd.concat([self.df, dummies], axis=1)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied one-hot encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "one_hot_encoding",
                                    "new_columns": dummies.columns.tolist()
                                }
                            })
                            
                            st.success(f"Created {dummies.shape[1]} new columns with one-hot encoding for '{col_to_transform}'")
                        
                        elif transform_method == "Label Encoding":
                            # Map each unique value to an integer
                            unique_values = self.df[col_to_transform].dropna().unique()
                            mapping = {val: i for i, val in enumerate(unique_values)}
                            
                            self.df[f"{col_to_transform}_encoded"] = self.df[col_to_transform].map(mapping)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied label encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "label_encoding",
                                    "mapping": mapping,
                                    "new_column": f"{col_to_transform}_encoded"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_encoded' with label encoding")
                        
                        elif transform_method == "Frequency Encoding":
                            # Replace each category with its frequency
                            freq = self.df[col_to_transform].value_counts(normalize=True)
                            self.df[f"{col_to_transform}_freq"] = self.df[col_to_transform].map(freq)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied frequency encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "frequency_encoding",
                                    "new_column": f"{col_to_transform}_freq"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_freq' with frequency encoding")
                        
                        elif transform_method == "Target Encoding (requires target column)":
                            # Check if target column exists and is numeric
                            if target_col not in self.df.columns:
                                st.error(f"Target column '{target_col}' not found")
                            elif self.df[target_col].dtype.kind not in 'bifc':
                                st.error(f"Target column '{target_col}' must be numeric")
                            else:
                                # Calculate mean target value for each category
                                target_means = self.df.groupby(col_to_transform)[target_col].mean()
                                self.df[f"{col_to_transform}_target"] = self.df[col_to_transform].map(target_means)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied target encoding to '{col_to_transform}' using '{target_col}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "target_encoding",
                                        "target_column": target_col,
                                        "new_column": f"{col_to_transform}_target"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_target' with target encoding")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
            
            # Date Transformations section omitted for brevity
    
    def _render_feature_engineering(self):
        """Render feature engineering interface"""
        st.subheader("Feature Engineering")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Create New Features")
            
            # Feature creation methods
            feature_method = st.selectbox(
                "Feature creation method:",
                [
                    "Arithmetic Operation",
                    "Mathematical Function",
                    "String Operation",
                    "Conditional Logic",
                    "Aggregation by Group",
                    "Rolling Window"
                ],
                key="feature_method"
            )
            
            # Placeholder for now
            st.info("Feature Engineering functionality is under development.")
    
    def _render_data_filtering(self):
        """Render data filtering interface"""
        st.subheader("Data Filtering")
        
        # Placeholder for now
        st.info("Data Filtering functionality is under development.")
    
    def _render_column_management(self):
        """Render column management interface"""
        st.subheader("Column Management")
        
        # Placeholder for now
        st.info("Column Management functionality is under development.")