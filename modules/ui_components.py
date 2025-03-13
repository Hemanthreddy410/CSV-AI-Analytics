import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import datetime
import re
import json
import os
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
import matplotlib.pyplot as plt

class DataProcessor:
    """Enhanced class for processing and transforming data with local saving capabilities"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataframe"""
        self.df = df
        self.original_df = df.copy() if df is not None else None
        
        # Store processing history
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
            
        # Create data directory if it doesn't exist
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create processing snapshots dir 
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # Initialize autosave setting
        if 'autosave_enabled' not in st.session_state:
            st.session_state.autosave_enabled = True
    
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
            "Column Management",
            "Data Export & Save"
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
            
        # Data Export & Save Tab
        with processing_tabs[5]:
            self._render_export_and_save()
        
        # Processing History
        if st.session_state.processing_history:
            st.header("Processing History")
            
            # Create collapsible section for history
            with st.expander("View Processing Steps", expanded=False):
                for i, step in enumerate(st.session_state.processing_history):
                    # Create columns for better UI
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        # Add revert button for each step
                        if st.button("Revert", key=f"revert_{i}", help=f"Revert to before this step"):
                            self._revert_to_step(i)
                            st.success(f"Reverted to before step {i+1}")
                            st.rerun()
            
            # Action buttons for history
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Reset to Original Data", key="reset_data", use_container_width=True):
                    self.df = self.original_df.copy()
                    st.session_state.df = self.original_df.copy()
                    st.session_state.processing_history = []
                    st.success("Data reset to original state!")
                    st.rerun()
            with col2:
                if st.button("Save Current Snapshot", key="save_snapshot", use_container_width=True):
                    snapshot_path = self._save_processing_snapshot()
                    st.success(f"Snapshot saved successfully!")
                    
            with col3:
                if st.button("Export Processed Data", key="export_data", use_container_width=True):
                    self._export_processed_data()
    
    def _render_data_cleaning(self):
        """Render data cleaning interface"""
        st.subheader("Data Cleaning")
        
        # Add data preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(self.df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {self.df.shape[0]} rows and all {self.df.shape[1]} columns")
        
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
                
                # Plot missing values
                if not missing_df.empty:
                    fig = px.bar(
                        missing_df, 
                        x='Column', 
                        y='Percentage',
                        title='Missing Values by Column (%)',
                        text_auto='.1f',
                        color='Percentage',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
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
                        "Fill with interpolation",
                        "Fill with prediction (ML)"
                    ]
                )
                
                # Additional input for constant value if selected
                if handling_method == "Fill with constant value":
                    constant_value = st.text_input("Enter constant value:")
                
                # Additional options for ML-based imputation
                if handling_method == "Fill with prediction (ML)":
                    st.markdown("##### Machine Learning Imputation Options")
                    
                    # Select predictors
                    predictor_cols = st.multiselect(
                        "Select predictor columns:",
                        [col for col in self.df.columns if col != col_to_handle and col not in cols_with_missing],
                        default=[col for col in self.df.columns if col != col_to_handle and col not in cols_with_missing][:3]
                    )
                    
                    # Select imputation model
                    imputer_model = st.selectbox(
                        "Select imputation model:",
                        ["Linear Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"]
                    )
                
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
                        
                        elif handling_method == "Fill with prediction (ML)":
                            if not predictor_cols:
                                st.error("Please select at least one predictor column")
                                return
                                
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.linear_model import LinearRegression
                            from sklearn.tree import DecisionTreeRegressor
                            from sklearn.neighbors import KNeighborsRegressor
                            
                            for col in columns_to_process:
                                # Only handle numeric columns
                                if self.df[col].dtype.kind in 'bifc':
                                    # Create train data from rows where target is not null
                                    train_df = self.df[~self.df[col].isna()]
                                    X_train = train_df[predictor_cols]
                                    y_train = train_df[col]
                                    
                                    # Handle non-numeric columns in predictors
                                    for p_col in predictor_cols:
                                        if X_train[p_col].dtype not in ['int64', 'float64']:
                                            # Encode categorical columns with one-hot encoding
                                            X_train = pd.get_dummies(X_train, columns=[p_col], drop_first=True)
                                    
                                    # Select model
                                    if imputer_model == "Linear Regression":
                                        model = LinearRegression()
                                    elif imputer_model == "K-Nearest Neighbors":
                                        model = KNeighborsRegressor(n_neighbors=min(5, len(X_train)))
                                    elif imputer_model == "Decision Tree":
                                        model = DecisionTreeRegressor(max_depth=5)
                                    else:  # Random Forest
                                        model = RandomForestRegressor(n_estimators=100, max_depth=5)
                                    
                                    # Train model
                                    model.fit(X_train, y_train)
                                    
                                    # Create prediction data
                                    pred_df = self.df[self.df[col].isna()]
                                    if not pred_df.empty:
                                        X_pred = pred_df[predictor_cols]
                                        
                                        # Handle non-numeric columns in predictors
                                        for p_col in predictor_cols:
                                            if X_pred[p_col].dtype not in ['int64', 'float64']:
                                                X_pred = pd.get_dummies(X_pred, columns=[p_col], drop_first=True)
                                        
                                        # Make predictions
                                        predictions = model.predict(X_pred)
                                        
                                        # Fill missing values with predictions
                                        self.df.loc[self.df[col].isna(), col] = predictions
                                        
                                        # Add to processing history
                                        st.session_state.processing_history.append({
                                            "description": f"Filled missing values in '{col}' with ML predictions",
                                            "timestamp": datetime.datetime.now(),
                                            "type": "missing_values",
                                            "details": {
                                                "column": col,
                                                "method": "ml_prediction",
                                                "model": imputer_model,
                                                "predictors": predictor_cols
                                            }
                                        })
                                else:
                                    st.warning(f"Column '{col}' is not numeric. Skipping ML imputation.")
                            
                            st.success(f"Filled missing values with ML predictions in {len(columns_to_process)} column(s)")
                    
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        
                        # Autosave if enabled
                        if st.session_state.autosave_enabled:
                            self._save_to_csv()
                            
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
                
                # Duplicate removal options
                dup_keep = st.radio(
                    "When removing duplicates, keep:",
                    ["First occurrence", "Last occurrence", "None (remove all duplicates)"],
                    horizontal=True
                )
                
                keep_map = {
                    "First occurrence": "first",
                    "Last occurrence": "last",
                    "None (remove all duplicates)": False
                }
                
                # Subset selection
                use_subset = st.checkbox("Consider only specific columns for determining duplicates")
                
                if use_subset:
                    subset_cols = st.multiselect(
                        "Select columns to check for duplicates:",
                        self.df.columns.tolist()
                    )
                else:
                    subset_cols = None
                
                # Button to remove duplicates
                if st.button("Remove Duplicate Rows", use_container_width=True):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Remove duplicates
                        self.df = self.df.drop_duplicates(
                            subset=subset_cols,
                            keep=keep_map[dup_keep]
                        )
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        
                        # Add to processing history
                        rows_removed = orig_shape[0] - self.df.shape[0]
                        st.session_state.processing_history.append({
                            "description": f"Removed {rows_removed} duplicate rows",
                            "timestamp": datetime.datetime.now(),
                            "type": "duplicates",
                            "details": {
                                "rows_removed": rows_removed,
                                "keep": keep_map[dup_keep],
                                "subset": subset_cols
                            }
                        })
                        
                        # Autosave if enabled
                        if st.session_state.autosave_enabled:
                            self._save_to_csv()
                            
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
                
                # Choose outlier detection method
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["IQR (Interquartile Range)", "Z-Score", "Modified Z-Score"]
                )
                
                # Calculate outlier boundaries based on selected method
                if outlier_method == "IQR (Interquartile Range)":
                    # IQR method
                    Q1 = self.df[col_for_outliers].quantile(0.25)
                    Q3 = self.df[col_for_outliers].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Allow user to adjust the multiplier
                    iqr_multiplier = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1)
                    
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    
                    # Create outlier mask
                    outlier_mask = (self.df[col_for_outliers] < lower_bound) | (self.df[col_for_outliers] > upper_bound)
                    
                elif outlier_method == "Z-Score":
                    # Z-Score method
                    z_threshold = st.slider("Z-Score Threshold:", 2.0, 5.0, 3.0, 0.1)
                    
                    mean = self.df[col_for_outliers].mean()
                    std = self.df[col_for_outliers].std()
                    
                    z_scores = (self.df[col_for_outliers] - mean) / std
                    outlier_mask = abs(z_scores) > z_threshold
                    
                    lower_bound = mean - z_threshold * std
                    upper_bound = mean + z_threshold * std
                    
                else:  # Modified Z-Score
                    # Modified Z-Score method (more robust to outliers)
                    mod_z_threshold = st.slider("Modified Z-Score Threshold:", 2.0, 5.0, 3.5, 0.1)
                    
                    median = self.df[col_for_outliers].median()
                    mad = np.median(np.abs(self.df[col_for_outliers] - median))
                    
                    if mad > 0:
                        mod_z_scores = 0.6745 * (self.df[col_for_outliers] - median) / mad
                        outlier_mask = abs(mod_z_scores) > mod_z_threshold
                    else:
                        st.warning("Cannot use Modified Z-Score due to zero MAD value. Using IQR method instead.")
                        # Fall back to IQR method
                        Q1 = self.df[col_for_outliers].quantile(0.25)
                        Q3 = self.df[col_for_outliers].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_mask = (self.df[col_for_outliers] < lower_bound) | (self.df[col_for_outliers] > upper_bound)
                    
                    # Set bounds for display
                    lower_bound = self.df[col_for_outliers].min()
                    upper_bound = self.df[col_for_outliers].max()
                    if outlier_mask.any():
                        non_outlier_values = self.df.loc[~outlier_mask, col_for_outliers]
                        if len(non_outlier_values) > 0:
                            lower_bound = non_outlier_values.min()
                            upper_bound = non_outlier_values.max()
                
                # Identify outliers
                outliers = self.df[outlier_mask]
                outlier_count = len(outliers)
                
                # Visualize the distribution with outlier bounds
                fig = px.box(self.df, y=col_for_outliers, title=f"Distribution of {col_for_outliers} with Outlier Bounds")
                fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower bound")
                fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper bound")
                st.plotly_chart(fig, use_container_width=True)
                
                if outlier_count == 0:
                    st.success(f"No outliers found in column '{col_for_outliers}' using {outlier_method}.")
                else:
                    st.warning(f"Found {outlier_count} outliers in column '{col_for_outliers}' using {outlier_method}.")
                    st.write(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                    # Display sample of outliers
                    if st.checkbox("Show sample of outliers"):
                        st.dataframe(outliers.head(5), use_container_width=True)
                    
                    # Options for handling outliers
                    outlier_handling = st.selectbox(
                        "Select handling method:",
                        [
                            "Remove outliers",
                            "Cap outliers (winsorize)",
                            "Replace with NaN",
                            "Replace with median",
                            "Replace with mean",
                            "Log transform column"
                        ]
                    )
                    
                    # Button to handle outliers
                    if st.button("Handle Outliers", use_container_width=True):
                        try:
                            # Store original shape for reporting
                            orig_shape = self.df.shape
                            
                            if outlier_handling == "Remove outliers":
                                # Remove rows with outliers
                                self.df = self.df[~outlier_mask]
                                
                                # Add to processing history
                                rows_removed = orig_shape[0] - self.df.shape[0]
                                st.session_state.processing_history.append({
                                    "description": f"Removed {rows_removed} outliers from column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "remove",
                                        "detection": outlier_method,
                                        "rows_affected": rows_removed
                                    }
                                })
                                
                                st.success(f"Removed {outlier_count} outliers from column '{col_for_outliers}'")
                                
                            elif outlier_handling == "Cap outliers (winsorize)":
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
                                        "detection": outlier_method,
                                        "lower_bound": lower_bound,
                                        "upper_bound": upper_bound,
                                        "values_affected": outlier_count
                                    }
                                })
                                
                                st.success(f"Capped {outlier_count} outliers in column '{col_for_outliers}'")
                                
                            elif outlier_handling == "Replace with NaN":
                                # Replace outliers with NaN
                                self.df.loc[outlier_mask, col_for_outliers] = np.nan
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "replace_nan",
                                        "detection": outlier_method,
                                        "values_affected": outlier_count
                                    }
                                })
                                
                                st.success(f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'")
                                
                            elif outlier_handling == "Replace with median":
                                # Get median value
                                median_val = self.df[col_for_outliers].median()
                                
                                # Replace outliers with median
                                self.df.loc[outlier_mask, col_for_outliers] = median_val
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Replaced {outlier_count} outliers with median in column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "replace_median",
                                        "detection": outlier_method,
                                        "values_affected": outlier_count,
                                        "median": median_val
                                    }
                                })
                                
                                st.success(f"Replaced {outlier_count} outliers with median ({median_val:.2f}) in column '{col_for_outliers}'")
                            
                            elif outlier_handling == "Replace with mean":
                                # Get mean from non-outlier values
                                mean_val = self.df.loc[~outlier_mask, col_for_outliers].mean()
                                
                                # Replace outliers with mean
                                self.df.loc[outlier_mask, col_for_outliers] = mean_val
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Replaced {outlier_count} outliers with mean in column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "replace_mean",
                                        "detection": outlier_method,
                                        "values_affected": outlier_count,
                                        "mean": mean_val
                                    }
                                })
                                
                                st.success(f"Replaced {outlier_count} outliers with mean ({mean_val:.2f}) in column '{col_for_outliers}'")
                                
                            elif outlier_handling == "Log transform column":
                                # Check if column has non-positive values
                                min_val = self.df[col_for_outliers].min()
                                
                                if min_val <= 0:
                                    # Add a constant to make all values positive
                                    shift_val = abs(min_val) + 1
                                    self.df[col_for_outliers] = np.log(self.df[col_for_outliers] + shift_val)
                                    
                                    # Add to processing history
                                    st.session_state.processing_history.append({
                                        "description": f"Applied log transform to '{col_for_outliers}' after shifting by {shift_val}",
                                        "timestamp": datetime.datetime.now(),
                                        "type": "transformation",
                                        "details": {
                                            "column": col_for_outliers,
                                            "method": "log_transform",
                                            "shift": shift_val
                                        }
                                    })
                                else:
                                    # No shift needed
                                    self.df[col_for_outliers] = np.log(self.df[col_for_outliers])
                                    
                                    # Add to processing history
                                    st.session_state.processing_history.append({
                                        "description": f"Applied log transform to '{col_for_outliers}'",
                                        "timestamp": datetime.datetime.now(),
                                        "type": "transformation",
                                        "details": {
                                            "column": col_for_outliers,
                                            "method": "log_transform"
                                        }
                                    })
                                
                                st.success(f"Applied log transform to column '{col_for_outliers}' to handle outliers")
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            
                            # Autosave if enabled
                            if st.session_state.autosave_enabled:
                                self._save_to_csv()
                                
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error handling outliers: {str(e)}")
            
            # Data Type Conversion
            st.markdown("### Data Type Conversion")
            
            # Show current data types
            st.write("Current Data Types:")
            dtypes_df = pd.DataFrame({
                'Column': self.df.columns,
                'Type': [str(self.df[col].dtype) for col in self.df.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)
            
            # Column selection for type conversion
            col_to_convert = st.selectbox(
                "Select column to convert:",
                self.df.columns,
                key="dtype_convert_col"
            )
            
            # Target data type selection
            target_type = st.selectbox(
                "Convert to type:",
                ["int", "float", "string", "category", "datetime", "boolean", "timedelta"]
            )
            
            # DateTime format if target type is datetime
            date_format = None
            if target_type == "datetime":
                date_format = st.text_input(
                    "Enter date format (e.g., '%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):",
                    placeholder="Leave blank to auto-detect"
                )
                
                # Add common date format examples
                st.markdown("""
                **Common date formats:**
                - '%Y-%m-%d' (2023-01-31)
                - '%d/%m/%Y' (31/01/2023)
                - '%m/%d/%Y' (01/31/2023)
                - '%Y-%m-%d %H:%M:%S' (2023-01-31 14:30:45)
                """)
            
            # Apply conversion button
            if st.button("Convert Data Type", use_container_width=True, key="convert_dtype_btn"):
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
                    elif target_type == "timedelta":
                        # Convert to timedelta (assuming input in seconds or a standard format)
                        self.df[col_to_convert] = pd.to_timedelta(self.df[col_to_convert])
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Converted column '{col_to_convert}' from {current_type} to {target_type}",
                        "timestamp": datetime.datetime.now(),
                        "type": "data_type_conversion",
                        "details": {
                            "column": col_to_convert,
                            "from_type": current_type,
                            "to_type": target_type,
                            "date_format": date_format if target_type == "datetime" and date_format else None
                        }
                    })
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    
                    # Autosave if enabled
                    if st.session_state.autosave_enabled:
                        self._save_to_csv()
                        
                    st.success(f"Converted column '{col_to_convert}' to {target_type}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
                    st.info("Try a different target type or check the data format")

    def _render_data_transformation(self):
        """Render data transformation interface"""
        st.subheader("Data Transformation")
        
        # Add data preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(self.df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {self.df.shape[0]} rows and all {self.df.shape[1]} columns")
        
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
                        "Exponential Transform",
                        "Normalization",
                        "Cumulative Sum/Product"
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
                
                elif transform_method == "Cumulative Sum/Product":
                    cum_method = st.radio("Method:", ["Cumulative Sum", "Cumulative Product"], horizontal=True)
                
                # Create new column or replace
                create_new_col = st.checkbox("Create new column", value=True)
                
                # Preview transformation
                preview_btn = st.button("Preview Transformation", key="preview_transform")
                
                if preview_btn:
                    try:
                        # Create temporary dataframe for preview
                        temp_df = self.df.copy()
                        
                        # Determine output column name for preview
                        if create_new_col:
                            output_col = f"{col_to_transform}_{transform_method.split(' ')[0].lower()}"
                        else:
                            output_col = col_to_transform
                        
                        # Apply transformation for preview
                        self._apply_numeric_transformation(
                            temp_df,
                            col_to_transform,
                            transform_method,
                            output_col,
                            locals()
                        )
                        
                        # Show preview
                        st.subheader("Transformation Preview")
                        
                        # Show statistics before and after
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Original Column: {col_to_transform}**")
                            original_stats = self.df[col_to_transform].describe()
                            st.dataframe(pd.DataFrame(original_stats), use_container_width=True)
                            
                        with col2:
                            st.markdown(f"**Transformed Column: {output_col}**")
                            if output_col in temp_df.columns:
                                transformed_stats = temp_df[output_col].describe()
                                st.dataframe(pd.DataFrame(transformed_stats), use_container_width=True)
                        
                        # Plot comparison
                        fig = go.Figure()
                        
                        # Add original data
                        fig.add_trace(go.Histogram(
                            x=self.df[col_to_transform],
                            name="Original",
                            opacity=0.7,
                            marker_color='blue'
                        ))
                        
                        # Add transformed data if it exists
                        if output_col in temp_df.columns:
                            fig.add_trace(go.Histogram(
                                x=temp_df[output_col],
                                name="Transformed",
                                opacity=0.7,
                                marker_color='red'
                            ))
                        
                        fig.update_layout(
                            title=f"Distribution Comparison: {transform_method}",
                            xaxis_title="Value",
                            yaxis_title="Count",
                            barmode='overlay',
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error previewing transformation: {str(e)}")
                
                # Apply button
                if st.button("Apply Transformation", key="apply_transform", use_container_width=True):
                    try:
                        # Determine output column name
                        if create_new_col:
                            output_col = f"{col_to_transform}_{transform_method.split(' ')[0].lower()}"
                        else:
                            output_col = col_to_transform
                        
                        # Apply the transformation
                        self._apply_numeric_transformation(
                            self.df,
                            col_to_transform,
                            transform_method,
                            output_col,
                            locals()
                        )
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        
                        # Autosave if enabled
                        if st.session_state.autosave_enabled:
                            self._save_to_csv()
                            
                        st.success(f"{'Created new column' if create_new_col else 'Transformed'} '{output_col}' with {transform_method.lower()}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
                        
    def _apply_numeric_transformation(self, df, col_to_transform, transform_method, output_col, params):
        """Apply the numeric transformation to the dataframe"""
        if transform_method == "Standardization (Z-score)":
            # Z-score standardization: (x - mean) / std
            data = df[col_to_transform].values.reshape(-1, 1)
            scaler = StandardScaler()
            transformed_data = scaler.fit_transform(data).flatten()
            
            # Handle potential NaN values
            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
            
            # Store transformation parameters
            mean = scaler.mean_[0]
            std = scaler.scale_[0]
            
            df[output_col] = transformed_data
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied standardization to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "standardization",
                    "new_column": output_col if output_col != col_to_transform else None,
                    "mean": mean,
                    "std": std
                }
            })
        
        elif transform_method == "Min-Max Scaling":
            # Min-Max scaling: (x - min) / (max - min)
            data = df[col_to_transform].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            transformed_data = scaler.fit_transform(data).flatten()
            
            # Handle potential NaN values
            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmean(transformed_data))
            
            # Store transformation parameters
            min_val = scaler.data_min_[0]
            max_val = scaler.data_max_[0]
            
            df[output_col] = transformed_data
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied min-max scaling to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "min_max_scaling",
                    "new_column": output_col if output_col != col_to_transform else None,
                    "min": min_val,
                    "max": max_val
                }
            })
            
        elif transform_method == "Robust Scaling":
            # Robust scaling using median and IQR (less sensitive to outliers)
            data = df[col_to_transform].values.reshape(-1, 1)
            scaler = RobustScaler()
            transformed_data = scaler.fit_transform(data).flatten()
            
            # Handle potential NaN values
            transformed_data = np.nan_to_num(transformed_data, nan=np.nanmedian(transformed_data))
            
            df[output_col] = transformed_data
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied robust scaling to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "robust_scaling",
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })
        
        elif transform_method == "Log Transform":
            # Check for non-positive values
            min_val = df[col_to_transform].min()
            
            if min_val <= 0:
                # Add a constant to make all values positive
                const = abs(min_val) + 1
                df[output_col] = np.log(df[col_to_transform] + const)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Applied log transform to '{col_to_transform}' with constant {const}",
                    "timestamp": datetime.datetime.now(),
                    "type": "transformation",
                    "details": {
                        "column": col_to_transform,
                        "method": "log_transform",
                        "constant": const,
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
            else:
                df[output_col] = np.log(df[col_to_transform])
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Applied log transform to '{col_to_transform}'",
                    "timestamp": datetime.datetime.now(),
                    "type": "transformation",
                    "details": {
                        "column": col_to_transform,
                        "method": "log_transform",
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
        
        elif transform_method == "Square Root Transform":
            # Check for negative values
            min_val = df[col_to_transform].min()
            
            if min_val < 0:
                # Add a constant to make all values non-negative
                const = abs(min_val) + 1
                df[output_col] = np.sqrt(df[col_to_transform] + const)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Applied square root transform to '{col_to_transform}' with constant {const}",
                    "timestamp": datetime.datetime.now(),
                    "type": "transformation",
                    "details": {
                        "column": col_to_transform,
                        "method": "sqrt_transform",
                        "constant": const,
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
            else:
                df[output_col] = np.sqrt(df[col_to_transform])
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Applied square root transform to '{col_to_transform}'",
                    "timestamp": datetime.datetime.now(),
                    "type": "transformation",
                    "details": {
                        "column": col_to_transform,
                        "method": "sqrt_transform",
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
        
        elif transform_method == "Box-Cox Transform":
            # Check if all values are positive
            min_val = df[col_to_transform].min()
            
            if min_val <= 0:
                # Add a constant to make all values positive
                const = abs(min_val) + 1
                
                from scipy import stats
                transformed_data, lambda_val = stats.boxcox(df[col_to_transform] + const)
                df[output_col] = transformed_data
                
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
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
            else:
                from scipy import stats
                transformed_data, lambda_val = stats.boxcox(df[col_to_transform])
                df[output_col] = transformed_data
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Applied Box-Cox transform to '{col_to_transform}'",
                    "timestamp": datetime.datetime.now(),
                    "type": "transformation",
                    "details": {
                        "column": col_to_transform,
                        "method": "boxcox_transform",
                        "lambda": lambda_val,
                        "new_column": output_col if output_col != col_to_transform else None
                    }
                })
        
        elif transform_method == "Binning/Discretization":
            # Get parameters from locals
            num_bins = params.get('num_bins', 5)
            bin_strategy = params.get('bin_strategy', 'uniform')
            bin_labels_str = params.get('bin_labels', '')
            bin_labels = bin_labels_str.split(',') if bin_labels_str else None
            
            # Create bins based on strategy
            if bin_strategy == "uniform":
                # Uniform binning
                bins = pd.cut(
                    df[col_to_transform], 
                    bins=num_bins, 
                    labels=bin_labels if bin_labels else None
                )
            elif bin_strategy == "quantile":
                # Quantile-based binning
                bins = pd.qcut(
                    df[col_to_transform], 
                    q=num_bins, 
                    labels=bin_labels if bin_labels else None,
                    duplicates='drop'
                )
            elif bin_strategy == "kmeans":
                # K-means clustering binning
                from sklearn.cluster import KMeans
                
                # Reshape data for KMeans
                data = df[col_to_transform].values.reshape(-1, 1)
                
                # Fit KMeans
                kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(data)
                
                # Get cluster labels
                clusters = kmeans.labels_
                
                # Create custom bin labels if provided
                if bin_labels:
                    label_map = {i: label for i, label in enumerate(bin_labels[:num_bins])}
                    bins = pd.Series([label_map.get(c, f"Cluster {c}") for c in clusters], index=df.index)
                else:
                    bins = pd.Series([f"Cluster {c}" for c in clusters], index=df.index)
            
            df[output_col] = bins
            
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
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })
        
        elif transform_method == "Power Transform":
            # Get power parameter
            power = params.get('power', 1.0)
            
            # Apply power transform: x^power
            df[output_col] = np.power(df[col_to_transform], power)
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied power transform to '{col_to_transform}' with power {power}",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "power_transform",
                    "power": power,
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })
            
        elif transform_method == "Exponential Transform":
            # Apply exponential transform: e^x
            df[output_col] = np.exp(df[col_to_transform])
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied exponential transform to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "exp_transform",
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })
        
        elif transform_method == "Normalization":
            # Normalize to sum to 1
            total = df[col_to_transform].sum()
            df[output_col] = df[col_to_transform] / total
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied normalization to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": "normalization",
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })
            
        elif transform_method == "Cumulative Sum/Product":
            # Get method parameter
            cum_method = params.get('cum_method', 'Cumulative Sum')
            
            if cum_method == "Cumulative Sum":
                df[output_col] = df[col_to_transform].cumsum()
                method_name = "cumulative sum"
            else:  # Cumulative Product
                df[output_col] = df[col_to_transform].cumprod()
                method_name = "cumulative product"
            
            # Add to processing history
            st.session_state.processing_history.append({
                "description": f"Applied {method_name} to '{col_to_transform}'",
                "timestamp": datetime.datetime.now(),
                "type": "transformation",
                "details": {
                    "column": col_to_transform,
                    "method": method_name.replace(" ", "_"),
                    "new_column": output_col if output_col != col_to_transform else None
                }
            })

    def _render_feature_engineering(self):
        """Render feature engineering interface"""
        # This will be implemented in part 2
        st.info("Feature Engineering functionality is available in Part 2 of the DataProcessor module.")
        
    def _render_data_filtering(self):
        """Render data filtering interface"""
        # This will be implemented in part 2
        st.info("Data Filtering functionality is available in Part 2 of the DataProcessor module.")
    
    def _render_column_management(self):
        """Render column management interface"""
        # This will be implemented in part 2
        st.info("Column Management functionality is available in Part 2 of the DataProcessor module.")
    
    def _render_export_and_save(self):
        """Render export and save interface"""
        # This will be implemented in part 2
        st.info("Data Export & Save functionality is available in Part 2 of the DataProcessor module.")
    
    def _export_processed_data(self):
        """Export processed data to various formats"""
        # This will be implemented in part 2
        st.info("Export Data functionality is available in Part 2 of the DataProcessor module.")
    
    def _save_to_csv(self):
        """Save current dataframe to CSV"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Save to CSV
            self.df.to_csv("data/processed_data.csv", index=False)
            return True
        except Exception as e:
            st.error(f"Error saving to CSV: {str(e)}")
            return False
            
    def _save_processing_snapshot(self):
        """Save a snapshot of the current processing state"""
        try:
            # Create snapshots directory if it doesn't exist
            os.makedirs("data/snapshots", exist_ok=True)
            
            # Create timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save dataframe
            csv_path = f"data/snapshots/data_snapshot_{timestamp}.csv"
            self.df.to_csv(csv_path, index=False)
            
            # Save processing history
            history_path = f"data/snapshots/history_snapshot_{timestamp}.json"
            with open(history_path, 'w') as f:
                json.dump(st.session_state.processing_history, f, default=str)
                
            return timestamp
            
        except Exception as e:
            st.error(f"Error saving snapshot: {str(e)}")
            return None
            
    def _revert_to_step(self, step_index):
        """Revert to before a specific processing step"""
        if step_index < 0 or step_index >= len(st.session_state.processing_history):
            st.error(f"Invalid step index: {step_index}")
            return False
            
        try:
            # Reset to original data
            self.df = self.original_df.copy()
            
            # Reapply all steps up to but not including the specified step
            for i in range(step_index):
                # Logic to reapply step i would go here
                # This is complex and would require storing more details about each operation
                # For now, we'll just update the history
                pass
                
            # Update history to remove steps from step_index onwards
            st.session_state.processing_history = st.session_state.processing_history[:step_index]
            
            # Update the dataframe in session state
            st.session_state.df = self.df
            
            # Autosave if enabled
            if st.session_state.autosave_enabled:
                self._save_to_csv()
                
            return True
            
        except Exception as e:
            st.error(f"Error reverting to step: {str(e)}")
            return False