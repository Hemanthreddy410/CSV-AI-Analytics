import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from modules.data_exporter import DataExporter

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
                ["Mean", "Median", "Min", "Max", "Custom"],
                key="reference_type"
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
                filter_col = st.selectbox("Column", all_cols, key="filter_col_12")
            
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
            combine_op = st.radio("Combine conditions with:", ["AND", "OR"], horizontal=True),
        key="combine_op"
        
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
                horizontal=True,
                key="sample_size_type"
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
                horizontal=True,
                key="select_type"
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
            horizontal=True,
            key="rename_method"
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
                horizontal=True,
                key="apply_to"
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
            horizontal=True,
            key="column_operation"
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
            horizontal=True,
            key="selection_method"
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
            horizontal=True,
            key="reorder_method"
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
                horizontal=True,
                key="sort_order"
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
                horizontal=True,
                key="move_direction"
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
                        horizontal=True,
                        key="math_operation",
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