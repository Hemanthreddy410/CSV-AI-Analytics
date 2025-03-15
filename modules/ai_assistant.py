import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import time
import uuid
import base64
from datetime import datetime
from scipy import stats
import re

class AIAssistant:
    """AI-powered data analysis assistant"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.session_id = str(uuid.uuid4())  # Generate a unique session ID
        
        # Store chat history in session state if not already there
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        # Store analysis cache
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
            
        # Store dynamic filters
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
            
        # Store data preprocessing state
        if 'preprocessing' not in st.session_state:
            st.session_state.preprocessing = {
                'original_df': None,
                'steps_applied': []
            }
            # Store original data for reset
            if self.df is not None and not self.df.empty:
                st.session_state.preprocessing['original_df'] = self.df.copy()
    
    def render_interface(self):
        """Render the AI assistant interface"""
        st.header("Advanced Data Assistant")
        
        # Mode selector
        modes = ["Chat Interface", "Data Browser", "Data Preprocessing", "Custom Analysis"]
        selected_mode = st.radio("Select Mode", modes, horizontal=True, key="mode_selector")
        
        if selected_mode == "Chat Interface":
            self._render_chat_interface()
        elif selected_mode == "Data Browser":
            self._render_data_browser()
        elif selected_mode == "Data Preprocessing":
            self._render_preprocessing_interface()
        elif selected_mode == "Custom Analysis":
            self._render_custom_analysis()
    
    def _render_chat_interface(self):
        """Render the chat interface for data analysis"""
        # Features section
        with st.expander("✨ What can the Data Assistant do?", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Ask questions about your data:**
                - 📊 "Summarize this dataset"
                - 🔍 "Find correlations between columns X and Y"
                - 📈 "What insights can you provide from this data?"
                - 🧮 "Calculate the average of column Z"
                - 📉 "Show me trends in this data"
                """)
            
            with col2:
                st.markdown("""
                **Generate visualizations:**
                - 📊 "Create a bar chart of column X"
                - 📈 "Show me a scatter plot of X vs Y"
                - 🔄 "Visualize the distribution of column Z"
                - 🧩 "Create a correlation heatmap"
                - 📉 "Make a line chart showing trends in column X"
                """)
        
        # Chat interface
        st.subheader("Ask me about your data")
        
        # Display chat history
        self._render_chat_history()
        
        # Input for new message
        user_input = st.text_area("Type your question here:", key="user_query_chat", height=100)
        
        # Suggested questions
        st.markdown("**⚡ Suggested questions:**")
        suggestion_cols = st.columns(3)
        
        suggestions = [
            "Summarize this dataset",
            "What are the main insights?",
            "Show correlations between columns",
            "Identify outliers in the data",
            "Visualize the distribution of numeric columns",
            "What trends can you identify?"
        ]
        
        # Add column-specific suggestions if we have data
        if self.df is not None:
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if num_cols and len(num_cols) >= 2:
                suggestions.append(f"Show a scatter plot of {num_cols[0]} vs {num_cols[1]}")
            
            if cat_cols and num_cols:
                suggestions.append(f"Compare {num_cols[0]} across different {cat_cols[0]} values")
        
        # Display suggestion buttons with guaranteed unique keys
        for i, suggestion in enumerate(suggestions):
            col_idx = i % 3
            with suggestion_cols[col_idx]:
                # Ensure unique key by using session_id and suggestion index
                unique_key = f"suggestion_{self.session_id}_{i}"
                if st.button(suggestion, key=unique_key, use_container_width=True):
                    # Use the suggestion as input
                    user_input = suggestion
        
        # Submit button with unique key
        submit_key = f"submit_chat_{self.session_id}"
        if st.button("Submit", key=submit_key, use_container_width=True, type="primary") or user_input:
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Process the query and generate a response
                with st.spinner("Thinking..."):
                    response = self._generate_response(user_input)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["text"],
                    "visualization": response.get("visualization", None),
                    "viz_type": response.get("viz_type", None)
                })
                
                # Clear input
                st.rerun()
    
    def _render_data_browser(self):
        """Render interface for browsing and exploring data"""
        st.subheader("Data Browser")
        
        if self.df is None or self.df.empty:
            st.warning("No data available. Please upload a dataset first.")
            return
            
        # DataFrame info
        rows, cols = self.df.shape
        
        with st.expander("Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{rows:,}")
            col2.metric("Columns", cols)
            col3.metric("Missing Values", f"{self.df.isna().sum().sum():,}")
            
            # Column types
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
            
            st.text(f"Numeric columns: {len(numeric_cols)}")
            st.text(f"Categorical columns: {len(categorical_cols)}")
            st.text(f"Date columns: {len(date_cols)}")
        
        # Data filtering
        with st.expander("Filter Data", expanded=False):
            st.subheader("Add Filters")
            
            # Add filter
            col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
            new_filter_col = col1.selectbox("Column", self.df.columns, key=f"filter_col_{uuid.uuid4()}")
            
            # Determine filter type based on column data type
            filter_type = None
            if new_filter_col in numeric_cols:
                filter_type = col2.selectbox("Filter Type", ["Range", "Equal to", "Greater than", "Less than"], key=f"filter_type_{uuid.uuid4()}")
            else:
                filter_type = col2.selectbox("Filter Type", ["Contains", "Equal to", "Not equal to"], key=f"filter_type_{uuid.uuid4()}")
            
            # Filter value input based on type
            filter_value = None
            if filter_type == "Range" and new_filter_col in numeric_cols:
                min_val = float(self.df[new_filter_col].min())
                max_val = float(self.df[new_filter_col].max())
                filter_value = col3.slider("Value Range", min_val, max_val, (min_val, max_val), key=f"filter_value_{uuid.uuid4()}")
            elif new_filter_col in numeric_cols:
                filter_value = col3.number_input("Value", value=float(self.df[new_filter_col].mean()), key=f"filter_value_{uuid.uuid4()}")
            elif new_filter_col in categorical_cols:
                unique_values = self.df[new_filter_col].dropna().unique().tolist()
                if len(unique_values) <= 10:
                    filter_value = col3.selectbox("Value", unique_values, key=f"filter_value_{uuid.uuid4()}")
                else:
                    filter_value = col3.text_input("Value", key=f"filter_value_{uuid.uuid4()}")
            
            # Add filter button
            if col4.button("Add", key=f"add_filter_{uuid.uuid4()}"):
                filter_id = str(uuid.uuid4())
                st.session_state.filters[filter_id] = {
                    "column": new_filter_col,
                    "type": filter_type,
                    "value": filter_value
                }
                st.rerun()
            
            # Display active filters
            if st.session_state.filters:
                st.subheader("Active Filters")
                
                for filter_id, filter_info in list(st.session_state.filters.items()):
                    col1, col2 = st.columns([5, 1])
                    
                    filter_text = f"{filter_info['column']} | {filter_info['type']} | "
                    if isinstance(filter_info['value'], tuple):
                        filter_text += f"{filter_info['value'][0]} to {filter_info['value'][1]}"
                    else:
                        filter_text += f"{filter_info['value']}"
                    
                    col1.text(filter_text)
                    if col2.button("Remove", key=f"remove_{filter_id}"):
                        del st.session_state.filters[filter_id]
                        st.rerun()
                
                # Apply filters
                filtered_df = self.df.copy()
                
                for filter_info in st.session_state.filters.values():
                    col, filter_type, value = filter_info["column"], filter_info["type"], filter_info["value"]
                    
                    if filter_type == "Range":
                        filtered_df = filtered_df[(filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])]
                    elif filter_type == "Equal to":
                        filtered_df = filtered_df[filtered_df[col] == value]
                    elif filter_type == "Not equal to":
                        filtered_df = filtered_df[filtered_df[col] != value]
                    elif filter_type == "Greater than":
                        filtered_df = filtered_df[filtered_df[col] > value]
                    elif filter_type == "Less than":
                        filtered_df = filtered_df[filtered_df[col] < value]
                    elif filter_type == "Contains":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(value), case=False, na=False)]
                
                # Clear filters button
                if st.button("Clear All Filters", key="clear_all_filters"):
                    st.session_state.filters = {}
                    st.rerun()
                
                # Show filter stats
                original_count = len(self.df)
                filtered_count = len(filtered_df)
                st.metric("Filtered Rows", f"{filtered_count:,}", f"{filtered_count - original_count:,}")
            else:
                filtered_df = self.df
        
        # Display the actual data table
        st.subheader("Data Preview")
        
        # Column selector
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display", 
            all_columns, 
            default=all_columns[:min(5, len(all_columns))],
            key="column_selector"
        )
        
        # Row limiter
        num_rows = len(filtered_df)
        display_rows = st.slider("Number of rows to display", 5, min(1000, num_rows), 50, key="row_slider")
        
        # Display dataframe
        if selected_columns:
            st.dataframe(filtered_df[selected_columns].head(display_rows))
        else:
            st.dataframe(filtered_df.head(display_rows))
        
        # Data summary
        with st.expander("Data Summary Statistics", expanded=False):
            if numeric_cols:
                st.write(filtered_df[numeric_cols].describe())
            
            if categorical_cols:
                st.subheader("Categorical Variables")
                for col in categorical_cols[:5]:  # Limit to 5 categorical columns
                    st.write(f"**{col}**")
                    st.write(filtered_df[col].value_counts().head(10))
        
        # Download filtered data
        if st.button("Download Filtered Data as CSV", key="download_filtered"):
            csv = filtered_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filtered_data_{timestamp}.csv"
            
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def _render_preprocessing_interface(self):
        """Render interface for data preprocessing"""
        st.subheader("Data Preprocessing")
        
        if self.df is None or self.df.empty:
            st.warning("No data available. Please upload a dataset first.")
            return
        
        # Check if we have original data stored
        if st.session_state.preprocessing['original_df'] is None:
            st.session_state.preprocessing['original_df'] = self.df.copy()
        
        # Show current preprocessing steps
        if st.session_state.preprocessing['steps_applied']:
            with st.expander("Applied Preprocessing Steps", expanded=True):
                for i, step in enumerate(st.session_state.preprocessing['steps_applied']):
                    st.text(f"{i+1}. {step}")
                
                if st.button("Reset to Original Data", key="reset_preprocessing"):
                    self.df = st.session_state.preprocessing['original_df'].copy()
                    st.session_state.preprocessing['steps_applied'] = []
                    st.rerun()
        
        # Data transformation options
        transform_options = [
            "Handle Missing Values",
            "Remove Outliers",
            "Normalize/Scale Data",
            "Convert Data Types",
            "Extract Features from Date/Time",
            "Create Calculated Column",
            "Filter Rows",
            "One-Hot Encode Categorical Variables"
        ]
        
        selected_transform = st.selectbox(
            "Select Preprocessing Action",
            ["Select an action..."] + transform_options,
            key="transform_selector"
        )
        
        if selected_transform == "Handle Missing Values":
            self._render_missing_values_handler()
        elif selected_transform == "Remove Outliers":
            self._render_outlier_removal()
        elif selected_transform == "Normalize/Scale Data":
            self._render_data_scaling()
        elif selected_transform == "Convert Data Types":
            self._render_type_conversion()
        elif selected_transform == "Extract Features from Date/Time":
            self._render_datetime_extraction()
        elif selected_transform == "Create Calculated Column":
            self._render_calculated_column()
        elif selected_transform == "Filter Rows":
            self._render_row_filter()
        elif selected_transform == "One-Hot Encode Categorical Variables":
            self._render_one_hot_encoding()
        
        # Show data preview after preprocessing
        st.subheader("Data Preview After Preprocessing")
        st.dataframe(self.df.head(10))
        
        # Show basic stats
        with st.expander("Data Statistics After Preprocessing", expanded=False):
            if not self.df.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{len(self.df):,}")
                col2.metric("Columns", f"{self.df.shape[1]:,}")
                col3.metric("Missing Values", f"{self.df.isna().sum().sum():,}")
                
                # Show data types
                st.subheader("Data Types")
                st.write(self.df.dtypes)
    
    def _render_missing_values_handler(self):
        """Handle missing values in the dataset"""
        st.subheader("Handle Missing Values")
        
        # Get columns with missing values
        missing_cols = [col for col in self.df.columns if self.df[col].isna().any()]
        
        if not missing_cols:
            st.info("No missing values found in the dataset!")
            return
        
        st.write(f"Found {len(missing_cols)} columns with missing values:")
        
        # Show missing value counts
        missing_df = pd.DataFrame({
            'Column': missing_cols,
            'Missing Count': [self.df[col].isna().sum() for col in missing_cols],
            'Missing Percentage': [self.df[col].isna().sum() / len(self.df) * 100 for col in missing_cols]
        })
        
        st.dataframe(missing_df)
        
        # Strategy selection
        col1, col2 = st.columns(2)
        
        selected_col = col1.selectbox("Select column to process", missing_cols, key="missing_col")
        
        # Get column data type to recommend appropriate methods
        col_dtype = self.df[selected_col].dtype
        
        if np.issubdtype(col_dtype, np.number):
            strategies = ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with constant value"]
        else:
            strategies = ["Drop rows", "Fill with mode", "Fill with constant value"]
        
        strategy = col2.selectbox("Select strategy", strategies, key="missing_strategy")
        
        # Additional input if needed
        if strategy == "Fill with constant value":
            if np.issubdtype(col_dtype, np.number):
                fill_value = st.number_input("Fill value", value=0, key="fill_value_num")
            else:
                fill_value = st.text_input("Fill value", value="unknown", key="fill_value_text")
        
        # Apply button
        if st.button("Apply Missing Value Strategy", key="apply_missing"):
            original_count = len(self.df)
            
            if strategy == "Drop rows":
                self.df = self.df.dropna(subset=[selected_col])
                step_text = f"Dropped rows with missing values in '{selected_col}'"
            
            elif strategy == "Fill with mean":
                mean_value = self.df[selected_col].mean()
                self.df[selected_col] = self.df[selected_col].fillna(mean_value)
                step_text = f"Filled missing values in '{selected_col}' with mean ({mean_value:.2f})"
            
            elif strategy == "Fill with median":
                median_value = self.df[selected_col].median()
                self.df[selected_col] = self.df[selected_col].fillna(median_value)
                step_text = f"Filled missing values in '{selected_col}' with median ({median_value:.2f})"
            
            elif strategy == "Fill with mode":
                mode_value = self.df[selected_col].mode().iloc[0]
                self.df[selected_col] = self.df[selected_col].fillna(mode_value)
                step_text = f"Filled missing values in '{selected_col}' with mode ({mode_value})"
            
            elif strategy == "Fill with constant value":
                self.df[selected_col] = self.df[selected_col].fillna(fill_value)
                step_text = f"Filled missing values in '{selected_col}' with constant ({fill_value})"
            
            # Store the preprocessing step
            st.session_state.preprocessing['steps_applied'].append(step_text)
            
            # Show success message
            new_count = len(self.df)
            if strategy == "Drop rows":
                st.success(f"Successfully applied strategy. Removed {original_count - new_count} rows.")
            else:
                st.success(f"Successfully filled {self.df[selected_col].isna().sum()} missing values in '{selected_col}'.")
            
            # Rerun to update the interface
            st.rerun()
    
    def _render_outlier_removal(self):
        """Handle outliers in the dataset"""
        st.subheader("Remove Outliers")
        
        # Get numeric columns for outlier detection
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for outlier detection!")
            return
        
        # Column selection
        selected_col = st.selectbox("Select column to check for outliers", numeric_cols, key="outlier_col")
        
        # Calculate basic stats for the selected column
        Q1 = self.df[selected_col].quantile(0.25)
        Q3 = self.df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = self.df[(self.df[selected_col] < lower_bound) | (self.df[selected_col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(self.df)) * 100
        
        # Display outlier information
        st.write(f"Outlier Detection for '{selected_col}':")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Outliers", f"{outlier_count:,}")
        col2.metric("Percentage", f"{outlier_percentage:.2f}%")
        col3.metric("Non-outlier Range", f"[{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Visualize the distribution with outliers highlighted
        fig = px.box(self.df, y=selected_col, title=f"Box Plot for {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier handling strategy
        st.subheader("Outlier Handling Strategy")
        
        strategy = st.radio(
            "Select strategy",
            ["Remove outlier rows", "Cap outliers at boundaries", "Replace with NaN (for later imputation)"],
            key="outlier_strategy"
        )
        
        # Apply button
        if st.button("Apply Outlier Strategy", key="apply_outlier"):
            original_count = len(self.df)
            
            if strategy == "Remove outlier rows":
                self.df = self.df[(self.df[selected_col] >= lower_bound) & (self.df[selected_col] <= upper_bound)]
                step_text = f"Removed {outlier_count} outliers from '{selected_col}'"
            
            elif strategy == "Cap outliers at boundaries":
                self.df[selected_col] = self.df[selected_col].clip(lower_bound, upper_bound)
                step_text = f"Capped {outlier_count} outliers in '{selected_col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]"
            
            elif strategy == "Replace with NaN (for later imputation)":
                mask = (self.df[selected_col] < lower_bound) | (self.df[selected_col] > upper_bound)
                self.df.loc[mask, selected_col] = np.nan
                step_text = f"Replaced {outlier_count} outliers in '{selected_col}' with NaN values"
            
            # Store the preprocessing step
            st.session_state.preprocessing['steps_applied'].append(step_text)
            
            # Show success message
            new_count = len(self.df)
            if strategy == "Remove outlier rows":
                st.success(f"Successfully removed {original_count - new_count} rows with outliers.")
            else:
                st.success(f"Successfully processed {outlier_count} outliers in '{selected_col}'.")
            
            # Rerun to update the interface
            st.rerun()
    
    def _render_data_scaling(self):
        """Scale numeric data in the dataset"""
        st.subheader("Normalize/Scale Data")
        
        # Get numeric columns for scaling
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for scaling!")
            return
        
        # Column selection (multiple)
        selected_cols = st.multiselect("Select columns to scale", numeric_cols, key="scale_cols")
        
        if not selected_cols:
            st.info("Please select at least one column to scale.")
            return
        
        # Scaling method
        scaling_method = st.radio(
            "Select scaling method",
            ["Min-Max Scaling (0-1)", "Standard Scaling (z-score)", "Robust Scaling (using quantiles)"],
            key="scaling_method"
        )
        
        # Preview before and after
        if selected_cols:
            st.subheader("Preview Before Scaling")
            
            # Show stats before scaling
            stats_before = self.df[selected_cols].describe().T
            st.dataframe(stats_before)
            
            # Show visualization before scaling
            fig_before = go.Figure()
            for col in selected_cols:
                fig_before.add_trace(go.Box(y=self.df[col], name=col))
            fig_before.update_layout(title="Distribution Before Scaling")
            st.plotly_chart(fig_before, use_container_width=True)
        
        # Apply button
        if st.button("Apply Scaling", key="apply_scaling"):
            # Create a copy of the original data for preview
            scaled_data = self.df.copy()
            
            # Apply the selected scaling method
            if scaling_method == "Min-Max Scaling (0-1)":
                for col in selected_cols:
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        scaled_data[col] = (self.df[col] - min_val) / (max_val - min_val)
                    else:
                        scaled_data[col] = 0  # Handle constant columns
                
                step_text = f"Applied Min-Max scaling to columns: {', '.join(selected_cols)}"
            
            elif scaling_method == "Standard Scaling (z-score)":
                for col in selected_cols:
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:  # Avoid division by zero
                        scaled_data[col] = (self.df[col] - mean_val) / std_val
                    else:
                        scaled_data[col] = 0  # Handle constant columns
                
                step_text = f"Applied Standard scaling (z-score) to columns: {', '.join(selected_cols)}"
            
            elif scaling_method == "Robust Scaling (using quantiles)":
                for col in selected_cols:
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:  # Avoid division by zero
                        scaled_data[col] = (self.df[col] - q1) / iqr
                    else:
                        scaled_data[col] = 0  # Handle constant columns
                
                step_text = f"Applied Robust scaling to columns: {', '.join(selected_cols)}"
            
            # Show preview after scaling
            st.subheader("Preview After Scaling")
            stats_after = scaled_data[selected_cols].describe().T
            st.dataframe(stats_after)
            
            # Show visualization after scaling
            fig_after = go.Figure()
            for col in selected_cols:
                fig_after.add_trace(go.Box(y=scaled_data[col], name=col))
            fig_after.update_layout(title="Distribution After Scaling")
            st.plotly_chart(fig_after, use_container_width=True)
            
            # Confirm application
            if st.button("Confirm and Apply Scaling", key="confirm_scaling"):
                # Apply the scaling to the actual dataframe
                self.df = scaled_data
                
                # Store the preprocessing step
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                st.success(f"Successfully applied {scaling_method} to {len(selected_cols)} columns.")
                
                # Rerun to update the interface
                st.rerun()
    
    def _render_type_conversion(self):
        """Convert data types of columns"""
        st.subheader("Convert Data Types")
        
        # Show current data types
        st.write("Current Data Types:")
        st.dataframe(pd.DataFrame({
            'Column': self.df.columns,
            'Current Type': self.df.dtypes.astype(str)
        }))
        
        # Column selection
        selected_col = st.selectbox("Select column to convert", self.df.columns, key="convert_col")
        
        # Current type
        current_type = self.df[selected_col].dtype
        st.write(f"Current type: {current_type}")
        
        # New type selection
        type_options = ["numeric", "text/categorical", "datetime", "boolean"]
        new_type = st.selectbox("Convert to type", type_options, key="new_type")
        
        # Additional options based on type
        conversion_option = None
        
        if new_type == "numeric":
            conversion_option = st.selectbox(
                "Numeric type", 
                ["int", "float"],
                key="numeric_subtype"
            )
        
        elif new_type == "datetime":
            # Show sample values
            st.write("Sample values from this column:")
            st.write(self.df[selected_col].head(3).tolist())
            
            # Format string for custom parsing
            conversion_option = st.text_input(
                "Datetime format (leave empty for auto-detection)", 
                key="datetime_format",
                help="e.g., '%Y-%m-%d' for '2023-01-31' or '%d/%m/%Y' for '31/01/2023'"
            )
        
        # Apply button
        if st.button("Apply Type Conversion", key="apply_conversion"):
            try:
                # Create a copy of the original column in case conversion fails
                original_data = self.df[selected_col].copy()
                
                # Apply the conversion based on selected type
                if new_type == "numeric":
                    if conversion_option == "int":
                        self.df[selected_col] = pd.to_numeric(self.df[selected_col], errors='coerce').astype('Int64')
                        step_text = f"Converted '{selected_col}' to integer type"
                    else:
                        self.df[selected_col] = pd.to_numeric(self.df[selected_col], errors='coerce')
                        step_text = f"Converted '{selected_col}' to float type"
                
                elif new_type == "text/categorical":
                    self.df[selected_col] = self.df[selected_col].astype(str)
                    step_text = f"Converted '{selected_col}' to text/string type"
                
                elif new_type == "datetime":
                    if conversion_option:
                        self.df[selected_col] = pd.to_datetime(self.df[selected_col], format=conversion_option, errors='coerce')
                    else:
                        self.df[selected_col] = pd.to_datetime(self.df[selected_col], errors='coerce')
                    step_text = f"Converted '{selected_col}' to datetime type"
                
                elif new_type == "boolean":
                    # Handle various boolean representations
                    if pd.api.types.is_numeric_dtype(self.df[selected_col]):
                        self.df[selected_col] = self.df[selected_col].astype(bool)
                    else:
                        # Map common string representations to boolean
                        bool_map = {
                            'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                            'false': False, 'no': False, 'n': False, '0': False, 'f': False
                        }
                        self.df[selected_col] = self.df[selected_col].str.lower().map(bool_map)
                    step_text = f"Converted '{selected_col}' to boolean type"
                
                # Check for conversion issues
                na_count = self.df[selected_col].isna().sum()
                
                # Store the preprocessing step
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                st.success(f"Successfully converted '{selected_col}' to {new_type} type.")
                if na_count > 0:
                    st.warning(f"{na_count} values could not be converted and were set to NaN.")
                
                # Show new data type
                st.write(f"New data type: {self.df[selected_col].dtype}")
                
                # Rerun to update the interface
                st.rerun()
                
            except Exception as e:
                # Restore original data if conversion failed
                self.df[selected_col] = original_data
                st.error(f"Error during conversion: {str(e)}")
    
    def _render_datetime_extraction(self):
        """Extract features from date/time columns"""
        st.subheader("Extract Features from Date/Time")
        
        # Identify potential datetime columns
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also check for columns that could be dates but aren't recognized
        potential_date_cols = []
        for col in self.df.columns:
            if col in datetime_cols:
                continue
            
            # Try to convert the first non-null value to datetime
            sample = self.df[col].dropna().head(1)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample.iloc[0])
                    potential_date_cols.append(col)
                except:
                    pass
        
        # Combine both lists
        date_cols = datetime_cols + potential_date_cols
        
        if not date_cols:
            st.warning("No date/time columns found in the dataset!")
            return
        
        # Column selection
        selected_col = st.selectbox("Select date/time column", date_cols, key="datetime_col")
        
        # Convert to datetime if not already
        is_datetime = selected_col in datetime_cols
        if not is_datetime:
            st.info(f"Column '{selected_col}' is not currently recognized as a datetime. It will be converted first.")
            
            # Show sample values
            st.write("Sample values from this column:")
            st.write(self.df[selected_col].head(3).tolist())
            
            # Format string for custom parsing
            date_format = st.text_input(
                "Datetime format (leave empty for auto-detection)", 
                key="date_format",
                help="e.g., '%Y-%m-%d' for '2023-01-31' or '%d/%m/%Y' for '31/01/2023'"
            )
        
        # Feature selection
        st.subheader("Select features to extract")
        
        col1, col2 = st.columns(2)
        
        with col1:
            extract_year = st.checkbox("Year", key="extract_year")
            extract_quarter = st.checkbox("Quarter", key="extract_quarter")
            extract_month = st.checkbox("Month", key="extract_month")
            extract_month_name = st.checkbox("Month Name", key="extract_month_name")
            extract_week = st.checkbox("Week of Year", key="extract_week")
        
        with col2:
            extract_day = st.checkbox("Day of Month", key="extract_day")
            extract_day_of_week = st.checkbox("Day of Week", key="extract_day_of_week")
            extract_day_name = st.checkbox("Day Name", key="extract_day_name")
            extract_hour = st.checkbox("Hour", key="extract_hour")
            extract_is_weekend = st.checkbox("Is Weekend", key="extract_is_weekend")
        
        # Apply button
        if st.button("Extract Date Features", key="apply_date_extract"):
            try:
                # Create datetime series
                if not is_datetime:
                    if date_format:
                        dt_series = pd.to_datetime(self.df[selected_col], format=date_format, errors='coerce')
                    else:
                        dt_series = pd.to_datetime(self.df[selected_col], errors='coerce')
                else:
                    dt_series = self.df[selected_col]
                
                features_added = []
                
                # Extract selected features
                if extract_year:
                    self.df[f"{selected_col}_year"] = dt_series.dt.year
                    features_added.append("year")
                
                if extract_quarter:
                    self.df[f"{selected_col}_quarter"] = dt_series.dt.quarter
                    features_added.append("quarter")
                
                if extract_month:
                    self.df[f"{selected_col}_month"] = dt_series.dt.month
                    features_added.append("month")
                
                if extract_month_name:
                    self.df[f"{selected_col}_month_name"] = dt_series.dt.strftime('%B')
                    features_added.append("month_name")
                
                if extract_week:
                    self.df[f"{selected_col}_week"] = dt_series.dt.isocalendar().week
                    features_added.append("week")
                
                if extract_day:
                    self.df[f"{selected_col}_day"] = dt_series.dt.day
                    features_added.append("day")
                
                if extract_day_of_week:
                    self.df[f"{selected_col}_day_of_week"] = dt_series.dt.dayofweek
                    features_added.append("day_of_week")
                
                if extract_day_name:
                    self.df[f"{selected_col}_day_name"] = dt_series.dt.strftime('%A')
                    features_added.append("day_name")
                
                if extract_hour:
                    self.df[f"{selected_col}_hour"] = dt_series.dt.hour
                    features_added.append("hour")
                
                if extract_is_weekend:
                    self.df[f"{selected_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6])
                    features_added.append("is_weekend")
                
                # Store the preprocessing step
                features_text = ", ".join(features_added)
                step_text = f"Extracted {features_text} from '{selected_col}'"
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                st.success(f"Successfully extracted {len(features_added)} features from '{selected_col}'.")
                
                # Rerun to update the interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during feature extraction: {str(e)}")
    
    def _render_calculated_column(self):
        """Create a new calculated column"""
        st.subheader("Create Calculated Column")
        
        # New column name
        new_col_name = st.text_input("New column name", key="new_col_name")
        
        # Calculation method
        calc_method = st.radio(
            "Calculation method",
            ["Simple arithmetic", "Complex expression", "String manipulation"],
            key="calc_method"
        )
        
        if calc_method == "Simple arithmetic":
            # Get numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found for arithmetic calculation!")
                return
            
            # Column selection and operator
            col1, col2, col3 = st.columns([2, 1, 2])
            
            left_col = col1.selectbox("First column", numeric_cols, key="left_col")
            
            operator = col2.selectbox(
                "Operator", 
                ["+", "-", "*", "/", "^"],
                key="operator"
            )
            
            # Right side can be a column or a constant
            right_type = st.radio("Second operand type", ["Column", "Constant"], key="right_type")
            
            if right_type == "Column":
                right_col = col3.selectbox("Second column", numeric_cols, key="right_col")
            else:
                right_val = col3.number_input("Value", value=0.0, key="right_val")
        
        elif calc_method == "Complex expression":
            # Show column names for reference
            st.write("Available columns:")
            st.write(", ".join([f"df['{col}']" for col in self.df.columns]))
            
            # Expression input
            expression = st.text_area(
                "Enter Python expression (use df['column_name'] to reference columns)",
                height=100,
                key="complex_expr",
                help="e.g., np.log(df['income']) or df['height'] * df['weight'] / 100"
            )
            
            # Sample code for common operations
            with st.expander("Example expressions"):
                st.code("""
# Simple math
df['A'] + df['B']
df['A'] / df['B']

# With NumPy functions
np.log(df['A'])
np.sqrt(df['A'] * df['B'])

# Conditional logic
np.where(df['A'] > 10, 'High', 'Low')
                """)
        
        elif calc_method == "String manipulation":
            # Get text columns
            text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_cols:
                st.warning("No text columns found for string manipulation!")
                return
            
            # Operation selection
            string_op = st.selectbox(
                "String operation",
                ["Concatenate columns", "Extract substring", "Convert case", "Replace text", "String length"],
                key="string_op"
            )
            
            if string_op == "Concatenate columns":
                # Select columns to concatenate
                concat_cols = st.multiselect("Select columns to concatenate", text_cols, key="concat_cols")
                
                # Separator
                separator = st.text_input("Separator", value=" ", key="separator")
            
            elif string_op == "Extract substring":
                # Select column
                text_col = st.selectbox("Select text column", text_cols, key="substr_col")
                
                # Sample values
                st.write("Sample values:")
                st.write(self.df[text_col].dropna().head(3).tolist())
                
                # Extraction method
                extract_method = st.radio("Extraction method", ["By position", "By regular expression"], key="extract_method")
                
                if extract_method == "By position":
                    col1, col2 = st.columns(2)
                    start_pos = col1.number_input("Start position", value=0, min_value=0, key="start_pos")
                    end_pos = col2.number_input("End position (leave at -1 for end of string)", value=-1, key="end_pos")
                
                else:  # By regex
                    regex_pattern = st.text_input("Regular expression pattern", key="regex_pattern", help="e.g., r'\\d+' for numbers")
                    
                    # Example patterns
                    with st.expander("Example regex patterns"):
                        st.markdown("""
                        - `\\d+`: Match one or more digits
                        - `[A-Z]\\w+`: Match words starting with capital letter
                        - `(\\w+)@(\\w+)`: Match email username and domain
                        - `\\$(\\d+\\.\\d+)`: Match dollar amounts
                        """)
            
            elif string_op == "Convert case":
                # Select column
                text_col = st.selectbox("Select text column", text_cols, key="case_col")
                
                # Case conversion type
                case_type = st.selectbox(
                    "Conversion type",
                    ["UPPERCASE", "lowercase", "Title Case", "Capitalize first letter"],
                    key="case_type"
                )
            
            elif string_op == "Replace text":
                # Select column
                text_col = st.selectbox("Select text column", text_cols, key="replace_col")
                
                # Sample values
                st.write("Sample values:")
                st.write(self.df[text_col].dropna().head(3).tolist())
                
                # Text to replace
                find_text = st.text_input("Find text", key="find_text")
                replace_text = st.text_input("Replace with", key="replace_text")
                
                # Case sensitive option
                case_sensitive = st.checkbox("Case sensitive", value=True, key="case_sensitive")
            
            elif string_op == "String length":
                # Select column
                text_col = st.selectbox("Select text column", text_cols, key="len_col")
        
        # Apply button
        if st.button("Create Column", key="apply_calc_col"):
            if not new_col_name:
                st.error("Please specify a name for the new column.")
                return
            
            try:
                # Create the new column based on selected method
                if calc_method == "Simple arithmetic":
                    if right_type == "Column":
                        if operator == "+":
                            self.df[new_col_name] = self.df[left_col] + self.df[right_col]
                            step_text = f"Created '{new_col_name}' as {left_col} + {right_col}"
                        elif operator == "-":
                            self.df[new_col_name] = self.df[left_col] - self.df[right_col]
                            step_text = f"Created '{new_col_name}' as {left_col} - {right_col}"
                        elif operator == "*":
                            self.df[new_col_name] = self.df[left_col] * self.df[right_col]
                            step_text = f"Created '{new_col_name}' as {left_col} * {right_col}"
                        elif operator == "/":
                            self.df[new_col_name] = self.df[left_col] / self.df[right_col]
                            step_text = f"Created '{new_col_name}' as {left_col} / {right_col}"
                        elif operator == "^":
                            self.df[new_col_name] = self.df[left_col] ** self.df[right_col]
                            step_text = f"Created '{new_col_name}' as {left_col} ^ {right_col}"
                    else:
                        if operator == "+":
                            self.df[new_col_name] = self.df[left_col] + right_val
                            step_text = f"Created '{new_col_name}' as {left_col} + {right_val}"
                        elif operator == "-":
                            self.df[new_col_name] = self.df[left_col] - right_val
                            step_text = f"Created '{new_col_name}' as {left_col} - {right_val}"
                        elif operator == "*":
                            self.df[new_col_name] = self.df[left_col] * right_val
                            step_text = f"Created '{new_col_name}' as {left_col} * {right_val}"
                        elif operator == "/":
                            self.df[new_col_name] = self.df[left_col] / right_val
                            step_text = f"Created '{new_col_name}' as {left_col} / {right_val}"
                        elif operator == "^":
                            self.df[new_col_name] = self.df[left_col] ** right_val
                            step_text = f"Created '{new_col_name}' as {left_col} ^ {right_val}"
                
                elif calc_method == "Complex expression":
                    if not expression:
                        st.error("Please enter an expression.")
                        return
                    
                    # Use pandas eval if possible, otherwise use Python eval
                    try:
                        # Try to use vectorized pandas eval first (faster)
                        self.df[new_col_name] = self.df.eval(expression)
                    except:
                        # Fall back to Python eval (more flexible but slower)
                        df = self.df  # Local reference for use in eval
                        self.df[new_col_name] = eval(expression)
                    
                    step_text = f"Created '{new_col_name}' using complex expression"
                
                elif calc_method == "String manipulation":
                    if string_op == "Concatenate columns":
                        if not concat_cols:
                            st.error("Please select columns to concatenate.")
                            return
                        
                        # Convert all selected columns to string
                        string_cols = [self.df[col].astype(str) for col in concat_cols]
                        
                        # Concatenate with separator
                        self.df[new_col_name] = string_cols[0]
                        for col in string_cols[1:]:
                            self.df[new_col_name] = self.df[new_col_name] + separator + col
                        
                        step_text = f"Created '{new_col_name}' by concatenating {len(concat_cols)} columns"
                    
                    elif string_op == "Extract substring":
                        if extract_method == "By position":
                            if end_pos == -1:
                                self.df[new_col_name] = self.df[text_col].str[start_pos:]
                            else:
                                self.df[new_col_name] = self.df[text_col].str[start_pos:end_pos]
                            
                            step_text = f"Created '{new_col_name}' by extracting substring from '{text_col}'"
                        
                        else:  # By regex
                            if not regex_pattern:
                                st.error("Please enter a regex pattern.")
                                return
                            
                            self.df[new_col_name] = self.df[text_col].str.extract(regex_pattern)
                            step_text = f"Created '{new_col_name}' by extracting regex pattern from '{text_col}'"
                    
                    elif string_op == "Convert case":
                        if case_type == "UPPERCASE":
                            self.df[new_col_name] = self.df[text_col].str.upper()
                        elif case_type == "lowercase":
                            self.df[new_col_name] = self.df[text_col].str.lower()
                        elif case_type == "Title Case":
                            self.df[new_col_name] = self.df[text_col].str.title()
                        elif case_type == "Capitalize first letter":
                            self.df[new_col_name] = self.df[text_col].str.capitalize()
                        
                        step_text = f"Created '{new_col_name}' by converting case of '{text_col}'"
                    
                    elif string_op == "Replace text":
                        if not find_text:
                            st.error("Please enter text to find.")
                            return
                        
                        self.df[new_col_name] = self.df[text_col].str.replace(
                            find_text, 
                            replace_text, 
                            case=case_sensitive,
                            regex=False
                        )
                        
                        step_text = f"Created '{new_col_name}' by replacing text in '{text_col}'"
                    
                    elif string_op == "String length":
                        self.df[new_col_name] = self.df[text_col].str.len()
                        step_text = f"Created '{new_col_name}' as length of '{text_col}'"
                
                # Store the preprocessing step
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                st.success(f"Successfully created new column '{new_col_name}'.")
                
                # Show a preview
                st.write("Preview of the new column:")
                st.dataframe(self.df[[new_col_name]].head(5))
                
                # Rerun to update the interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating calculated column: {str(e)}")
    
    def _render_row_filter(self):
        """Interface for filtering rows based on conditions"""
        st.subheader("Filter Rows")
        
        if self.df is None or self.df.empty:
            st.warning("No data to filter.")
            return
        
        # Show current row count
        st.write(f"Current dataset has {len(self.df):,} rows.")
        
        # Filter type selection
        filter_type = st.radio(
            "Filter type",
            ["Simple condition", "Multiple conditions", "Drop duplicates", "Random sample"],
            key="filter_type"
        )
        
        if filter_type == "Simple condition":
            # Column selection
            col = st.selectbox("Select column", self.df.columns, key="filter_col")
            
            # Condition type based on column data type
            if np.issubdtype(self.df[col].dtype, np.number):
                condition_type = st.selectbox(
                    "Condition type",
                    ["Greater than", "Less than", "Equal to", "Not equal to", "Between"],
                    key="condition_type"
                )
                
                if condition_type == "Between":
                    col1, col2 = st.columns(2)
                    min_val = col1.number_input("Minimum value", value=float(self.df[col].min()), key="min_val")
                    max_val = col2.number_input("Maximum value", value=float(self.df[col].max()), key="max_val")
                else:
                    threshold = st.number_input("Threshold value", value=float(self.df[col].mean()), key="threshold")
            else:
                condition_type = st.selectbox(
                    "Condition type",
                    ["Contains", "Equals", "Not equals", "Starts with", "Ends with"],
                    key="condition_type"
                )
                
                # Show unique values for reference
                st.write("Sample unique values:")
                unique_vals = self.df[col].dropna().unique()[:5].tolist()
                st.write(", ".join([str(val) for val in unique_vals]))
                
                value = st.text_input("Value", key="filter_value")
        
        elif filter_type == "Multiple conditions":
            st.write("Build complex filter with multiple conditions (AND logic)")
            
            # Add conditions dynamically
            if 'conditions' not in st.session_state:
                st.session_state.conditions = []
            
            # Show current conditions
            if st.session_state.conditions:
                st.subheader("Current conditions")
                for i, condition in enumerate(st.session_state.conditions):
                    st.write(f"{i+1}. {condition['column']} {condition['operator']} {condition['value']}")
                
                if st.button("Clear All Conditions", key="clear_conditions"):
                    st.session_state.conditions = []
                    st.rerun()
            
            # Add new condition
            st.subheader("Add Condition")
            
            col1, col2, col3 = st.columns([2, 2, 3])
            
            new_col = col1.selectbox("Column", self.df.columns, key="new_cond_col")
            
            # Operator based on column type
            if np.issubdtype(self.df[new_col].dtype, np.number):
                operators = [">", "<", "==", "!=", ">=", "<="]
            else:
                operators = ["==", "!=", "contains", "startswith", "endswith"]
            
            new_op = col2.selectbox("Operator", operators, key="new_cond_op")
            
            # Value based on column type
            if np.issubdtype(self.df[new_col].dtype, np.number):
                new_val = col3.number_input("Value", value=0.0, key="new_cond_val")
            else:
                new_val = col3.text_input("Value", key="new_cond_val")
            
            # Add condition button
            if st.button("Add Condition", key="add_condition"):
                st.session_state.conditions.append({
                    "column": new_col,
                    "operator": new_op,
                    "value": new_val
                })
                st.rerun()
        
        elif filter_type == "Drop duplicates":
            # Select columns to consider for duplicates
            cols_for_dupes = st.multiselect(
                "Select columns to check for duplicates (leave empty to check all columns)", 
                self.df.columns,
                key="dupe_cols"
            )
            
            # Keep option
            keep_option = st.radio(
                "Which duplicate to keep",
                ["first", "last", "none"],
                key="keep_option"
            )
            
            # Current duplicates info
            if cols_for_dupes:
                dupe_count = self.df.duplicated(subset=cols_for_dupes).sum()
            else:
                dupe_count = self.df.duplicated().sum()
            
            st.write(f"Found {dupe_count:,} duplicate rows.")
        
        elif filter_type == "Random sample":
            # Sample size
            sample_type = st.radio(
                "Sample type",
                ["Number of rows", "Percentage of data"],
                key="sample_type"
            )
            
            if sample_type == "Number of rows":
                n_rows = st.number_input(
                    "Number of rows", 
                    min_value=1, 
                    max_value=len(self.df),
                    value=min(100, len(self.df)),
                    key="n_rows"
                )
            else:
                pct = st.slider(
                    "Percentage of data", 
                    min_value=1, 
                    max_value=100,
                    value=10,
                    key="sample_pct"
                )
            
            # Random seed for reproducibility
            use_seed = st.checkbox("Use random seed (for reproducibility)", key="use_seed")
            if use_seed:
                seed = st.number_input("Random seed", value=42, key="rand_seed")
        
        # Apply button
        if st.button("Apply Filter", key="apply_filter"):
            original_count = len(self.df)
            
            try:
                if filter_type == "Simple condition":
                    if np.issubdtype(self.df[col].dtype, np.number):
                        if condition_type == "Greater than":
                            self.df = self.df[self.df[col] > threshold]
                            step_text = f"Filtered rows where {col} > {threshold}"
                        elif condition_type == "Less than":
                            self.df = self.df[self.df[col] < threshold]
                            step_text = f"Filtered rows where {col} < {threshold}"
                        elif condition_type == "Equal to":
                            self.df = self.df[self.df[col] == threshold]
                            step_text = f"Filtered rows where {col} = {threshold}"
                        elif condition_type == "Not equal to":
                            self.df = self.df[self.df[col] != threshold]
                            step_text = f"Filtered rows where {col} != {threshold}"
                        elif condition_type == "Between":
                            self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
                            step_text = f"Filtered rows where {col} between {min_val} and {max_val}"
                    else:
                        if condition_type == "Contains":
                            self.df = self.df[self.df[col].astype(str).str.contains(value, case=False, na=False)]
                            step_text = f"Filtered rows where {col} contains '{value}'"
                        elif condition_type == "Equals":
                            self.df = self.df[self.df[col].astype(str) == value]
                            step_text = f"Filtered rows where {col} equals '{value}'"
                        elif condition_type == "Not equals":
                            self.df = self.df[self.df[col].astype(str) != value]
                            step_text = f"Filtered rows where {col} does not equal '{value}'"
                        elif condition_type == "Starts with":
                            self.df = self.df[self.df[col].astype(str).str.startswith(value, na=False)]
                            step_text = f"Filtered rows where {col} starts with '{value}'"
                        elif condition_type == "Ends with":
                            self.df = self.df[self.df[col].astype(str).str.endswith(value, na=False)]
                            step_text = f"Filtered rows where {col} ends with '{value}'"
                
                elif filter_type == "Multiple conditions":
                    if not st.session_state.conditions:
                        st.error("No conditions defined. Please add at least one condition.")
                        return
                    
                    # Start with all rows
                    mask = pd.Series(True, index=self.df.index)
                    
                    # Apply each condition
                    for condition in st.session_state.conditions:
                        col, op, val = condition['column'], condition['operator'], condition['value']
                        
                        if op == ">":
                            mask = mask & (self.df[col] > val)
                        elif op == "<":
                            mask = mask & (self.df[col] < val)
                        elif op == ">=":
                            mask = mask & (self.df[col] >= val)
                        elif op == "<=":
                            mask = mask & (self.df[col] <= val)
                        elif op == "==":
                            mask = mask & (self.df[col].astype(str) == str(val))
                        elif op == "!=":
                            mask = mask & (self.df[col].astype(str) != str(val))
                        elif op == "contains":
                            mask = mask & (self.df[col].astype(str).str.contains(str(val), case=False, na=False))
                        elif op == "startswith":
                            mask = mask & (self.df[col].astype(str).str.startswith(str(val), na=False))
                        elif op == "endswith":
                            mask = mask & (self.df[col].astype(str).str.endswith(str(val), na=False))
                    
                    # Apply the combined mask
                    self.df = self.df[mask]
                    
                    # Generate step text
                    conditions_text = " AND ".join([f"{c['column']} {c['operator']} {c['value']}" for c in st.session_state.conditions])
                    step_text = f"Filtered rows where {conditions_text}"
                
                elif filter_type == "Drop duplicates":
                    if cols_for_dupes:
                        self.df = self.df.drop_duplicates(subset=cols_for_dupes, keep=keep_option)
                        step_text = f"Dropped duplicates based on columns: {', '.join(cols_for_dupes)}"
                    else:
                        self.df = self.df.drop_duplicates(keep=keep_option)
                        step_text = "Dropped duplicates across all columns"
                
                elif filter_type == "Random sample":
                    if sample_type == "Number of rows":
                        if use_seed:
                            self.df = self.df.sample(n=n_rows, random_state=seed)
                        else:
                            self.df = self.df.sample(n=n_rows)
                        step_text = f"Took random sample of {n_rows} rows"
                    else:
                        frac = pct / 100.0
                        if use_seed:
                            self.df = self.df.sample(frac=frac, random_state=seed)
                        else:
                            self.df = self.df.sample(frac=frac)
                        step_text = f"Took random sample of {pct}% of rows"
                
                # Store the preprocessing step
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                new_count = len(self.df)
                st.success(f"Filter applied. Kept {new_count:,} rows out of {original_count:,} ({new_count/original_count*100:.1f}%).")
                
                # Clear conditions if needed
                if filter_type == "Multiple conditions":
                    st.session_state.conditions = []
                
                # Rerun to update the interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error applying filter: {str(e)}")
    
    def _render_one_hot_encoding(self):
        """One-hot encode categorical variables"""
        st.subheader("One-Hot Encode Categorical Variables")
        
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns found for one-hot encoding!")
            return
        
        # Column selection
        selected_cols = st.multiselect("Select categorical columns to encode", cat_cols, key="one_hot_cols")
        
        if not selected_cols:
            st.info("Please select at least one column to encode.")
            return
        
        # Show cardinality (number of unique values) for selected columns
        if selected_cols:
            cardinality_data = []
            for col in selected_cols:
                unique_count = self.df[col].nunique()
                cardinality_data.append({
                    "Column": col,
                    "Unique Values": unique_count,
                    "Warning": "High cardinality" if unique_count > 10 else ""
                })
            
            st.write("Cardinality (unique values) in selected columns:")
            st.dataframe(pd.DataFrame(cardinality_data))
            
            # Warning for high cardinality
            high_card_cols = [d["Column"] for d in cardinality_data if d["Warning"]]
            if high_card_cols:
                st.warning(f"Columns with high cardinality ({', '.join(high_card_cols)}) will generate many new columns. Consider alternative encoding methods.")
        
        # Options
        drop_original = st.checkbox("Drop original columns after encoding", value=True, key="drop_original")
        prefix_sep = st.text_input("Separator for column names", value="_", key="prefix_sep")
        
        # Apply button
        if st.button("Apply One-Hot Encoding", key="apply_onehot"):
            try:
                # Count original columns
                original_cols = self.df.shape[1]
                
                # Apply one-hot encoding
                dummies_df = pd.get_dummies(
                    self.df[selected_cols], 
                    prefix=selected_cols, 
                    prefix_sep=prefix_sep,
                    drop_first=False,
                    dummy_na=False
                )
                
                # Combine with original dataframe
                if drop_original:
                    # Drop encoded columns and add new ones
                    self.df = pd.concat([self.df.drop(columns=selected_cols), dummies_df], axis=1)
                else:
                    # Keep all columns and add new ones
                    self.df = pd.concat([self.df, dummies_df], axis=1)
                
                # Count new columns
                new_columns = self.df.shape[1] - original_cols
                
                # Store the preprocessing step
                step_text = f"One-hot encoded {len(selected_cols)} columns, creating {new_columns} new columns"
                st.session_state.preprocessing['steps_applied'].append(step_text)
                
                # Show success message
                st.success(f"Successfully encoded {len(selected_cols)} columns, creating {new_columns} new binary columns.")
                
                # Rerun to update the interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error applying one-hot encoding: {str(e)}")
    
    def _render_custom_analysis(self):
        """Render interface for custom analysis"""
        st.subheader("Custom Analysis")
        
        if self.df is None or self.df.empty:
            st.warning("No data available. Please upload a dataset first.")
            return
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Column Statistics",
                "Correlation Analysis",
                "Distribution Analysis",
                "Group Comparison",
                "Time Series Analysis",
                "Hypothesis Testing",
                "Custom Visualization"
            ],
            key="analysis_type"
        )
        
        if analysis_type == "Column Statistics":
            self._render_column_statistics()
        elif analysis_type == "Correlation Analysis":
            self._render_correlation_analysis()
        elif analysis_type == "Distribution Analysis":
            self._render_distribution_analysis()
        elif analysis_type == "Group Comparison":
            self._render_group_comparison()
        elif analysis_type == "Time Series Analysis":
            self._render_time_series_analysis()
        elif analysis_type == "Hypothesis Testing":
            self._render_hypothesis_testing()
        elif analysis_type == "Custom Visualization":
            self._render_custom_visualization()
    
    def _render_column_statistics(self):
        """Render interface for column statistics"""
        st.subheader("Column Statistics")
        
        # Column selection
        all_cols = self.df.columns.tolist()
        selected_cols = st.multiselect("Select columns to analyze", all_cols, key="stats_cols")
        
        if not selected_cols:
            st.info("Please select at least one column to analyze.")
            return
        
        # Calculate statistics
        stats_df = pd.DataFrame()
        
        for col in selected_cols:
            col_stats = {}
            series = self.df[col]
            
            # Basic stats
            col_stats["Column"] = col
            col_stats["Type"] = str(series.dtype)
            col_stats["Count"] = series.count()
            col_stats["Missing"] = series.isna().sum()
            col_stats["Missing %"] = (series.isna().sum() / len(self.df)) * 100
            col_stats["Unique Values"] = series.nunique()
            
            # Add numeric stats if applicable
            if np.issubdtype(series.dtype, np.number):
                col_stats["Min"] = series.min()
                col_stats["Max"] = series.max()
                col_stats["Mean"] = series.mean()
                col_stats["Median"] = series.median()
                col_stats["Std Dev"] = series.std()
                col_stats["Skewness"] = series.skew()
                col_stats["Kurtosis"] = series.kurtosis()
            
            # Add most common values for all types
            value_counts = series.value_counts()
            if not value_counts.empty:
                col_stats["Most Common"] = value_counts.index[0]
                col_stats["Most Common Count"] = value_counts.iloc[0]
                col_stats["Most Common %"] = (value_counts.iloc[0] / series.count()) * 100
            
            # Add to stats dataframe
            stats_df = pd.concat([stats_df, pd.DataFrame([col_stats])], ignore_index=True)
        
        # Display statistics
        st.write(stats_df)
        
        # Generate visualizations
        st.subheader("Visualizations")
        
        for col in selected_cols:
            series = self.df[col]
            
            if np.issubdtype(series.dtype, np.number):
                # Numeric column: Show histograms/box plots
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=series, name=col),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=series, name=col),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title=f"Analysis of {col}",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical column: Show bar chart
                value_counts = series.value_counts().head(20)  # Limit to top 20
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 20 values for {col}",
                    labels={"x": col, "y": "Count"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_analysis(self):
        """Render interface for correlation analysis"""
        st.subheader("Correlation Analysis")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns for correlation analysis", 
            numeric_cols, 
            default=numeric_cols[:min(10, len(numeric_cols))],
            key="corr_cols"
        )
        
        if len(selected_cols) < 2:
            st.info("Please select at least 2 columns for correlation analysis.")
            return
        
        # Correlation method
        corr_method = st.radio(
            "Correlation method",
            ["Pearson", "Spearman", "Kendall"],
            key="corr_method"
        )
        
        # Calculate correlation matrix
        corr_matrix = self.df[selected_cols].corr(method=corr_method.lower())
        
        # Display correlation matrix
        st.write("Correlation Matrix:")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
        
        # Visualization
        st.subheader("Correlation Heatmap")
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title=f"{corr_method} Correlation Heatmap"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.subheader("Top Correlations")
        
        # Extract all pairs of correlations
        corr_pairs = []
        for i in range(len(selected_cols)):
            for j in range(i+1, len(selected_cols)):
                col1 = selected_cols[i]
                col2 = selected_cols[j]
                corr_val = corr_matrix.loc[col1, col2]
                corr_pairs.append((col1, col2, corr_val))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Display top correlations
        top_corr_df = pd.DataFrame(
            corr_pairs, 
            columns=["Variable 1", "Variable 2", "Correlation"]
        ).head(10)
        
        st.write(top_corr_df)
        
        # Scatter plots for top correlations
        st.subheader("Scatter Plots for Top Correlations")
        
        for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline="ols",
                title=f"{col1} vs {col2} (r = {corr:.2f})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_distribution_analysis(self):
        """Render interface for distribution analysis"""
        st.subheader("Distribution Analysis")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for distribution analysis.")
            return
        
        # Column selection
        selected_col = st.selectbox("Select column to analyze", numeric_cols, key="dist_col")
        
        # Get series
        series = self.df[selected_col]
        
        # Calculate basic statistics
        stats = series.describe()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{stats['mean']:.2f}")
        col2.metric("Median", f"{stats['50%']:.2f}")
        col3.metric("Std Dev", f"{stats['std']:.2f}")
        col4.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
        
        # Additional statistics
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        col1, col2 = st.columns(2)
        col1.metric("Skewness", f"{skewness:.2f}", help="0 = symmetric, >0 = right skewed, <0 = left skewed")
        col2.metric("Kurtosis", f"{kurtosis:.2f}", help="0 = normal, >0 = heavy tails, <0 = light tails")
        
        # Distribution plots
        st.subheader("Distribution Plots")
        
        plot_type = st.radio(
            "Plot type",
            ["Histogram", "Box Plot", "Violin Plot", "QQ Plot"],
            key="dist_plot_type"
        )
        
        if plot_type == "Histogram":
            # Histogram settings
            col1, col2 = st.columns(2)
            bins = col1.slider("Number of bins", 5, 100, 20, key="hist_bins")
            kde = col2.checkbox("Show density curve", value=True, key="hist_kde")
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=series,
                nbinsx=bins,
                name=selected_col,
                histnorm="probability density" if kde else None
            ))
            
            if kde:
                # Calculate KDE
                from scipy import stats as scipy_stats
                kde_x = np.linspace(series.min(), series.max(), 1000)
                kde_y = scipy_stats.gaussian_kde(series.dropna())(kde_x)
                
                # Add KDE line
                fig.add_trace(go.Scatter(
                    x=kde_x, 
                    y=kde_y,
                    mode='lines',
                    name='Density',
                    line=dict(color='red', width=2)
                ))
                
                # Add normal distribution
                mu, sigma = series.mean(), series.std()
                normal_y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(kde_x - mu)**2 / (2 * sigma**2))
                
                fig.add_trace(go.Scatter(
                    x=kde_x, 
                    y=normal_y,
                    mode='lines',
                    name='Normal',
                    line=dict(color='green', dash='dot', width=2)
                ))
            
            fig.update_layout(
                title=f"Histogram of {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Density" if kde else "Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=series,
                name=selected_col,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig.update_layout(
                title=f"Box Plot of {selected_col}",
                yaxis_title=selected_col
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Violin Plot":
            fig = go.Figure()
            
            fig.add_trace(go.Violin(
                y=series,
                name=selected_col,
                box_visible=True,
                meanline_visible=True
            ))
            
            fig.update_layout(
                title=f"Violin Plot of {selected_col}",
                yaxis_title=selected_col
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "QQ Plot":
            from scipy import stats as scipy_stats
            
            # Calculate QQ data
            clean_data = series.dropna()
            qq_x, qq_y = scipy_stats.probplot(clean_data, dist="norm", fit=False)
            
            fig = go.Figure()
            
            # Add QQ points
            fig.add_trace(go.Scatter(
                x=qq_x,
                y=qq_y,
                mode='markers',
                name='Data Points'
            ))
            
            # Add reference line
            min_val = min(qq_x)
            max_val = max(qq_x)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"QQ Plot of {selected_col}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution tests
        st.subheader("Normality Tests")
        
        # Calculate normality tests
        clean_data = series.dropna()
        
        if len(clean_data) > 3:  # Need at least 3 points for tests
            from scipy import stats as scipy_stats
            
            # Shapiro-Wilk test (most powerful for small samples)
            try:
                if len(clean_data) <= 5000:  # Shapiro-Wilk limited to 5000 samples
                    shapiro_stat, shapiro_p = scipy_stats.shapiro(clean_data)
                else:
                    shapiro_stat, shapiro_p = np.nan, np.nan
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # D'Agostino-Pearson test
            try:
                dagostino_stat, dagostino_p = scipy_stats.normaltest(clean_data)
            except:
                dagostino_stat, dagostino_p = np.nan, np.nan
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = scipy_stats.kstest(clean_data, 'norm', args=(clean_data.mean(), clean_data.std()))
            except:
                ks_stat, ks_p = np.nan, np.nan
            
            # Create results table
            test_results = pd.DataFrame({
                'Test': ['Shapiro-Wilk', "D'Agostino-Pearson", 'Kolmogorov-Smirnov'],
                'Statistic': [shapiro_stat, dagostino_stat, ks_stat],
                'p-value': [shapiro_p, dagostino_p, ks_p],
                'Normal?': [
                    'Yes' if shapiro_p > 0.05 else 'No', 
                    'Yes' if dagostino_p > 0.05 else 'No',
                    'Yes' if ks_p > 0.05 else 'No'
                ]
            })
            
            st.write(test_results)
            
            # Interpretation
            st.info("""
            **Interpreting normality tests:**
            - p-value > 0.05: The data follows a normal distribution
            - p-value ≤ 0.05: The data does not follow a normal distribution
            
            Note: For large datasets, normality tests may often reject normality due to their high sensitivity to small deviations.
            Visual inspection via QQ plots can be more informative in these cases.
            """)
    
    def _render_group_comparison(self):
        """Render interface for group comparison analysis"""
        st.subheader("Group Comparison")
        
        # Get categorical columns for grouping
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for metrics
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or not numeric_cols:
            st.warning("Need at least one categorical column and one numeric column for group comparison.")
            return
        
        # Column selection
        group_col = st.selectbox("Select grouping column", cat_cols, key="group_col")
        metric_cols = st.multiselect(
            "Select metric columns", 
            numeric_cols,
            default=numeric_cols[:1],
            key="metric_cols"
        )
        
        if not metric_cols:
            st.info("Please select at least one metric column.")
            return
        
        # Count unique groups
        groups = self.df[group_col].unique()
        group_counts = self.df[group_col].value_counts()
        
        st.write(f"Found {len(groups)} unique groups in column '{group_col}'")
        
        # Group filter for large number of groups
        if len(groups) > 10:
            # Show top groups by frequency
            top_groups = group_counts.head(10).index.tolist()
            
            selected_groups = st.multiselect(
                "Select groups to compare (max 10 recommended)", 
                groups,
                default=top_groups[:5],
                key="selected_groups"
            )
            
            if selected_groups:
                filtered_df = self.df[self.df[group_col].isin(selected_groups)]
            else:
                filtered_df = self.df
                selected_groups = groups
        else:
            filtered_df = self.df
            selected_groups = groups
        
        # Calculate group statistics
        st.subheader("Group Statistics")
        
        # Create aggregation dict for multiple metrics
        agg_dict = {}
        for col in metric_cols:
            agg_dict[col] = ['count', 'mean', 'median', 'std', 'min', 'max']
        
        # Calculate group stats
        group_stats = filtered_df.groupby(group_col).agg(agg_dict)
        
        # Display stats
        st.write(group_stats)
        
        # Visualization options
        st.subheader("Visualizations")
        
        # Visualization type
        viz_type = st.radio(
            "Visualization type",
            ["Bar Chart", "Box Plot", "Violin Plot"],
            key="group_viz_type"
        )
        
        for metric in metric_cols:
            if viz_type == "Bar Chart":
                # Bar chart with error bars
                summary = filtered_df.groupby(group_col)[metric].agg(['mean', 'std']).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=summary[group_col],
                    y=summary['mean'],
                    error_y=dict(
                        type='data',
                        array=summary['std'],
                        visible=True
                    ),
                    name=metric
                ))
                
                fig.update_layout(
                    title=f"Mean {metric} by {group_col}",
                    xaxis_title=group_col,
                    yaxis_title=f"Mean {metric}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plot":
                fig = px.box(
                    filtered_df,
                    x=group_col,
                    y=metric,
                    title=f"Distribution of {metric} by {group_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Violin Plot":
                fig = px.violin(
                    filtered_df,
                    x=group_col,
                    y=metric,
                    box=True,
                    title=f"Distribution of {metric} by {group_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistical tests if there are exactly 2 groups and at least one metric
        if len(selected_groups) == 2 and metric_cols:
            st.subheader("Statistical Tests")
            
            # Perform tests for each metric
            for metric in metric_cols:
                st.write(f"**Statistical tests for {metric}:**")
                
                # Get data for each group
                group1_data = filtered_df[filtered_df[group_col] == selected_groups[0]][metric].dropna()
                group2_data = filtered_df[filtered_df[group_col] == selected_groups[1]][metric].dropna()
                
                if len(group1_data) < 3 or len(group2_data) < 3:
                    st.warning(f"Not enough data in groups for statistical testing for {metric}.")
                    continue
                
                # Perform t-test
                try:
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("t-statistic", f"{t_stat:.3f}")
                    col2.metric("p-value", f"{p_value:.4f}")
                    col3.metric("Significant Difference", "Yes" if p_value < 0.05 else "No")
                    
                    # Interpretation
                    if p_value < 0.05:
                        st.success(f"There is a statistically significant difference in {metric} between the two groups (p < 0.05).")
                    else:
                        st.info(f"There is no statistically significant difference in {metric} between the two groups (p > 0.05).")
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = group1_data.mean(), group2_data.mean()
                    std1, std2 = group1_data.std(), group2_data.std()
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((len(group1_data) - 1) * std1**2 + (len(group2_data) - 1) * std2**2) / 
                                         (len(group1_data) + len(group2_data) - 2))
                    
                    # Cohen's d
                    cohen_d = abs(mean1 - mean2) / pooled_std
                    
                    # Interpret effect size
                    if cohen_d < 0.2:
                        effect_text = "very small"
                    elif cohen_d < 0.5:
                        effect_text = "small"
                    elif cohen_d < 0.8:
                        effect_text = "medium"
                    else:
                        effect_text = "large"
                    
                    st.write(f"Effect size (Cohen's d): {cohen_d:.3f} ({effect_text} effect)")
                    
                except Exception as e:
                    st.error(f"Error performing t-test: {str(e)}")
    
    def _render_time_series_analysis(self):
        """Render interface for time series analysis"""
        st.subheader("Time Series Analysis")
        
        # Look for potential time/date columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also look for columns that might be dates but not detected as such
        potential_date_cols = []
        for col in self.df.columns:
            if col in date_cols:
                continue
                
            # Look for columns with "date", "time", "year", "month", "day" in the name
            if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                potential_date_cols.append(col)
        
        # Combine confirmed and potential date columns
        time_cols = date_cols + potential_date_cols
        
        if not time_cols:
            st.warning("No time/date columns found for time series analysis.")
            return
        
        # Get numeric columns for values
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for time series analysis.")
            return
        
        # Column selection
        time_col = st.selectbox("Select time/date column", time_cols, key="ts_time_col")
        value_cols = st.multiselect(
            "Select value columns", 
            numeric_cols,
            default=numeric_cols[:1],
            key="ts_value_cols"
        )
        
        if not value_cols:
            st.info("Please select at least one value column.")
            return
        
        # Check if time column is datetime type
        is_datetime = time_col in date_cols
        
        if not is_datetime:
            st.warning(f"Column '{time_col}' is not recognized as a datetime. Converting for analysis.")
            
            # Show sample values
            st.write("Sample values from this column:")
            st.write(self.df[time_col].head(3).tolist())
            
            # Format string for custom parsing
            date_format = st.text_input(
                "Datetime format (leave empty for auto-detection)", 
                key="ts_date_format",
                help="e.g., '%Y-%m-%d' for '2023-01-31' or '%d/%m/%Y' for '31/01/2023'"
            )
            
            # Convert to datetime
            try:
                if date_format:
                    time_series = pd.to_datetime(self.df[time_col], format=date_format, errors='coerce')
                else:
                    time_series = pd.to_datetime(self.df[time_col], errors='coerce')
                    
                # Check for conversion success
                if time_series.isna().all():
                    st.error(f"Failed to convert '{time_col}' to datetime. Please check the format.")
                    return
            except Exception as e:
                st.error(f"Error converting to datetime: {str(e)}")
                return
        else:
            time_series = self.df[time_col]
        
        # Time aggregation
        agg_options = ["No aggregation", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        aggregation = st.selectbox("Time aggregation", agg_options, key="ts_aggregation")
        
        # Create a copy of the dataframe with proper datetime column
        ts_df = self.df.copy()
        ts_df['_datetime_'] = time_series
        
        # Aggregate if selected
        if aggregation != "No aggregation":
            # Set up the grouping
            if aggregation == "Daily":
                ts_df['_period_'] = ts_df['_datetime_'].dt.date
            elif aggregation == "Weekly":
                ts_df['_period_'] = ts_df['_datetime_'].dt.to_period('W').dt.start_time
            elif aggregation == "Monthly":
                ts_df['_period_'] = ts_df['_datetime_'].dt.to_period('M').dt.start_time
            elif aggregation == "Quarterly":
                ts_df['_period_'] = ts_df['_datetime_'].dt.to_period('Q').dt.start_time
            elif aggregation == "Yearly":
                ts_df['_period_'] = ts_df['_datetime_'].dt.to_period('Y').dt.start_time
            
            # Aggregation function
            agg_func = st.selectbox(
                "Aggregation function", 
                ["Mean", "Sum", "Min", "Max", "Median", "Count"],
                key="ts_agg_func"
            )
            
            # Map function name to actual function
            func_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Min": "min",
                "Max": "max",
                "Median": "median",
                "Count": "count"
            }
            
            # Create aggregation dictionary
            agg_dict = {col: func_map[agg_func] for col in value_cols}
            
            # Perform aggregation
            aggregated_df = ts_df.groupby('_period_').agg(agg_dict)
            aggregated_df = aggregated_df.reset_index()
            
            # Use the aggregated data for analysis
            plot_df = aggregated_df
            x_col = '_period_'
        else:
            # Use original data with datetime column
            plot_df = ts_df.sort_values('_datetime_')
            x_col = '_datetime_'
        
        # Visualization
        st.subheader("Time Series Visualization")
        
        # Plot time series for each selected column
        for col in value_cols:
            fig = px.line(
                plot_df, 
                x=x_col, 
                y=col,
                title=f"Time Series of {col}"
            )
            
            # Add markers for better visibility with sparse data
            if len(plot_df) < 50:
                fig.update_traces(mode='lines+markers')
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=col
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.subheader("Trend Analysis")
        
        if len(plot_df) >= 5:  # Need sufficient data for trend analysis
            # Select column for trend analysis
            trend_col = st.selectbox(
                "Select column for trend analysis",
                value_cols,
                key="trend_col"
            )
            
            # Clean data (remove NaNs)
            clean_data = plot_df.dropna(subset=[trend_col])
            
            if len(clean_data) < 5:
                st.warning("Not enough non-missing data points for trend analysis.")
            else:
                # Simple linear regression for trend
                try:
                    # Create numeric x-axis for regression
                    clean_data = clean_data.reset_index(drop=True)
                    clean_data['_index_'] = clean_data.index
                    
                    # Perform regression
                    from scipy import stats as scipy_stats
                    
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        clean_data['_index_'], 
                        clean_data[trend_col]
                    )
                    
                    # Calculate fitted values
                    clean_data['_trend_'] = intercept + slope * clean_data['_index_']
                    
                    # Create plot with trendline
                    fig = go.Figure()
                    
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=clean_data[x_col],
                        y=clean_data[trend_col],
                        mode='lines+markers',
                        name='Data'
                    ))
                    
                    # Trend line
                    fig.add_trace(go.Scatter(
                        x=clean_data[x_col],
                        y=clean_data['_trend_'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Trend Analysis for {trend_col}",
                        xaxis_title="Time",
                        yaxis_title=trend_col
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend statistics
                    col1, col2, col3 = st.columns(3)
                    
                    # Direction and strength
                    direction = "Increasing" if slope > 0 else "Decreasing"
                    
                    # Monthly change (approximation)
                    period_change = slope
                    if aggregation == "Daily":
                        period_change = slope * 30  # Approx month
                    elif aggregation == "Weekly":
                        period_change = slope * 4  # Approx month
                    
                    # Significance
                    significant = p_value < 0.05
                    
                    col1.metric("Trend Direction", direction)
                    col2.metric("Change per Period", f"{slope:.4f}")
                    col3.metric("Est. Monthly Change", f"{period_change:.4f}")
                    
                    st.metric("R-squared", f"{r_value**2:.4f}", 
                             help="Proportion of variance explained by the trend (0-1)")
                    
                    # Interpretation
                    if significant:
                        st.success(f"There is a statistically significant {direction.lower()} trend (p < 0.05).")
                    else:
                        st.info(f"The {direction.lower()} trend is not statistically significant (p > 0.05).")
                    
                except Exception as e:
                    st.error(f"Error in trend analysis: {str(e)}")
        
        # Seasonality analysis
        if aggregation != "No aggregation" and len(plot_df) >= 12:  # Need at least a year of data
            st.subheader("Seasonality Analysis")
            
            # Select column for seasonality analysis
            season_col = st.selectbox(
                "Select column for seasonality analysis",
                value_cols,
                key="season_col"
            )
            
            try:
                # Extract date components
                seasonality_df = plot_df.copy()
                seasonality_df['_month_'] = pd.DatetimeIndex(seasonality_df[x_col]).month
                seasonality_df['_quarter_'] = pd.DatetimeIndex(seasonality_df[x_col]).quarter
                seasonality_df['_year_'] = pd.DatetimeIndex(seasonality_df[x_col]).year
                
                # Monthly seasonality
                monthly_avg = seasonality_df.groupby('_month_')[season_col].mean().reset_index()
                monthly_avg['Month'] = monthly_avg['_month_'].apply(lambda x: pd.Timestamp(2000, x, 1).strftime('%b'))
                
                fig = px.bar(
                    monthly_avg,
                    x='Month',
                    y=season_col,
                    title=f"Monthly Seasonality Pattern for {season_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Quarterly seasonality
                quarterly_avg = seasonality_df.groupby('_quarter_')[season_col].mean().reset_index()
                quarterly_avg['Quarter'] = quarterly_avg['_quarter_'].apply(lambda x: f"Q{x}")
                
                fig = px.bar(
                    quarterly_avg,
                    x='Quarter',
                    y=season_col,
                    title=f"Quarterly Seasonality Pattern for {season_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in seasonality analysis: {str(e)}")
    
    def _render_hypothesis_testing(self):
        """Render interface for hypothesis testing"""
        st.subheader("Hypothesis Testing")
        
        # Test type selection
        test_type = st.selectbox(
            "Select test type",
            [
                "One-sample t-test", 
                "Two-sample t-test", 
                "Paired t-test",
                "ANOVA (One-way)",
                "Chi-square test of independence"
            ],
            key="test_type"
        )
        
        if test_type == "One-sample t-test":
            # Get numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found for t-test.")
                return
            
            # Column selection
            col = st.selectbox("Select column to test", numeric_cols, key="one_sample_col")
            
            # Reference value
            mu = st.number_input(
                "Reference value (null hypothesis)", 
                value=float(self.df[col].mean()),
                key="ref_value"
            )
            
            # Significance level
            alpha = st.selectbox(
                "Significance level (α)",
                [0.01, 0.05, 0.1],
                index=1,
                key="alpha"
            )
            
            # Alternative hypothesis
            alternative = st.radio(
                "Alternative hypothesis",
                ["two-sided", "less", "greater"],
                key="alternative"
            )
            
            # Run test
            if st.button("Run test", key="run_one_sample"):
                # Clean data
                clean_data = self.df[col].dropna()
                
                if len(clean_data) < 3:
                    st.error("Not enough data for t-test (need at least 3 non-missing values).")
                    return
                
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_1samp(clean_data, mu, alternative=alternative)
                    
                    # Display results
                    st.write("### Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("t-statistic", f"{t_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.4f}")
                    col3.metric("Significant?", "Yes" if p_value < alpha else "No")
                    
                    # Conclusion
                    if p_value < alpha:
                        if alternative == "two-sided":
                            conclusion = f"The mean of {col} is significantly different from {mu}."
                        elif alternative == "less":
                            conclusion = f"The mean of {col} is significantly less than {mu}."
                        else:  # greater
                            conclusion = f"The mean of {col} is significantly greater than {mu}."
                    else:
                        if alternative == "two-sided":
                            conclusion = f"There is not enough evidence to conclude that the mean of {col} is different from {mu}."
                        elif alternative == "less":
                            conclusion = f"There is not enough evidence to conclude that the mean of {col} is less than {mu}."
                        else:  # greater
                            conclusion = f"There is not enough evidence to conclude that the mean of {col} is greater than {mu}."
                    
                    st.success(conclusion)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Histogram
                    fig.add_trace(go.Histogram(
                        x=clean_data,
                        name="Data Distribution",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    
                    # Add vertical line for reference value
                    fig.add_vline(
                        x=mu,
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"μ = {mu}",
                        annotation_position="top"
                    )
                    
                    # Add vertical line for sample mean
                    sample_mean = clean_data.mean()
                    fig.add_vline(
                        x=sample_mean,
                        line_width=2,
                        line_color="green",
                        annotation_text=f"Sample mean = {sample_mean:.2f}",
                        annotation_position="bottom"
                    )
                    
                    fig.update_layout(
                        title=f"Distribution of {col} with Reference Value",
                        xaxis_title=col,
                        yaxis_title="Density"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running t-test: {str(e)}")
        
        elif test_type == "Two-sample t-test":
            # Get categorical columns for grouping
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Get numeric columns for testing
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not cat_cols or not numeric_cols:
                st.warning("Need at least one categorical column and one numeric column for two-sample t-test.")
                return
            
            # Column selection
            group_col = st.selectbox("Select grouping column", cat_cols, key="two_sample_group")
            value_col = st.selectbox("Select value column", numeric_cols, key="two_sample_value")
            
            # Get unique values in grouping column
            unique_groups = self.df[group_col].dropna().unique()
            
            if len(unique_groups) < 2:
                st.warning(f"Need at least 2 groups in '{group_col}' for two-sample t-test.")
                return
            
            # Select two groups
            col1, col2 = st.columns(2)
            group1 = col1.selectbox("Group 1", unique_groups, index=0, key="group1")
            group2 = col2.selectbox("Group 2", unique_groups, index=min(1, len(unique_groups)-1), key="group2")
            
            # Test options
            equal_var = st.checkbox("Assume equal variances", value=False, key="equal_var")
            
            # Significance level
            alpha = st.selectbox(
                "Significance level (α)",
                [0.01, 0.05, 0.1],
                index=1,
                key="two_sample_alpha"
            )
            
            # Run test
            if st.button("Run test", key="run_two_sample"):
                # Get data for each group
                group1_data = self.df[self.df[group_col] == group1][value_col].dropna()
                group2_data = self.df[self.df[group_col] == group2][value_col].dropna()
                
                if len(group1_data) < 3 or len(group2_data) < 3:
                    st.error("Not enough data in one or both groups (need at least 3 non-missing values in each).")
                    return
                
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                    
                    # Display results
                    st.write("### Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("t-statistic", f"{t_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.4f}")
                    col3.metric("Significant?", "Yes" if p_value < alpha else "No")
                    
                    # Group statistics
                    st.write("### Group Statistics")
                    
                    stats_df = pd.DataFrame({
                        'Group': [group1, group2],
                        'Count': [len(group1_data), len(group2_data)],
                        'Mean': [group1_data.mean(), group2_data.mean()],
                        'Std Dev': [group1_data.std(), group2_data.std()]
                    })
                    
                    st.write(stats_df)
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = group1_data.mean(), group2_data.mean()
                    std1, std2 = group1_data.std(), group2_data.std()
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((len(group1_data) - 1) * std1**2 + (len(group2_data) - 1) * std2**2) / 
                                         (len(group1_data) + len(group2_data) - 2))
                    
                    # Cohen's d
                    cohen_d = abs(mean1 - mean2) / pooled_std
                    
                    # Interpret effect size
                    if cohen_d < 0.2:
                        effect_text = "very small"
                    elif cohen_d < 0.5:
                        effect_text = "small"
                    elif cohen_d < 0.8:
                        effect_text = "medium"
                    else:
                        effect_text = "large"
                    
                    st.write(f"Effect size (Cohen's d): {cohen_d:.3f} ({effect_text} effect)")
                    
                    # Conclusion
                    if p_value < alpha:
                        conclusion = f"There is a statistically significant difference in {value_col} between {group1} and {group2}."
                    else:
                        conclusion = f"There is not enough evidence to conclude that {value_col} differs between {group1} and {group2}."
                    
                    st.success(conclusion)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Box plots
                    fig.add_trace(go.Box(
                        y=group1_data,
                        name=str(group1),
                        boxpoints='outliers'
                    ))
                    
                    fig.add_trace(go.Box(
                        y=group2_data,
                        name=str(group2),
                        boxpoints='outliers'
                    ))
                    
                    fig.update_layout(
                        title=f"Distribution of {value_col} by Group",
                        yaxis_title=value_col
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running t-test: {str(e)}")
        
        elif test_type == "Paired t-test":
            # Get numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for paired t-test.")
                return
            
            # Column selection
            col1, col2 = st.columns(2)
            first_col = col1.selectbox("First measurement", numeric_cols, index=0, key="first_col")
            second_col = col2.selectbox("Second measurement", numeric_cols, index=min(1, len(numeric_cols)-1), key="second_col")
            
            # Significance level
            alpha = st.selectbox(
                "Significance level (α)",
                [0.01, 0.05, 0.1],
                index=1,
                key="paired_alpha"
            )
            
            # Alternative hypothesis
            alternative = st.radio(
                "Alternative hypothesis",
                ["two-sided", "less", "greater"],
                key="paired_alternative"
            )
            
            # Run test
            if st.button("Run test", key="run_paired"):
                # Clean data (keep only rows with valid data in both columns)
                valid_mask = ~(self.df[first_col].isna() | self.df[second_col].isna())
                first_data = self.df.loc[valid_mask, first_col]
                second_data = self.df.loc[valid_mask, second_col]
                
                if len(first_data) < 3:
                    st.error("Not enough paired data (need at least 3 valid pairs).")
                    return
                
                try:
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(first_data, second_data, alternative=alternative)
                    
                    # Display results
                    st.write("### Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("t-statistic", f"{t_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.4f}")
                    col3.metric("Significant?", "Yes" if p_value < alpha else "No")
                    
                    # Statistics
                    st.write("### Measurement Statistics")
                    
                    diff_data = first_data - second_data
                    
                    stats_df = pd.DataFrame({
                        'Measurement': [first_col, second_col, "Difference"],
                        'Mean': [first_data.mean(), second_data.mean(), diff_data.mean()],
                        'Std Dev': [first_data.std(), second_data.std(), diff_data.std()],
                        'Min': [first_data.min(), second_data.min(), diff_data.min()],
                        'Max': [first_data.max(), second_data.max(), diff_data.max()]
                    })
                    
                    st.write(stats_df)
                    
                    # Conclusion
                    if p_value < alpha:
                        if alternative == "two-sided":
                            conclusion = f"There is a statistically significant difference between {first_col} and {second_col}."
                        elif alternative == "less":
                            conclusion = f"{first_col} is significantly less than {second_col}."
                        else:  # greater
                            conclusion = f"{first_col} is significantly greater than {second_col}."
                    else:
                        if alternative == "two-sided":
                            conclusion = f"There is not enough evidence to conclude that {first_col} and {second_col} are different."
                        elif alternative == "less":
                            conclusion = f"There is not enough evidence to conclude that {first_col} is less than {second_col}."
                        else:  # greater
                            conclusion = f"There is not enough evidence to conclude that {first_col} is greater than {second_col}."
                    
                    st.success(conclusion)
                    
                    # Visualization
                    fig = make_subplots(rows=1, cols=2, subplot_titles=["Paired Measurements", "Difference Distribution"])
                    
                    # Paired line plot
                    for i in range(min(50, len(first_data))):  # Limit to 50 lines for visibility
                        fig.add_trace(
                            go.Scatter(
                                x=[first_col, second_col],
                                y=[first_data.iloc[i], second_data.iloc[i]],
                                mode='lines+markers',
                                line=dict(color='rgba(0,0,255,0.2)'),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    # Add mean line
                    fig.add_trace(
                        go.Scatter(
                            x=[first_col, second_col],
                            y=[first_data.mean(), second_data.mean()],
                            mode='lines+markers',
                            line=dict(color='red', width=3),
                            name="Mean"
                        ),
                        row=1, col=1
                    )
                    
                    # Difference histogram
                    fig.add_trace(
                        go.Histogram(
                            x=diff_data,
                            marker_color='rgba(0,0,255,0.5)',
                            name="Difference"
                        ),
                        row=1, col=2
                    )
                    
                    # Add vertical line for mean difference
                    fig.add_vline(
                        x=diff_data.mean(),
                        line_width=2,
                        line_color="red",
                        line_dash="dash",
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f"Paired Analysis of {first_col} and {second_col}",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running paired t-test: {str(e)}")
        
        elif test_type == "ANOVA (One-way)":
            # Get categorical columns for grouping
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Get numeric columns for testing
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not cat_cols or not numeric_cols:
                st.warning("Need at least one categorical column and one numeric column for ANOVA.")
                return
            
            # Column selection
            group_col = st.selectbox("Select grouping column", cat_cols, key="anova_group")
            value_col = st.selectbox("Select value column", numeric_cols, key="anova_value")
            
            # Get unique groups
            unique_groups = self.df[group_col].dropna().unique()
            
            if len(unique_groups) < 3:
                st.warning(f"Need at least 3 groups in '{group_col}' for ANOVA. For 2 groups, use t-test instead.")
                return
            
            # Significance level
            alpha = st.selectbox(
                "Significance level (α)",
                [0.01, 0.05, 0.1],
                index=1,
                key="anova_alpha"
            )
            
            # Run test
            if st.button("Run test", key="run_anova"):
                try:
                    # Prepare data for ANOVA
                    groups = []
                    group_names = []
                    
                    for group in unique_groups:
                        group_data = self.df[self.df[group_col] == group][value_col].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups) < 3:
                        st.error("After removing missing values, fewer than 3 groups have data.")
                        return
                    
                    # Perform ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Display results
                    st.write("### ANOVA Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("F-statistic", f"{f_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.4f}")
                    col3.metric("Significant?", "Yes" if p_value < alpha else "No")
                    
                    # Group statistics
                    st.write("### Group Statistics")
                    
                    stats = []
                    for i, group_data in enumerate(groups):
                        stats.append({
                            'Group': group_names[i],
                            'Count': len(group_data),
                            'Mean': group_data.mean(),
                            'Std Dev': group_data.std(),
                            'Min': group_data.min(),
                            'Max': group_data.max()
                        })
                    
                    stats_df = pd.DataFrame(stats)
                    st.write(stats_df)
                    
                    # Conclusion
                    if p_value < alpha:
                        conclusion = f"There are statistically significant differences in {value_col} among the groups."
                    else:
                        conclusion = f"There is not enough evidence to conclude that {value_col} differs among the groups."
                    
                    st.success(conclusion)
                    
                    # Post-hoc tests if ANOVA is significant
                    if p_value < alpha:
                        st.write("### Post-hoc Tests (Tukey HSD)")
                        
                        try:
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            
                            # Prepare data for Tukey's test
                            all_data = []
                            group_labels = []
                            
                            for i, group_data in enumerate(groups):
                                all_data.extend(group_data)
                                group_labels.extend([group_names[i]] * len(group_data))
                            
                            # Perform Tukey's test
                            tukey = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
                            
                            # Display results
                            tukey_df = pd.DataFrame(
                                data=tukey._results_table.data[1:],
                                columns=tukey._results_table.data[0]
                            )
                            
                            st.write(tukey_df)
                            
                        except Exception as e:
                            st.error(f"Error running post-hoc tests: {str(e)}")
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Box plots for each group
                    for i, group_data in enumerate(groups):
                        fig.add_trace(go.Box(
                            y=group_data,
                            name=group_names[i],
                            boxpoints='outliers'
                        ))
                    
                    fig.update_layout(
                        title=f"Distribution of {value_col} by Group",
                        yaxis_title=value_col
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running ANOVA: {str(e)}")
        
        elif test_type == "Chi-square test of independence":
            # Get categorical columns
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(cat_cols) < 2:
                st.warning("Need at least two categorical columns for chi-square test.")
                return
            
            # Column selection
            col1, col2 = st.columns(2)
            first_col = col1.selectbox("First categorical variable", cat_cols, index=0, key="chi_first_col")
            second_col = col2.selectbox("Second categorical variable", cat_cols, index=min(1, len(cat_cols)-1), key="chi_second_col")
            
            # Significance level
            alpha = st.selectbox(
                "Significance level (α)",
                [0.01, 0.05, 0.1],
                index=1,
                key="chi_alpha"
            )
            
            # Run test
            if st.button("Run test", key="run_chi"):
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(self.df[first_col], self.df[second_col])
                    
                    # Check if table is valid for chi-square
                    if contingency_table.size <= 1:
                        st.error("Contingency table must have at least 2 cells.")
                        return
                    
                    # Check for small expected frequencies
                    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    # Warning for small expected frequencies
                    small_exp = (expected < 5).sum()
                    total_cells = expected.size
                    
                    # Display contingency table
                    st.write("### Contingency Table (Observed Frequencies)")
                    st.write(contingency_table)
                    
                    # Display expected frequencies
                    st.write("### Expected Frequencies")
                    expected_df = pd.DataFrame(
                        expected, 
                        index=contingency_table.index, 
                        columns=contingency_table.columns
                    )
                    st.write(expected_df)
                    
                    if small_exp > 0:
                        pct_small = (small_exp / total_cells) * 100
                        if pct_small > 20:
                            st.warning(f"{small_exp} cells ({pct_small:.1f}%) have expected frequencies less than 5. Chi-square results may not be reliable.")
                    
                    # Display results
                    st.write("### Chi-square Test Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Chi-square", f"{chi2:.4f}")
                    col2.metric("p-value", f"{p:.4f}")
                    col3.metric("Degrees of freedom", dof)
                    col4.metric("Significant?", "Yes" if p < alpha else "No")
                    
                    # Calculate Cramer's V (effect size)
                    n = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    
                    # Interpret Cramer's V
                    if cramers_v < 0.1:
                        effect_size = "negligible"
                    elif cramers_v < 0.3:
                        effect_size = "small"
                    elif cramers_v < 0.5:
                        effect_size = "medium"
                    else:
                        effect_size = "large"
                    
                    st.metric("Cramer's V (effect size)", f"{cramers_v:.4f}", help=f"{effect_size} association")
                    
                    # Conclusion
                    if p < alpha:
                        conclusion = f"There is a statistically significant association between {first_col} and {second_col}."
                    else:
                        conclusion = f"There is not enough evidence to conclude that {first_col} and {second_col} are associated."
                    
                    st.success(conclusion)
                    
                    # Visualization
                    st.write("### Visualizations")
                    
                    # Heatmap of observed frequencies
                    fig = px.imshow(
                        contingency_table,
                        text_auto=True,
                        title=f"Observed Frequencies: {first_col} vs {second_col}",
                        labels=dict(x=second_col, y=first_col, color="Frequency")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stacked bar chart
                    normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)
                    
                    fig = go.Figure()
                    
                    # Add bars for each category in second column
                    for col in normalized_table.columns:
                        fig.add_trace(go.Bar(
                            x=normalized_table.index,
                            y=normalized_table[col],
                            name=str(col)
                        ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title=f"Proportions of {second_col} within each {first_col} category",
                        xaxis_title=first_col,
                        yaxis_title="Proportion"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running chi-square test: {str(e)}")
    
    def _render_custom_visualization(self):
        """Render interface for custom visualization"""
        st.subheader("Custom Visualization")
        
        # Visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            [
                "Scatter Plot",
                "Line Chart",
                "Bar Chart",
                "Histogram",
                "Box Plot",
                "Heatmap",
                "Pie Chart",
                "3D Scatter Plot",
                "Bubble Chart",
                "Violin Plot",
                "Radar Chart"
            ],
            key="custom_viz_type"
        )
        
        # Get appropriate columns based on visualization type
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        if viz_type == "Scatter Plot":
            # Simple customization for scatter plot
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            
            # Optional parameters
            show_advanced = st.checkbox("Show advanced options", key="scatter_advanced")
            
            if show_advanced:
                col1, col2 = st.columns(2)
                
                # Color by
                color_by = col1.selectbox(
                    "Color by", 
                    ["None"] + all_cols,
                    key="scatter_color"
                )
                
                # Size by
                size_by = col2.selectbox(
                    "Size by", 
                    ["None"] + numeric_cols,
                    key="scatter_size"
                )
                
                # Add trendline
                trendline = st.checkbox("Add trendline", key="scatter_trendline")
                
                # Title
                title = st.text_input("Chart title", value=f"{y_col} vs {x_col}", key="scatter_title")
            else:
                color_by = "None"
                size_by = "None"
                trendline = False
                title = f"{y_col} vs {x_col}"
            
            # Create visualization
            try:
                if show_advanced and color_by != "None" and size_by != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        color=color_by,
                        size=size_by,
                        title=title,
                        trendline="ols" if trendline else None
                    )
                elif show_advanced and color_by != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        color=color_by,
                        title=title,
                        trendline="ols" if trendline else None
                    )
                elif show_advanced and size_by != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_by,
                        title=title,
                        trendline="ols" if trendline else None
                    )
                else:
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        trendline="ols" if trendline else None
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
        
        elif viz_type == "Line Chart":
            # For line chart, need an x-axis (usually time or ordered variable)
            x_col = st.selectbox("X-axis", all_cols, key="line_x")
            y_cols = st.multiselect("Y-axis (one or more)", numeric_cols, key="line_y")
            
            if not y_cols:
                st.info("Please select at least one Y-axis column.")
                return
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="line_advanced")
            
            if show_advanced:
                # Title
                title = st.text_input("Chart title", value="Line Chart", key="line_title")
                
                # Group by
                group_by = st.selectbox(
                    "Group lines by", 
                    ["None"] + cat_cols,
                    key="line_group"
                )
                
                # Line type
                line_mode = st.selectbox(
                    "Line mode",
                    ["lines", "lines+markers", "markers"],
                    key="line_mode"
                )
                
                # Sort data
                sort_x = st.checkbox("Sort by X-axis", value=True, key="sort_x")
            else:
                title = "Line Chart"
                group_by = "None"
                line_mode = "lines"
                sort_x = True
            
            # Create visualization
            try:
                # Prepare data - sort if needed
                plot_df = self.df.copy()
                if sort_x:
                    try:
                        plot_df = plot_df.sort_values(by=x_col)
                    except:
                        st.warning(f"Could not sort by {x_col}. Showing unsorted data.")
                
                if show_advanced and group_by != "None":
                    fig = px.line(
                        plot_df,
                        x=x_col,
                        y=y_cols,
                        color=group_by,
                        title=title
                    )
                else:
                    fig = go.Figure()
                    
                    for y_col in y_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df[x_col],
                                y=plot_df[y_col],
                                mode=line_mode,
                                name=y_col
                            )
                        )
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title=x_col,
                        yaxis_title="Value"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating line chart: {str(e)}")
        
        elif viz_type == "Bar Chart":
            # For bar chart, need an x-axis (usually categorical) and y-axis (numeric)
            orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="bar_orientation")
            
            if orientation == "Vertical":
                x_col = st.selectbox("X-axis (categories)", all_cols, key="bar_x")
                y_col = st.selectbox("Y-axis (values)", numeric_cols, key="bar_y")
            else:
                y_col = st.selectbox("Y-axis (categories)", all_cols, key="bar_y")
                x_col = st.selectbox("X-axis (values)", numeric_cols, key="bar_x")
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="bar_advanced")
            
            if show_advanced:
                # Aggregation method
                agg_method = st.selectbox(
                    "Aggregation method",
                    ["sum", "mean", "count", "median", "min", "max"],
                    key="bar_agg"
                )
                
                # Color by
                color_by = st.selectbox(
                    "Color by", 
                    ["None"] + cat_cols,
                    key="bar_color"
                )
                
                # Title
                title = st.text_input("Chart title", value="Bar Chart", key="bar_title")
                
                # Sort bars
                sort_bars = st.checkbox("Sort bars", value=False, key="sort_bars")
                sort_dir = st.radio("Sort direction", ["Ascending", "Descending"], key="sort_dir")
                
                # Limit categories
                limit_cats = st.checkbox("Limit categories", value=False, key="limit_cats")
                if limit_cats:
                    top_n = st.number_input("Show top N categories", min_value=1, value=10, key="top_n")
            else:
                agg_method = "sum"
                color_by = "None"
                title = "Bar Chart"
                sort_bars = False
                sort_dir = "Descending"
                limit_cats = False
                top_n = 10
            
            # Create visualization
            try:
                # Prepare data - aggregate if needed
                if agg_method != "count":
                    if orientation == "Vertical":
                        if show_advanced and color_by != "None":
                            agg_df = self.df.groupby([x_col, color_by])[y_col].agg(agg_method).reset_index()
                        else:
                            agg_df = self.df.groupby(x_col)[y_col].agg(agg_method).reset_index()
                    else:
                        if show_advanced and color_by != "None":
                            agg_df = self.df.groupby([y_col, color_by])[x_col].agg(agg_method).reset_index()
                        else:
                            agg_df = self.df.groupby(y_col)[x_col].agg(agg_method).reset_index()
                else:
                    # Count aggregation
                    if orientation == "Vertical":
                        if show_advanced and color_by != "None":
                            agg_df = self.df.groupby([x_col, color_by]).size().reset_index(name=y_col)
                        else:
                            agg_df = self.df.groupby(x_col).size().reset_index(name=y_col)
                    else:
                        if show_advanced and color_by != "None":
                            agg_df = self.df.groupby([y_col, color_by]).size().reset_index(name=x_col)
                        else:
                            agg_df = self.df.groupby(y_col).size().reset_index(name=x_col)
                
                # Sort if requested
                if show_advanced and sort_bars:
                    if orientation == "Vertical":
                        agg_df = agg_df.sort_values(by=y_col, ascending=(sort_dir == "Ascending"))
                    else:
                        agg_df = agg_df.sort_values(by=x_col, ascending=(sort_dir == "Ascending"))
                
                # Limit categories if requested
                if show_advanced and limit_cats:
                    if orientation == "Vertical":
                        if sort_bars:
                            agg_df = agg_df.head(top_n)
                        else:
                            value_col = y_col
                            if show_advanced and color_by != "None":
                                # Get top categories considering the color groups
                                top_cats = agg_df.groupby(x_col)[value_col].sum().sort_values(
                                    ascending=False).head(top_n).index
                                agg_df = agg_df[agg_df[x_col].isin(top_cats)]
                            else:
                                agg_df = agg_df.nlargest(top_n, value_col)
                    else:
                        if sort_bars:
                            agg_df = agg_df.head(top_n)
                        else:
                            value_col = x_col
                            if show_advanced and color_by != "None":
                                # Get top categories considering the color groups
                                top_cats = agg_df.groupby(y_col)[value_col].sum().sort_values(
                                    ascending=False).head(top_n).index
                                agg_df = agg_df[agg_df[y_col].isin(top_cats)]
                            else:
                                agg_df = agg_df.nlargest(top_n, value_col)
                
                # Create the plot
                if show_advanced and color_by != "None":
                    if orientation == "Vertical":
                        fig = px.bar(
                            agg_df,
                            x=x_col,
                            y=y_col,
                            color=color_by,
                            title=title,
                            barmode="group"
                        )
                    else:
                        fig = px.bar(
                            agg_df,
                            x=x_col,
                            y=y_col,
                            color=color_by,
                            title=title,
                            barmode="group",
                            orientation='h'
                        )
                else:
                    if orientation == "Vertical":
                        fig = px.bar(
                            agg_df,
                            x=x_col,
                            y=y_col,
                            title=title
                        )
                    else:
                        fig = px.bar(
                            agg_df,
                            x=x_col,
                            y=y_col,
                            title=title,
                            orientation='h'
                        )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating bar chart: {str(e)}")
        
        elif viz_type == "Histogram":
            # For histogram, need a numeric column
            col = st.selectbox("Select column", numeric_cols, key="hist_col")
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="hist_advanced")
            
            if show_advanced:
                col1, col2 = st.columns(2)
                
                # Number of bins
                bins = col1.slider("Number of bins", 5, 100, 20, key="hist_bins")
                
                # Density vs count
                hist_type = col2.radio("Histogram type", ["count", "density"], key="hist_type")
                
                # Color by
                color_by = st.selectbox(
                    "Color by (categorical)", 
                    ["None"] + cat_cols,
                    key="hist_color"
                )
                
                # Show kde
                show_kde = st.checkbox("Show density curve", value=True, key="show_kde")
                
                # Histogram mode for multiple groups
                if color_by != "None":
                    hist_mode = st.radio("Histogram mode", ["overlay", "stack"], key="hist_mode")
                else:
                    hist_mode = "overlay"
                
                # Title
                title = st.text_input("Chart title", value=f"Distribution of {col}", key="hist_title")
            else:
                bins = 20
                hist_type = "count"
                color_by = "None"
                show_kde = False
                hist_mode = "overlay"
                title = f"Distribution of {col}"
            
            # Create visualization
            try:
                if show_advanced and color_by != "None":
                    fig = px.histogram(
                        self.df,
                        x=col,
                        color=color_by,
                        nbins=bins,
                        histnorm="probability density" if hist_type == "density" else None,
                        title=title,
                        barmode=hist_mode
                    )
                    
                    if show_kde:
                        # For grouped data, not adding KDE as it becomes too complex for a simple UI
                        st.info("KDE curves not available with grouped histograms")
                else:
                    fig = go.Figure()
                    
                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=self.df[col],
                        nbinsx=bins,
                        histnorm="probability density" if hist_type == "density" else None,
                        name=col
                    ))
                    
                    if show_advanced and show_kde:
                        # Calculate KDE
                        from scipy import stats as scipy_stats
                        
                        # Remove NaN values
                        kde_data = self.df[col].dropna()
                        
                        if len(kde_data) > 1:  # Need at least 2 points for KDE
                            kde_x = np.linspace(kde_data.min(), kde_data.max(), 1000)
                            kde_y = scipy_stats.gaussian_kde(kde_data)(kde_x)
                            
                            # Add KDE line
                            fig.add_trace(go.Scatter(
                                x=kde_x, 
                                y=kde_y,
                                mode='lines',
                                name='Density Curve',
                                line=dict(color='red', width=2)
                            ))
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title=col,
                        yaxis_title="Density" if hist_type == "density" else "Count"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating histogram: {str(e)}")
        
        elif viz_type == "Box Plot":
            # For box plot, need a numeric column and optionally a grouping column
            y_col = st.selectbox("Value column", numeric_cols, key="box_y")
            
            # Optional grouping
            use_groups = st.checkbox("Group by category", key="use_box_groups")
            
            if use_groups:
                x_col = st.selectbox("Grouping column", cat_cols, key="box_x")
            else:
                x_col = None
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="box_advanced")
            
            if show_advanced:
                # Show points
                show_points = st.checkbox("Show all points", value=False, key="show_points")
                
                # Notched box plot
                notched = st.checkbox("Show notched box plot", value=False, key="notched_box", 
                                     help="Notches display confidence interval around median")
                
                # Title
                title = st.text_input("Chart title", value=f"Distribution of {y_col}", key="box_title")
                
                # Orientation
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="box_orientation")
            else:
                show_points = False
                notched = False
                title = f"Distribution of {y_col}"
                orientation = "Vertical"
            
            # Create visualization
            try:
                if use_groups:
                    if show_advanced and orientation == "Horizontal":
                        fig = px.box(
                            self.df,
                            x=y_col,
                            y=x_col,
                            title=title,
                            notched=notched,
                            points="all" if show_points else "outliers"
                        )
                    else:
                        fig = px.box(
                            self.df,
                            x=x_col,
                            y=y_col,
                            title=title,
                            notched=notched,
                            points="all" if show_points else "outliers"
                        )
                else:
                    fig = go.Figure()
                    
                    if show_advanced and orientation == "Horizontal":
                        fig.add_trace(go.Box(
                            x=self.df[y_col],
                            name=y_col,
                            notched=notched,
                            boxpoints="all" if show_points else "outliers"
                        ))
                    else:
                        fig.add_trace(go.Box(
                            y=self.df[y_col],
                            name=y_col,
                            notched=notched,
                            boxpoints="all" if show_points else "outliers"
                        ))
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title=None,
                        yaxis_title=y_col
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating box plot: {str(e)}")
        
        elif viz_type == "Heatmap":
            # For heatmap, typically need numeric data in a matrix form
            heatmap_type = st.radio(
                "Heatmap type",
                ["Correlation matrix", "Two categorical variables (counts)", "Custom values"],
                key="heatmap_type"
            )
            
            if heatmap_type == "Correlation matrix":
                # Select numeric columns for correlation
                corr_cols = st.multiselect(
                    "Select columns for correlation", 
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    key="corr_cols"
                )
                
                if not corr_cols or len(corr_cols) < 2:
                    st.info("Please select at least 2 columns for correlation heatmap.")
                    return
                
                # Correlation method
                corr_method = st.selectbox(
                    "Correlation method",
                    ["pearson", "spearman", "kendall"],
                    key="corr_method"
                )
                
                # Calculate correlation matrix
                corr_matrix = self.df[corr_cols].corr(method=corr_method)
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method.capitalize()} Correlation Heatmap",
                    zmin=-1,
                    zmax=1
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif heatmap_type == "Two categorical variables (counts)":
                # Select two categorical columns
                if len(cat_cols) < 2:
                    st.warning("Need at least two categorical columns for this heatmap type.")
                    return
                
                col1, col2 = st.columns(2)
                x_col = col1.selectbox("X-axis column", cat_cols, key="heat_x")
                y_col = col2.selectbox("Y-axis column", cat_cols, index=min(1, len(cat_cols)-1), key="heat_y")
                
                # Count aggregation or another metric
                use_metric = st.checkbox("Use metric instead of count", key="use_metric")
                
                if use_metric:
                    metric_col = st.selectbox("Metric column", numeric_cols, key="metric_col")
                    agg_func = st.selectbox(
                        "Aggregation function",
                        ["mean", "sum", "median", "min", "max", "count"],
                        key="heat_agg"
                    )
                
                # Create cross-tabulation
                if use_metric:
                    # Pivot table with metric
                    pivot_table = self.df.pivot_table(
                        index=y_col,
                        columns=x_col,
                        values=metric_col,
                        aggfunc=agg_func,
                        fill_value=0
                    )
                    title = f"{agg_func.capitalize()} of {metric_col} by {x_col} and {y_col}"
                else:
                    # Simple cross-tabulation of counts
                    pivot_table = pd.crosstab(self.df[y_col], self.df[x_col])
                    title = f"Count of instances by {x_col} and {y_col}"
                
                # Create heatmap
                fig = px.imshow(
                    pivot_table,
                    text_auto=True,
                    color_continuous_scale="Viridis",
                    title=title
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif heatmap_type == "Custom values":
                # Select three columns: x, y, and values
                col1, col2, col3 = st.columns(3)
                x_col = col1.selectbox("X-axis column", all_cols, key="custom_heat_x")
                y_col = col2.selectbox("Y-axis column", all_cols, index=min(1, len(all_cols)-1), key="custom_heat_y")
                value_col = col3.selectbox("Value column", numeric_cols, key="custom_heat_val")
                
                # Aggregation function
                agg_func = st.selectbox(
                    "Aggregation function",
                    ["mean", "sum", "median", "min", "max", "count"],
                    key="custom_heat_agg"
                )
                
                # Create pivot table
                pivot_table = self.df.pivot_table(
                    index=y_col,
                    columns=x_col,
                    values=value_col,
                    aggfunc=agg_func,
                    fill_value=0
                )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_table,
                    text_auto=True,
                    color_continuous_scale="Viridis",
                    title=f"{agg_func.capitalize()} of {value_col} by {x_col} and {y_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            # For pie chart, need a categorical column for slices
            cat_col = st.selectbox("Category column", cat_cols, key="pie_cat")
            
            # Value column (optional)
            use_values = st.checkbox("Use custom values instead of counts", key="use_pie_values")
            
            if use_values:
                value_col = st.selectbox("Value column", numeric_cols, key="pie_value")
                agg_func = st.selectbox(
                    "Aggregation function",
                    ["sum", "mean", "median", "min", "max"],
                    key="pie_agg"
                )
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="pie_advanced")
            
            if show_advanced:
                # Title
                title = st.text_input("Chart title", value=f"Distribution of {cat_col}", key="pie_title")
                
                # Donut chart
                donut = st.checkbox("Show as donut chart", key="pie_donut")
                
                # Limit categories
                limit_cats = st.checkbox("Limit categories", value=True, key="pie_limit")
                if limit_cats:
                    top_n = st.number_input("Show top N categories", min_value=1, value=10, key="pie_top_n")
                    other_cat = st.checkbox("Group remaining as 'Other'", value=True, key="pie_other")
                else:
                    top_n = 10
                    other_cat = True
            else:
                title = f"Distribution of {cat_col}"
                donut = False
                limit_cats = True
                top_n = 10
                other_cat = True
            
            # Create visualization
            try:
                # Prepare data
                if use_values:
                    # Aggregate values by category
                    agg_df = self.df.groupby(cat_col)[value_col].agg(agg_func).reset_index()
                    value_label = f"{agg_func.capitalize()} of {value_col}"
                    values = agg_df[value_col]
                    names = agg_df[cat_col]
                else:
                    # Count by category
                    value_counts = self.df[cat_col].value_counts()
                    value_label = "Count"
                    values = value_counts.values
                    names = value_counts.index
                
                # Limit categories if requested
                if show_advanced and limit_cats and len(names) > top_n:
                    if other_cat:
                        # Combine small categories as 'Other'
                        top_values = values[:top_n-1]
                        top_names = names[:top_n-1]
                        other_value = sum(values[top_n-1:])
                        
                        values = np.append(top_values, other_value)
                        names = np.append(top_names, "Other")
                    else:
                        # Just truncate
                        values = values[:top_n]
                        names = names[:top_n]
                
                # Create pie chart
                fig = px.pie(
                    values=values,
                    names=names,
                    title=title
                )
                
                # Convert to donut if requested
                if show_advanced and donut:
                    fig.update_traces(hole=0.4)
                
                # Update layout
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating pie chart: {str(e)}")
        
        elif viz_type == "3D Scatter Plot":
            # For 3D scatter, need three numeric columns
            if len(numeric_cols) < 3:
                st.warning("Need at least three numeric columns for 3D scatter plot.")
                return
            
            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox("X-axis", numeric_cols, index=0, key="scatter3d_x")
            y_col = col2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter3d_y")
            z_col = col3.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1), key="scatter3d_z")
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="scatter3d_advanced")
            
            if show_advanced:
                # Color by
                color_by = st.selectbox(
                    "Color by", 
                    ["None"] + all_cols,
                    key="scatter3d_color"
                )
                
                # Size by
                size_by = st.selectbox(
                    "Size by", 
                    ["None"] + numeric_cols,
                    key="scatter3d_size"
                )
                
                # Title
                title = st.text_input("Chart title", value="3D Scatter Plot", key="scatter3d_title")
            else:
                color_by = "None"
                size_by = "None"
                title = "3D Scatter Plot"
            
            # Create visualization
            try:
                if show_advanced and color_by != "None" and size_by != "None":
                    fig = px.scatter_3d(
                        self.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=color_by,
                        size=size_by,
                        title=title
                    )
                elif show_advanced and color_by != "None":
                    fig = px.scatter_3d(
                        self.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=color_by,
                        title=title
                    )
                elif show_advanced and size_by != "None":
                    fig = px.scatter_3d(
                        self.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        size=size_by,
                        title=title
                    )
                else:
                    fig = px.scatter_3d(
                        self.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        title=title
                    )
                
                # Update layout for better view
                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D scatter plot: {str(e)}")
        
        elif viz_type == "Bubble Chart":
            # For bubble chart, need X, Y and size columns
            if len(numeric_cols) < 3:
                st.warning("Need at least three numeric columns for bubble chart.")
                return
            
            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox("X-axis", numeric_cols, index=0, key="bubble_x")
            y_col = col2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="bubble_y")
            size_col = col3.selectbox("Size", numeric_cols, index=min(2, len(numeric_cols)-1), key="bubble_size")
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="bubble_advanced")
            
            if show_advanced:
                # Color by
                color_by = st.selectbox(
                    "Color by", 
                    ["None"] + all_cols,
                    key="bubble_color"
                )
                
                # Text labels
                text_col = st.selectbox(
                    "Text labels", 
                    ["None"] + all_cols,
                    key="bubble_text"
                )
                
                # Title
                title = st.text_input("Chart title", value="Bubble Chart", key="bubble_title")
                
                # Bubble size scaling
                size_max = st.slider("Maximum bubble size", 10, 60, 40, key="bubble_size_max")
            else:
                color_by = "None"
                text_col = "None"
                title = "Bubble Chart"
                size_max = 40
            
            # Create visualization
            try:
                if show_advanced and color_by != "None" and text_col != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        color=color_by,
                        text=text_col,
                        title=title,
                        size_max=size_max
                    )
                elif show_advanced and color_by != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        color=color_by,
                        title=title,
                        size_max=size_max
                    )
                elif show_advanced and text_col != "None":
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        text=text_col,
                        title=title,
                        size_max=size_max
                    )
                else:
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        title=title,
                        size_max=size_max
                    )
                
                # Show text when hovering
                if show_advanced and text_col != "None":
                    fig.update_traces(textposition='top center')
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating bubble chart: {str(e)}")
        
        elif viz_type == "Violin Plot":
            # For violin plot, need a numeric column for values
            y_col = st.selectbox("Value column", numeric_cols, key="violin_y")
            
            # Optional grouping
            use_groups = st.checkbox("Group by category", key="use_violin_groups")
            
            if use_groups:
                x_col = st.selectbox("Grouping column", cat_cols, key="violin_x")
            else:
                x_col = None
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="violin_advanced")
            
            if show_advanced:
                # Show box plot inside violin
                show_box = st.checkbox("Show box plot inside", value=True, key="violin_box")
                
                # Show individual points
                show_points = st.checkbox("Show all points", value=False, key="violin_points")
                
                # Title
                title = st.text_input("Chart title", value=f"Distribution of {y_col}", key="violin_title")
                
                # Orientation
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="violin_orientation")
                
                # Color by (if not grouped)
                if not use_groups:
                    color_by = st.selectbox(
                        "Color by", 
                        ["None"] + cat_cols,
                        key="violin_color"
                    )
                else:
                    color_by = "None"
            else:
                show_box = True
                show_points = False
                title = f"Distribution of {y_col}"
                orientation = "Vertical"
                color_by = "None"
            
            # Create visualization
            try:
                if use_groups:
                    if show_advanced and orientation == "Horizontal":
                        fig = px.violin(
                            self.df,
                            x=y_col,
                            y=x_col,
                            title=title,
                            box=show_box,
                            points="all" if show_points else False
                        )
                    else:
                        fig = px.violin(
                            self.df,
                            x=x_col,
                            y=y_col,
                            title=title,
                            box=show_box,
                            points="all" if show_points else False
                        )
                else:
                    if show_advanced and color_by != "None":
                        if orientation == "Horizontal":
                            fig = px.violin(
                                self.df,
                                x=y_col,
                                color=color_by,
                                title=title,
                                box=show_box,
                                points="all" if show_points else False
                            )
                        else:
                            fig = px.violin(
                                self.df,
                                y=y_col,
                                color=color_by,
                                title=title,
                                box=show_box,
                                points="all" if show_points else False
                            )
                    else:
                        fig = go.Figure()
                        
                        if orientation == "Horizontal":
                            fig.add_trace(go.Violin(
                                x=self.df[y_col],
                                name=y_col,
                                box_visible=show_box,
                                meanline_visible=True,
                                points="all" if show_points else False
                            ))
                        else:
                            fig.add_trace(go.Violin(
                                y=self.df[y_col],
                                name=y_col,
                                box_visible=show_box,
                                meanline_visible=True,
                                points="all" if show_points else False
                            ))
                        
                        fig.update_layout(
                            title=title,
                            xaxis_title=None,
                            yaxis_title=y_col
                        )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating violin plot: {str(e)}")
        
        elif viz_type == "Radar Chart":
            # For radar chart, need numeric columns for axes
            if len(numeric_cols) < 3:
                st.warning("Need at least three numeric columns for radar chart.")
                return
            
            # Select columns for radar axes
            radar_cols = st.multiselect(
                "Select columns for radar axes", 
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="radar_cols"
            )
            
            if len(radar_cols) < 3:
                st.info("Please select at least 3 columns for radar chart.")
                return
            
            # Group by column (optional)
            use_groups = st.checkbox("Group by category", key="use_radar_groups")
            
            if use_groups:
                group_col = st.selectbox("Grouping column", cat_cols, key="radar_group")
                
                # Limit groups (if too many)
                unique_groups = self.df[group_col].nunique()
                if unique_groups > 10:
                    st.warning(f"Found {unique_groups} unique groups. Please select specific groups to plot.")
                    selected_groups = st.multiselect(
                        "Select groups to include",
                        self.df[group_col].dropna().unique(),
                        default=self.df[group_col].value_counts().head(5).index.tolist(),
                        key="radar_selected_groups"
                    )
                    
                    if not selected_groups:
                        st.info("Please select at least one group to plot.")
                        return
                    
                    # Filter data to selected groups
                    plot_df = self.df[self.df[group_col].isin(selected_groups)]
                else:
                    selected_groups = self.df[group_col].dropna().unique()
                    plot_df = self.df
            else:
                group_col = None
                plot_df = self.df
                selected_groups = None
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="radar_advanced")
            
            if show_advanced:
                # Title
                title = st.text_input("Chart title", value="Radar Chart", key="radar_title")
                
                # Scaling method
                scale_method = st.selectbox(
                    "Scale variables",
                    ["None", "Min-Max (0-1)", "Z-score", "Robust scaling"],
                    key="radar_scale"
                )
                
                # Fill area
                fill_type = st.selectbox(
                    "Fill type",
                    ["toself", "tonext", "none"],
                    key="radar_fill"
                )
            else:
                title = "Radar Chart"
                scale_method = "Min-Max (0-1)"
                fill_type = "toself"
            
            # Create visualization
            try:
                # Create scaled dataframe for radar
                radar_df = plot_df.copy()
                
                # Scale the data
                if show_advanced and scale_method != "None":
                    for col in radar_cols:
                        if scale_method == "Min-Max (0-1)":
                            min_val = radar_df[col].min()
                            max_val = radar_df[col].max()
                            if max_val > min_val:  # Avoid division by zero
                                radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
                        
                        elif scale_method == "Z-score":
                            mean_val = radar_df[col].mean()
                            std_val = radar_df[col].std()
                            if std_val > 0:  # Avoid division by zero
                                radar_df[col] = (radar_df[col] - mean_val) / std_val
                        
                        elif scale_method == "Robust scaling":
                            q1 = radar_df[col].quantile(0.25)
                            q3 = radar_df[col].quantile(0.75)
                            iqr = q3 - q1
                            if iqr > 0:  # Avoid division by zero
                                radar_df[col] = (radar_df[col] - q1) / iqr
                
                # Create radar chart
                fig = go.Figure()
                
                if use_groups:
                    # Add trace for each group
                    for group in selected_groups:
                        group_data = radar_df[radar_df[group_col] == group]
                        
                        if len(group_data) > 0:
                            # Calculate mean for each column in this group
                            means = [group_data[col].mean() for col in radar_cols]
                            
                            # Close the loop
                            radar_values = means + [means[0]]
                            radar_axes = radar_cols + [radar_cols[0]]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=radar_values,
                                theta=radar_axes,
                                name=str(group),
                                fill=fill_type
                            ))
                else:
                    # Single trace for all data
                    means = [radar_df[col].mean() for col in radar_cols]
                    
                    # Close the loop
                    radar_values = means + [means[0]]
                    radar_axes = radar_cols + [radar_cols[0]]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=radar_values,
                        theta=radar_axes,
                        fill=fill_type
                    ))
                
                fig.update_layout(
                    title=title,
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1] if scale_method == "Min-Max (0-1)" else None
                        )
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating radar chart: {str(e)}")
    
    def _render_chat_history(self):
        """Render the chat history"""
        if not st.session_state.chat_history:
            return
        
        # Create chat container
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <div class='chat-message-content'>
                        <p>{message["content"]}</p>
                    </div>
                    <div class='chat-message-avatar'>👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div class='chat-message-avatar'>🤖</div>
                    <div class='chat-message-content'>
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display visualization if available
                if "visualization" in message and message["visualization"] is not None:
                    if message["viz_type"] == "plotly":
                        st.plotly_chart(
                            message["visualization"], 
                            use_container_width=True, 
                            key=f"plot_{idx}_{uuid.uuid4()}"  # Add a unique key with UUID 
                        )
                    elif message["viz_type"] == "matplotlib":
                        st.pyplot(message["visualization"], key=f"mpl_{idx}")  # Also add key here for consistency
                
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _generate_response(self, query):
        """Generate a response to the user query"""
        query_lower = query.lower()
        
        # Check if we have a cached response for this query
        cache_key = query_lower.strip()
        if cache_key in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[cache_key]
        
        # Basic dataset summary
        if any(phrase in query_lower for phrase in ["summarize", "summary", "describe", "overview"]):
            response = self._summarize_dataset()
        
        # Correlation analysis
        elif "correlation" in query_lower or "correlate" in query_lower or "relationship between" in query_lower:
            response = self._analyze_correlations(query)
        
        # Distribution analysis
        elif any(phrase in query_lower for phrase in ["distribution", "histogram", "spread"]):
            response = self._analyze_distributions(query)
        
        # Statistical insights
        elif any(phrase in query_lower for phrase in ["statistics", "average", "mean", "median", "min", "max"]):
            response = self._provide_statistics(query)
        
        # Trend analysis
        elif any(phrase in query_lower for phrase in ["trend", "change over", "time series"]):
            response = self._analyze_trends(query)
        
        # Outlier detection
        elif any(phrase in query_lower for phrase in ["outlier", "anomaly", "unusual"]):
            response = self._detect_outliers(query)
        
        # Visualization requests
        elif any(phrase in query_lower for phrase in ["plot", "chart", "graph", "visualize", "visualization", "figure"]):
            response = self._create_visualization(query)
        
        # Compare groups
        elif any(phrase in query_lower for phrase in ["compare", "comparison", "versus", "vs"]):
            response = self._compare_groups(query)
        
        # General insights
        elif any(phrase in query_lower for phrase in ["insight", "pattern", "discover", "interesting", "important"]):
            response = self._provide_insights()
        
        # Default response for other queries
        else:
            response = self._default_response(query)
        
        # Cache the response
        st.session_state.analysis_cache[cache_key] = response
        
        return response
    
    def _summarize_dataset(self):
        """Provide a summary of the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Basic dataset info
        rows, cols = self.df.shape
        missing_values = self.df.isna().sum().sum()
        missing_percentage = (missing_values / (rows * cols)) * 100
        
        # Column types
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Generate summary text
        summary = f"""
        ### Dataset Summary
        
        This dataset contains **{rows:,} rows** and **{cols} columns**. 
        
        **Data Quality:**
        - Missing values: {missing_values:,} ({missing_percentage:.2f}% of all cells)
        - Duplicate rows: {self.df.duplicated().sum():,}
        
        **Column Types:**
        - Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{"..." if len(numeric_cols) > 5 else ""}
        - Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{"..." if len(categorical_cols) > 5 else ""}
        - Date columns ({len(date_cols)}): {', '.join(date_cols)}
        
        **Key Statistics:**
        """
        
        # Add statistics for key numeric columns (up to 3)
        for col in numeric_cols[:3]:
            col_stats = self.df[col].describe()
            summary += f"""
            - **{col}**: 
                - Mean: {col_stats['mean']:.2f}
                - Median: {col_stats['50%']:.2f}
                - Min: {col_stats['min']:.2f}
                - Max: {col_stats['max']:.2f}
            """
        
        # Add info for key categorical columns (up to 3)
        for col in categorical_cols[:3]:
            value_counts = self.df[col].value_counts()
            top_value = value_counts.index[0] if not value_counts.empty else "N/A"
            top_count = value_counts.iloc[0] if not value_counts.empty else 0
            unique_count = self.df[col].nunique()
            summary += f"""
            - **{col}**: 
                - Unique values: {unique_count}
                - Most common: {top_value} ({top_count} occurrences)
                - Missing: {self.df[col].isna().sum()}
            """
        
        # Create a visualization showing column data types
        fig = go.Figure()
        
        # Count column types
        type_counts = {
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'Date/Time': len(date_cols),
            'Other': cols - len(numeric_cols) - len(categorical_cols) - len(date_cols)
        }
        
        # Filter out zero counts
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        # Create pie chart
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Data Types",
            template="plotly_white"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return {
            "text": summary,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_correlations(self, query):
        """Analyze correlations in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns for correlation analysis
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {
                "text": "Correlation analysis requires at least two numeric columns, but I could only find " +
                      f"{len(numeric_cols)} numeric column in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use all numeric columns
        if len(mentioned_cols) >= 2:
            cols_to_analyze = mentioned_cols
        else:
            # Use all numeric columns (limit to 10 for readability)
            cols_to_analyze = numeric_cols[:min(10, len(numeric_cols))]
        
        # Calculate correlation matrix
        corr_matrix = self.df[cols_to_analyze].corr()
        
        # Find highest correlations (excluding self-correlations)
        correlations = []
        for i in range(len(cols_to_analyze)):
            for j in range(i+1, len(cols_to_analyze)):
                col1 = cols_to_analyze[i]
                col2 = cols_to_analyze[j]
                corr_value = corr_matrix.loc[col1, col2]
                correlations.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate response text
        response_text = f"""
        ### Correlation Analysis
        
        I've analyzed the relationships between numeric variables in your dataset. 
        Here are the key findings:
        
        **Top Correlations:**
        """
        
        # Add top correlations to response
        for col1, col2, corr in correlations[:5]:
            strength = "strong positive" if corr > 0.7 else \
                      "moderate positive" if corr > 0.3 else \
                      "weak positive" if corr > 0 else \
                      "strong negative" if corr < -0.7 else \
                      "moderate negative" if corr < -0.3 else \
                      "weak negative"
            
            response_text += f"""
            - **{col1}** and **{col2}**: {corr:.3f} ({strength} correlation)
            """
        
        # Create heatmap visualization
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            template="plotly_white",
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )
        
        # Add interpretation if requested
        if "interpret" in query.lower() or "explain" in query.lower() or "insights" in query.lower():
            response_text += """
            
            **Interpretation:**
            
            - **Positive correlation** means that as one variable increases, the other tends to increase as well.
            - **Negative correlation** means that as one variable increases, the other tends to decrease.
            - **Correlation values** range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
            - A value close to 0 suggests little to no linear relationship.
            
            Remember that correlation does not imply causation. Just because two variables are correlated doesn't mean that one causes the other.
            """
            
            # If we have highly correlated variables, add specific insights
            if correlations and abs(correlations[0][2]) > 0.7:
                col1, col2, corr = correlations[0]
                direction = "positively" if corr > 0 else "negatively"
                response_text += f"""
                
                **Key Insight:**
                
                The strongest relationship in your data is between **{col1}** and **{col2}** with a correlation of {corr:.3f}. These variables are strongly {direction} correlated, meaning they tend to {
                "increase together" if corr > 0 else "move in opposite directions"}.
                """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_distributions(self, query):
        """Analyze distributions of variables in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Distribution analysis requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use a subset of numeric columns
        if mentioned_cols:
            cols_to_analyze = mentioned_cols[:4]  # Limit to 4 for visualization
        else:
            # Use a subset of numeric columns (up to 4 for readability)
            cols_to_analyze = numeric_cols[:min(4, len(numeric_cols))]
        
        # Generate response text
        response_text = f"""
        ### Distribution Analysis
        
        I've analyzed the distributions of the numeric variables in your dataset.
        Here are the key findings:
        
        """
        
        # Create visualization
        if len(cols_to_analyze) == 1:
            # Single column distribution
            col = cols_to_analyze[0]
            
            # Check for skewness
            skewness = self.df[col].skew()
            skew_description = "highly right-skewed" if skewness > 1 else \
                              "moderately right-skewed" if skewness > 0.5 else \
                              "approximately symmetric" if abs(skewness) <= 0.5 else \
                              "moderately left-skewed" if skewness > -1 else \
                              "highly left-skewed"
            
            # Calculate basic statistics
            col_stats = self.df[col].describe()
            iqr = col_stats['75%'] - col_stats['25%']
            
            response_text += f"""
            **{col}**:
            - Mean: {col_stats['mean']:.2f}
            - Median: {col_stats['50%']:.2f}
            - Standard Deviation: {col_stats['std']:.2f}
            - Minimum: {col_stats['min']:.2f}
            - Maximum: {col_stats['max']:.2f}
            - Interquartile Range (IQR): {iqr:.2f}
            - Distribution shape: {skew_description} (skewness = {skewness:.2f})
            """
            
            # Generate distribution plot
            fig = px.histogram(
                self.df,
                x=col,
                marginal="box",
                title=f"Distribution of {col}",
                template="plotly_white"
            )
        else:
            # Multiple columns distribution
            response_text += "**Summary Statistics:**\n\n"
            
            for col in cols_to_analyze:
                col_stats = self.df[col].describe()
                skewness = self.df[col].skew()
                skew_description = "right-skewed" if skewness > 0.5 else \
                                  "approximately symmetric" if abs(skewness) <= 0.5 else \
                                  "left-skewed"
                
                response_text += f"""
                **{col}**:
                - Mean: {col_stats['mean']:.2f}
                - Median: {col_stats['50%']:.2f}
                - Distribution: {skew_description}
                """
            
            # Create subplots for multiple distributions
            fig = make_subplots(
                rows=2, 
                cols=2, 
                subplot_titles=[f"Distribution of {col}" for col in cols_to_analyze[:4]]
            )
            
            # Add histograms
            for i, col in enumerate(cols_to_analyze[:4]):
                row = i // 2 + 1
                col_idx = i % 2 + 1
                
                # Add histogram trace
                fig.add_trace(
                    go.Histogram(
                        x=self.df[col],
                        name=col,
                        showlegend=False,
                        opacity=0.7
                    ),
                    row=row, 
                    col=col_idx
                )
                
                # Add KDE
                # Simple KDE calculation for display
                from scipy import stats
                if not self.df[col].dropna().empty:
                    kde_x = np.linspace(
                        self.df[col].min(), 
                        self.df[col].max(), 
                        100
                    )
                    kde = stats.gaussian_kde(self.df[col].dropna())
                    kde_y = kde(kde_x)
                    
                    # Scale KDE to match histogram height
                    hist_values, bin_edges = np.histogram(
                        self.df[col].dropna(), 
                        bins=20, 
                        density=True
                    )
                    scale_factor = 1
                    kde_y = kde_y * scale_factor
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x, 
                            y=kde_y,
                            mode='lines',
                            name=f"{col} KDE",
                            line=dict(width=2),
                            showlegend=False
                        ),
                        row=row, 
                        col=col_idx
                    )
            
            # Update layout
            fig.update_layout(
                title="Distribution Analysis",
                template="plotly_white",
                height=600
            )
        
        # Add interpretation
        response_text += """
        
        **Interpretation:**
        - A **symmetric distribution** means values are evenly distributed around the mean.
        - A **right-skewed distribution** has a long tail to the right, with most values concentrated on the left.
        - A **left-skewed distribution** has a long tail to the left, with most values concentrated on the right.
        - **Outliers** are points that lie far from the majority of the data and may need special attention.
        """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _compute_kde(self, data, points=100):
        """Helper method to compute KDE for a data series"""
        import numpy as np
        from scipy import stats
        
        # Remove NaN values
        data = data.dropna()
        
        if len(data) == 0:
            return np.array([]), np.array([])
        
        # Define the range for the KDE
        min_val = data.min()
        max_val = data.max()
        x = np.linspace(min_val, max_val, points)
        
        # Compute KDE
        kde = stats.gaussian_kde(data)
        y = kde(x)
        
        return x, y
    
    def _provide_statistics(self, query):
        """Provide statistical analysis based on the query"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get all columns
        all_cols = self.df.columns.tolist()
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Statistical analysis requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check which statistic is requested
        stat_type = None
        if "mean" in query.lower() or "average" in query.lower():
            stat_type = "mean"
            stat_function = np.mean
            stat_name = "Mean"
        elif "median" in query.lower():
            stat_type = "median"
            stat_function = np.median
            stat_name = "Median"
        elif "min" in query.lower() or "minimum" in query.lower():
            stat_type = "min"
            stat_function = np.min
            stat_name = "Minimum"
        elif "max" in query.lower() or "maximum" in query.lower():
            stat_type = "max"
            stat_function = np.max
            stat_name = "Maximum"
        elif "sum" in query.lower() or "total" in query.lower():
            stat_type = "sum"
            stat_function = np.sum
            stat_name = "Sum"
        elif "std" in query.lower() or "standard deviation" in query.lower():
            stat_type = "std"
            stat_function = np.std
            stat_name = "Standard Deviation"
        elif "var" in query.lower() or "variance" in query.lower():
            stat_type = "var"
            stat_function = np.var
            stat_name = "Variance"
        else:
            # Default to providing a complete statistical summary
            stat_type = "summary"
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in all_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use numeric columns
        if mentioned_cols:
            cols_to_analyze = [col for col in mentioned_cols if col in numeric_cols]
            if not cols_to_analyze:
                return {
                    "text": f"I found the columns you mentioned ({', '.join(mentioned_cols)}), but they don't appear to be numeric columns that I can analyze statistically."
                }
        else:
            # Use all numeric columns
            cols_to_analyze = numeric_cols
        
        # Generate response based on the requested statistic
        if stat_type == "summary":
            # Full statistical summary
            stats_df = self.df[cols_to_analyze].describe().T
            
            # Round to 2 decimal places for readability
            stats_df = stats_df.round(2)
            
            # Convert to HTML for display
            stats_html = stats_df.to_html()
            
            response_text = f"""
            ### Statistical Summary
            
            Here's a statistical summary of the numeric columns in your dataset:
            
            {stats_html}
            
            **Interpretation:**
            - **count**: Number of non-missing values
            - **mean**: Average value
            - **std**: Standard deviation (measure of spread)
            - **min**: Minimum value
            - **25%**: First quartile (25th percentile)
            - **50%**: Median (50th percentile)
            - **75%**: Third quartile (75th percentile)
            - **max**: Maximum value
            """
            
            # Create visualization (box plots)
            fig = px.box(
                self.df,
                y=cols_to_analyze[:10],  # Limit to 10 columns for readability
                title="Statistical Distribution",
                template="plotly_white"
            )
        else:
            # Specific statistic for each column
            results = {}
            for col in cols_to_analyze:
                results[col] = stat_function(self.df[col].dropna())
            
            # Create response text
            response_text = f"""
            ### {stat_name} Analysis
            
            I've calculated the {stat_name.lower()} for the following columns:
            
            """
            
            for col, value in results.items():
                response_text += f"- **{col}**: {value:.2f}\n"
            
            # Create bar chart visualization
            fig = px.bar(
                x=list(results.keys()),
                y=list(results.values()),
                title=f"{stat_name} by Column",
                labels={"x": "Column", "y": stat_name},
                template="plotly_white"
            )
            
            # Add value labels on top of bars
            fig.update_traces(
                text=[f"{val:.2f}" for val in results.values()],
                textposition="outside"
            )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_trends(self, query):
        """Analyze trends in the data"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Look for potential time/date columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also look for columns that might be dates but not detected as such
        potential_date_cols = []
        for col in self.df.columns:
            if col in date_cols:
                continue
                
            # Look for columns with "date", "time", "year", "month", "day" in the name
            if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                potential_date_cols.append(col)
        
        # Combine confirmed and potential date columns
        time_cols = date_cols + potential_date_cols
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in self.df.columns if col.lower() in query.lower()]
        time_col = None
        
        # Try to identify the time column and value column from the query
        if mentioned_cols:
            for col in mentioned_cols:
                if col in time_cols:
                    time_col = col
                    break
            
            # If no time column found among mentioned columns, try to find a numeric column
            value_cols = [col for col in mentioned_cols if col in self.df.select_dtypes(include=['number']).columns.tolist()]
            if not value_cols and not time_col:
                return {
                    "text": "I couldn't identify suitable columns for trend analysis from your query. Please specify a time/date column and a numeric column to analyze trends."
                }
        
        # If no time column identified yet, use the first available time column
        if not time_col and time_cols:
            time_col = time_cols[0]
        
        # If still no time column, look for any ordered numeric column that could serve as a time index
        if not time_col:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                # Check if values are mostly sequential (could be a time index)
                if len(self.df) > 1 and self.df[col].is_monotonic_increasing:
                    time_col = col
                    break
        
        # If still no time column, return an error message
        if not time_col:
            return {
                "text": "I couldn't identify a suitable time or date column for trend analysis. Please specify a column that represents time or a sequential order."
            }
        
        # Get numeric columns for analysis
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != time_col]  # Exclude time column if it's numeric
        
        if not numeric_cols:
            return {
                "text": "Trend analysis requires numeric columns to analyze, but I couldn't find any suitable columns in this dataset."
            }
        
        # If value columns were identified from the query, use them; otherwise, use a subset of numeric columns
        if 'value_cols' in locals() and value_cols:
            cols_to_analyze = value_cols[:3]  # Limit to 3 for visualization
        else:
            # Use a subset of numeric columns (up to 3 for readability)
            cols_to_analyze = numeric_cols[:min(3, len(numeric_cols))]
        
        # Sort by time column
        df_sorted = self.df.sort_values(time_col)
        
        # Generate response text
        response_text = f"""
        ### Trend Analysis
        
        I've analyzed trends over time using **{time_col}** as the time dimension.
        Here are the key findings:
        """
        
        # Create trend visualization
        fig = go.Figure()
        
        for col in cols_to_analyze:
            # Add line trace
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[time_col],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                )
            )
            
            # Calculate trend (simple linear regression)
            if len(df_sorted) >= 2:
                try:
                    # Convert to numeric index for regression
                    x = np.arange(len(df_sorted))
                    y = df_sorted[col].values
                    
                    # Remove NaN values
                    mask = ~np.isnan(y)
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) >= 2:
                        slope, intercept = np.polyfit(x_clean, y_clean, 1)
                        
                        # Interpret trend
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        trend_strength = "strongly" if abs(slope) > 0.1 else "moderately" if abs(slope) > 0.01 else "slightly"
                        
                        response_text += f"""
                        
                        **{col}**:
                        - Overall trend: {trend_strength} {trend_direction}
                        - Change rate: {slope:.4f} units per step
                        """
                        
                        # Add trendline to plot
                        trend_y = intercept + slope * x
                        fig.add_trace(
                            go.Scatter(
                                x=df_sorted[time_col],
                                y=trend_y,
                                mode='lines',
                                line=dict(dash='dash'),
                                name=f'{col} Trend',
                                showlegend=True
                            )
                        )
                except Exception as e:
                    # Skip trendline if there's an error (e.g., non-numeric data)
                    pass
        
        # Update layout
        fig.update_layout(
            title=f"Trends over {time_col}",
            xaxis_title=time_col,
            yaxis_title="Value",
            template="plotly_white"
        )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _detect_outliers(self, query):
        """Detect outliers in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Outlier detection requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use a subset of numeric columns
        if mentioned_cols:
            cols_to_analyze = mentioned_cols[:4]  # Limit to 4 for visualization
        else:
            # Use a subset of numeric columns (up to 4 for readability)
            cols_to_analyze = numeric_cols[:min(4, len(numeric_cols))]
        
        # Generate response text
        response_text = f"""
        ### Outlier Detection
        
        I've analyzed the following columns for outliers:
        """
        
        # Find outliers using IQR method
        outlier_results = {}
        
        for col in cols_to_analyze:
            # Calculate Q1, Q3, and IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_results[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage
            }
            
            response_text += f"""
            
            **{col}**:
            - Outlier count: {outlier_count} ({outlier_percentage:.2f}% of data)
            - Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]
            """
        
        # Create box plot visualization to show outliers
        fig = px.box(
            self.df,
            y=cols_to_analyze,
            title="Outlier Analysis",
            template="plotly_white"
        )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_visualization(self, query):
        """Create a visualization based on the query"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to visualize. Please upload a dataset first."
            }
        
        # Determine visualization type from the query
        viz_type = None
        if any(term in query.lower() for term in ["bar", "column"]):
            viz_type = "bar"
        elif any(term in query.lower() for term in ["line", "trend", "time series"]):
            viz_type = "line"
        elif any(term in query.lower() for term in ["scatter", "relationship", "correlation"]):
            viz_type = "scatter"
        elif any(term in query.lower() for term in ["histogram", "distribution"]):
            viz_type = "histogram"
        elif any(term in query.lower() for term in ["pie", "proportion", "percentage"]):
            viz_type = "pie"
        elif any(term in query.lower() for term in ["box", "boxplot", "range"]):
            viz_type = "box"
        elif any(term in query.lower() for term in ["heatmap", "correlation matrix"]):
            viz_type = "heatmap"
        else:
            # Default to bar chart
            viz_type = "bar"
        
        # Extract columns from query
        all_cols = self.df.columns.tolist()
        mentioned_cols = [col for col in all_cols if col.lower() in query.lower()]
        
        # Get numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create the visualization based on type
        if viz_type == "bar":
            return self._create_bar_visualization(query, mentioned_cols, numeric_cols, categorical_cols)
        elif viz_type == "line":
            return self._create_line_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "scatter":
            return self._create_scatter_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "histogram":
            return self._create_histogram_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "pie":
            return self._create_pie_visualization(query, mentioned_cols, categorical_cols)
        elif viz_type == "box":
            return self._create_box_visualization(query, mentioned_cols, numeric_cols, categorical_cols)
        elif viz_type == "heatmap":
            return self._create_heatmap_visualization(query, mentioned_cols, numeric_cols)
    
    def _create_bar_visualization(self, query, mentioned_cols, numeric_cols, categorical_cols):
        """Create a bar chart visualization"""
        # Need at least one categorical and one numeric column for a bar chart
        if not categorical_cols or not numeric_cols:
            return {
                "text": "I couldn't create a bar chart because I need at least one categorical column and one numeric column."
            }
        
        # Select columns for the bar chart
        x_col = None
        y_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for a categorical column first
            for col in mentioned_cols:
                if col in categorical_cols:
                    x_col = col
                    break
            
            # Then look for a numeric column
            for col in mentioned_cols:
                if col in numeric_cols and col != x_col:
                    y_col = col
                    break
        
        # If no suitable columns found from mentions, use defaults
        if not x_col:
            x_col = categorical_cols[0]
        
        if not y_col:
            # Default to count if no specific numeric column
            y_col = "count"
        
        # Create the bar chart
        if y_col == "count":
            # Count by category
            df_grouped = self.df.groupby(x_col).size().reset_index(name='count')
            
            # Sort by count for better visualization
            df_grouped = df_grouped.sort_values('count', ascending=False)
            
            # Limit to top 15 categories if there are too many
            if len(df_grouped) > 15:
                df_grouped = df_grouped.head(15)
            
            fig = px.bar(
                df_grouped,
                x=x_col,
                y='count',
                title=f"Count by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Bar Chart: Count by {x_col}
            
            I've created a bar chart showing the count of items in each {x_col} category.
            
            **Key observations:**
            - Most common {x_col}: {df_grouped.iloc[0][x_col]} ({df_grouped.iloc[0]['count']} instances)
            - Least common {x_col} (shown): {df_grouped.iloc[-1][x_col]} ({df_grouped.iloc[-1]['count']} instances)
            """
        else:
            # Aggregate by category and value
            df_grouped = self.df.groupby(x_col)[y_col].mean().reset_index()
            
            # Sort by the numeric column for better visualization
            df_grouped = df_grouped.sort_values(y_col, ascending=False)
            
            # Limit to top 15 categories if there are too many
            if len(df_grouped) > 15:
                df_grouped = df_grouped.head(15)
            
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                title=f"Average {y_col} by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Bar Chart: Average {y_col} by {x_col}
            
            I've created a bar chart showing the average {y_col} for each {x_col} category.
            
            **Key observations:**
            - Highest average {y_col}: {df_grouped.iloc[0][x_col]} ({df_grouped.iloc[0][y_col]:.2f})
            - Lowest average {y_col} (shown): {df_grouped.iloc[-1][x_col]} ({df_grouped.iloc[-1][y_col]:.2f})
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_line_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a line chart visualization"""
        # Need at least one column to use as x-axis and one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a line chart because I need at least one numeric column."
            }
        
        # Look for potential time/date columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also look for columns that might be dates but not detected as such
        potential_date_cols = []
        for col in self.df.columns:
            if col in date_cols:
                continue
                
            # Look for columns with "date", "time", "year", "month", "day" in the name
            if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                potential_date_cols.append(col)
        
        # Combine confirmed and potential date columns
        time_cols = date_cols + potential_date_cols
        
        # Select columns for the line chart
        x_col = None
        y_cols = []
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for a time column first
            for col in mentioned_cols:
                if col in time_cols:
                    x_col = col
                    break
            
            # Look for numeric columns
            for col in mentioned_cols:
                if col in numeric_cols and col != x_col:
                    y_cols.append(col)
        
        # If no time column found, look for a column that's sortable
        if not x_col:
            # Use the first time column if available
            if time_cols:
                x_col = time_cols[0]
            else:
                # Otherwise, use the first column as x-axis
                x_col = self.df.columns[0]
        
        # If no y columns found, use all numeric columns (up to 3)
        if not y_cols:
            y_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        # Limit to 3 y columns for readability
        y_cols = y_cols[:3]
        
        # Sort by x column
        df_sorted = self.df.sort_values(x_col)
        
        # Create the line chart
        fig = go.Figure()
        
        for col in y_cols:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{', '.join(y_cols)} by {x_col}",
            xaxis_title=x_col,
            yaxis_title="Value",
            template="plotly_white"
        )
        
        response_text = f"""
        ### Line Chart: {', '.join(y_cols)} by {x_col}
        
        I've created a line chart showing how {', '.join(y_cols)} change across {x_col}.
        
        **Key observations:**
        """
        
        # Add observations for each y column
        for col in y_cols:
            try:
                min_val = df_sorted[col].min()
                max_val = df_sorted[col].max()
                min_x = df_sorted[df_sorted[col] == min_val].iloc[0][x_col]
                max_x = df_sorted[df_sorted[col] == max_val].iloc[0][x_col]
                
                response_text += f"""
                - **{col}**:
                  - Maximum: {max_val:.2f} at {max_x}
                  - Minimum: {min_val:.2f} at {min_x}
                """
            except:
                # Skip if there's an error (e.g., with datetime values)
                pass
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_scatter_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a scatter plot visualization"""
        # Need at least two numeric columns
        if len(numeric_cols) < 2:
            return {
                "text": "I couldn't create a scatter plot because I need at least two numeric columns."
            }
        
        # Select columns for the scatter plot
        x_col = None
        y_col = None
        color_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Find numeric columns
            numeric_mentions = [col for col in mentioned_cols if col in numeric_cols]
            
            if len(numeric_mentions) >= 2:
                x_col = numeric_mentions[0]
                y_col = numeric_mentions[1]
            
            # Look for a color column
            if len(mentioned_cols) > 2:
                for col in mentioned_cols:
                    if col != x_col and col != y_col:
                        color_col = col
                        break
        
        # If no suitable columns found from mentions, use defaults
        if not x_col or not y_col:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        
        # Create the scatter plot
        if color_col:
            fig = px.scatter(
                self.df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col} (colored by {color_col})",
                template="plotly_white"
            )
        else:
            fig = px.scatter(
                self.df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                template="plotly_white"
            )
        
        # Add trendline
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        # Calculate correlation
        correlation = self.df[[x_col, y_col]].corr().iloc[0, 1]
        
        response_text = f"""
        ### Scatter Plot: {y_col} vs {x_col}
        
        I've created a scatter plot showing the relationship between {x_col} and {y_col}.
        
        **Key observations:**
        - Correlation coefficient: {correlation:.4f}
        - Relationship: """
        
        if correlation > 0.7:
            response_text += "Strong positive correlation"
        elif correlation > 0.3:
            response_text += "Moderate positive correlation"
        elif correlation > 0:
            response_text += "Weak positive correlation"
        elif correlation > -0.3:
            response_text += "Weak negative correlation"
        elif correlation > -0.7:
            response_text += "Moderate negative correlation"
        else:
            response_text += "Strong negative correlation"
        
        if color_col:
            response_text += f"\n- Points are colored by {color_col}"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_histogram_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a histogram visualization"""
        # Need at least one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a histogram because I need at least one numeric column."
            }
        
        # Select column for the histogram
        col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            for mentioned_col in mentioned_cols:
                if mentioned_col in numeric_cols:
                    col = mentioned_col
                    break
        
        # If no suitable column found from mentions, use default
        if not col:
            col = numeric_cols[0]
        
        # Create the histogram
        fig = px.histogram(
            self.df,
            x=col,
            marginal="box",
            title=f"Distribution of {col}",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Count"
        )
        
        # Calculate basic statistics
        mean = self.df[col].mean()
        median = self.df[col].median()
        std = self.df[col].std()
        skewness = self.df[col].skew()
        
        response_text = f"""
        ### Histogram: Distribution of {col}
        
        I've created a histogram showing the distribution of {col}.
        
        **Key statistics:**
        - Mean: {mean:.2f}
        - Median: {median:.2f}
        - Standard Deviation: {std:.2f}
        - Skewness: {skewness:.2f} ("""
        
        if skewness > 1:
            response_text += "highly right-skewed"
        elif skewness > 0.5:
            response_text += "moderately right-skewed"
        elif skewness > -0.5:
            response_text += "approximately symmetric"
        elif skewness > -1:
            response_text += "moderately left-skewed"
        else:
            response_text += "highly left-skewed"
        
        response_text += ")"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_pie_visualization(self, query, mentioned_cols, categorical_cols):
        """Create a pie chart visualization"""
        # Need at least one categorical column
        if not categorical_cols:
            return {
                "text": "I couldn't create a pie chart because I need at least one categorical column."
            }
        
        # Select column for the pie chart
        col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            for mentioned_col in mentioned_cols:
                if mentioned_col in categorical_cols:
                    col = mentioned_col
                    break
        
        # If no suitable column found from mentions, use default
        if not col:
            col = categorical_cols[0]
        
        # Count values in the categorical column
        value_counts = self.df[col].value_counts()
        
        # Limit to top 10 categories if there are too many
        if len(value_counts) > 10:
            others_sum = value_counts[10:].sum()
            value_counts = value_counts[:10]
            value_counts["Other"] = others_sum
        
        # Create the pie chart
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {col}",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        response_text = f"""
        ### Pie Chart: Distribution of {col}
        
        I've created a pie chart showing the distribution of {col}.
        
        **Key observations:**
        - Most common category: {value_counts.index[0]} ({value_counts.values[0]} instances, {(value_counts.values[0] / value_counts.sum()) * 100:.1f}%)
        """
        
        if len(value_counts) > 5:
            response_text += f"- There are {len(value_counts)} different categories shown"
        
        if "Other" in value_counts:
            response_text += f"\n- The 'Other' category combines {len(self.df[col].unique()) - 9} less common categories"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_box_visualization(self, query, mentioned_cols, numeric_cols, categorical_cols):
        """Create a box plot visualization"""
        # Need at least one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a box plot because I need at least one numeric column."
            }
        
        # Select columns for the box plot
        y_col = None
        x_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for numeric column first
            for col in mentioned_cols:
                if col in numeric_cols:
                    y_col = col
                    break
            
            # Then look for categorical column
            for col in mentioned_cols:
                if col in categorical_cols:
                    x_col = col
                    break
        
        # If no suitable columns found from mentions, use defaults
        if not y_col:
            y_col = numeric_cols[0]
        
        # Create the box plot
        if x_col:
            fig = px.box(
                self.df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Box Plot: Distribution of {y_col} by {x_col}
            
            I've created a box plot showing how the distribution of {y_col} varies across different {x_col} categories.
            
            **Key observations:**
            """
            
            # Calculate statistics for each group
            grouped_stats = self.df.groupby(x_col)[y_col].agg(['median', 'mean', 'std']).reset_index()
            grouped_stats = grouped_stats.sort_values('median', ascending=False)
            
            response_text += f"""
            - Highest median {y_col}: {grouped_stats.iloc[0][x_col]} ({grouped_stats.iloc[0]['median']:.2f})
            - Lowest median {y_col}: {grouped_stats.iloc[-1][x_col]} ({grouped_stats.iloc[-1]['median']:.2f})
            """
        else:
            fig = px.box(
                self.df,
                y=y_col,
                title=f"Distribution of {y_col}",
                template="plotly_white"
            )
            
            # Calculate basic statistics
            median = self.df[y_col].median()
            mean = self.df[y_col].mean()
            q1 = self.df[y_col].quantile(0.25)
            q3 = self.df[y_col].quantile(0.75)
            iqr = q3 - q1
            
            response_text = f"""
            ### Box Plot: Distribution of {y_col}
            
            I've created a box plot showing the distribution of {y_col}.
            
            **Key statistics:**
            - Median: {median:.2f}
            - Mean: {mean:.2f}
            - Interquartile Range (IQR): {iqr:.2f}
            - Q1 (25th percentile): {q1:.2f}
            - Q3 (75th percentile): {q3:.2f}
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_heatmap_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a heatmap visualization"""
        # Need at least two numeric columns
        if len(numeric_cols) < 2:
            return {
                "text": "I couldn't create a heatmap because I need at least two numeric columns."
            }
        
        # Select columns for the heatmap
        cols_to_use = []
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            cols_to_use = [col for col in mentioned_cols if col in numeric_cols]
        
        # If no suitable columns found from mentions, use defaults
        if not cols_to_use or len(cols_to_use) < 2:
            # Use all numeric columns (up to 10 for readability)
            cols_to_use = numeric_cols[:min(10, len(numeric_cols))]
        
        # Calculate correlation matrix
        corr_matrix = self.df[cols_to_use].corr()
        
        # Create the heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
            template="plotly_white"
        )
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(cols_to_use)):
            for j in range(i+1, len(cols_to_use)):
                col1 = cols_to_use[i]
                col2 = cols_to_use[j]
                corr_value = corr_matrix.loc[col1, col2]
                corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        response_text = """
        ### Correlation Heatmap
        
        I've created a heatmap showing the correlations between numeric variables.
        
        **Key correlations:**
        """
        
        # Add top 3 correlations (or fewer if there aren't that many)
        for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
            strength = "strong positive" if corr > 0.7 else \
                      "moderate positive" if corr > 0.3 else \
                      "weak positive" if corr > 0 else \
                      "strong negative" if corr < -0.7 else \
                      "moderate negative" if corr < -0.3 else \
                      "weak negative"
            
            response_text += f"""
            - {col1} & {col2}: {corr:.2f} ({strength})
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _compare_groups(self, query):
        """Compare groups in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get categorical columns for grouping
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for metrics
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or not numeric_cols:
            return {
                "text": "Group comparison requires at least one categorical column and one numeric column."
            }
        
        # Try to identify the grouping column and the metrics from the query
        group_col = None
        metric_cols = []
        
        # Check which columns are mentioned in the query
        mentioned_cols = [col for col in self.df.columns if col.lower() in query.lower()]
        
        if mentioned_cols:
            # Look for a categorical column first
            for col in mentioned_cols:
                if col in cat_cols:
                    group_col = col
                    break
            
            # Then look for numeric columns
            for col in mentioned_cols:
                if col in numeric_cols and col != group_col:
                    metric_cols.append(col)
        
        # If no suitable columns found from mentions, use defaults
        if not group_col:
            group_col = cat_cols[0]
        
        if not metric_cols:
            # Use the first 3 numeric columns as metrics
            metric_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        # Generate comparison
        response_text = f"""
        ### Group Comparison: by {group_col}
        
        I've compared different groups based on **{group_col}**, analyzing these metrics:
        {', '.join([f"**{col}**" for col in metric_cols])}
        
        **Key findings:**
        """
        
        # Calculate group statistics
        group_stats = {}
        
        for metric in metric_cols:
            # Calculate mean, median, std for each group
            stats = self.df.groupby(group_col)[metric].agg(['mean', 'median', 'std']).reset_index()
            
            # Sort by mean
            stats = stats.sort_values('mean', ascending=False)
            
            group_stats[metric] = stats
            
            response_text += f"""
            
            **{metric}**:
            - Highest average: {stats.iloc[0][group_col]} ({stats.iloc[0]['mean']:.2f})
            - Lowest average: {stats.iloc[-1][group_col]} ({stats.iloc[-1]['mean']:.2f})
            """
            
            # Add statistical significance info if there are only 2 major groups
            unique_groups = self.df[group_col].nunique()
            if unique_groups == 2:
                group_values = self.df[group_col].unique().tolist()
                group1_data = self.df[self.df[group_col] == group_values[0]][metric].dropna()
                group2_data = self.df[self.df[group_col] == group_values[1]][metric].dropna()
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    try:
                        # Perform t-test
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        
                        response_text += f"""
                        - Statistical significance: {"Significant" if p_value < 0.05 else "Not significant"} (p-value: {p_value:.4f})
                        """
                    except:
                        # Skip if t-test fails
                        pass
        
        # Create visualization
        if len(metric_cols) == 1:
            # Single metric: bar chart
            metric = metric_cols[0]
            stats = group_stats[metric]
            
            fig = px.bar(
                stats,
                x=group_col,
                y='mean',
                error_y=stats['std'],
                title=f"Comparison of {metric} by {group_col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=group_col,
                yaxis_title=f"Average {metric}"
            )
        else:
            # Multiple metrics: radar chart or multi-bar chart
            if len(self.df[group_col].unique()) <= 5:
                # Use radar chart for few groups
                fig = go.Figure()
                
                categories = metric_cols
                
                for group in self.df[group_col].unique():
                    group_means = [self.df[self.df[group_col] == group][metric].mean() for metric in metric_cols]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=group_means,
                        theta=categories,
                        fill='toself',
                        name=str(group)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True
                        )
                    ),
                    title=f"Comparison by {group_col}",
                    template="plotly_white"
                )
            else:
                # Use bar chart for many groups
                # Create comparison of first metric
                metric = metric_cols[0]
                stats = group_stats[metric]
                
                # Limit to top 10 groups if there are too many
                if len(stats) > 10:
                    stats = stats.head(10)
                
                fig = px.bar(
                    stats,
                    x=group_col,
                    y='mean',
                    error_y=stats['std'],
                    title=f"Comparison of {metric} by {group_col} (Top 10)",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    xaxis_title=group_col,
                    yaxis_title=f"Average {metric}"
                )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _provide_insights(self):
        """Provide general insights about the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Basic dataset info
        rows, cols = self.df.shape
        
        # Get column types
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate insights text
        insights_text = f"""
        ### Key Insights from the Dataset
        
        I've analyzed your dataset with {rows:,} rows and {cols} columns, and here are the main insights:
        """
        
        insights = []
        
        # Missing values insight
        missing_values = self.df.isna().sum().sum()
        missing_percentage = (missing_values / (rows * cols)) * 100
        if missing_percentage > 0:
            insights.append(f"📊 Your dataset has {missing_values:,} missing values ({missing_percentage:.2f}% of all cells).")
            
            # Add columns with most missing values
            cols_with_missing = self.df.columns[self.df.isna().any()].tolist()
            if cols_with_missing:
                missing_by_col = self.df[cols_with_missing].isna().sum().sort_values(ascending=False)
                top_missing = missing_by_col.head(3)
                insights.append(f"🔍 Columns with most missing values: " + 
                               ", ".join([f"{col} ({missing:,}, {(missing/rows)*100:.1f}%)" 
                                         for col, missing in top_missing.items()]))
        
        # Correlations insight (if there are numeric columns)
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find highest absolute correlations (excluding self-correlations)
            high_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Only include strong correlations
                        high_corrs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation value
            high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            if high_corrs:
                insights.append("📈 Strong correlations found:")
                for col1, col2, corr in high_corrs[:3]:  # Show top 3
                    corr_type = "positive" if corr > 0 else "negative"
                    insights.append(f"  - {col1} and {col2}: {corr:.2f} ({corr_type})")
        
        # Distribution insights (if there are numeric columns)
        if numeric_cols:
            # Look for columns with skewed distributions
            skewed_cols = []
            for col in numeric_cols:
                skewness = self.df[col].skew()
                if abs(skewness) > 1:  # Significantly skewed
                    skew_direction = "right" if skewness > 0 else "left"
                    skewed_cols.append((col, skewness, skew_direction))
            
            if skewed_cols:
                skewed_cols.sort(key=lambda x: abs(x[1]), reverse=True)
                insights.append("📊 Skewed distributions:")
                for col, skewness, direction in skewed_cols[:3]:  # Show top 3
                    insights.append(f"  - {col}: {direction}-skewed (skewness = {skewness:.2f})")
        
        # Categorical insights (if there are categorical columns)
        if categorical_cols:
            cat_insights = []
            for col in categorical_cols[:3]:  # Analyze top 3 categorical columns
                value_counts = self.df[col].value_counts()
                unique_count = len(value_counts)
                top_value = value_counts.index[0] if not value_counts.empty else "N/A"
                top_percentage = (value_counts.iloc[0] / len(self.df)) * 100 if not value_counts.empty else 0
                
                if unique_count == 1:
                    cat_insights.append(f"  - {col}: Only one value ({top_value})")
                elif unique_count <= 5:
                    cat_insights.append(f"  - {col}: {unique_count} unique values, most common is {top_value} ({top_percentage:.1f}%)")
                else:
                    cat_insights.append(f"  - {col}: {unique_count} unique values, most common is {top_value} ({top_percentage:.1f}%)")
            
            if cat_insights:
                insights.append("🔍 Categorical variables:")
                for insight in cat_insights:
                    insights.append(insight)
        
        # Outlier insights (if there are numeric columns)
        if numeric_cols:
            outlier_insights = []
            for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
                # Calculate Q1, Q3, and IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                if outlier_percentage > 5:
                    outlier_insights.append(f"  - {col}: {outlier_count} outliers ({outlier_percentage:.1f}% of data)")
            
            if outlier_insights:
                insights.append("⚠️ Significant outliers detected:")
                for insight in outlier_insights:
                    insights.append(insight)
        
        # Add insights to response
        for insight in insights:
            insights_text += f"\n- {insight}"
        
        # Create visualization
        if numeric_cols and len(numeric_cols) >= 2:
            # Create scatter plot of two key numeric variables
            col1 = numeric_cols[0]
            col2 = numeric_cols[1]
            
            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                title=f"Relationship: {col2} vs {col1}",
                template="plotly_white"
            )
            
            # Add trendline
            fig.update_layout(
                xaxis_title=col1,
                yaxis_title=col2
            )
        elif categorical_cols:
            # Create bar chart of categorical distribution
            col = categorical_cols[0]
            value_counts = self.df[col].value_counts()
            
            # Limit to top 10 categories if there are too many
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count"
            )
        else:
            # Create a simple data overview
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=["Column", "Type", "Non-Null", "Unique Values"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        self.df.columns,
                        self.df.dtypes.astype(str),
                        self.df.count().values,
                        [self.df[col].nunique() for col in self.df.columns]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )])
            
            fig.update_layout(
                title="Dataset Overview",
                template="plotly_white"
            )
        
        return {
            "text": insights_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _default_response(self, query):
        """Generate a default response when query isn't recognized"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        return {
            "text": f"""
            I'm not sure how to answer that specific question about your data. Here are some things you can ask me:
            
            - Summarize this dataset
            - Show me correlations between variables
            - Analyze the distribution of a specific column
            - Compare different groups in the data
            - Create visualizations like bar charts, scatter plots, etc.
            - Find outliers in the data
            - Provide statistics for specific columns
            
            Feel free to try one of these questions or rephrase your query.
            """
        }