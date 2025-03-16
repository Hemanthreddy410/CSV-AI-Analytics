# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import plotly.express as px

# def render_data_overview():
#     """Render data overview section"""
#     st.header("Data Overview")
    
#     # Display file information
#     if st.session_state.file_details is not None:
#         # Create an info box
#         st.markdown('<div class="data-info">', unsafe_allow_html=True)
        
#         # Create two columns for layout
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown(f"**Filename:** {st.session_state.file_details['filename']}")
#             st.markdown(f"**Type:** {st.session_state.file_details['type']}")
            
#             # Format size nicely
#             size = st.session_state.file_details['size']
#             if size is not None:
#                 if size < 1024:
#                     size_str = f"{size} bytes"
#                 elif size < 1024 ** 2:
#                     size_str = f"{size / 1024:.2f} KB"
#                 elif size < 1024 ** 3:
#                     size_str = f"{size / (1024 ** 2):.2f} MB"
#                 else:
#                     size_str = f"{size / (1024 ** 3):.2f} GB"
                
#                 st.markdown(f"**Size:** {size_str}")
            
#             # Show additional info if available
#             if 'info' in st.session_state.file_details:
#                 st.markdown(f"**Info:** {st.session_state.file_details['info']}")
        
#         with col2:
#             st.markdown(f"**Rows:** {st.session_state.df.shape[0]:,}")
#             st.markdown(f"**Columns:** {st.session_state.df.shape[1]}")
            
#             # Format datetime properly
#             last_modified = st.session_state.file_details['last_modified']
#             if isinstance(last_modified, datetime.datetime):
#                 formatted_date = last_modified.strftime('%Y-%m-%d %H:%M:%S')
#                 st.markdown(f"**Last Modified:** {formatted_date}")
            
#             # Show current project if available
#             if st.session_state.current_project is not None:
#                 st.markdown(f"**Project:** {st.session_state.current_project}")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Create tabs for different overview sections
#     overview_tabs = st.tabs([
#         "Data Preview", 
#         "Column Info", 
#         "Summary Statistics", 
#         "Missing Values", 
#         "Duplicates"
#     ])
    
#     # Data Preview Tab
#     with overview_tabs[0]:
#         st.subheader("Data Preview")
        
#         # Create columns for controls
#         control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        
#         with control_col1:
#             # Number of rows to display
#             n_rows = st.slider("Number of rows:", 5, 100, 10, key="preview_rows")
        
#         with control_col2:
#             # View options
#             view_option = st.radio("View:", ["Head", "Tail", "Sample"], horizontal=True, key="view_option_preview")
        
#         with control_col3:
#             # Column filter
#             if st.checkbox("Select columns", value=False, key="select_columns_preview"):
#                 selected_cols = st.multiselect(
#                     "Columns to display:",
#                     st.session_state.df.columns.tolist(),
#                     default=st.session_state.df.columns.tolist()[:5],
#                     key="selected_cols_preview"
#                 )
#             else:
#                 selected_cols = st.session_state.df.columns.tolist()
        
#         # Display the data
#         if selected_cols:
#             if view_option == "Head":
#                 st.dataframe(st.session_state.df[selected_cols].head(n_rows), use_container_width=True)
#             elif view_option == "Tail":
#                 st.dataframe(st.session_state.df[selected_cols].tail(n_rows), use_container_width=True)
#             else:  # Sample
#                 st.dataframe(st.session_state.df[selected_cols].sample(min(n_rows, len(st.session_state.df))), use_container_width=True)
#         else:
#             st.info("Please select at least one column to display.")
    
#     # Column Info Tab
#     with overview_tabs[1]:
#         st.subheader("Column Information")
        
#         # Create a dataframe with column information
#         df = st.session_state.df
#         columns_df = pd.DataFrame({
#             'Column': df.columns,
#             'Type': df.dtypes.astype(str).values,
#             'Non-Null Count': df.count().values,
#             'Null Count': df.isna().sum().values,
#             'Null %': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%',
#             'Unique Values': [df[col].nunique() for col in df.columns],
#             'Memory Usage': [df[col].memory_usage(deep=True) for col in df.columns]
#         })
        
#         # Format memory usage
#         columns_df['Memory Usage'] = columns_df['Memory Usage'].apply(
#             lambda x: f"{x} bytes" if x < 1024 else (
#                 f"{x/1024:.2f} KB" if x < 1024**2 else f"{x/(1024**2):.2f} MB"
#             )
#         )
        
#         # Display the dataframe
#         st.dataframe(columns_df, use_container_width=True)
        
#         # Download button for column info
#         csv = columns_df.to_csv(index=False)
#         st.download_button(
#             label="Download Column Info",
#             data=csv,
#             file_name="column_info.csv",
#             mime="text/csv",
#             key="download_column_info",
#             use_container_width=True
#         )
        
#         # Display column type distribution
#         st.subheader("Column Type Distribution")
        
#         # Create a dataframe with type counts
#         type_counts = df.dtypes.astype(str).value_counts().reset_index()
#         type_counts.columns = ['Data Type', 'Count']
        
#         # Create two columns for layout
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             # Display type counts
#             st.dataframe(type_counts, use_container_width=True)
        
#         with col2:
#             # Create a pie chart of column types
#             fig = px.pie(
#                 type_counts, 
#                 values='Count', 
#                 names='Data Type', 
#                 title='Column Type Distribution'
#             )
#             st.plotly_chart(fig, use_container_width=True)
    
#     # Summary Statistics Tab
#     with overview_tabs[2]:
#         st.subheader("Summary Statistics")
        
#         # Get numeric columns
#         num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
#         if num_cols:
#             # Calculate statistics
#             stats_df = st.session_state.df[num_cols].describe().T
            
#             # Round to 3 decimal places
#             stats_df = stats_df.round(3)
            
#             # Add more statistics
#             stats_df['median'] = st.session_state.df[num_cols].median()
#             stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
#             stats_df['missing_pct'] = (st.session_state.df[num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
            
#             try:
#                 stats_df['skew'] = st.session_state.df[num_cols].skew().round(3)
#                 stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis().round(3)
#             except:
#                 pass
            
#             # Reorder columns
#             stats_cols = ['count', 'missing', 'missing_pct', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
#             extra_cols = ['skew', 'kurtosis']
            
#             # Only include available columns
#             avail_cols = [col for col in stats_cols + extra_cols if col in stats_df.columns]
            
#             stats_df = stats_df[avail_cols]
            
#             # Display the dataframe
#             st.dataframe(stats_df, use_container_width=True)
            
#             # Download button for statistics
#             csv = stats_df.to_csv()
#             st.download_button(
#                 label="Download Statistics",
#                 data=csv,
#                 file_name="summary_statistics.csv",
#                 mime="text/csv",
#                 key="download_statistics",
#                 use_container_width=True
#             )
            
#             # Distribution visualization
#             st.subheader("Distributions")
            
#             # Select column for distribution
#             dist_col = st.selectbox("Select column for distribution:", num_cols, key="dist_col_select")
            
#             # Create histogram with KDE
#             fig = px.histogram(
#                 st.session_state.df, 
#                 x=dist_col,
#                 marginal="box",
#                 title=f"Distribution of {dist_col}"
#             )
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("No numeric columns available for summary statistics.")
    
#     # Missing Values Tab
#     with overview_tabs[3]:
#         st.subheader("Missing Values Analysis")
        
#         # Calculate missing values
#         missing_df = pd.DataFrame({
#             'Column': st.session_state.df.columns,
#             'Missing Count': st.session_state.df.isna().sum().values,
#             'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
#         })
        
#         # Sort by missing count (descending)
#         missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
#         # Display the dataframe
#         st.dataframe(missing_df, use_container_width=True)
        
#         # Create visualization
#         fig = px.bar(
#             missing_df, 
#             x='Column', 
#             y='Missing %',
#             title='Missing Values by Column (%)',
#             color='Missing %',
#             color_continuous_scale="Reds"
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Missing values heatmap
#         st.subheader("Missing Values Heatmap")
#         st.write("This visualization shows patterns of missing data across the dataset.")
        
#         # Limit to columns with missing values
#         cols_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        
#         if cols_with_missing:
#             # Create sample for heatmap (limit to 100 rows for performance)
#             sample_size = min(100, len(st.session_state.df))
#             sample_df = st.session_state.df[cols_with_missing].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[cols_with_missing]
            
#             # Create heatmap
#             fig = px.imshow(
#                 sample_df.isna(),
#                 labels=dict(x="Column", y="Row", color="Missing"),
#                 color_continuous_scale=["blue", "red"],
#                 title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
#             )
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.success("No missing values found in the dataset!")
    
#     # Duplicates Tab
#     with overview_tabs[4]:
#         st.subheader("Duplicate Rows Analysis")
        
#         # Calculate duplicates
#         duplicates = st.session_state.df.duplicated()
#         duplicate_count = duplicates.sum()
#         duplicate_pct = (duplicate_count / len(st.session_state.df) * 100).round(2)
        
#         # Display duplicates summary
#         st.write(f"**Total duplicate rows:** {duplicate_count} ({duplicate_pct}% of all rows)")
        
#         # Show duplicates if any
#         if duplicate_count > 0:
#             # Get the duplicate rows
#             duplicate_rows = st.session_state.df[duplicates]
            
#             # Show options
#             show_option = st.radio(
#                 "Display options:",
#                 ["Show sample of duplicates", "Show all duplicates", "Show duplicate values counts"],
#                 key="dup_display_options",
#                 horizontal=True,
#             )
            
#             if show_option == "Show sample of duplicates":
#                 # Show sample of duplicates
#                 sample_size = min(10, len(duplicate_rows))
#                 st.dataframe(duplicate_rows.head(sample_size), use_container_width=True)
                
#             elif show_option == "Show all duplicates":
#                 # Show all duplicates
#                 st.dataframe(duplicate_rows, use_container_width=True)
                
#             else:  # Show counts
#                 # Count occurrences of each duplicate combination
#                 dup_counts = st.session_state.df.groupby(list(st.session_state.df.columns)).size().reset_index(name='count')
#                 dup_counts = dup_counts[dup_counts['count'] > 1].sort_values('count', ascending=False)
                
#                 st.dataframe(dup_counts, use_container_width=True)
            
#             # Download button for duplicates
#             csv = duplicate_rows.to_csv(index=False)
#             st.download_button(
#                 label="Download Duplicate Rows",
#                 data=csv,
#                 file_name="duplicate_rows.csv",
#                 mime="text/csv",
#                 key="download_duplicates",
#                 use_container_width=True
#             )
            
#             # Option to remove duplicates
#             if st.button("Remove Duplicate Rows", key="remove_duplicates_overview", use_container_width=True):
#                 # Remove duplicates
#                 st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                
#                 # Add to processing history
#                 st.session_state.processing_history.append({
#                     "description": f"Removed {duplicate_count} duplicate rows",
#                     "timestamp": datetime.datetime.now(),
#                     "type": "remove_duplicates",
#                     "details": {
#                         "rows_removed": int(duplicate_count)
#                     }
#                 })
                
#                 # Success message
#                 st.success(f"Removed {duplicate_count} duplicate rows!")
#                 st.rerun()
#         else:
#             st.success("No duplicate rows found in the dataset!")



import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import gc

def render_data_overview():
    """Render data overview section"""
    st.header("Data Overview")
    
    # Display file information
    if st.session_state.file_details is not None:
        # Create an info box
        st.markdown('<div class="data-info">', unsafe_allow_html=True)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Filename:** {st.session_state.file_details['filename']}")
            st.markdown(f"**Type:** {st.session_state.file_details['type']}")
            
            # Format size nicely
            size = st.session_state.file_details['size']
            if size is not None:
                if size < 1024:
                    size_str = f"{size} bytes"
                elif size < 1024 ** 2:
                    size_str = f"{size / 1024:.2f} KB"
                elif size < 1024 ** 3:
                    size_str = f"{size / (1024 ** 2):.2f} MB"
                else:
                    size_str = f"{size / (1024 ** 3):.2f} GB"
                
                st.markdown(f"**Size:** {size_str}")
            
            # Show additional info if available
            if 'info' in st.session_state.file_details:
                st.markdown(f"**Info:** {st.session_state.file_details['info']}")
        
        with col2:
            st.markdown(f"**Rows:** {st.session_state.df.shape[0]:,}")
            st.markdown(f"**Columns:** {st.session_state.df.shape[1]}")
            
            # Format datetime properly
            last_modified = st.session_state.file_details['last_modified']
            if isinstance(last_modified, datetime.datetime):
                formatted_date = last_modified.strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"**Last Modified:** {formatted_date}")
            
            # Show current project if available
            if st.session_state.current_project is not None:
                st.markdown(f"**Project:** {st.session_state.current_project}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different overview sections
    overview_tabs = st.tabs([
        "Data Preview", 
        "Column Info", 
        "Summary Statistics", 
        "Missing Values", 
        "Duplicates"
    ])
    
    # Data Preview Tab
    with overview_tabs[0]:
        render_data_preview()
    
    # Column Info Tab
    with overview_tabs[1]:
        render_column_info()
    
    # Summary Statistics Tab
    with overview_tabs[2]:
        render_summary_statistics()
    
    # Missing Values Tab
    with overview_tabs[3]:
        render_missing_values()
    
    # Duplicates Tab
    with overview_tabs[4]:
        render_duplicates()


# OPTIMIZATION: Split large function into smaller functions and add caching where appropriate
def render_data_preview():
    """Render data preview tab with optimizations for large datasets"""
    st.subheader("Data Preview")
    
    # Create columns for controls
    control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
    
    with control_col1:
        # Number of rows to display
        n_rows = st.slider("Number of rows:", 5, 100, 10, key="preview_rows")
    
    with control_col2:
        # View options
        view_option = st.radio("View:", ["Head", "Tail", "Sample"], horizontal=True, key="view_option_preview")
    
    with control_col3:
        # Column filter
        if st.checkbox("Select columns", value=False, key="select_columns_preview"):
            # OPTIMIZATION: Only show first 100 columns in selection to avoid UI slowdown
            cols_to_show = st.session_state.df.columns.tolist()[:100] if len(st.session_state.df.columns) > 100 else st.session_state.df.columns.tolist()
            if len(st.session_state.df.columns) > 100:
                st.info(f"Showing first 100 of {len(st.session_state.df.columns)} columns")
            
            selected_cols = st.multiselect(
                "Columns to display:",
                cols_to_show,
                default=cols_to_show[:min(5, len(cols_to_show))],
                key="selected_cols_preview"
            )
        else:
            # OPTIMIZATION: Limit default columns to prevent memory issues
            selected_cols = st.session_state.df.columns.tolist()[:20] if len(st.session_state.df.columns) > 20 else st.session_state.df.columns.tolist()
            if len(st.session_state.df.columns) > 20:
                st.info(f"Showing first 20 of {len(st.session_state.df.columns)} columns. Check 'Select columns' to customize.")
    
    # Display the data
    if selected_cols:
        # OPTIMIZATION: Check if dataframe is large and warn user
        if len(st.session_state.df) > 100000:
            st.warning(f"Large dataset detected ({len(st.session_state.df):,} rows). Showing limited preview for performance.")
        
        try:
            # Use a subset for display
            if view_option == "Head":
                st.dataframe(st.session_state.df[selected_cols].head(n_rows), use_container_width=True)
            elif view_option == "Tail":
                st.dataframe(st.session_state.df[selected_cols].tail(n_rows), use_container_width=True)
            else:  # Sample
                # OPTIMIZATION: Use efficient sampling
                st.dataframe(st.session_state.df[selected_cols].sample(min(n_rows, len(st.session_state.df))), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying data: {str(e)}")
            st.info("Try selecting fewer columns or rows for preview.")
    else:
        st.info("Please select at least one column to display.")


# OPTIMIZATION: Add caching for column info which doesn't change often
@st.cache_data(ttl=300, show_spinner=False)
def get_column_info(df):
    """Get column information with caching for performance"""
    try:
        # OPTIMIZATION: More memory-efficient column info calculation
        columns_info = []
        for col in df.columns:
            # Calculate one column at a time to reduce memory pressure
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique() if df[col].dtype != 'object' or len(df) < 10000 else None
            memory_usage = df[col].memory_usage(deep=True)
            
            # Format memory usage
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            else:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            
            # For large object columns, estimate unique values
            if unique_count is None:
                # Sample 10,000 rows for estimation
                sample = df[col].dropna().sample(min(10000, len(df)))
                unique_count = sample.nunique()
                unique_count = f"~{unique_count}+ (estimated)"
            
            columns_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null Count': non_null,
                'Null Count': null_count,
                'Null %': f"{(null_count / len(df) * 100):.2f}%",
                'Unique Values': unique_count,
                'Memory Usage': memory_str
            })
        
        return pd.DataFrame(columns_info)
    except Exception as e:
        # Return error info
        return pd.DataFrame([{'Column': 'Error', 'Type': str(e)}])


def render_column_info():
    """Render column information tab with optimizations"""
    st.subheader("Column Information")
    
    # Show processing indicator for large datasets
    if len(st.session_state.df) > 50000:
        with st.spinner("Analyzing column information..."):
            columns_df = get_column_info(st.session_state.df)
    else:
        columns_df = get_column_info(st.session_state.df)
        
    # Display the dataframe
    st.dataframe(columns_df, use_container_width=True)
    
    # Download button for column info
    csv = columns_df.to_csv(index=False)
    st.download_button(
        label="Download Column Info",
        data=csv,
        file_name="column_info.csv",
        mime="text/csv",
        key="download_column_info",
        use_container_width=True
    )
    
    # Display column type distribution
    st.subheader("Column Type Distribution")
    
    # Create a dataframe with type counts
    type_counts = st.session_state.df.dtypes.astype(str).value_counts().reset_index()
    type_counts.columns = ['Data Type', 'Count']
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display type counts
        st.dataframe(type_counts, use_container_width=True)
    
    with col2:
        # Create a pie chart of column types
        fig = px.pie(
            type_counts, 
            values='Count', 
            names='Data Type', 
            title='Column Type Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)


# OPTIMIZATION: Add caching for summary statistics which are expensive to calculate
@st.cache_data(ttl=300, show_spinner=False)
def get_summary_statistics(df, num_cols):
    """Generate summary statistics with optimizations for large datasets"""
    try:
        # For large datasets, calculate with sampling
        if len(df) > 100000:
            # Use a sample for initial calculations
            sample_size = min(100000, len(df))
            df_sample = df.sample(sample_size)
            stats_df = df_sample[num_cols].describe().T
            
            # Add note about sampling
            stats_df.insert(0, 'note', f"Based on {sample_size:,} sampled rows")
        else:
            stats_df = df[num_cols].describe().T
        
        # Round to 3 decimal places
        stats_df = stats_df.round(3)
        
        # Add more statistics - calculated efficiently
        stats_df['median'] = df[num_cols].median()
        stats_df['missing'] = df[num_cols].isna().sum()
        stats_df['missing_pct'] = (df[num_cols].isna().sum() / len(df) * 100).round(2)
        
        # Only calculate these if not too expensive (smaller datasets)
        if len(df) < 100000:
            try:
                stats_df['skew'] = df[num_cols].skew().round(3)
                stats_df['kurtosis'] = df[num_cols].kurtosis().round(3)
            except:
                pass
        
        # Reorder columns to ensure consistent display
        stats_cols = ['count', 'missing', 'missing_pct', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
        extra_cols = ['skew', 'kurtosis']
        
        # Only include available columns
        avail_cols = ['note'] if 'note' in stats_df.columns else []
        avail_cols.extend([col for col in stats_cols + extra_cols if col in stats_df.columns])
        
        return stats_df[avail_cols]
    except Exception as e:
        # Return error info
        return pd.DataFrame([{'Error': str(e)}])


def render_summary_statistics():
    """Render summary statistics tab with optimizations"""
    st.subheader("Summary Statistics")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if num_cols:
        # Show spinner for large datasets
        if len(st.session_state.df) > 50000:
            with st.spinner("Calculating summary statistics..."):
                stats_df = get_summary_statistics(st.session_state.df, num_cols)
        else:
            stats_df = get_summary_statistics(st.session_state.df, num_cols)
        
        # Display the dataframe
        st.dataframe(stats_df, use_container_width=True)
        
        # Download button for statistics
        csv = stats_df.to_csv()
        st.download_button(
            label="Download Statistics",
            data=csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
            key="download_statistics",
            use_container_width=True
        )
        
        # Distribution visualization - allow user to select column
        st.subheader("Distributions")
        
        # OPTIMIZATION: Allow users to select column, but warn about performance
        dist_col = st.selectbox("Select column for distribution:", num_cols, key="dist_col_select")
        
        # Use spinner for visualization generation
        with st.spinner(f"Generating distribution visualization for {dist_col}..."):
            # OPTIMIZATION: Sample data for large datasets
            if len(st.session_state.df) > 10000:
                sample_size = min(10000, len(st.session_state.df))
                plot_data = st.session_state.df.sample(sample_size)
                st.info(f"Visualization based on a sample of {sample_size:,} rows for performance")
            else:
                plot_data = st.session_state.df
            
            # Create histogram with box plot (more efficient than KDE for large data)
            fig = px.histogram(
                plot_data, 
                x=dist_col,
                marginal="box",
                title=f"Distribution of {dist_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Clean up to save memory
            del fig
            if len(st.session_state.df) > 10000:
                del plot_data
                gc.collect()
    else:
        st.info("No numeric columns available for summary statistics.")


# OPTIMIZATION: Optimize missing values visualization for large datasets
@st.cache_data(ttl=300, show_spinner=False)
def get_missing_values_data(df):
    """Calculate missing values information with optimizations"""
    # Calculate missing values efficiently
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df) * 100).round(2)
    
    # Create dataframe
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_counts.values,
        'Missing %': missing_percent.values
    })
    
    # Sort by missing count (descending)
    return missing_df.sort_values('Missing Count', ascending=False)


def render_missing_values():
    """Render missing values analysis tab with optimizations"""
    st.subheader("Missing Values Analysis")
    
    # Show spinner for large datasets
    if len(st.session_state.df) > 50000:
        with st.spinner("Analyzing missing values..."):
            missing_df = get_missing_values_data(st.session_state.df)
    else:
        missing_df = get_missing_values_data(st.session_state.df)
    
    # Display the dataframe
    st.dataframe(missing_df, use_container_width=True)
    
    # Create visualization - only if there are missing values
    if missing_df['Missing Count'].sum() > 0:
        # OPTIMIZATION: Only visualize columns with missing values
        missing_cols = missing_df[missing_df['Missing Count'] > 0]
        
        # OPTIMIZATION: Limit to top 20 columns to avoid overcrowded charts
        if len(missing_cols) > 20:
            vis_data = missing_cols.head(20)
            st.info(f"Showing top 20 columns with missing values out of {len(missing_cols)} total")
        else:
            vis_data = missing_cols
        
        fig = px.bar(
            vis_data, 
            x='Column', 
            y='Missing %',
            title='Missing Values by Column (%)',
            color='Missing %',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values heatmap
        st.subheader("Missing Values Heatmap")
        st.write("This visualization shows patterns of missing data across the dataset.")
        
        # OPTIMIZATION: Limit to columns with reasonable amount of missing values
        # to avoid memory issues
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        
        if cols_with_missing:
            # OPTIMIZATION: Limit columns to avoid memory issues
            if len(cols_with_missing) > 10:
                heatmap_cols = cols_with_missing[:10]
                st.info(f"Showing first 10 of {len(cols_with_missing)} columns with missing values for performance")
            else:
                heatmap_cols = cols_with_missing
            
            # Create sample for heatmap (limit for performance)
            sample_size = min(100, len(st.session_state.df))
            sample_df = st.session_state.df[heatmap_cols].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[heatmap_cols]
            
            # Create heatmap
            fig = px.imshow(
                sample_df.isna(),
                labels=dict(x="Column", y="Row", color="Missing"),
                color_continuous_scale=["blue", "red"],
                title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Clean up to free memory
            del fig, sample_df
            gc.collect()
        else:
            st.success("No missing values found in the dataset!")
    else:
        st.success("No missing values found in the dataset!")


# OPTIMIZATION: Optimize duplicates analysis for large datasets
def render_duplicates():
    """Render duplicate rows analysis tab with optimizations for large datasets"""
    st.subheader("Duplicate Rows Analysis")
    
    # For very large datasets, show warning and sampling option
    is_large_dataset = len(st.session_state.df) > 100000
    
    if is_large_dataset:
        st.warning(f"Large dataset detected ({len(st.session_state.df):,} rows). Checking for duplicates might be slow.")
        use_sampling = st.checkbox("Use sampling to check for duplicates (faster but approximate)", value=True)
        
        if use_sampling:
            # Use sampling for initial check
            sample_size = min(100000, len(st.session_state.df))
            with st.spinner(f"Checking duplicates in sample of {sample_size:,} rows..."):
                sample = st.session_state.df.sample(sample_size)
                duplicates = sample.duplicated()
                duplicate_count = duplicates.sum()
                duplicate_pct = (duplicate_count / len(sample) * 100).round(2)
                st.write(f"**Estimated duplicate rows (based on {sample_size:,} row sample):** {duplicate_count} (~{duplicate_pct}% of sampled rows)")
                
                if duplicate_count == 0:
                    st.success("No duplicates found in the sampled rows. Your data might be duplicate-free, but a full scan is needed for certainty.")
                    if st.button("Check entire dataset for duplicates", use_container_width=True):
                        check_full_dataset = True
                    else:
                        return
                else:
                    st.warning(f"Found {duplicate_count} duplicates in the sample. Proceed with caution on the full dataset.")
                    if st.button("Analyze duplicates in full dataset", use_container_width=True):
                        check_full_dataset = True
                    else:
                        return
        else:
            check_full_dataset = True
    else:
        check_full_dataset = True
    
    # Calculate duplicates on full dataset
    if check_full_dataset:
        with st.spinner("Analyzing duplicates in full dataset..."):
            duplicates = st.session_state.df.duplicated()
            duplicate_count = duplicates.sum()
            duplicate_pct = (duplicate_count / len(st.session_state.df) * 100).round(2)
    
    # Display duplicates summary
    st.write(f"**Total duplicate rows:** {duplicate_count} ({duplicate_pct}% of all rows)")
    
    # Show duplicates if any
    if duplicate_count > 0:
        # Get the duplicate rows
        with st.spinner("Retrieving duplicate rows..."):
            duplicate_rows = st.session_state.df[st.session_state.df.duplicated(keep='first')]
        
        # Show options
        show_option = st.radio(
            "Display options:",
            ["Show sample of duplicates", "Show all duplicates", "Show duplicate values counts"],
            key="dup_display_options",
            horizontal=True,
        )
        
        if show_option == "Show sample of duplicates":
            # Show sample of duplicates
            sample_size = min(10, len(duplicate_rows))
            st.dataframe(duplicate_rows.head(sample_size), use_container_width=True)
            
        elif show_option == "Show all duplicates":
            # OPTIMIZATION: For large duplicate sets, warn and paginate
            if len(duplicate_rows) > 1000:
                st.warning(f"Large number of duplicates ({len(duplicate_rows):,} rows). Showing first 1000 for performance.")
                st.dataframe(duplicate_rows.head(1000), use_container_width=True)
            else:
                # Show all duplicates
                st.dataframe(duplicate_rows, use_container_width=True)
            
        else:  # Show counts
            # OPTIMIZATION: For large datasets, sample or use more efficient approach
            if len(st.session_state.df) > 500000 and len(st.session_state.df.columns) > 10:
                st.warning("Counting duplicates in very large datasets may be slow. Processing a subset of columns.")
                # Select subset of columns likely to identify duplicates
                count_cols = st.multiselect(
                    "Select columns to identify duplicate combinations:",
                    st.session_state.df.columns.tolist(),
                    default=st.session_state.df.columns.tolist()[:min(5, len(st.session_state.df.columns))]
                )
                if not count_cols:
                    st.error("Please select at least one column")
                    return
                    
                with st.spinner("Counting duplicate combinations..."):
                    # Count occurrences of each duplicate combination
                    dup_counts = st.session_state.df.groupby(count_cols).size().reset_index(name='count')
                    dup_counts = dup_counts[dup_counts['count'] > 1].sort_values('count', ascending=False)
            else:
                with st.spinner("Counting duplicate combinations..."):
                    # Count occurrences of each duplicate combination
                    dup_counts = st.session_state.df.groupby(list(st.session_state.df.columns)).size().reset_index(name='count')
                    dup_counts = dup_counts[dup_counts['count'] > 1].sort_values('count', ascending=False)
            
            # OPTIMIZATION: Limit display for performance
            if len(dup_counts) > 1000:
                st.warning(f"Found {len(dup_counts):,} duplicate groups. Showing first 1000 for performance.")
                st.dataframe(dup_counts.head(1000), use_container_width=True)
            else:
                st.dataframe(dup_counts, use_container_width=True)
        
        # Download button for duplicates
        csv = duplicate_rows.head(10000).to_csv(index=False) if len(duplicate_rows) > 10000 else duplicate_rows.to_csv(index=False)
        max_rows = min(10000, len(duplicate_rows))
        
        dl_label = f"Download Duplicate Rows (First {max_rows:,})" if len(duplicate_rows) > 10000 else "Download Duplicate Rows"
        st.download_button(
            label=dl_label,
            data=csv,
            file_name="duplicate_rows.csv",
            mime="text/csv",
            key="download_duplicates",
            use_container_width=True
        )
        
        # Option to remove duplicates
        if st.button("Remove Duplicate Rows", key="remove_duplicates_overview", use_container_width=True):
            with st.spinner("Removing duplicates..."):
                # Remove duplicates
                orig_len = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                rows_removed = orig_len - len(st.session_state.df)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Removed {rows_removed} duplicate rows",
                    "timestamp": datetime.datetime.now(),
                    "type": "remove_duplicates",
                    "details": {
                        "rows_removed": int(rows_removed)
                    }
                })
                
                # Success message
                st.success(f"Removed {rows_removed} duplicate rows!")
                st.rerun()
    else:
        st.success("No duplicate rows found in the dataset!")