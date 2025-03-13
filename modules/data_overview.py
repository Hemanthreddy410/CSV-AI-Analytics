import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

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
        st.subheader("Data Preview")
        
        # Create columns for controls
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        
        with control_col1:
            # Number of rows to display
            n_rows = st.slider("Number of rows:", 5, 100, 10, key="preview_rows")
        
        with control_col2:
            # View options
            view_option = st.radio("View:", ["Head", "Tail", "Sample"], horizontal=True)
        
        with control_col3:
            # Column filter
            if st.checkbox("Select columns", value=False, key="select_columns_preview"):
                selected_cols = st.multiselect(
                    "Columns to display:",
                    st.session_state.df.columns.tolist(),
                    default=st.session_state.df.columns.tolist()[:5],
                    key="selected_cols_preview"
                )
            else:
                selected_cols = st.session_state.df.columns.tolist()
        
        # Display the data
        if selected_cols:
            if view_option == "Head":
                st.dataframe(st.session_state.df[selected_cols].head(n_rows), use_container_width=True)
            elif view_option == "Tail":
                st.dataframe(st.session_state.df[selected_cols].tail(n_rows), use_container_width=True)
            else:  # Sample
                st.dataframe(st.session_state.df[selected_cols].sample(min(n_rows, len(st.session_state.df))), use_container_width=True)
        else:
            st.info("Please select at least one column to display.")
    
    # Column Info Tab
    with overview_tabs[1]:
        st.subheader("Column Information")
        
        # Create a dataframe with column information
        df = st.session_state.df
        columns_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%',
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Memory Usage': [df[col].memory_usage(deep=True) for col in df.columns]
        })
        
        # Format memory usage
        columns_df['Memory Usage'] = columns_df['Memory Usage'].apply(
            lambda x: f"{x} bytes" if x < 1024 else (
                f"{x/1024:.2f} KB" if x < 1024**2 else f"{x/(1024**2):.2f} MB"
            )
        )
        
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
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
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
    
    # Summary Statistics Tab
    with overview_tabs[2]:
        st.subheader("Summary Statistics")
        
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if num_cols:
            # Calculate statistics
            stats_df = st.session_state.df[num_cols].describe().T
            
            # Round to 3 decimal places
            stats_df = stats_df.round(3)
            
            # Add more statistics
            stats_df['median'] = st.session_state.df[num_cols].median()
            stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
            stats_df['missing_pct'] = (st.session_state.df[num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
            
            try:
                stats_df['skew'] = st.session_state.df[num_cols].skew().round(3)
                stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis().round(3)
            except:
                pass
            
            # Reorder columns
            stats_cols = ['count', 'missing', 'missing_pct', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
            extra_cols = ['skew', 'kurtosis']
            
            # Only include available columns
            avail_cols = [col for col in stats_cols + extra_cols if col in stats_df.columns]
            
            stats_df = stats_df[avail_cols]
            
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
            
            # Distribution visualization
            st.subheader("Distributions")
            
            # Select column for distribution
            dist_col = st.selectbox("Select column for distribution:", num_cols, key="dist_col_select")
            
            # Create histogram with KDE
            fig = px.histogram(
                st.session_state.df, 
                x=dist_col,
                marginal="box",
                title=f"Distribution of {dist_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for summary statistics.")
    
    # Missing Values Tab
    with overview_tabs[3]:
        st.subheader("Missing Values Analysis")
        
        # Calculate missing values
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isna().sum().values,
            'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
        })
        
        # Sort by missing count (descending)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        # Display the dataframe
        st.dataframe(missing_df, use_container_width=True)
        
        # Create visualization
        fig = px.bar(
            missing_df, 
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
        
        # Limit to columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        
        if cols_with_missing:
            # Create sample for heatmap (limit to 100 rows for performance)
            sample_size = min(100, len(st.session_state.df))
            sample_df = st.session_state.df[cols_with_missing].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[cols_with_missing]
            
            # Create heatmap
            fig = px.imshow(
                sample_df.isna(),
                labels=dict(x="Column", y="Row", color="Missing"),
                color_continuous_scale=["blue", "red"],
                title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Duplicates Tab
    with overview_tabs[4]:
        st.subheader("Duplicate Rows Analysis")
        
        # Calculate duplicates
        duplicates = st.session_state.df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_pct = (duplicate_count / len(st.session_state.df) * 100).round(2)
        
        # Display duplicates summary
        st.write(f"**Total duplicate rows:** {duplicate_count} ({duplicate_pct}% of all rows)")
        
        # Show duplicates if any
        if duplicate_count > 0:
            # Get the duplicate rows
            duplicate_rows = st.session_state.df[duplicates]
            
            # Show options
            show_option = st.radio(
                "Display options:",
                ["Show sample of duplicates", "Show all duplicates", "Show duplicate values counts"],
                key="dup_display_options",
                horizontal=True
            )
            
            if show_option == "Show sample of duplicates":
                # Show sample of duplicates
                sample_size = min(10, len(duplicate_rows))
                st.dataframe(duplicate_rows.head(sample_size), use_container_width=True)
                
            elif show_option == "Show all duplicates":
                # Show all duplicates
                st.dataframe(duplicate_rows, use_container_width=True)
                
            else:  # Show counts
                # Count occurrences of each duplicate combination
                dup_counts = st.session_state.df.groupby(list(st.session_state.df.columns)).size().reset_index(name='count')
                dup_counts = dup_counts[dup_counts['count'] > 1].sort_values('count', ascending=False)
                
                st.dataframe(dup_counts, use_container_width=True)
            
            # Download button for duplicates
            csv = duplicate_rows.to_csv(index=False)
            st.download_button(
                label="Download Duplicate Rows",
                data=csv,
                file_name="duplicate_rows.csv",
                mime="text/csv",
                key="download_duplicates",
                use_container_width=True
            )
            
            # Option to remove duplicates
            if st.button("Remove Duplicate Rows", key="remove_duplicates_overview", use_container_width=True):
                # Remove duplicates
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                
                # Add to processing history
                st.session_state.processing_history.append({
                    "description": f"Removed {duplicate_count} duplicate rows",
                    "timestamp": datetime.datetime.now(),
                    "type": "remove_duplicates",
                    "details": {
                        "rows_removed": int(duplicate_count)
                    }
                })
                
                # Success message
                st.success(f"Removed {duplicate_count} duplicate rows!")
                st.rerun()
        else:
            st.success("No duplicate rows found in the dataset!")