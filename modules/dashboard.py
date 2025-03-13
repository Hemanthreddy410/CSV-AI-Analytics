def render_dashboard():
    """Render dashboard section"""
    st.header("Interactive Dashboard")
    
    # Dashboard generator
    dashboard_generator = DashboardGenerator(st.session_state.df)
    
    # Create or load dashboard
    with st.expander("Dashboard Settings", expanded=True):
        # Dashboard title
        dashboard_title = st.text_input("Dashboard Title:", "Data Insights Dashboard")
        
        # Dashboard layout
        layout_type = st.radio(
            "Layout Type:",
            ["2 columns", "3 columns", "Custom"],
            horizontal=True
        )
        
        # Components selection
        st.subheader("Add Dashboard Components")
        
        # Create tabs for component types
        component_tabs = st.tabs(["Charts", "Metrics", "Tables", "Filters"])
        
        with component_tabs[0]:
            # Charts section
            st.markdown("### Chart Components")
            
            # Get numeric and categorical columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Number of charts to add
            n_charts = st.number_input("Number of charts to add:", 1, 8, 3)
            
            # List to store chart configurations
            chart_configs = []
            
            for i in range(n_charts):
                with st.container():
                    st.markdown(f"#### Chart {i+1}")
                    
                    # Chart type selection
                    chart_type = st.selectbox(
                        "Chart type:",
                        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"],
                        key=f"chart_type_{i}"
                    )
                    
                    # Chart title
                    chart_title = st.text_input("Chart title:", value=f"Chart {i+1}", key=f"chart_title_{i}")
                    
                    # Configure chart based on type
                    if chart_type == "Bar Chart":
                        # Column selections
                        x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "bar",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Line Chart":
                        # Column selections
                        x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "line",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Scatter Plot":
                        # Column selections
                        x_col = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox("Y-axis:", [col for col in num_cols if col != x_col] if len(num_cols) > 1 else num_cols, key=f"y_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "scatter",
                            "title": chart_title,
                            "x": x_col,
                            "y": y_col
                        })
                    
                    elif chart_type == "Pie Chart":
                        # Column selections
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        name_col = st.selectbox("Name column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key=f"name_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "pie",
                            "title": chart_title,
                            "value": value_col,
                            "name": name_col
                        })
                    
                    elif chart_type == "Histogram":
                        # Column selection
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        
                        # Add to configurations
                        chart_configs.append({
                            "type": "histogram",
                            "title": chart_title,
                            "value": value_col
                        })
                    
                    elif chart_type == "Box Plot":
                        # Column selection
                        value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key=f"value_col_{i}")
                        
                        # Optional grouping
                        use_groups = st.checkbox("Group by category", key=f"use_groups_{i}")
                        if use_groups and cat_cols:
                            group_col = st.selectbox("Group by:", cat_cols, key=f"group_col_{i}")
                            
                            # Add to configurations with grouping
                            chart_configs.append({
                                "type": "box",
                                "title": chart_title,
                                "value": value_col,
                                "group": group_col
                            })
                        else:
                            # Add to configurations without grouping
                            chart_configs.append({
                                "type": "box",
                                "title": chart_title,
                                "value": value_col
                            })
                    
                    st.markdown("---")
        
        with component_tabs[1]:
            # Metrics section
            st.markdown("### Metric Components")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.info("No numeric columns found for metrics.")
            else:
                # Number of metrics to add
                n_metrics = st.number_input("Number of metrics to add:", 1, 8, 4)
                
                # List to store metric configurations
                metric_configs = []
                
                for i in range(n_metrics):
                    with st.container():
                        st.markdown(f"#### Metric {i+1}")
                        
                        # Metric label
                        metric_label = st.text_input("Metric label:", value=f"Metric {i+1}", key=f"metric_label_{i}")
                        
                        # Metric column
                        metric_col = st.selectbox("Column:", num_cols, key=f"metric_col_{i}")
                        
                        # Aggregation function
                        agg_func = st.selectbox(
                            "Aggregation:",
                            ["Mean", "Median", "Sum", "Min", "Max", "Count"],
                            key=f"agg_func_{i}"
                        )
                        
                        # Formatting
                        format_type = st.selectbox(
                            "Format as:",
                            ["Number", "Percentage", "Currency"],
                            key=f"format_type_{i}"
                        )
                        
                        # Decimal places
                        decimal_places = st.slider("Decimal places:", 0, 4, 2, key=f"decimal_places_{i}")
                        
                        # Add to configurations
                        metric_configs.append({
                            "label": metric_label,
                            "column": metric_col,
                            "aggregation": agg_func.lower(),
                            "format": format_type.lower(),
                            "decimals": decimal_places
                        })
                        
                        st.markdown("---")
        
        with component_tabs[2]:
            # Tables section
            st.markdown("### Table Components")
            
            # Number of tables to add
            n_tables = st.number_input("Number of tables to add:", 0, 4, 1)
            
            # List to store table configurations
            table_configs = []
            
            for i in range(n_tables):
                with st.container():
                    st.markdown(f"#### Table {i+1}")
                    
                    # Table title
                    table_title = st.text_input("Table title:", value=f"Table {i+1}", key=f"table_title_{i}")
                    
                    # Columns to include
                    include_cols = st.multiselect(
                        "Columns to include:",
                        st.session_state.df.columns.tolist(),
                        default=st.session_state.df.columns[:min(5, len(st.session_state.df.columns))].tolist(),
                        key=f"include_cols_{i}"
                    )
                    
                    # Max rows
                    max_rows = st.slider("Maximum rows:", 5, 100, 10, key=f"max_rows_{i}")
                    
                    # Include index
                    include_index = st.checkbox("Include row index", value=False, key=f"include_index_{i}")
                    
                    # Add to configurations
                    table_configs.append({
                        "title": table_title,
                        "columns": include_cols,
                        "max_rows": max_rows,
                        "include_index": include_index
                    })
                    
                    st.markdown("---")
        
        with component_tabs[3]:
            # Filters section
            st.markdown("### Filter Components")
            
            # Number of filters to add
            n_filters = st.number_input("Number of filters to add:", 0, 8, 2)
            
            # List to store filter configurations
            filter_configs = []
            
            for i in range(n_filters):
                with st.container():
                    st.markdown(f"#### Filter {i+1}")
                    
                    # Filter label
                    filter_label = st.text_input("Filter label:", value=f"Filter {i+1}", key=f"filter_label_{i}")
                    
                    # Filter column
                    filter_col = st.selectbox("Column to filter:", st.session_state.df.columns.tolist(), key=f"filter_col_{i}")
                    
                    # Filter type (based on column data type)
                    if st.session_state.df[filter_col].dtype.kind in 'bifc':  # Numeric
                        filter_type = "range"
                    elif st.session_state.df[filter_col].dtype.kind == 'O' and st.session_state.df[filter_col].nunique() <= 20:  # Categorical with few values
                        filter_type = "multiselect"
                    else:  # Categorical with many values or other types
                        filter_type = "text"
                    
                    # Add to configurations
                    filter_configs.append({
                        "label": filter_label,
                        "column": filter_col,
                        "type": filter_type
                    })
                    
                    st.markdown("---")
    
    # Generate the dashboard
    st.subheader(dashboard_title)
    
    # Create a filtered dataframe based on any active filters
    if 'filter_configs' in locals() and filter_configs:
        filtered_df = dashboard_generator.apply_filters(filter_configs)
    else:
        filtered_df = st.session_state.df.copy()
    
    # Display filters
    if 'filter_configs' in locals() and filter_configs:
        st.markdown("### Filters")
        
        # Create columns for filters
        filter_cols = st.columns(min(3, len(filter_configs)))
        
        # Display each filter
        for i, filter_config in enumerate(filter_configs):
            col_idx = i % len(filter_cols)
            with filter_cols[col_idx]:
                dashboard_generator.render_filter(filter_config, key_suffix=f"dash_{i}")
        
        # Apply filters button
        if st.button("Apply Filters", use_container_width=True):
            filtered_df = dashboard_generator.apply_filters(filter_configs)
    
    # Display metrics
    if 'metric_configs' in locals() and metric_configs:
        st.markdown("### Key Metrics")
        
        # Create columns for metrics
        metric_cols = st.columns(min(4, len(metric_configs)))
        
        # Display each metric
        for i, metric_config in enumerate(metric_configs):
            col_idx = i % len(metric_cols)
            with metric_cols[col_idx]:
                dashboard_generator.render_metric(filtered_df, metric_config)
    
    # Display charts
    if 'chart_configs' in locals() and chart_configs:
        st.markdown("### Charts")
        
        # Determine layout
        if layout_type == "2 columns":
            n_cols = 2
        elif layout_type == "3 columns":
            n_cols = 3
        else:  # Custom
            n_cols = st.slider("Number of columns for charts:", 1, 4, 2)
        
        # Create columns for charts
        chart_cols = []
        for i in range(0, len(chart_configs), n_cols):
            # For each row, create n_cols columns (unless we're at the end)
            row_cols = st.columns(min(n_cols, len(chart_configs) - i))
            chart_cols.extend(row_cols)
        
        # Display each chart
        for i, chart_config in enumerate(chart_configs):
            with chart_cols[i]:
                dashboard_generator.render_chart(filtered_df, chart_config)
    
    # Display tables
    if 'table_configs' in locals() and table_configs:
        st.markdown("### Data Tables")
        
        # Display each table
        for table_config in table_configs:
            st.subheader(table_config["title"])
            dashboard_generator.render_table(filtered_df, table_config)

# Main execution
if __name__ == "__main__":
    main()

def render_exploratory_report():
    """Render exploratory analysis report tab"""
    st.subheader("Exploratory Data Analysis Report")
    
    # Report configuration
    st.write("Generate an exploratory data analysis (EDA) report with visualizations and insights.")
    
    with st.expander("Report Configuration", expanded=True):
        # Report title and description
        report_title = st.text_input("Report Title:", "Exploratory Data Analysis Report")
        report_description = st.text_area("Report Description:", "This report provides an exploratory analysis of the dataset.")
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Comprehensive"],
            value="Standard"
        )
        
        # Visualization settings
        st.write("Visualization Settings:")
        max_categorical_values = st.slider("Max categorical values to display:", 5, 30, 10)
        correlation_threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.5)
    
    # Generate report
    if st.button("Generate EDA Report", use_container_width=True):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        report += f"{report_description}\n\n"
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Analysis Depth:** {analysis_depth}\n\n"
        
        # Dataset Overview
        report += "## 1. Dataset Overview\n\n"
        
        # Basic stats
        report += f"* **Rows:** {len(st.session_state.df):,}\n"
        report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
        
        # Data types
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        other_cols = [col for col in st.session_state.df.columns if col not in num_cols + cat_cols + date_cols]
        
        report += f"* **Numeric Columns:** {len(num_cols)}\n"
        report += f"* **Categorical Columns:** {len(cat_cols)}\n"
        report += f"* **DateTime Columns:** {len(date_cols)}\n"
        if other_cols:
            report += f"* **Other Columns:** {len(other_cols)}\n"
        
        # Missing data overview
        missing_values = st.session_state.df.isna().sum().sum()
        missing_pct = (missing_values / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100).round(2)
        report += f"* **Missing Values:** {missing_values:,} ({missing_pct}% of all values)\n\n"
        
        progress_bar.progress(10)
        
        # Column Analysis
        report += "## 2. Column Analysis\n\n"
        
        # Sort columns by type for better organization
        all_cols = num_cols + cat_cols + date_cols + other_cols
        
        # For standard and comprehensive analysis, analyze all columns
        # For basic analysis, limit to a subset
        if analysis_depth == "Basic":
            max_cols_per_type = 3
            analyzed_num_cols = num_cols[:min(max_cols_per_type, len(num_cols))]
            analyzed_cat_cols = cat_cols[:min(max_cols_per_type, len(cat_cols))]
            analyzed_date_cols = date_cols[:min(max_cols_per_type, len(date_cols))]
            analyzed_cols = analyzed_num_cols + analyzed_cat_cols + analyzed_date_cols + other_cols
        else:
            analyzed_num_cols = num_cols
            analyzed_cat_cols = cat_cols
            analyzed_date_cols = date_cols
            analyzed_cols = all_cols
        
        # Initialize charts counter for progress tracking
        total_charts = len(analyzed_cols)
        charts_done = 0
        
        # Analysis for each column
        for col in analyzed_cols:
            # Determine column type
            if col in num_cols:
                col_type = "Numeric"
            elif col in cat_cols:
                col_type = "Categorical"
            elif col in date_cols:
                col_type = "DateTime"
            else:
                col_type = "Other"
            
            report += f"### 2.{analyzed_cols.index(col) + 1}. {col} ({col_type})\n\n"
            
            # Basic column statistics
            non_null_count = st.session_state.df[col].count()
            null_count = st.session_state.df[col].isna().sum()
            null_pct = (null_count / len(st.session_state.df) * 100).round(2)
            
            report += f"* **Non-null Count:** {non_null_count:,} ({100 - null_pct:.2f}%)\n"
            report += f"* **Null Count:** {null_count:,} ({null_pct}%)\n"
            
            if col_type == "Numeric":
                # Numeric column analysis
                numeric_stats = st.session_state.df[col].describe().to_dict()
                
                report += f"* **Mean:** {numeric_stats['mean']:.3f}\n"
                report += f"* **Median:** {st.session_state.df[col].median():.3f}\n"
                report += f"* **Std Dev:** {numeric_stats['std']:.3f}\n"
                report += f"* **Min:** {numeric_stats['min']:.3f}\n"
                report += f"* **Max:** {numeric_stats['max']:.3f}\n"
                
                # Additional statistics for comprehensive analysis
                if analysis_depth == "Comprehensive":
                    try:
                        skewness = st.session_state.df[col].skew()
                        kurtosis = st.session_state.df[col].kurtosis()
                        
                        report += f"* **Skewness:** {skewness:.3f}\n"
                        report += f"* **Kurtosis:** {kurtosis:.3f}\n"
                        
                        # Interpret skewness
                        if abs(skewness) < 0.5:
                            report += f"* **Distribution:** Approximately symmetric\n"
                        elif skewness < 0:
                            report += f"* **Distribution:** Negatively skewed (left-tailed)\n"
                        else:
                            report += f"* **Distribution:** Positively skewed (right-tailed)\n"
                    except:
                        pass
                
                report += f"\n**Histogram will be included in the final report.**\n\n"
                
            elif col_type == "Categorical":
                # Categorical column analysis
                unique_count = st.session_state.df[col].nunique()
                report += f"* **Unique Values:** {unique_count}\n"
                
                # Show top categories
                value_counts = st.session_state.df[col].value_counts().nlargest(max_categorical_values)
                value_pcts = (value_counts / len(st.session_state.df) * 100).round(2)
                
                report += f"\n**Top {min(max_categorical_values, unique_count)} Values:**\n\n"
                
                # Create value counts dataframe
                vc_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': value_pcts.values
                })
                
                # Convert to markdown table
                report += vc_df.to_markdown(index=False)
                report += "\n\n**Bar chart will be included in the final report.**\n\n"
                
            elif col_type == "DateTime":
                # DateTime column analysis
                try:
                    min_date = st.session_state.df[col].min()
                    max_date = st.session_state.df[col].max()
                    date_range = max_date - min_date
                    
                    report += f"* **Minimum Date:** {min_date}\n"
                    report += f"* **Maximum Date:** {max_date}\n"
                    report += f"* **Date Range:** {date_range}\n"
                    
                    # For comprehensive analysis, include time-based statistics
                    if analysis_depth in ["Standard", "Comprehensive"]:
                        # Extract year, month, day
                        years = pd.DatetimeIndex(st.session_state.df[col].dropna()).year
                        months = pd.DatetimeIndex(st.session_state.df[col].dropna()).month
                        days = pd.DatetimeIndex(st.session_state.df[col].dropna()).day
                        
                        # Most common year
                        if len(years) > 0:
                            common_year = pd.Series(years).value_counts().nlargest(1).index[0]
                            common_year_count = pd.Series(years).value_counts().nlargest(1).values[0]
                            common_year_pct = (common_year_count / len(years) * 100).round(2)
                            
                            report += f"* **Most Common Year:** {common_year} ({common_year_count:,} occurrences, {common_year_pct}%)\n"
                        
                        # For comprehensive analysis, include month and day statistics
                        if analysis_depth == "Comprehensive":
                            # Most common month
                            if len(months) > 0:
                                common_month = pd.Series(months).value_counts().nlargest(1).index[0]
                                common_month_count = pd.Series(months).value_counts().nlargest(1).values[0]
                                common_month_pct = (common_month_count / len(months) * 100).round(2)
                                
                                # Convert month number to name
                                import calendar
                                month_name = calendar.month_name[common_month]
                                
                                report += f"* **Most Common Month:** {month_name} ({common_month_count:,} occurrences, {common_month_pct}%)\n"
                            
                            # Most common day of week
                            try:
                                weekdays = pd.DatetimeIndex(st.session_state.df[col].dropna()).dayofweek
                                
                                if len(weekdays) > 0:
                                    common_weekday = pd.Series(weekdays).value_counts().nlargest(1).index[0]
                                    common_weekday_count = pd.Series(weekdays).value_counts().nlargest(1).values[0]
                                    common_weekday_pct = (common_weekday_count / len(weekdays) * 100).round(2)
                                    
                                    # Convert weekday number to name
                                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                    weekday_name = weekday_names[common_weekday]
                                    
                                    report += f"* **Most Common Day of Week:** {weekday_name} ({common_weekday_count:,} occurrences, {common_weekday_pct}%)\n"
                            except:
                                pass
                                
                    report += f"\n**Time series plot will be included in the final report.**\n\n"
                except Exception as e:
                    report += f"*Error analyzing datetime column: {str(e)}*\n\n"
            
            # Update progress
            charts_done += 1
            progress_bar.progress(10 + int(40 * charts_done / total_charts))
        
        # Correlation Analysis (for numeric columns)
        if len(analyzed_num_cols) >= 2:
            report += "## 3. Correlation Analysis\n\n"
            
            # Calculate correlation matrix
            corr_matrix = st.session_state.df[analyzed_num_cols].corr().round(3)
            
            # For basic analysis, limit to top correlations
            if analysis_depth == "Basic":
                report += "### Top Correlations\n\n"
                
                # Convert correlation matrix to a list of pairs
                pairs = []
                for i, col1 in enumerate(analyzed_num_cols):
                    for j, col2 in enumerate(analyzed_num_cols):
                        if i < j:  # Only include each pair once
                            pairs.append({
                                'Column 1': col1,
                                'Column 2': col2,
                                'Correlation': corr_matrix.loc[col1, col2]
                            })
                
                # Convert to dataframe
                pairs_df = pd.DataFrame(pairs)
                
                # Sort by absolute correlation
                pairs_df['Abs Correlation'] = pairs_df['Correlation'].abs()
                pairs_df = pairs_df.sort_values('Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
                
                # Display top correlations above threshold
                strong_pairs = pairs_df[pairs_df['Correlation'].abs() >= correlation_threshold]
                
                if len(strong_pairs) > 0:
                    report += f"**Strong correlations (|r| ≥ {correlation_threshold}):**\n\n"
                    report += strong_pairs.to_markdown(index=False)
                    report += "\n\n"
                else:
                    report += f"*No strong correlations found (using threshold |r| ≥ {correlation_threshold}).*\n\n"
            else:
                # For standard and comprehensive analysis, include full correlation matrix
                report += "### Correlation Matrix\n\n"
                report += "*Correlation heatmap will be included in the final report.*\n\n"
                
                if analysis_depth == "Comprehensive":
                    report += "### Full Correlation Matrix\n\n"
                    report += corr_matrix.to_markdown()
                    report += "\n\n"
        
        progress_bar.progress(60)
        
        # Missing Value Analysis
        report += "## 4. Missing Value Analysis\n\n"
        
        # Calculate missing values for each column
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isna().sum().values,
            'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
        })
        
        # Sort by missing count (descending)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        # Get columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
        
        if len(cols_with_missing) > 0:
            report += f"**Total Columns with Missing Values:** {len(cols_with_missing)} out of {len(st.session_state.df.columns)}\n\n"
            
            # Convert to markdown table
            report += cols_with_missing.to_markdown(index=False)
            report += "\n\n*Missing values bar chart will be included in the final report.*\n\n"
            
            # For comprehensive analysis, include detailed pattern analysis
            if analysis_depth == "Comprehensive" and len(cols_with_missing) > 1:
                report += "### Missing Value Patterns\n\n"
                report += "*Missing values heatmap will be included in the final report.*\n\n"
                
                # Calculate correlations between missing values
                # This indicates if missingness in one column correlates with missingness in another
                missing_pattern = st.session_state.df[cols_with_missing['Column']].isna()
                missing_corr = missing_pattern.corr().round(3)
                
                report += "**Missing Value Correlation Matrix** (correlation between missing patterns)\n\n"
                report += missing_corr.to_markdown()
                report += "\n\n*High correlation suggests missing values occur together in the same rows.*\n\n"
        else:
            report += "*No missing values found in the dataset.*\n\n"
        
        progress_bar.progress(70)
        
        # Outlier Analysis (for numeric columns)
        if len(analyzed_num_cols) > 0:
            report += "## 5. Outlier Analysis\n\n"
            
            outlier_summary = []
            
            for col in analyzed_num_cols:
                # Calculate IQR
                Q1 = st.session_state.df[col].quantile(0.25)
                Q3 = st.session_state.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = st.session_state.df[(st.session_state.df[col] < lower_bound) | 
                                     (st.session_state.df[col] > upper_bound)]
                
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(st.session_state.df) * 100).round(2)
                
                # Add to summary
                outlier_summary.append({
                    'Column': col,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Outlier Count': outlier_count,
                    'Outlier %': outlier_pct
                })
            
            # Convert to dataframe
            outlier_df = pd.DataFrame(outlier_summary)
            
            # Sort by outlier count (descending)
            outlier_df = outlier_df.sort_values('Outlier Count', ascending=False)
            
            # For standard and comprehensive analysis, include detailed outlier information
            if analysis_depth in ["Standard", "Comprehensive"]:
                report += "### Outlier Summary\n\n"
                
                # Convert to markdown table
                summary_cols = ['Column', 'Outlier Count', 'Outlier %', 'Lower Bound', 'Upper Bound']
                report += outlier_df[summary_cols].to_markdown(index=False)
                report += "\n\n"
                
                report += "*Box plots for columns with outliers will be included in the final report.*\n\n"
                
                # For comprehensive analysis, include detailed statistics
                if analysis_depth == "Comprehensive":
                    report += "### Detailed Outlier Statistics\n\n"
                    report += outlier_df.to_markdown(index=False)
                    report += "\n\n"
            else:
                # For basic analysis, just include a summary
                report += f"**Total columns with outliers:** {len(outlier_df[outlier_df['Outlier Count'] > 0])}\n\n"
                
                # Show top columns with outliers
                top_outliers = outlier_df[outlier_df['Outlier Count'] > 0].head(5)
                
                if len(top_outliers) > 0:
                    report += "**Top columns with outliers:**\n\n"
                    report += top_outliers[['Column', 'Outlier Count', 'Outlier %']].to_markdown(index=False)
                    report += "\n\n"
                else:
                    report += "*No outliers found in the dataset using the IQR method.*\n\n"
        
        progress_bar.progress(80)
        
        # Distribution Analysis (for categorical columns)
        if len(analyzed_cat_cols) > 0:
            report += "## 6. Categorical Data Analysis\n\n"
            
            # For each categorical column, analyze distribution
            for i, col in enumerate(analyzed_cat_cols[:5]):  # Limit to top 5 columns
                unique_count = st.session_state.df[col].nunique()
                
                report += f"### 6.{i+1}. {col}\n\n"
                report += f"* **Unique Values:** {unique_count}\n"
                
                # For columns with many unique values, show value counts
                if unique_count <= 30 or analysis_depth == "Comprehensive":
                    # Get value counts
                    value_counts = st.session_state.df[col].value_counts().nlargest(max_categorical_values)
                    value_pcts = (value_counts / len(st.session_state.df) * 100).round(2)
                    
                    report += f"\n**Top {min(max_categorical_values, unique_count)} Values:**\n\n"
                    
                    # Create value counts dataframe
                    vc_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': value_pcts.values
                    })
                    
                    # Convert to markdown table
                    report += vc_df.to_markdown(index=False)
                    report += "\n\n"
                
                report += "*Bar chart will be included in the final report.*\n\n"
            
            # If too many columns, note that only top ones were shown
            if len(analyzed_cat_cols) > 5:
                report += f"*Note: Only the first 5 out of {len(analyzed_cat_cols)} categorical columns are shown in detail.*\n\n"
        
        progress_bar.progress(90)
        
        # Insights and Recommendations
        report += "## 7. Insights and Recommendations\n\n"
        
        # Basic insights
        report += "### Key Insights\n\n"
        
        # Dataset size
        report += f"* **Dataset Size:** The dataset contains {len(st.session_state.df):,} rows and {len(st.session_state.df.columns)} columns.\n"
        
        # Data types
        report += f"* **Data Composition:** The dataset includes {len(num_cols)} numeric columns, {len(cat_cols)} categorical columns, and {len(date_cols)} datetime columns.\n"
        
        # Missing values
        if missing_values > 0:
            report += f"* **Missing Data:** {missing_values:,} values ({missing_pct}% of all values) are missing in the dataset.\n"
            
            # Top columns with missing values
            top_missing = cols_with_missing.head(3)
            if len(top_missing) > 0:
                missing_cols = ", ".join([f"'{col}' ({pct}%)" for col, pct in zip(top_missing['Column'], top_missing['Missing %'])])
                report += f"  * Columns with most missing values: {missing_cols}\n"
        else:
            report += f"* **Missing Data:** No missing values found in the dataset.\n"
        
        # Outliers (if analyzed)
        if len(analyzed_num_cols) > 0:
            outlier_cols = outlier_df[outlier_df['Outlier %'] > 5]
            if len(outlier_cols) > 0:
                report += f"* **Outliers:** {len(outlier_cols)} columns have more than 5% outliers.\n"
                
                # Top columns with outliers
                top_outlier_cols = outlier_cols.head(3)
                outlier_col_list = ", ".join([f"'{col}' ({pct}%)" for col, pct in zip(top_outlier_cols['Column'], top_outlier_cols['Outlier %'])])
                report += f"  * Columns with most outliers: {outlier_col_list}\n"
        
        # Correlations (if analyzed)
        if len(analyzed_num_cols) >= 2:
            strong_corrs = pairs_df[pairs_df['Correlation'].abs() >= 0.7]
            if len(strong_corrs) > 0:
                report += f"* **Strong Correlations:** {len(strong_corrs)} pairs of numeric columns have strong correlations (|r| ≥ 0.7).\n"
                
                # Top correlations
                top_corr = strong_corrs.head(2)
                if len(top_corr) > 0:
                    corr_list = ", ".join([f"'{col1}' and '{col2}' (r={corr:.2f})" for col1, col2, corr in zip(
                        top_corr['Column 1'], top_corr['Column 2'], top_corr['Correlation'])])
                    report += f"  * Strongest correlations: {corr_list}\n"
        
        # Recommendations
        report += "\n### Recommendations\n\n"
        
        # Missing data recommendations
        if missing_values > 0:
            report += "* **Missing Data Handling:**\n"
            report += "  * Consider imputation strategies for columns with missing values, such as mean/median imputation for numeric columns or mode imputation for categorical columns.\n"
            report += "  * For columns with high missing percentages, evaluate whether they should be kept or dropped.\n"
            report += "  * Investigate if missing data follows a pattern that could introduce bias.\n"
        
        # Outlier recommendations
        if len(analyzed_num_cols) > 0 and len(outlier_df[outlier_df['Outlier Count'] > 0]) > 0:
            report += "* **Outlier Treatment:**\n"
            report += "  * Inspect outliers to determine if they are valid data points or errors.\n"
            report += "  * Consider appropriate treatment such as capping, transforming, or removing outliers depending on your analysis goals.\n"
        
        # Correlation recommendations
        if len(analyzed_num_cols) >= 2:
            report += "* **Feature Selection/Engineering:**\n"
            if len(strong_corrs) > 0:
                report += "  * Highly correlated features may contain redundant information. Consider removing some of them or creating composite features.\n"
            report += "  * Explore feature engineering opportunities to create more predictive variables.\n"
        
        # Categorical data recommendations
        if len(analyzed_cat_cols) > 0:
            high_cardinality_cols = [col for col in analyzed_cat_cols if st.session_state.df[col].nunique() > 20]
            if high_cardinality_cols:
                report += "* **Categorical Variable Treatment:**\n"
                report += "  * Some categorical variables have high cardinality (many unique values). Consider grouping less frequent categories.\n"
                report += "  * For machine learning, use appropriate encoding techniques (one-hot encoding for low cardinality, target or frequency encoding for high cardinality).\n"
        
        # General recommendations
        report += "* **Further Analysis:**\n"
        report += "  * Consider more advanced analytics like clustering, dimensionality reduction, or predictive modeling.\n"
        report += "  * Validate any patterns or insights with domain experts.\n"
        
        progress_bar.progress(100)
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate charts for visualizations mentioned in the report
        st.subheader("Report Charts")
        
        # Create a list to store chart figures
        charts = []
        
        # Numeric column histograms
        for col in analyzed_num_cols[:3]:  # Limit to first 3 columns
            fig = px.histogram(
                st.session_state.df,
                x=col,
                title=f"Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Categorical column bar charts
        for col in analyzed_cat_cols[:3]:  # Limit to first 3 columns
            # Limit to top categories
            top_cats = st.session_state.df[col].value_counts().nlargest(max_categorical_values).index.tolist()
            filtered_df = st.session_state.df[st.session_state.df[col].isin(top_cats)]
            
            fig = px.bar(
                filtered_df[col].value_counts().reset_index(),
                x="index",
                y=col,
                title=f"Frequency of {col} Categories (Top {len(top_cats)})",
                labels={"index": col, col: "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Correlation heatmap
        if len(analyzed_num_cols) >= 2:
            corr_matrix = st.session_state.df[analyzed_num_cols].corr().round(3)
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Missing values bar chart
        if len(cols_with_missing) > 0:
            fig = px.bar(
                cols_with_missing,
                x='Column',
                y='Missing %',
                title='Missing Values by Column (%)',
                color='Missing %',
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
            charts.append(fig)
        
        # Export options
        st.subheader("Export Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as markdown
            st.download_button(
                label="Download as Markdown",
                data=report,
                file_name=f"{report_title.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            try:
                import markdown
                html = markdown.markdown(report)
                
                st.download_button(
                    label="Download as HTML",
                    data=html,
                    file_name=f"{report_title.replace(' ', '_').lower()}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_trend_analysis():
    """Render trend analysis tab"""
    st.subheader("Trend Analysis")
    
    # Check if we have date columns
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Also look for potential date columns
    potential_date_cols = []
    for col in st.session_state.df.columns:
        if col in date_cols:
            continue
        
        if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
            potential_date_cols.append(col)
    
    all_date_cols = date_cols + potential_date_cols
    
    if not all_date_cols:
        st.info("No date/time columns found for trend analysis. Convert a column to datetime first.")
        return
    
    # Column selections
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Date/Time column:", all_date_cols)
    
    with col2:
        # Get numeric columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns found for trend analysis.")
            return
        
        y_col = st.selectbox("Value column:", num_cols)
    
    # Convert date column to datetime if it's not already
    if x_col in potential_date_cols:
        try:
            # Try to convert to datetime
            st.session_state.df[x_col] = pd.to_datetime(st.session_state.df[x_col])
            st.success(f"Converted '{x_col}' to datetime format.")
        except Exception as e:
            st.error(f"Could not convert '{x_col}' to datetime: {str(e)}")
            st.info("Try selecting a different column or convert it to datetime in the Data Processing tab.")
            return
    
    # Trend analysis type
    analysis_type = st.radio(
        "Analysis type:",
        ["Time Series Plot", "Moving Averages", "Seasonality Analysis", "Trend Decomposition"],
        horizontal=True
    )
    
    if analysis_type == "Time Series Plot":
        # Optional grouping
        use_grouping = st.checkbox("Group by a categorical column")
        group_col = None
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                group_col = st.selectbox("Group by:", cat_cols)
        
        # Create time series plot
        if use_grouping and group_col:
            # Get top groups for better visualization
            top_n = st.slider("Show top N groups:", 1, 10, 5)
            top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
            
            # Filter to top groups
            filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
            
            # Create grouped time series
            fig = px.line(
                filtered_df,
                x=x_col,
                y=y_col,
                color=group_col,
                title=f"Time Series Plot of {y_col} over {x_col}, grouped by {group_col}"
            )
        else:
            # Simple time series
            fig = px.line(
                st.session_state.df,
                x=x_col,
                y=y_col,
                title=f"Time Series Plot of {y_col} over {x_col}"
            )
        
        # Add trend line
        add_trend = st.checkbox("Add trend line")
        if add_trend:
            try:
                from scipy import stats as scipy_stats
                
                # Convert dates to ordinal values for regression
                x_values = pd.to_numeric(pd.to_datetime(st.session_state.df[x_col])).values
                y_values = st.session_state.df[y_col].values
                
                # Remove NaN values
                mask = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[mask]
                y_values = y_values[mask]
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_values, y_values)
                
                # Create line
                x_range = np.array([x_values.min(), x_values.max()])
                y_range = intercept + slope * x_range
                
                # Convert back to datetime for plotting
                x_dates = pd.to_datetime(x_range)
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=x_dates,
                        y=y_range,
                        mode="lines",
                        name=f"Trend (r²={r_value**2:.3f})",
                        line=dict(color="red", dash="dash")
                    )
                )
                
            except Exception as e:
                st.warning(f"Could not add trend line: {str(e)}")
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader("Time Series Statistics")
        
        # Calculate statistics
        try:
            # First and last values
            first_value = st.session_state.df.sort_values(x_col).iloc[0][y_col]
            last_value = st.session_state.df.sort_values(x_col).iloc[-1][y_col]
            
            # Change and percent change
            change = last_value - first_value
            pct_change = (change / first_value) * 100 if first_value != 0 else float('inf')
            
            # Min and max values
            min_value = st.session_state.df[y_col].min()
            max_value = st.session_state.df[y_col].max()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("First Value", f"{first_value:.2f}")
            with col2:
                st.metric("Last Value", f"{last_value:.2f}")
            with col3:
                st.metric("Change", f"{change:.2f}")
            with col4:
                st.metric("% Change", f"{pct_change:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Minimum", f"{min_value:.2f}")
            with col2:
                st.metric("Maximum", f"{max_value:.2f}")
            with col3:
                st.metric("Range", f"{max_value - min_value:.2f}")
            with col4:
                st.metric("Mean", f"{st.session_state.df[y_col].mean():.2f}")
            
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
    
    elif analysis_type == "Moving Averages":
        # Window sizes
        window_sizes = st.multiselect(
            "Select moving average window sizes:",
            [7, 14, 30, 60, 90, 180, 365],
            default=[30]
        )
        
        if not window_sizes:
            st.warning("Please select at least one window size.")
            return
        
        # Create base time series plot
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_col],
                y=st.session_state.df[y_col],
                mode="lines",
                name=f"Original {y_col}",
                line=dict(color="blue")
            )
        )
        
        # Add moving averages
        colors = ["red", "green", "purple", "orange", "brown", "pink", "grey"]
        
        for i, window in enumerate(window_sizes):
            # Calculate moving average
            ma = st.session_state.df[y_col].rolling(window=window).mean()
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=ma,
                    mode="lines",
                    name=f"{window}-period MA",
                    line=dict(color=colors[i % len(colors)])
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Moving Averages of {y_col} over {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis options
        st.subheader("Additional Analysis")
        
        # Volatility analysis
        show_volatility = st.checkbox("Show volatility (rolling standard deviation)")
        if show_volatility:
            # Create volatility plot
            fig_vol = go.Figure()
            
            # Calculate rolling standard deviation
            vol_window = st.slider("Volatility window size:", 5, 100, 30)
            volatility = st.session_state.df[y_col].rolling(window=vol_window).std()
            
            # Add to plot
            fig_vol.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=volatility,
                    mode="lines",
                    name=f"{vol_window}-period Volatility",
                    line=dict(color="red")
                )
            )
            
            # Update layout
            fig_vol.update_layout(
                title=f"Volatility (Rolling Std Dev) of {y_col} over {x_col}",
                xaxis_title=x_col,
                yaxis_title=f"Standard Deviation of {y_col}"
            )
            
            # Show plot
            st.plotly_chart(fig_vol, use_container_width=True)
    
    elif analysis_type == "Seasonality Analysis":
        try:
            # Make sure we have enough data
            if len(st.session_state.df) < 10:
                st.warning("Need more data points for seasonality analysis.")
                return
            
            # Check if the data is regularly spaced in time
            df_sorted = st.session_state.df.sort_values(x_col)
            date_diffs = df_sorted[x_col].diff().dropna()
            
            if date_diffs.nunique() > 5:
                st.warning("Data points are not regularly spaced in time. Seasonality analysis may not be accurate.")
            
            # Time period selection
            period_type = st.selectbox(
                "Analyze seasonality by:",
                ["Day of Week", "Month", "Quarter", "Year"]
            )
            
            # Extract the relevant period component
            if period_type == "Day of Week":
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.day_name()
                period_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            elif period_type == "Month":
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                period_order = ["January", "February", "March", "April", "May", "June", 
                               "July", "August", "September", "October", "November", "December"]
            elif period_type == "Quarter":
                df_sorted['period'] = "Q" + pd.to_datetime(df_sorted[x_col]).dt.quarter.astype(str)
                period_order = ["Q1", "Q2", "Q3", "Q4"]
            else:  # Year
                df_sorted['period'] = pd.to_datetime(df_sorted[x_col]).dt.year
                years = sorted(df_sorted['period'].unique())
                period_order = years
            
            # Calculate statistics by period
            period_stats = df_sorted.groupby('period')[y_col].agg(['mean', 'median', 'min', 'max', 'std', 'count'])
            
            # Reorder based on natural order
            if period_type != "Year":
                # For named periods, use predefined order
                period_stats = period_stats.reindex(period_order)
            else:
                # For years, sort numerically
                period_stats = period_stats.sort_index()
            
            # Fill NaN with 0
            period_stats = period_stats.fillna(0)
            
            # Create bar chart
            fig = go.Figure()
            
            # Add mean bars
            fig.add_trace(
                go.Bar(
                    x=period_stats.index,
                    y=period_stats['mean'],
                    name="Mean",
                    error_y=dict(
                        type='data',
                        array=period_stats['std'],
                        visible=True
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Seasonality Analysis of {y_col} by {period_type}",
                xaxis_title=period_type,
                yaxis_title=f"Mean {y_col} (with Std Dev)"
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics table
            st.subheader(f"Statistics by {period_type}")
            st.dataframe(period_stats.round(2), use_container_width=True)
            
            # Heatmap visualization
            st.subheader("Seasonality Heatmap")
            
            # Create heatmap based on period type
            if period_type in ["Month", "Day of Week"]:
                # For month or day, we can also show year-on-year patterns
                
                if period_type == "Month":
                    # Extract month and year
                    df_sorted['month'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                    df_sorted['year'] = pd.to_datetime(df_sorted[x_col]).dt.year
                    
                    # Calculate monthly averages by year
                    heatmap_data = df_sorted.groupby(['year', 'month'])[y_col].mean().unstack()
                    
                    # Reorder months
                    heatmap_data = heatmap_data[period_order]
                    
                else:  # Day of Week
                    # Extract day of week and week of year
                    df_sorted['day'] = pd.to_datetime(df_sorted[x_col]).dt.day_name()
                    
                    # Use month or week number as the other dimension
                    use_month = st.radio(
                        "Second dimension for day of week heatmap:",
                        ["Month", "Week of Year"],
                        horizontal=True
                    )
                    
                    if use_month:
                        df_sorted['second_dim'] = pd.to_datetime(df_sorted[x_col]).dt.month_name()
                        dim_order = ["January", "February", "March", "April", "May", "June", 
                                   "July", "August", "September", "October", "November", "December"]
                        dim_name = "Month"
                    else:
                        df_sorted['second_dim'] = pd.to_datetime(df_sorted[x_col]).dt.isocalendar().week
                        weeks = sorted(df_sorted['second_dim'].unique())
                        dim_order = weeks
                        dim_name = "Week of Year"
                    
                    # Calculate averages
                    heatmap_data = df_sorted.groupby(['second_dim', 'day'])[y_col].mean().unstack()
                    
                    # Reorder days
                    heatmap_data = heatmap_data[period_order]
                
                # Create heatmap
                try:
                    # Convert to numpy for Plotly
                    z_data = heatmap_data.values
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='Blues',
                        colorbar=dict(title=f"Mean {y_col}")
                    ))
                    
                    # Update layout
                    if period_type == "Month":
                        fig.update_layout(
                            title=f"Monthly {y_col} Heatmap by Year",
                            xaxis_title="Month",
                            yaxis_title="Year"
                        )
                    else:
                        fig.update_layout(
                            title=f"Day of Week {y_col} Heatmap by {dim_name}",
                            xaxis_title="Day of Week",
                            yaxis_title=dim_name
                        )
                    
                    # Show heatmap
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in seasonality analysis: {str(e)}")
    
    elif analysis_type == "Trend Decomposition":
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Need to make sure data is sorted and has a regular time index
            st.info("Trend decomposition requires regularly spaced time series data.")
            
            # Sort data by date
            df_sorted = st.session_state.df.sort_values(x_col)
            
            # Check if we need to resample
            resample = st.checkbox("Resample data to regular intervals")
            
            if resample:
                # Frequency selection
                freq = st.selectbox(
                    "Resampling frequency:",
                    ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
                )
                
                # Map to pandas frequency string
                freq_map = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "MS",
                    "Quarterly": "QS",
                    "Yearly": "YS"
                }
                
                # Set date as index
                df_sorted = df_sorted.set_index(x_col)
                
                # Resample
                df_resampled = df_sorted[y_col].resample(freq_map[freq]).mean()
                
                # Convert back to dataframe
                data_ts = pd.DataFrame({y_col: df_resampled})
                
                # Reset index
                data_ts = data_ts.reset_index()
                
                # Rename index column
                data_ts = data_ts.rename(columns={"index": x_col})
                
            else:
                # Use original data
                data_ts = df_sorted[[x_col, y_col]]
            
            # Make sure we have enough data
            if len(data_ts) < 10:
                st.warning("Need at least 10 data points for decomposition. Try resampling to a lower frequency.")
                return
            
            # Set date as index
            data_ts = data_ts.set_index(x_col)
            
            # Decomposition model
            model = st.selectbox(
                "Decomposition model:",
                ["Additive", "Multiplicative"]
            )
            
            # Period selection
            period = st.slider("Period (number of time units in a seasonal cycle):", 2, 52, 12)
            
            # Perform decomposition
            if model == "Additive":
                result = seasonal_decompose(data_ts[y_col], model='additive', period=period)
            else:
                result = seasonal_decompose(data_ts[y_col], model='multiplicative', period=period)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                shared_xaxes=True,
                vertical_spacing=0.05
            )
            
            # Add observed data
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.observed,
                    mode="lines",
                    name="Observed"
                ),
                row=1, col=1
            )
            
            # Add trend
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.trend,
                    mode="lines",
                    name="Trend",
                    line=dict(color="red")
                ),
                row=2, col=1
            )
            
            # Add seasonal
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.seasonal,
                    mode="lines",
                    name="Seasonal",
                    line=dict(color="green")
                ),
                row=3, col=1
            )
            
            # Add residual
            fig.add_trace(
                go.Scatter(
                    x=data_ts.index,
                    y=result.resid,
                    mode="lines",
                    name="Residual",
                    line=dict(color="purple")
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title=f"{model} Decomposition of {y_col} (Period={period})",
                showlegend=False
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Component Statistics")
            
            # Calculate statistics for each component
            stats = pd.DataFrame({
                "Component": ["Observed", "Trend", "Seasonal", "Residual"],
                "Mean": [
                    result.observed.mean(),
                    result.trend.mean(),
                    result.seasonal.mean(),
                    result.resid.dropna().mean()
                ],
                "Std Dev": [
                    result.observed.std(),
                    result.trend.std(),
                    result.seasonal.std(),
                    result.resid.dropna().std()
                ],
                "Min": [
                    result.observed.min(),
                    result.trend.min(),
                    result.seasonal.min(),
                    result.resid.dropna().min()
                ],
                "Max": [
                    result.observed.max(),
                    result.trend.max(),
                    result.seasonal.max(),
                    result.resid.dropna().max()
                ]
            })
            
            # Display statistics
            st.dataframe(stats.round(2), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in trend decomposition: {str(e)}")
            st.info("Try resampling the data or adjusting the period parameter.")
    
    # Export options
    if 'fig' in locals():
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
        
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Chart",
                data=html_bytes,
                file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            # Export as image
            img_bytes = fig.to_image(format=export_format.lower())
            
            st.download_button(
                label=f"Download Chart",
                data=img_bytes,
                file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True
            )

def render_distribution_analysis():
    """Render distribution analysis tab"""
    st.subheader("Distribution Analysis")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns found for distribution analysis.")
        return
    
    # Column selection
    selected_col = st.selectbox(
        "Select column for distribution analysis:",
        num_cols
    )
    
    # Analysis type selection
    analysis_type = st.radio(
        "Analysis type:",
        ["Histogram", "Density Plot", "Box Plot", "Q-Q Plot", "Distribution Fitting"],
        horizontal=True
    )
    
    if analysis_type == "Histogram":
        # Histogram options
        n_bins = st.slider("Number of bins:", 5, 100, 20)
        
        # Create histogram
        fig = px.histogram(
            st.session_state.df,
            x=selected_col,
            nbins=n_bins,
            title=f"Histogram of {selected_col}",
            marginal="box"
        )
        
        # Show histogram
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate basic statistics
        stats = st.session_state.df[selected_col].describe().to_dict()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{stats['mean']:.3f}")
        with col2:
            st.metric("Median", f"{stats['50%']:.3f}")
        with col3:
            st.metric("Std Dev", f"{stats['std']:.3f}")
        with col4:
            st.metric("Count", f"{stats['count']}")
        
    elif analysis_type == "Density Plot":
        # Create KDE plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Calculate KDE
            kde_x = np.linspace(data.min(), data.max(), 1000)
            kde = scipy_stats.gaussian_kde(data)
            kde_y = kde(kde_x)
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            show_hist = st.checkbox("Show histogram with density plot", value=True)
            if show_hist:
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name="Histogram",
                        opacity=0.7,
                        histnorm="probability density"
                    )
                )
            
            # Add KDE
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name="KDE",
                    line=dict(color="red", width=2)
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Density Plot of {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Density"
            )
            
            # Show density plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate skewness and kurtosis
            skewness = scipy_stats.skew(data)
            kurtosis = scipy_stats.kurtosis(data)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{data.mean():.3f}")
            with col2:
                st.metric("Median", f"{data.median():.3f}")
            with col3:
                st.metric("Skewness", f"{skewness:.3f}")
            with col4:
                st.metric("Kurtosis", f"{kurtosis:.3f}")
            
            # Interpretation of skewness and kurtosis
            st.subheader("Distribution Interpretation")
            
            # Skewness interpretation
            if abs(skewness) < 0.5:
                skew_text = "approximately symmetric"
            elif skewness < 0:
                skew_text = "negatively skewed (left-tailed)"
            else:
                skew_text = "positively skewed (right-tailed)"
            
            # Kurtosis interpretation
            if abs(kurtosis) < 0.5:
                kurt_text = "approximately mesokurtic (similar to normal distribution)"
            elif kurtosis < 0:
                kurt_text = "platykurtic (flatter than normal distribution)"
            else:
                kurt_text = "leptokurtic (more peaked than normal distribution)"
            
            st.write(f"The distribution is {skew_text} and {kurt_text}.")
            
        except Exception as e:
            st.error(f"Error creating density plot: {str(e)}")
            return
        
    elif analysis_type == "Box Plot":
        # Optional grouping
        use_grouping = st.checkbox("Group by a categorical column")
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                group_col = st.selectbox("Group by:", cat_cols)
                
                # Limit number of groups
                top_n = st.slider("Show top N groups:", 2, 20, 10)
                
                # Get top groups
                top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
                
                # Filter to top groups
                filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
                
                # Create grouped box plot
                fig = px.box(
                    filtered_df,
                    x=group_col,
                    y=selected_col,
                    title=f"Box Plot of {selected_col} by {group_col}",
                    points="all"
                )
        
        if not use_grouping or not cat_cols:
            # Simple box plot
            fig = px.box(
                st.session_state.df,
                y=selected_col,
                title=f"Box Plot of {selected_col}",
                points="all"
            )
        
        # Show box plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate quartiles and IQR
        q1 = st.session_state.df[selected_col].quantile(0.25)
        q3 = st.session_state.df[selected_col].quantile(0.75)
        iqr = q3 - q1
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{st.session_state.df[selected_col].min():.3f}")
        with col2:
            st.metric("Q1 (25%)", f"{q1:.3f}")
        with col3:
            st.metric("Median", f"{st.session_state.df[selected_col].median():.3f}")
        with col4:
            st.metric("Q3 (75%)", f"{q3:.3f}")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Max", f"{st.session_state.df[selected_col].max():.3f}")
        with col2:
            st.metric("IQR", f"{iqr:.3f}")
        with col3:
            # Calculate outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = st.session_state.df[(st.session_state.df[selected_col] < lower_bound) | 
                                  (st.session_state.df[selected_col] > upper_bound)]
            
            st.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(st.session_state.df)*100:.1f}%)")
        
    elif analysis_type == "Q-Q Plot":
        # Distribution selection
        dist = st.selectbox(
            "Reference distribution:",
            ["Normal", "Uniform", "Exponential", "Log-normal"]
        )
        
        # Create Q-Q plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Calculate theoretical quantiles
            if dist == "Normal":
                theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Normal"
            elif dist == "Uniform":
                theoretical_quantiles = scipy_stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Uniform"
            elif dist == "Exponential":
                theoretical_quantiles = scipy_stats.expon.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Exponential"
            elif dist == "Log-normal":
                theoretical_quantiles = scipy_stats.lognorm.ppf(np.linspace(0.01, 0.99, len(data)), s=1)
                theoretical_name = "Log-normal"
            
            # Sort the data
            sample_quantiles = np.sort(data)
            
            # Create Q-Q plot
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Data'
                )
            )
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Q-Q Plot: {selected_col} vs {theoretical_name} Distribution",
                xaxis_title=f"Theoretical Quantiles ({theoretical_name})",
                yaxis_title=f"Sample Quantiles ({selected_col})"
            )
            
            # Show Q-Q plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Perform Shapiro-Wilk test for normality if normal distribution selected
            if dist == "Normal":
                shapiro_test = scipy_stats.shapiro(data)
                
                st.subheader("Normality Test (Shapiro-Wilk)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{shapiro_test.statistic:.4f}")
                with col2:
                    st.metric("p-value", f"{shapiro_test.pvalue:.4f}")
                
                alpha = 0.05
                if shapiro_test.pvalue < alpha:
                    st.write(f"With p-value < {alpha}, we can reject the null hypothesis that the data is normally distributed.")
                else:
                    st.write(f"With p-value >= {alpha}, we cannot reject the null hypothesis that the data is normally distributed.")
            
        except Exception as e:
            st.error(f"Error creating Q-Q plot: {str(e)}")
            return
        
    elif analysis_type == "Distribution Fitting":
        # Try to fit distributions to the data
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            # Distributions to try
            distributions = [
                ("Normal", scipy_stats.norm),
                ("Log-normal", scipy_stats.lognorm),
                ("Exponential", scipy_stats.expon),
                ("Gamma", scipy_stats.gamma),
                ("Beta", scipy_stats.beta),
                ("Weibull", scipy_stats.weibull_min)
            ]
            
            # Select distributions to try
            selected_dists = st.multiselect(
                "Select distributions to fit:",
                [d[0] for d in distributions],
                default=["Normal", "Log-normal", "Gamma"]
            )
            
            if not selected_dists:
                st.warning("Please select at least one distribution to fit.")
                return
            
            # Filter to selected distributions
            distributions = [d for d in distributions if d[0] in selected_dists]
            
            # Fit distributions and calculate goodness of fit
            results = []
            
            for dist_name, distribution in distributions:
                try:
                    # Handle special case for lognorm
                    if dist_name == "Log-normal":
                        # For lognormal, use only positive values
                        pos_data = data[data > 0]
                        if len(pos_data) < 10:
                            results.append({
                                "Distribution": dist_name,
                                "Parameters": "N/A",
                                "AIC": float('inf'),
                                "BIC": float('inf'),
                                "Error": "Not enough positive values for Log-normal"
                            })
                            continue
                            
                        # Fit using positive data
                        params = distribution.fit(pos_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(pos_data, *params))
                    elif dist_name == "Beta":
                        # For beta, scale data to [0, 1]
                        min_val = data.min()
                        max_val = data.max()
                        scaled_data = (data - min_val) / (max_val - min_val)
                        # Handle edge cases
                        scaled_data = np.clip(scaled_data, 0.001, 0.999)
                        # Fit using scaled data
                        params = distribution.fit(scaled_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(scaled_data, *params))
                    elif dist_name == "Weibull":
                        # For Weibull, use only positive values and shift
                        pos_data = data[data > 0]
                        if len(pos_data) < 10:
                            results.append({
                                "Distribution": dist_name,
                                "Parameters": "N/A",
                                "AIC": float('inf'),
                                "BIC": float('inf'),
                                "Error": "Not enough positive values for Weibull"
                            })
                            continue
                        # Fit using positive data
                        params = distribution.fit(pos_data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(pos_data, *params))
                    else:
                        # Fit distribution
                        params = distribution.fit(data)
                        # Calculate log-likelihood
                        loglik = np.sum(distribution.logpdf(data, *params))
                    
                    # Calculate AIC and BIC
                    k = len(params)
                    n = len(data)
                    aic = 2 * k - 2 * loglik
                    bic = k * np.log(n) - 2 * loglik
                    
                    results.append({
                        "Distribution": dist_name,
                        "Parameters": params,
                        "AIC": aic,
                        "BIC": bic,
                        "Error": None
                    })
                    
                except Exception as e:
                    results.append({
                        "Distribution": dist_name,
                        "Parameters": "N/A",
                        "AIC": float('inf'),
                        "BIC": float('inf'),
                        "Error": str(e)
                    })
            
            # Filter out errors
            valid_results = [r for r in results if r["Error"] is None]
            
            if not valid_results:
                st.error("Could not fit any of the selected distributions to the data.")
                return
            
            # Find best fit by AIC
            best_fit = min(valid_results, key=lambda x: x["AIC"])
            
            # Create visualization
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name="Data",
                    opacity=0.7,
                    histnorm="probability density",
                    nbinsx=30
                )
            )
            
            # Add fitted distributions
            x = np.linspace(data.min(), data.max(), 1000)
            
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
            
            for i, result in enumerate(valid_results):
                dist_name = result["Distribution"]
                params = result["Parameters"]
                dist_obj = [d[1] for d in distributions if d[0] == dist_name][0]
                
                # Calculate PDF
                try:
                    if dist_name == "Log-normal":
                        # Handle special case for lognormal
                        pdf = dist_obj.pdf(x, *params)
                    elif dist_name == "Beta":
                        # For beta, need to scale x to [0, 1]
                        min_val = data.min()
                        max_val = data.max()
                        scaled_x = (x - min_val) / (max_val - min_val)
                        # Handle edge cases
                        scaled_x = np.clip(scaled_x, 0.001, 0.999)
                        # Calculate PDF
                        pdf = dist_obj.pdf(scaled_x, *params) / (max_val - min_val)
                    elif dist_name == "Weibull":
                        # Handle special case for Weibull
                        pdf = dist_obj.pdf(x, *params)
                    else:
                        pdf = dist_obj.pdf(x, *params)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pdf,
                            mode="lines",
                            name=f"{dist_name} (AIC: {result['AIC']:.2f})",
                            line=dict(color=colors[i % len(colors)])
                        )
                    )
                except Exception as e:
                    st.warning(f"Could not plot {dist_name} distribution: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title=f"Distribution Fitting for {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Probability Density"
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display fitting results
            st.subheader("Goodness of Fit")
            
            # Create results table
            fit_results = pd.DataFrame([
                {
                    "Distribution": r["Distribution"],
                    "AIC": r["AIC"] if r["Error"] is None else "Error",
                    "BIC": r["BIC"] if r["Error"] is None else "Error",
                    "Status": "Success" if r["Error"] is None else f"Error: {r['Error']}"
                }
                for r in results
            ])
            
            # Sort by AIC (if available)
            fit_results = fit_results.sort_values(
                by="AIC", 
                key=lambda x: pd.to_numeric(x, errors='coerce'),
                ascending=True
            )
            
            st.dataframe(fit_results, use_container_width=True)
            
            # Display best fit
            st.subheader("Best Fit Distribution")
            st.info(f"The best fit distribution is **{best_fit['Distribution']}** with AIC = {best_fit['AIC']:.2f}.")
            
        except Exception as e:
            st.error(f"Error in distribution fitting: {str(e)}")
            return
    
    # Export options
    export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
    
    if 'fig' in locals():
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Chart",
                data=html_bytes,
                file_name=f"distribution_{analysis_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            # Export as image
            img_bytes = fig.to_image(format=export_format.lower())
            
            st.download_button(
                label=f"Download Chart",
                data=img_bytes,
                file_name=f"distribution_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True
            )