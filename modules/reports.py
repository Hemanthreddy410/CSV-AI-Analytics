import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import datetime
import markdown  # Optional, used for HTML export
from statsmodels.tsa.seasonal import seasonal_decompose  # For trend decomposition
import scipy.stats  # For statistical tests and distributions
from modules.data_exporter import DataExporter

def render_reports():
    """Render reports section"""
    st.header("Reports")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Create tabs for different report types
    report_tabs = st.tabs([
        "Summary Report",
        "Exploratory Analysis",
        "Custom Report",
        "Export Options"
    ])
    
    # Summary Report Tab
    with report_tabs[0]:
        render_summary_report()
    
    # Exploratory Analysis Tab
    with report_tabs[1]:
        render_exploratory_report()
    
    # Custom Report Tab
    with report_tabs[2]:
        render_custom_report()
    
    # Export Options Tab
    with report_tabs[3]:
        render_export_options()

def render_summary_report():
    """Render summary report tab"""
    st.subheader("Summary Report")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Report configuration
    st.write("Generate a comprehensive summary report of your dataset.")
    
    with st.expander("Report Configuration", expanded=True):
        # Report title
        report_title = st.text_input("Report Title:", "Data Summary Report", key="summary_report_title")
        
        # Include sections
        st.write("Select sections to include:")
        
        include_dataset_info = st.checkbox("Dataset Information", value=True, key="summary_include_dataset_info")
        include_column_summary = st.checkbox("Column Summary", value=True, key="summary_include_column_summary")
        include_numeric_summary = st.checkbox("Numeric Data Summary", value=True, key="summary_include_numeric_summary")
        include_categorical_summary = st.checkbox("Categorical Data Summary", value=True, key="summary_include_categorical_summary")
        include_missing_data = st.checkbox("Missing Data Analysis", value=True, key="summary_include_missing_data")
        include_charts = st.checkbox("Include Charts", value=True, key="summary_include_charts")
    
    # Generate report
    if st.button("Generate Summary Report", use_container_width=True, key="summary_generate_button"):
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Dataset Information
        if include_dataset_info:
            report += "## Dataset Information\n\n"
            
            # Basic stats
            report += f"* **Rows:** {len(st.session_state.df):,}\n"
            report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
            
            # File details if available
            if hasattr(st.session_state, 'file_details') and st.session_state.file_details is not None:
                report += f"* **Filename:** {st.session_state.file_details['filename']}\n"
                
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
                    
                    report += f"* **Size:** {size_str}\n"
            
            # Data types summary
            dtypes = st.session_state.df.dtypes.value_counts().to_dict()
            report += "\n### Data Types\n\n"
            for dtype, count in dtypes.items():
                report += f"* **{dtype}:** {count} columns\n"
            
            report += "\n"
        
        # Column Summary
        if include_column_summary:
            report += "## Column Summary\n\n"
            
            # Create a dataframe with column information
            columns_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes.astype(str).values,
                'Non-Null Count': st.session_state.df.count().values,
                'Null Count': st.session_state.df.isna().sum().values,
                'Null %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).astype(str) + '%',
                'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
            })
            
            # Convert to markdown table
            report += columns_df.to_markdown(index=False)
            report += "\n\n"
        
        # Numeric Data Summary
        if include_numeric_summary:
            report += "## Numeric Data Summary\n\n"
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if num_cols:
                # Calculate statistics
                stats_df = st.session_state.df[num_cols].describe().T.round(3)
                
                # Add more statistics
                stats_df['median'] = st.session_state.df[num_cols].median().round(3)
                stats_df['missing'] = st.session_state.df[num_cols].isna().sum()
                stats_df['missing_pct'] = (st.session_state.df[num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
                
                try:
                    stats_df['skew'] = st.session_state.df[num_cols].skew().round(3)
                    stats_df['kurtosis'] = st.session_state.df[num_cols].kurtosis().round(3)
                except:
                    pass
                
                # Convert to markdown table
                report += stats_df.reset_index().rename(columns={'index': 'Column'}).to_markdown(index=False)
                report += "\n\n"
                
                # Include charts for numeric columns
                if include_charts:
                    report += "### Numeric Data Distributions\n\n"
                    report += "See attached visualizations for distributions of numeric columns.\n\n"
            else:
                report += "*No numeric columns found in the dataset.*\n\n"
        
        # Categorical Data Summary
        if include_categorical_summary:
            report += "## Categorical Data Summary\n\n"
            
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                # For each categorical column, show top values
                for col in cat_cols:
                    value_counts = st.session_state.df[col].value_counts().nlargest(10)
                    report += f"### {col}\n\n"
                    report += f"* **Unique Values:** {st.session_state.df[col].nunique()}\n"
                    report += f"* **Missing Values:** {st.session_state.df[col].isna().sum()} ({st.session_state.df[col].isna().sum() / len(st.session_state.df) * 100:.2f}%)\n"
                    report += f"* **Most Common Values:**\n\n"
                    
                    # Create value counts dataframe
                    vc_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(st.session_state.df) * 100).round(2)
                    })
                    
                    # Convert to markdown table
                    report += vc_df.to_markdown(index=False)
                    report += "\n\n"
            else:
                report += "*No categorical columns found in the dataset.*\n\n"
        
        # Missing Data Analysis
        if include_missing_data:
            report += "## Missing Data Analysis\n\n"
            
            # Calculate missing values
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
                report += "\n\n"
                
                if include_charts:
                    report += "### Missing Data Visualization\n\n"
                    report += "See attached visualizations for patterns of missing data.\n\n"
            else:
                report += "*No missing values found in the dataset.*\n\n"
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate charts if requested
        if include_charts:
            st.subheader("Report Charts")
            
            # Create a list to store chart images
            charts = []
            
            # Numeric column distributions
            if include_numeric_summary:
                num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                
                if num_cols:
                    # Select a subset of numeric columns for visualization (max 5)
                    selected_num_cols = num_cols[:min(5, len(num_cols))]
                    
                    # Create histogram for each selected column
                    for col in selected_num_cols:
                        try:
                            fig = px.histogram(
                                st.session_state.df,
                                x=col,
                                title=f"Distribution of {col}",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create histogram for column '{col}': {str(e)}")
            
            # Missing data visualization
            if include_missing_data:
                missing_df = pd.DataFrame({
                    'Column': st.session_state.df.columns,
                    'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
                })
                
                # Sort by missing percentage (descending)
                missing_df = missing_df.sort_values('Missing %', ascending=False)
                
                # Get columns with missing values
                cols_with_missing = missing_df[missing_df['Missing %'] > 0]
                
                if len(cols_with_missing) > 0:
                    # Create missing values chart
                    try:
                        fig = px.bar(
                            cols_with_missing,
                            x='Column',
                            y='Missing %',
                            title='Missing Values by Column (%)',
                            color='Missing %',
                            color_continuous_scale="Reds"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create missing values chart: {str(e)}")
        
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
                use_container_width=True,
                key="summary_download_md"
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
                    use_container_width=True,
                    key="summary_download_html"
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_custom_report():
    """Render custom report tab"""
    st.subheader("Custom Report")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Report configuration
    st.write("Build a custom report with selected sections and visualizations.")
    
    with st.expander("Report Settings", expanded=True):
        # Report title and description
        report_title = st.text_input("Report Title:", "Custom Data Analysis Report", key="custom_report_title")
        report_description = st.text_area("Report Description:", "This report provides a custom analysis of the dataset.", key="custom_report_description")
        
        # Author information
        include_author = st.checkbox("Include Author Information", value=False, key="custom_include_author")
        if include_author:
            author_name = st.text_input("Author Name:", key="custom_author_name")
            author_info = st.text_input("Additional Info (e.g., email, organization):", key="custom_author_info")
    
    # Section selection
    st.subheader("Report Sections")
    
    # Create a list to store selected sections
    selected_sections = []
    
    # Dataset Overview section
    include_overview = st.checkbox("Dataset Overview", value=True, key="custom_include_overview")
    if include_overview:
        selected_sections.append("overview")
        with st.container():
            st.markdown("##### Dataset Overview Options")
            include_basic_stats = st.checkbox("Basic Statistics", value=True, key="custom_include_basic_stats")
            include_data_types = st.checkbox("Data Types Summary", value=True, key="custom_include_data_types")
            include_sample_data = st.checkbox("Sample Data Preview", value=True, key="custom_include_sample_data")
            
            # Store options
            overview_options = {
                "basic_stats": include_basic_stats,
                "data_types": include_data_types,
                "sample_data": include_sample_data
            }
    else:
        overview_options = {}
    
    # Numeric Data Analysis section
    include_numeric = st.checkbox("Numeric Data Analysis", value=True, key="custom_include_numeric")
    if include_numeric:
        selected_sections.append("numeric")
        with st.container():
            st.markdown("##### Numeric Analysis Options")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                # Select columns to include
                selected_num_cols = st.multiselect(
                    "Select columns to analyze:",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))],
                    key="custom_selected_num_cols"
                )
                
                # Chart types
                include_histograms = st.checkbox("Include Histograms", value=True, key="custom_include_histograms")
                include_boxplots = st.checkbox("Include Box Plots", value=True, key="custom_include_boxplots")
                include_descriptive = st.checkbox("Include Descriptive Statistics", value=True, key="custom_include_descriptive")
                
                # Store options
                numeric_options = {
                    "columns": selected_num_cols,
                    "histograms": include_histograms,
                    "boxplots": include_boxplots,
                    "descriptive": include_descriptive
                }
    else:
        numeric_options = {}
    
    # Categorical Data Analysis section
    include_categorical = st.checkbox("Categorical Data Analysis", value=True, key="custom_include_categorical")
    if include_categorical:
        selected_sections.append("categorical")
        with st.container():
            st.markdown("##### Categorical Analysis Options")
            
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found in the dataset.")
            else:
                # Select columns to include
                selected_cat_cols = st.multiselect(
                    "Select columns to analyze:",
                    cat_cols,
                    default=cat_cols[:min(5, len(cat_cols))],
                    key="custom_selected_cat_cols"
                )
                
                # Chart types
                include_bar_charts = st.checkbox("Include Bar Charts", value=True, key="custom_include_bar_charts")
                include_pie_charts = st.checkbox("Include Pie Charts", value=False, key="custom_include_pie_charts")
                include_frequency = st.checkbox("Include Frequency Tables", value=True, key="custom_include_frequency")
                
                # Maximum categories to display
                max_categories = st.slider("Maximum categories per chart:", 5, 30, 10, key="custom_max_categories")
                
                # Store options
                categorical_options = {
                    "columns": selected_cat_cols,
                    "bar_charts": include_bar_charts,
                    "pie_charts": include_pie_charts,
                    "frequency": include_frequency,
                    "max_categories": max_categories
                }
    else:
        categorical_options = {}
    
    # Correlation Analysis section
    include_correlation = st.checkbox("Correlation Analysis", value=True, key="custom_include_correlation")
    if include_correlation:
        selected_sections.append("correlation")
        with st.container():
            st.markdown("##### Correlation Analysis Options")
            
            # Get numeric columns
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for correlation analysis.")
            else:
                # Select columns to include
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    num_cols,
                    default=num_cols[:min(7, len(num_cols))],
                    key="custom_selected_corr_cols"
                )
                
                # Correlation method
                corr_method = st.selectbox(
                    "Correlation method:",
                    ["Pearson", "Spearman", "Kendall"],
                    key="custom_corr_method"
                )
                
                # Visualization options
                include_heatmap = st.checkbox("Include Correlation Heatmap", value=True, key="custom_include_heatmap")
                include_corr_table = st.checkbox("Include Correlation Table", value=True, key="custom_include_corr_table")
                
                # Correlation threshold
                corr_threshold = st.slider("Highlight correlations with absolute value above:", 0.0, 1.0, 0.7, key="custom_corr_threshold")
                
                # Store options
                correlation_options = {
                    "columns": selected_corr_cols,
                    "method": corr_method,
                    "heatmap": include_heatmap,
                    "table": include_corr_table,
                    "threshold": corr_threshold
                }
    else:
        correlation_options = {}
    
    # Missing Data Analysis section
    include_missing = st.checkbox("Missing Data Analysis", value=True, key="custom_include_missing")
    if include_missing:
        selected_sections.append("missing")
        with st.container():
            st.markdown("##### Missing Data Analysis Options")
            
            # Visualization options
            include_missing_bar = st.checkbox("Include Missing Values Bar Chart", value=True, key="custom_include_missing_bar")
            include_missing_heatmap = st.checkbox("Include Missing Values Heatmap", value=False, key="custom_include_missing_heatmap")
            include_missing_table = st.checkbox("Include Missing Values Table", value=True, key="custom_include_missing_table")
            
            # Store options
            missing_options = {
                "bar_chart": include_missing_bar,
                "heatmap": include_missing_heatmap,
                "table": include_missing_table
            }
    else:
        missing_options = {}
    
    # Additional Custom Section
    include_custom = st.checkbox("Custom Text Section", value=False, key="custom_include_custom_section")
    if include_custom:
        selected_sections.append("custom")
        with st.container():
            st.markdown("##### Custom Section Content")
            
            custom_section_title = st.text_input("Section Title:", "Additional Insights", key="custom_section_title")
            custom_section_content = st.text_area("Section Content:", "Enter your custom analysis and insights here.", key="custom_section_content")
            
            # Store options
            custom_options = {
                "title": custom_section_title,
                "content": custom_section_content
            }
    else:
        custom_options = {}
    
    # Generate report
    if st.button("Generate Custom Report", use_container_width=True, key="custom_generate_button"):
        # Create a report markdown string
        report = f"# {report_title}\n\n"
        
        # Add description
        report += f"{report_description}\n\n"
        
        # Add date generated
        report += f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add author information if included
        if include_author and author_name:
            report += f"**Author:** {author_name}\n"
            if author_info:
                report += f"**Contact:** {author_info}\n"
            report += "\n"
        
        # Generate each selected section
        section_num = 1
        
        # Dataset Overview section
        if "overview" in selected_sections:
            report += f"## {section_num}. Dataset Overview\n\n"
            section_num += 1
            
            if overview_options.get("basic_stats", False):
                # Basic statistics
                report += "### Basic Statistics\n\n"
                report += f"* **Rows:** {len(st.session_state.df):,}\n"
                report += f"* **Columns:** {len(st.session_state.df.columns)}\n"
                
                # File details if available
                if hasattr(st.session_state, 'file_details') and st.session_state.file_details is not None:
                    report += f"* **Filename:** {st.session_state.file_details['filename']}\n"
                
                # Missing values
                missing_values = st.session_state.df.isna().sum().sum()
                missing_pct = (missing_values / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100).round(2)
                report += f"* **Missing Values:** {missing_values:,} ({missing_pct}% of all values)\n\n"
            
            if overview_options.get("data_types", False):
                # Data types summary
                report += "### Data Types\n\n"
                
                # Count by type
                type_counts = st.session_state.df.dtypes.astype(str).value_counts().to_dict()
                for dtype, count in type_counts.items():
                    report += f"* **{dtype}:** {count} columns\n"
                
                report += "\n"
            
            if overview_options.get("sample_data", False):
                # Sample data preview
                report += "### Data Preview\n\n"
                
                # Convert first 5 rows to markdown table
                sample_df = st.session_state.df.head(5)
                report += sample_df.to_markdown(index=False)
                report += "\n\n"
        
        # Numeric Data Analysis section
        if "numeric" in selected_sections and numeric_options.get("columns"):
            report += f"## {section_num}. Numeric Data Analysis\n\n"
            section_num += 1
            
            selected_num_cols = numeric_options.get("columns")
            
            if numeric_options.get("descriptive", False):
                # Descriptive statistics
                report += "### Descriptive Statistics\n\n"
                
                # Calculate statistics
                stats_df = st.session_state.df[selected_num_cols].describe().T.round(3)
                
                # Add additional statistics
                stats_df['median'] = st.session_state.df[selected_num_cols].median().round(3)
                stats_df['missing'] = st.session_state.df[selected_num_cols].isna().sum()
                stats_df['missing_pct'] = (st.session_state.df[selected_num_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
                
                try:
                    stats_df['skew'] = st.session_state.df[selected_num_cols].skew().round(3)
                    stats_df['kurtosis'] = st.session_state.df[selected_num_cols].kurtosis().round(3)
                except:
                    pass
                
                # Convert to markdown table
                report += stats_df.reset_index().rename(columns={'index': 'Column'}).to_markdown(index=False)
                report += "\n\n"
            
            # Add note about visualizations
            if numeric_options.get("histograms", False) or numeric_options.get("boxplots", False):
                report += "### Visualizations\n\n"
                report += "*The report includes the following visualizations for numeric columns:*\n\n"
                
                if numeric_options.get("histograms", False):
                    report += "* **Histograms** showing the distribution of values for each numeric column\n"
                
                if numeric_options.get("boxplots", False):
                    report += "* **Box plots** showing the central tendency and spread of each numeric column\n"
                
                report += "\n*These visualizations will be included in the exported report.*\n\n"
        
        # Categorical Data Analysis section
        if "categorical" in selected_sections and categorical_options.get("columns"):
            report += f"## {section_num}. Categorical Data Analysis\n\n"
            section_num += 1
            
            selected_cat_cols = categorical_options.get("columns")
            max_categories = categorical_options.get("max_categories", 10)
            
            if categorical_options.get("frequency", False):
                report += "### Frequency Analysis\n\n"
                
                # For each categorical column, show frequency table
                for col in selected_cat_cols:
                    report += f"#### {col}\n\n"
                    report += f"* **Unique Values:** {st.session_state.df[col].nunique()}\n"
                    report += f"* **Missing Values:** {st.session_state.df[col].isna().sum()} ({st.session_state.df[col].isna().sum() / len(st.session_state.df) * 100:.2f}%)\n\n"
                    
                    # Get top categories
                    value_counts = st.session_state.df[col].value_counts().nlargest(max_categories)
                    total_values = len(st.session_state.df)
                    
                    # Create frequency table
                    freq_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / total_values * 100).round(2)
                    })
                    
                    # Check if there are more categories not shown
                    if st.session_state.df[col].nunique() > max_categories:
                        report += f"*Top {max_categories} categories shown (out of {st.session_state.df[col].nunique()} total)*\n\n"
                    
                    # Convert to markdown table
                    report += freq_df.to_markdown(index=False)
                    report += "\n\n"
            
            # Add note about visualizations
            if categorical_options.get("bar_charts", False) or categorical_options.get("pie_charts", False):
                report += "### Visualizations\n\n"
                report += "*The report includes the following visualizations for categorical columns:*\n\n"
                
                if categorical_options.get("bar_charts", False):
                    report += "* **Bar charts** showing the frequency of each category\n"
                
                if categorical_options.get("pie_charts", False):
                    report += "* **Pie charts** showing the proportion of each category\n"
                
                report += "\n*These visualizations will be included in the exported report.*\n\n"
        
        # Correlation Analysis section
        if "correlation" in selected_sections and correlation_options.get("columns") and len(correlation_options.get("columns", [])) >= 2:
            report += f"## {section_num}. Correlation Analysis\n\n"
            section_num += 1
            
            selected_corr_cols = correlation_options.get("columns")
            corr_method = correlation_options.get("method", "Pearson")
            corr_threshold = correlation_options.get("threshold", 0.7)
            
            # Calculate correlation matrix
            corr_matrix = st.session_state.df[selected_corr_cols].corr(method=corr_method.lower()).round(3)
            
            if correlation_options.get("table", False):
                report += f"### {corr_method} Correlation Matrix\n\n"
                
                # Convert to markdown table
                report += corr_matrix.to_markdown()
                report += "\n\n"
            
            # Strong correlations
            report += "### Strong Correlations\n\n"
            
            # Find pairs with strong correlation
            strong_pairs = []
            for i, col1 in enumerate(selected_corr_cols):
                for j, col2 in enumerate(selected_corr_cols):
                    if i < j:  # Only include each pair once
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) >= corr_threshold:
                            strong_pairs.append({
                                'Column 1': col1,
                                'Column 2': col2,
                                'Correlation': corr_val
                            })
            
            if strong_pairs:
                # Convert to dataframe and sort
                strong_df = pd.DataFrame(strong_pairs)
                strong_df = strong_df.sort_values('Correlation', key=abs, ascending=False)
                
                report += f"*Pairs with absolute correlation ≥ {corr_threshold}:*\n\n"
                report += strong_df.to_markdown(index=False)
                report += "\n\n"
            else:
                report += f"*No pairs with absolute correlation ≥ {corr_threshold} found.*\n\n"
            
            # Add note about visualization
            if correlation_options.get("heatmap", False):
                report += "### Correlation Heatmap\n\n"
                report += "*A correlation heatmap will be included in the exported report.*\n\n"
        
        # Missing Data Analysis section
        if "missing" in selected_sections:
            report += f"## {section_num}. Missing Data Analysis\n\n"
            section_num += 1
            
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
                
                if missing_options.get("table", False):
                    # Convert to markdown table
                    report += cols_with_missing.to_markdown(index=False)
                    report += "\n\n"
                
                # Add note about visualization
                if missing_options.get("bar_chart", False) or missing_options.get("heatmap", False):
                    report += "### Missing Value Visualizations\n\n"
                    report += "*The report includes the following visualizations for missing data:*\n\n"
                    
                    if missing_options.get("bar_chart", False):
                        report += "* **Bar chart** showing the percentage of missing values in each column\n"
                    
                    if missing_options.get("heatmap", False):
                        report += "* **Heatmap** showing patterns of missing values across the dataset\n"
                    
                    report += "\n*These visualizations will be included in the exported report.*\n\n"
            else:
                report += "*No missing values found in the dataset.*\n\n"
        
        # Custom text section
        if "custom" in selected_sections:
            report += f"## {section_num}. {custom_options.get('title', 'Additional Insights')}\n\n"
            section_num += 1
            
            report += custom_options.get('content', '')
            report += "\n\n"
        
        # Display the report in a scrollable area
        st.markdown("### Preview Report")
        st.markdown(report)
        
        # Generate visualizations mentioned in the report
        if (("numeric" in selected_sections and (numeric_options.get("histograms", False) or numeric_options.get("boxplots", False))) or
            ("categorical" in selected_sections and (categorical_options.get("bar_charts", False) or categorical_options.get("pie_charts", False))) or
            ("correlation" in selected_sections and correlation_options.get("heatmap", False)) or
            ("missing" in selected_sections and (missing_options.get("bar_chart", False) or missing_options.get("heatmap", False)))):
            
            st.subheader("Report Visualizations")
            
            # Numeric visualizations
            if "numeric" in selected_sections:
                selected_num_cols = numeric_options.get("columns", [])
                
                if numeric_options.get("histograms", False) and selected_num_cols:
                    st.markdown("#### Histograms")
                    # Show one example histogram
                    example_col = selected_num_cols[0]
                    try:
                        fig = px.histogram(
                            st.session_state.df,
                            x=example_col,
                            title=f"Distribution of {example_col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(selected_num_cols) > 1:
                            st.info(f"Histograms for all {len(selected_num_cols)} selected numeric columns will be included in the exported report.")
                    except Exception as e:
                        st.warning(f"Could not create histogram for column '{example_col}': {str(e)}")
                
                if numeric_options.get("boxplots", False) and selected_num_cols:
                    st.markdown("#### Box Plots")
                    # Create box plots
                    try:
                        fig = go.Figure()
                        
                        for col in selected_num_cols[:3]:  # Show up to 3 columns
                            fig.add_trace(
                                go.Box(
                                    y=st.session_state.df[col],
                                    name=col
                                )
                            )
                        
                        fig.update_layout(title="Box Plots of Numeric Columns")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(selected_num_cols) > 3:
                            st.info(f"Box plots for all {len(selected_num_cols)} selected numeric columns will be included in the exported report.")
                    except Exception as e:
                        st.warning(f"Could not create box plots: {str(e)}")
            
            # Categorical visualizations
            if "categorical" in selected_sections:
                selected_cat_cols = categorical_options.get("columns", [])
                max_categories = categorical_options.get("max_categories", 10)
                
                if categorical_options.get("bar_charts", False) and selected_cat_cols:
                    st.markdown("#### Bar Charts")
                    # Show one example bar chart
                    example_col = selected_cat_cols[0]
                    
                    try:
                        # Get top categories
                        value_counts = st.session_state.df[example_col].value_counts().nlargest(max_categories)
                        
                        # Create bar chart
                        fig = px.bar(
                            value_counts.reset_index(),
                            x="index",
                            y=example_col,
                            title=f"Frequency of {example_col} Categories",
                            labels={"index": example_col, example_col: "Count"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(selected_cat_cols) > 1:
                            st.info(f"Bar charts for all {len(selected_cat_cols)} selected categorical columns will be included in the exported report.")
                    except Exception as e:
                        st.warning(f"Could not create bar chart for column '{example_col}': {str(e)}")
                
                if categorical_options.get("pie_charts", False) and selected_cat_cols:
                    st.markdown("#### Pie Charts")
                    # Show one example pie chart
                    example_col = selected_cat_cols[0]
                    
                    try:
                        # Get top categories
                        value_counts = st.session_state.df[example_col].value_counts().nlargest(max_categories)
                        
                        # Create pie chart
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"Distribution of {example_col} Categories"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(selected_cat_cols) > 1:
                            st.info(f"Pie charts for all {len(selected_cat_cols)} selected categorical columns will be included in the exported report.")
                    except Exception as e:
                        st.warning(f"Could not create pie chart for column '{example_col}': {str(e)}")
            
            # Correlation heatmap
            if "correlation" in selected_sections and correlation_options.get("heatmap", False):
                selected_corr_cols = correlation_options.get("columns", [])
                
                if len(selected_corr_cols) >= 2:
                    st.markdown("#### Correlation Heatmap")
                    
                    try:
                        # Calculate correlation matrix
                        corr_method = correlation_options.get("method", "Pearson")
                        corr_matrix = st.session_state.df[selected_corr_cols].corr(method=corr_method.lower()).round(3)
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            title=f"{corr_method} Correlation Heatmap",
                            labels=dict(color="Correlation")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create correlation heatmap: {str(e)}")
            
            # Missing data visualizations
            if "missing" in selected_sections:
                # Get columns with missing values
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
                    if missing_options.get("bar_chart", False):
                        st.markdown("#### Missing Values Bar Chart")
                        
                        try:
                            # Create bar chart
                            fig = px.bar(
                                cols_with_missing,
                                x='Column',
                                y='Missing %',
                                title='Missing Values by Column (%)',
                                color='Missing %',
                                color_continuous_scale="Reds"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create missing values bar chart: {str(e)}")
                    
                    if missing_options.get("heatmap", False):
                        st.markdown("#### Missing Values Heatmap")
                        
                        # Limit to columns with missing values
                        cols_to_plot = cols_with_missing['Column'].tolist()
                        
                        if cols_to_plot:
                            try:
                                # Create sample for heatmap (limit to 100 rows for performance)
                                sample_size = min(100, len(st.session_state.df))
                                sample_df = st.session_state.df[cols_to_plot].sample(sample_size) if len(st.session_state.df) > sample_size else st.session_state.df[cols_to_plot]
                                
                                # Create heatmap
                                fig = px.imshow(
                                    sample_df.isna(),
                                    labels=dict(x="Column", y="Row", color="Missing"),
                                    color_continuous_scale=["blue", "red"],
                                    title=f"Missing Values Heatmap (Sample of {sample_size} rows)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create missing values heatmap: {str(e)}")
        
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
                use_container_width=True,
                key="custom_download_md"
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
                    use_container_width=True,
                    key="custom_download_html"
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_exploratory_report():
    """Render exploratory analysis report tab"""
    st.subheader("Exploratory Data Analysis Report")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Report configuration
    st.write("Generate an exploratory data analysis (EDA) report with visualizations and insights.")
    
    with st.expander("Report Configuration", expanded=True):
        # Report title and description
        report_title = st.text_input("Report Title:", "Exploratory Data Analysis Report", key="eda_report_title")
        report_description = st.text_area("Report Description:", "This report provides an exploratory analysis of the dataset.", key="eda_report_description")
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Comprehensive"],
            value="Standard",
            key="eda_analysis_depth"
        )
        
        # Visualization settings
        st.write("Visualization Settings:")
        max_categorical_values = st.slider("Max categorical values to display:", 5, 30, 10, key="eda_max_categorical_values")
        correlation_threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.5, key="eda_correlation_threshold")
    
    # Generate report
    if st.button("Generate EDA Report", use_container_width=True, key="eda_generate_button"):
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
        total_charts = len(analyzed_cols) if len(analyzed_cols) > 0 else 1
        charts_done = 0
        
        # Analysis for each column
        for col in analyzed_cols:
            # Make sure column exists in dataframe
            if col not in st.session_state.df.columns:
                continue
                
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
                try:
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
                        except Exception as e:
                            report += f"* *Note: Could not calculate advanced statistics: {str(e)}*\n"
                except Exception as e:
                    report += f"* *Error calculating numeric statistics: {str(e)}*\n"
                
                report += f"\n**Histogram will be included in the final report.**\n\n"
                
            elif col_type == "Categorical":
                # Categorical column analysis
                unique_count = st.session_state.df[col].nunique()
                report += f"* **Unique Values:** {unique_count}\n"
                
                # Show top categories
                try:
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
                except Exception as e:
                    report += f"*Error calculating category frequencies: {str(e)}*\n"
                
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
                            except Exception as e:
                                report += f"* *Error calculating day of week: {str(e)}*\n"
                                
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
            try:
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
            except Exception as e:
                report += f"*Error calculating correlations: {str(e)}*\n\n"
        
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
                try:
                    report += "### Missing Value Patterns\n\n"
                    report += "*Missing values heatmap will be included in the final report.*\n\n"
                    
                    # Calculate correlations between missing values
                    # This indicates if missingness in one column correlates with missingness in another
                    missing_pattern = st.session_state.df[cols_with_missing['Column']].isna()
                    missing_corr = missing_pattern.corr().round(3)
                    
                    report += "**Missing Value Correlation Matrix** (correlation between missing patterns)\n\n"
                    report += missing_corr.to_markdown()
                    report += "\n\n*High correlation suggests missing values occur together in the same rows.*\n\n"
                except Exception as e:
                    report += f"*Error analyzing missing value patterns: {str(e)}*\n\n"
        else:
            report += "*No missing values found in the dataset.*\n\n"
        
        progress_bar.progress(70)
        
        # Outlier Analysis (for numeric columns)
        if len(analyzed_num_cols) > 0:
            report += "## 5. Outlier Analysis\n\n"
            
            outlier_summary = []
            
            for col in analyzed_num_cols:
                try:
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
                except Exception as e:
                    outlier_summary.append({
                        'Column': col,
                        'Q1': None,
                        'Q3': None,
                        'IQR': None,
                        'Lower Bound': None,
                        'Upper Bound': None,
                        'Outlier Count': None,
                        'Outlier %': None,
                        'Error': str(e)
                    })
            
            # Convert to dataframe
            outlier_df = pd.DataFrame(outlier_summary)
            
            # Filter out errors
            valid_outlier_df = outlier_df[outlier_df['Q1'].notna()]
            
            if len(valid_outlier_df) > 0:
                # Sort by outlier count (descending)
                valid_outlier_df = valid_outlier_df.sort_values('Outlier Count', ascending=False)
                
                # For standard and comprehensive analysis, include detailed outlier information
                if analysis_depth in ["Standard", "Comprehensive"]:
                    report += "### Outlier Summary\n\n"
                    
                    # Convert to markdown table
                    summary_cols = ['Column', 'Outlier Count', 'Outlier %', 'Lower Bound', 'Upper Bound']
                    report += valid_outlier_df[summary_cols].to_markdown(index=False)
                    report += "\n\n"
                    
                    report += "*Box plots for columns with outliers will be included in the final report.*\n\n"
                    
                    # For comprehensive analysis, include detailed statistics
                    if analysis_depth == "Comprehensive":
                        report += "### Detailed Outlier Statistics\n\n"
                        report += valid_outlier_df[['Column', 'Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count', 'Outlier %']].to_markdown(index=False)
                        report += "\n\n"
                else:
                    # For basic analysis, just include a summary
                    report += f"**Total columns with outliers:** {len(valid_outlier_df[valid_outlier_df['Outlier Count'] > 0])}\n\n"
                    
                    # Show top columns with outliers
                    top_outliers = valid_outlier_df[valid_outlier_df['Outlier Count'] > 0].head(5)
                    
                    if len(top_outliers) > 0:
                        report += "**Top columns with outliers:**\n\n"
                        report += top_outliers[['Column', 'Outlier Count', 'Outlier %']].to_markdown(index=False)
                        report += "\n\n"
                    else:
                        report += "*No outliers found in the dataset using the IQR method.*\n\n"
            else:
                report += "*Error calculating outliers for numeric columns.*\n\n"
        
        progress_bar.progress(80)
        
        # Distribution Analysis (for categorical columns)
        if len(analyzed_cat_cols) > 0:
            report += "## 6. Categorical Data Analysis\n\n"
            
            # For each categorical column, analyze distribution
            for i, col in enumerate(analyzed_cat_cols[:5]):  # Limit to top 5 columns
                try:
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
                except Exception as e:
                    report += f"*Error analyzing categorical column {col}: {str(e)}*\n\n"
            
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
        if len(analyzed_num_cols) > 0 and 'valid_outlier_df' in locals():
            try:
                outlier_cols = valid_outlier_df[valid_outlier_df['Outlier %'] > 5]
                if len(outlier_cols) > 0:
                    report += f"* **Outliers:** {len(outlier_cols)} columns have more than 5% outliers.\n"
                    
                    # Top columns with outliers
                    top_outlier_cols = outlier_cols.head(3)
                    outlier_col_list = ", ".join([f"'{col}' ({pct}%)" for col, pct in zip(top_outlier_cols['Column'], top_outlier_cols['Outlier %'])])
                    report += f"  * Columns with most outliers: {outlier_col_list}\n"
            except Exception as e:
                report += f"* **Outliers:** Error summarizing outliers: {str(e)}\n"
        
        # Correlations (if analyzed)
        if len(analyzed_num_cols) >= 2 and 'pairs_df' in locals():
            try:
                strong_corrs = pairs_df[pairs_df['Correlation'].abs() >= 0.7]
                if len(strong_corrs) > 0:
                    report += f"* **Strong Correlations:** {len(strong_corrs)} pairs of numeric columns have strong correlations (|r| ≥ 0.7).\n"
                    
                    # Top correlations
                    top_corr = strong_corrs.head(2)
                    if len(top_corr) > 0:
                        corr_list = ", ".join([f"'{col1}' and '{col2}' (r={corr:.2f})" for col1, col2, corr in zip(
                            top_corr['Column 1'], top_corr['Column 2'], top_corr['Correlation'])])
                        report += f"  * Strongest correlations: {corr_list}\n"
            except Exception as e:
                report += f"* **Correlations:** Error summarizing correlations: {str(e)}\n"
        
        # Recommendations
        report += "\n### Recommendations\n\n"
        
        # Missing data recommendations
        if missing_values > 0:
            report += "* **Missing Data Handling:**\n"
            report += "  * Consider imputation strategies for columns with missing values, such as mean/median imputation for numeric columns or mode imputation for categorical columns.\n"
            report += "  * For columns with high missing percentages, evaluate whether they should be kept or dropped.\n"
            report += "  * Investigate if missing data follows a pattern that could introduce bias.\n"
        
        # Outlier recommendations
        if len(analyzed_num_cols) > 0 and 'valid_outlier_df' in locals() and len(valid_outlier_df[valid_outlier_df['Outlier Count'] > 0]) > 0:
            report += "* **Outlier Treatment:**\n"
            report += "  * Inspect outliers to determine if they are valid data points or errors.\n"
            report += "  * Consider appropriate treatment such as capping, transforming, or removing outliers depending on your analysis goals.\n"
        
        # Correlation recommendations
        if len(analyzed_num_cols) >= 2:
            report += "* **Feature Selection/Engineering:**\n"
            if 'strong_corrs' in locals() and len(strong_corrs) > 0:
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
            try:
                fig = px.histogram(
                    st.session_state.df,
                    x=col,
                    title=f"Distribution of {col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                charts.append(fig)
            except Exception as e:
                st.warning(f"Could not create histogram for column '{col}': {str(e)}")
        
        # Categorical column bar charts
        for col in analyzed_cat_cols[:3]:  # Limit to first 3 columns
            try:
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
            except Exception as e:
                st.warning(f"Could not create bar chart for column '{col}': {str(e)}")
        
        # Correlation heatmap
        if len(analyzed_num_cols) >= 2:
            try:
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
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {str(e)}")
        
        # Missing values bar chart
        if len(cols_with_missing) > 0:
            try:
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
            except Exception as e:
                st.warning(f"Could not create missing values chart: {str(e)}")
        
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
                use_container_width=True,
                key="eda_download_md"
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
                    use_container_width=True,
                    key="eda_download_html"
                )
            except:
                st.warning("HTML export requires the markdown package. Try downloading as Markdown instead.")

def render_export_options():
    """Render export options tab"""
    st.subheader("Export Options")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Data")
        
        # Create exporter instance
        try:
            exporter = DataExporter(st.session_state.df)
            
            # Render export options
            exporter.render_export_options()
        except Exception as e:
            st.error(f"Error initializing data exporter: {str(e)}")
            
            # Provide fallback export options
            st.write("Basic export options:")
            
            export_format = st.selectbox(
                "Export format:",
                ["CSV", "Excel", "JSON"],
                key="basic_export_format"
            )
            
            if st.button("Export Data", key="basic_export_button"):
                try:
                    if export_format == "CSV":
                        csv_data = st.session_state.df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            data=csv_data,
                            file_name="exported_data.csv",
                            mime="text/csv",
                            key="download_csv"
                        )
                    elif export_format == "Excel":
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            st.session_state.df.to_excel(writer, index=False)
                        excel_data = output.getvalue()
                        st.download_button(
                            "Download Excel",
                            data=excel_data,
                            file_name="exported_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                    elif export_format == "JSON":
                        json_data = st.session_state.df.to_json(orient="records")
                        st.download_button(
                            "Download JSON",
                            data=json_data,
                            file_name="exported_data.json",
                            mime="application/json",
                            key="download_json"
                        )
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
    
    with col2:
        st.markdown("### Export Charts")
        
        # Select chart type to export
        chart_type = st.selectbox(
            "Select chart type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"],
            key="export_chart_type"
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        try:
            # Configure chart based on type
            if chart_type == "Bar Chart":
                # Column selections
                x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="export_bar_x_col")
                y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_bar_y_col")
                
                # Create chart
                fig = px.bar(
                    st.session_state.df,
                    x=x_col,
                    y=y_col,
                    title=f"Bar Chart: {x_col} vs {y_col}"
                )
            
            elif chart_type == "Line Chart":
                # Column selections
                x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist(), key="export_line_x_col")
                y_col = st.selectbox("Y-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_line_y_col")
                
                # Create chart
                fig = px.line(
                    st.session_state.df,
                    x=x_col,
                    y=y_col,
                    title=f"Line Chart: {x_col} vs {y_col}"
                )
            
            elif chart_type == "Scatter Plot":
                # Column selections
                x_col = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_scatter_x_col")
                y_col = st.selectbox("Y-axis:", [col for col in num_cols if col != x_col] if len(num_cols) > 1 else num_cols, key="export_scatter_y_col")
                
                # Create chart
                fig = px.scatter(
                    st.session_state.df,
                    x=x_col,
                    y=y_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}"
                )
            
            elif chart_type == "Pie Chart":
                # Column selections
                value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_pie_value_col")
                name_col = st.selectbox("Name column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="export_pie_name_col")
                
                # Limit categories for better visualization
                top_n = st.slider("Top N categories:", 3, 20, 10, key="export_pie_top_n")
                
                # Calculate values for pie chart
                value_counts = st.session_state.df.groupby(name_col)[value_col].sum().nlargest(top_n)
                
                # Create chart
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Pie Chart: {value_col} by {name_col}"
                )
            
            elif chart_type == "Histogram":
                # Column selection
                value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_hist_value_col")
                
                # Number of bins
                bins = st.slider("Number of bins:", 5, 100, 20, key="export_hist_bins")
                
                # Create chart
                fig = px.histogram(
                    st.session_state.df,
                    x=value_col,
                    nbins=bins,
                    title=f"Histogram of {value_col}"
                )
            
            elif chart_type == "Box Plot":
                # Column selection
                value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="export_box_value_col")
                
                # Optional grouping
                use_groups = st.checkbox("Group by category", key="export_box_use_groups")
                if use_groups and cat_cols:
                    group_col = st.selectbox("Group by:", cat_cols, key="export_box_group_col")
                    
                    # Create chart with grouping
                    fig = px.box(
                        st.session_state.df,
                        x=group_col,
                        y=value_col,
                        title=f"Box Plot of {value_col} by {group_col}"
                    )
                else:
                    # Create chart without grouping
                    fig = px.box(
                        st.session_state.df,
                        y=value_col,
                        title=f"Box Plot of {value_col}"
                    )
            
            elif chart_type == "Heatmap":
                if len(num_cols) < 1:
                    st.warning("Need numeric columns for heatmap.")
                    return
                
                # Create correlation matrix for heatmap
                corr_cols = st.multiselect(
                    "Select columns for correlation heatmap:",
                    num_cols,
                    default=num_cols[:min(8, len(num_cols))],
                    key="export_heatmap_cols"
                )
                
                if len(corr_cols) < 2:
                    st.warning("Please select at least two columns for correlation heatmap.")
                else:
                    # Create chart
                    corr_matrix = st.session_state.df[corr_cols].corr().round(2)
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap"
                    )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.markdown("### Export Chart")
            
            # Format selection
            export_format = st.radio(
                "Export format:",
                ["PNG", "JPEG", "SVG", "HTML"],
                horizontal=True,
                key='export_format'
            )
            
            # Export button
            if st.button("Export Chart", use_container_width=True, key="export_chart_button"):
                try:
                    if export_format == "HTML":
                        # Export as HTML file
                        buffer = StringIO()
                        fig.write_html(buffer)
                        html_bytes = buffer.getvalue().encode()
                        
                        st.download_button(
                            label="Download HTML",
                            data=html_bytes,
                            file_name=f"chart_{chart_type.lower().replace(' ', '_')}.html",
                            mime="text/html",
                            key="download_chart_html"
                        )
                    else:
                        # Export as image
                        img_bytes = fig.to_image(format=export_format.lower())
                        
                        st.download_button(
                            label=f"Download {export_format}",
                            data=img_bytes,
                            file_name=f"chart_{chart_type.lower().replace(' ', '_')}.{export_format.lower()}",
                            mime=f"image/{export_format.lower()}",
                            key=f"download_chart_{export_format.lower()}"
                        )
                except Exception as e:
                    st.error(f"Error exporting chart: {str(e)}")
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("This may be due to incompatible column types or missing values. Try selecting different columns.")

def render_trend_analysis():
    """Render trend analysis tab"""
    st.subheader("Trend Analysis")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
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
        # Make sure we only suggest columns that actually exist in the dataframe
        available_date_cols = [col for col in all_date_cols if col in st.session_state.df.columns]
        if not available_date_cols:
            st.error("No valid date/time columns found in the dataset.")
            return
            
        x_col = st.selectbox("Date/Time column:", available_date_cols, key="trend_x_col")
    
    with col2:
        # Get numeric columns and ensure they exist in the dataframe
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns found for trend analysis.")
            return
        
        # Ensure each column in the list actually exists in the dataframe
        available_num_cols = [col for col in num_cols if col in st.session_state.df.columns]
        if not available_num_cols:
            st.error("No valid numeric columns found in the dataset.")
            return
            
        y_col = st.selectbox("Value column:", available_num_cols, key="trend_y_col")
    
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
    
    # Validate that selected columns exist before proceeding
    if x_col not in st.session_state.df.columns:
        st.error(f"Selected date column '{x_col}' not found in the dataset.")
        return
        
    if y_col not in st.session_state.df.columns:
        st.error(f"Selected value column '{y_col}' not found in the dataset.")
        return
    
    # Trend analysis type
    analysis_type = st.radio(
        "Analysis type:",
        ["Time Series Plot", "Moving Averages", "Seasonality Analysis", "Trend Decomposition"],
        horizontal=True,
        key="trend_analysis_type"
    )
    
    if analysis_type == "Time Series Plot":
        # Optional grouping
        use_grouping = st.checkbox("Group by a categorical column", key="ts_use_grouping")
        group_col = None
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                # Ensure categorical columns exist in the dataframe
                available_cat_cols = [col for col in cat_cols if col in st.session_state.df.columns]
                if not available_cat_cols:
                    st.warning("No valid categorical columns found for grouping.")
                    use_grouping = False
                else:
                    group_col = st.selectbox("Group by:", available_cat_cols, key="ts_group_col")
        
        # Create time series plot
        if use_grouping and group_col:
            # Validate group column exists
            if group_col not in st.session_state.df.columns:
                st.error(f"Selected grouping column '{group_col}' not found in the dataset.")
                return
                
            # Get top groups for better visualization
            top_n = st.slider("Show top N groups:", 1, 10, 5, key="ts_top_n")
            top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
            
            # Filter to top groups
            filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
            
            # Create grouped time series - use try/except to catch any plotting errors
            try:
                fig = px.line(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color=group_col,
                    title=f"Time Series Plot of {y_col} over {x_col}, grouped by {group_col}"
                )
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("This may be due to incompatible data types or missing values. Try selecting different columns or preprocessing your data.")
                return
        else:
            # Simple time series - use try/except to catch any plotting errors
            try:
                fig = px.line(
                    st.session_state.df,
                    x=x_col,
                    y=y_col,
                    title=f"Time Series Plot of {y_col} over {x_col}"
                )
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("This may be due to incompatible data types or missing values. Try selecting different columns or preprocessing your data.")
                return
        
        # Add trend line
        add_trend = st.checkbox("Add trend line", key="ts_add_trend")
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
                
                # Check if we have enough data points
                if len(x_values) < 2 or len(y_values) < 2:
                    st.warning("Not enough valid data points to calculate trend line.")
                else:
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
            default=[30],
            key="ma_window_sizes"
        )
        
        if not window_sizes:
            st.warning("Please select at least one window size.")
            return
        
        try:
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
            show_volatility = st.checkbox("Show volatility (rolling standard deviation)", key="ma_show_volatility")
            if show_volatility:
                # Create volatility plot
                fig_vol = go.Figure()
                
                # Calculate rolling standard deviation
                vol_window = st.slider("Volatility window size:", 5, 100, 30, key="ma_vol_window")
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
        except Exception as e:
            st.error(f"Error creating moving averages: {str(e)}")
            st.info("This may be due to incompatible data types or missing values. Try selecting different columns or preprocessing your data.")
    
    elif analysis_type == "Seasonality Analysis":
        try:
            # Make sure we have enough data
            if len(st.session_state.df) < 10:
                st.warning("Need more data points for seasonality analysis.")
                return
            
            # Check if the data is regularly spaced in time
            df_sorted = st.session_state.df.sort_values(x_col)
            
            # Time period selection
            period_type = st.selectbox(
                "Analyze seasonality by:",
                ["Day of Week", "Month", "Quarter", "Year"],
                key="season_period_type"
            )
            
            # Extract the relevant period component
            try:
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
            except Exception as e:
                st.error(f"Error extracting {period_type} from the date column: {str(e)}")
                st.info("Make sure the date column is in a proper datetime format.")
                return
            
            # Calculate statistics by period
            try:
                period_stats = df_sorted.groupby('period')[y_col].agg(['mean', 'median', 'min', 'max', 'std', 'count'])
            except Exception as e:
                st.error(f"Error calculating statistics by period: {str(e)}")
                return
            
            # Reorder based on natural order
            try:
                if period_type != "Year":
                    # For named periods, use predefined order
                    period_stats = period_stats.reindex(period_order)
                else:
                    # For years, sort numerically
                    period_stats = period_stats.sort_index()
                
                # Fill NaN with 0
                period_stats = period_stats.fillna(0)
            except Exception as e:
                st.warning(f"Error reordering periods: {str(e)}")
            
            # Create bar chart
            try:
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
            except Exception as e:
                st.error(f"Error creating seasonality bar chart: {str(e)}")
            
            # Show statistics table
            st.subheader(f"Statistics by {period_type}")
            st.dataframe(period_stats.round(2), use_container_width=True)
            
            # Heatmap visualization
            st.subheader("Seasonality Heatmap")
            
            # Create heatmap based on period type
            if period_type in ["Month", "Day of Week"]:
                try:
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
                            horizontal=True,
                            key="season_use_month"
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
                    st.error(f"Error preparing data for heatmap: {str(e)}")
            
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
            resample = st.checkbox("Resample data to regular intervals", key="decomp_resample")
            
            if resample:
                # Frequency selection
                freq = st.selectbox(
                    "Resampling frequency:",
                    ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                    key="decomp_freq"
                )
                
                # Map to pandas frequency string
                freq_map = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "MS",
                    "Quarterly": "QS",
                    "Yearly": "YS"
                }
                
                try:
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
                except Exception as e:
                    st.error(f"Error resampling data: {str(e)}")
                    st.info("Make sure the date column is in a proper datetime format and contains enough data points.")
                    return
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
                ["Additive", "Multiplicative"],
                key="decomp_model"
            )
            
            # Period selection
            period = st.slider("Period (number of time units in a seasonal cycle):", 2, 52, 12, key="decomp_period")
            
            # Perform decomposition
            try:
                if model == "Additive":
                    result = seasonal_decompose(data_ts[y_col], model='additive', period=period)
                else:
                    result = seasonal_decompose(data_ts[y_col], model='multiplicative', period=period)
            except Exception as e:
                st.error(f"Error performing decomposition: {str(e)}")
                st.info("Try adjusting the period parameter or check for missing/invalid values in your data.")
                return
            
            # Create figure with subplots
            try:
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
            except Exception as e:
                st.error(f"Error creating decomposition plot: {str(e)}")
                return
            
            # Summary statistics
            st.subheader("Component Statistics")
            
            # Calculate statistics for each component
            try:
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
                st.error(f"Error calculating component statistics: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in trend decomposition: {str(e)}")
            st.info("Try resampling the data or adjusting the period parameter.")
    
    # Export options
    if 'fig' in locals():
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"], key="trend_export_format")
        
        if export_format == "HTML":
            # Export as HTML file
            try:
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Chart",
                    data=html_bytes,
                    file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="trend_download_html"
                )
            except Exception as e:
                st.error(f"Error exporting chart as HTML: {str(e)}")
        else:
            # Export as image
            try:
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Chart",
                    data=img_bytes,
                    file_name=f"trend_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True,
                    key=f"trend_download_{export_format.lower()}"
                )
            except Exception as e:
                st.error(f"Error exporting chart as image: {str(e)}")

def render_distribution_analysis():
    """Render distribution analysis tab"""
    st.subheader("Distribution Analysis")
    
    # Check if dataframe exists and is not empty
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Please upload a CSV file first.")
        return
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns found for distribution analysis.")
        return
    
    # Column selection
    selected_col = st.selectbox(
        "Select column for distribution analysis:",
        num_cols,
        key="dist_selected_col"
    )
    
    # Validate that selected column exists
    if selected_col not in st.session_state.df.columns:
        st.error(f"Selected column '{selected_col}' not found in the dataset.")
        return
    
    # Analysis type selection
    analysis_type = st.radio(
        "Analysis type:",
        ["Histogram", "Density Plot", "Box Plot", "Q-Q Plot", "Distribution Fitting"],
        horizontal=True,
        key="distribution_analysis_type"
    )
    
    if analysis_type == "Histogram":
        # Histogram options
        n_bins = st.slider("Number of bins:", 5, 100, 20, key="hist_n_bins")
        
        # Create histogram
        try:
            fig = px.histogram(
                st.session_state.df,
                x=selected_col,
                nbins=n_bins,
                title=f"Histogram of {selected_col}",
                marginal="box"
            )
            
            # Show histogram
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return
        
        # Calculate basic statistics
        try:
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
                st.metric("Count", f"{int(stats['count'])}")
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
        
    elif analysis_type == "Density Plot":
        # Create KDE plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            if len(data) < 2:
                st.error("Not enough valid data points for density estimation.")
                return
            
            # Calculate KDE
            kde_x = np.linspace(data.min(), data.max(), 1000)
            kde = scipy_stats.gaussian_kde(data)
            kde_y = kde(kde_x)
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            show_hist = st.checkbox("Show histogram with density plot", value=True, key="kde_show_hist")
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
        use_grouping = st.checkbox("Group by a categorical column", key="box_use_grouping")
        
        if use_grouping:
            # Get categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.warning("No categorical columns found for grouping.")
                use_grouping = False
            else:
                group_col = st.selectbox("Group by:", cat_cols, key="box_group_col")
                
                # Validate group column exists
                if group_col not in st.session_state.df.columns:
                    st.error(f"Selected grouping column '{group_col}' not found in the dataset.")
                    use_grouping = False
                else:
                    # Limit number of groups
                    top_n = st.slider("Show top N groups:", 2, 20, 10, key="box_top_n")
                    
                    # Get top groups
                    top_groups = st.session_state.df[group_col].value_counts().nlargest(top_n).index.tolist()
                    
                    # Filter to top groups
                    filtered_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups)]
                    
                    try:
                        # Create grouped box plot
                        fig = px.box(
                            filtered_df,
                            x=group_col,
                            y=selected_col,
                            title=f"Box Plot of {selected_col} by {group_col}",
                            points="all"
                        )
                    except Exception as e:
                        st.error(f"Error creating grouped box plot: {str(e)}")
                        use_grouping = False
        
        if not use_grouping:
            try:
                # Simple box plot
                fig = px.box(
                    st.session_state.df,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}",
                    points="all"
                )
            except Exception as e:
                st.error(f"Error creating box plot: {str(e)}")
                return
        
        # Show box plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate quartiles and IQR
        try:
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
        except Exception as e:
            st.error(f"Error calculating box plot statistics: {str(e)}")
        
    elif analysis_type == "Q-Q Plot":
        # Distribution selection
        dist = st.selectbox(
            "Reference distribution:",
            ["Normal", "Uniform", "Exponential", "Log-normal"],
            key="qq_dist"
        )
        
        # Create Q-Q plot
        try:
            from scipy import stats as scipy_stats
            
            # Get data
            data = st.session_state.df[selected_col].dropna()
            
            if len(data) < 2:
                st.error("Not enough valid data points for Q-Q plot.")
                return
            
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
            
            if len(data) < 10:
                st.error("Need at least 10 valid data points for distribution fitting.")
                return
            
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
                default=["Normal", "Log-normal", "Gamma"],
                key="dist_selected_dists"
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
    if 'fig' in locals():
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"], key="dist_export_format")
        
        try:
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
                    use_container_width=True,
                    key="dist_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Chart",
                    data=img_bytes,
                    file_name=f"distribution_{analysis_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True,
                    key=f"dist_download_{export_format.lower()}"
                )
        except Exception as e:
            st.error(f"Error exporting chart: {str(e)}")