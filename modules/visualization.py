import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from io import BytesIO
import math
import base64
from scipy import stats
import networkx as nx

class EnhancedVisualizer:
    """Class for creating data visualizations"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Store visualizations in session state if not already there
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = []
    
    def render_interface(self):
        """Render visualization interface"""
        st.header("Visualization Studio")
        
        if self.df is None or self.df.empty:
            st.info("No data available for visualization. Please process your data first.")
            return
        
        # Available visualization types
        viz_types = {
            "Basic": ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart", "Histogram"],
            "Statistical": ["Box Plot", "Violin Plot", "Scatter Plot", "Heatmap", "Density Plot"],
            "Multi-Variable": ["Grouped Bar Chart", "Stacked Bar Chart", "Bubble Chart", "Radar Chart", "Pairplot"],
            "Distribution": ["Distribution Plot", "Q-Q Plot", "ECDF Plot", "Residual Plot", "Correlation Matrix"]
        }
        
        # Create tabs for visualization categories
        viz_category = st.radio("Select Visualization Category", list(viz_types.keys()), horizontal=True)
        
        # Select visualization type
        viz_type = st.selectbox("Select Visualization Type", viz_types[viz_category])
        
        # Run specific visualization function based on selection
        if viz_type == "Bar Chart":
            self._create_bar_chart()
        elif viz_type == "Line Chart":
            self._create_line_chart()
        elif viz_type == "Pie Chart":
            self._create_pie_chart()
        elif viz_type == "Area Chart":
            self._create_area_chart()
        elif viz_type == "Histogram":
            self._create_histogram()
        elif viz_type == "Box Plot":
            self._create_box_plot()
        elif viz_type == "Violin Plot":
            self._create_violin_plot()
        elif viz_type == "Scatter Plot":
            self._create_scatter_plot()
        elif viz_type == "Heatmap":
            self._create_heatmap()
        elif viz_type == "Density Plot":
            self._create_density_plot()
        elif viz_type == "Grouped Bar Chart":
            self._create_grouped_bar_chart()
        elif viz_type == "Stacked Bar Chart":
            self._create_stacked_bar_chart()
        elif viz_type == "Bubble Chart":
            self._create_bubble_chart()
        elif viz_type == "Radar Chart":
            self._create_radar_chart()
        elif viz_type == "Pairplot":
            self._create_pairplot()
        elif viz_type == "Distribution Plot":
            self._create_distribution_plot()
        elif viz_type == "Q-Q Plot":
            self._create_qq_plot()
        elif viz_type == "ECDF Plot":
            self._create_ecdf_plot()
        elif viz_type == "Residual Plot":
            self._create_residual_plot()
        elif viz_type == "Correlation Matrix":
            self._create_correlation_matrix()
        
        # Display saved visualizations
        if st.session_state.visualizations:
            st.header("Saved Visualizations")
            
            # Create columns based on number of visualizations (max 3 per row)
            num_cols = min(3, len(st.session_state.visualizations))
            cols = st.columns(num_cols)
            
            # Display visualizations in columns
            for i, viz in enumerate(st.session_state.visualizations):
                with cols[i % num_cols]:
                    st.subheader(viz["title"])
                    if viz["type"] == "matplotlib":
                        st.pyplot(viz["figure"])
                    elif viz["type"] == "plotly":
                        st.plotly_chart(viz["figure"], use_container_width=True)
    
    def _create_bar_chart(self):
        """Create bar chart visualization"""
        st.subheader("Bar Chart")
        
        # Get categorical columns for x-axis
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or not num_cols:
            st.info("Bar charts require at least one categorical column and one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis (categorical)", cat_cols)
        y_col = st.selectbox("Select Y-axis (numeric)", num_cols)
        
        # Aggregation function
        agg_func = st.selectbox(
            "Select aggregation function", 
            ["Count", "Sum", "Mean", "Median", "Min", "Max"]
        )
        
        # Color options
        use_color = st.checkbox("Add color dimension")
        color_col = None
        if use_color and len(cat_cols) > 1:
            color_options = [col for col in cat_cols if col != x_col]
            color_col = st.selectbox("Select color column", color_options)
        
        # Sort options
        sort_values = st.checkbox("Sort bars by value")
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Prepare data
            if agg_func == "Count":
                if color_col:
                    df_grouped = self.df.groupby([x_col, color_col]).size().reset_index(name='count')
                    fig = px.bar(
                        df_grouped, 
                        x=x_col, 
                        y='count', 
                        color=color_col,
                        title=custom_title or f"Count by {x_col} and {color_col}",
                        labels={x_col: x_col, 'count': 'Count'},
                        template="plotly_white"
                    )
                else:
                    df_grouped = self.df.groupby(x_col).size().reset_index(name='count')
                    if sort_values:
                        df_grouped = df_grouped.sort_values('count')
                    
                    fig = px.bar(
                        df_grouped, 
                        x=x_col, 
                        y='count',
                        title=custom_title or f"Count by {x_col}",
                        labels={x_col: x_col, 'count': 'Count'},
                        template="plotly_white"
                    )
            else:
                # Map aggregation function
                agg_map = {
                    "Sum": "sum",
                    "Mean": "mean",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max"
                }
                
                if color_col:
                    df_grouped = self.df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
                    fig = px.bar(
                        df_grouped, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=custom_title or f"{agg_func} of {y_col} by {x_col} and {color_col}",
                        labels={x_col: x_col, y_col: f"{agg_func} of {y_col}"},
                        template="plotly_white"
                    )
                else:
                    df_grouped = self.df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
                    if sort_values:
                        df_grouped = df_grouped.sort_values(y_col)
                    
                    fig = px.bar(
                        df_grouped, 
                        x=x_col, 
                        y=y_col,
                        title=custom_title or f"{agg_func} of {y_col} by {x_col}",
                        labels={x_col: x_col, y_col: f"{agg_func} of {y_col}"},
                        template="plotly_white"
                    )
            
            # Customize layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col if agg_func != "Count" else "Count",
                legend_title=color_col if color_col else ""
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="bar_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="bar_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Bar Chart: {agg_func} of {y_col} by {x_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_line_chart(self):
        """Create line chart visualization"""
        st.subheader("Line Chart")
        
        # Get all columns
        all_cols = self.df.columns.tolist()
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Line charts require at least one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis", all_cols)
        y_cols = st.multiselect("Select Y-axis (can select multiple)", num_cols)
        
        if not y_cols:
            st.info("Please select at least one column for Y-axis.")
            return
        
        # Line style options
        markers = st.checkbox("Show markers")
        
        # Color options
        use_color = st.checkbox("Group/color by category")
        color_col = None
        if use_color:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                color_col = st.selectbox("Select grouping column", cat_cols)
            else:
                st.warning("No categorical columns available for grouping.")
                use_color = False
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Sort by x-axis if it's a datetime or numeric column
            if pd.api.types.is_datetime64_any_dtype(self.df[x_col]) or pd.api.types.is_numeric_dtype(self.df[x_col]):
                df_sorted = self.df.sort_values(x_col)
            else:
                df_sorted = self.df.copy()
            
            # Create plot based on whether we're using a color/group column
            if use_color and color_col:
                if len(y_cols) > 1:
                    st.warning("When using grouping, only the first Y-axis column will be used.")
                
                # Use only the first y column when grouping
                y_col = y_cols[0]
                
                fig = px.line(
                    df_sorted,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    markers=markers,
                    title=custom_title or f"Line Chart: {y_col} by {x_col} (Grouped by {color_col})",
                    template="plotly_white"
                )
            else:
                # Create plot with multiple y columns
                fig = go.Figure()
                
                # Add lines for each y column
                for y_col in y_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted[x_col],
                            y=df_sorted[y_col],
                            mode='lines+markers' if markers else 'lines',
                            name=y_col
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=custom_title or f"Line Chart: {', '.join(y_cols)} by {x_col}",
                    xaxis_title=x_col,
                    yaxis_title=y_cols[0] if len(y_cols) == 1 else "Value",
                    legend_title="Series",
                    template="plotly_white"
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="line_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="line_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Line Chart: {', '.join(y_cols)} by {x_col}"
                if use_color and color_col:
                    title += f" (Grouped by {color_col})"
                
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_scatter_plot(self):
        """Create scatter plot visualization"""
        st.subheader("Scatter Plot")
        
        # Get numeric columns for axes
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Scatter plots require at least two numeric columns.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis", num_cols)
        y_col = st.selectbox("Select Y-axis", [col for col in num_cols if col != x_col])
        
        # Color options
        use_color = st.checkbox("Add color dimension")
        color_col = None
        
        if use_color:
            color_options = [col for col in self.df.columns if col not in [x_col, y_col]]
            color_col = st.selectbox("Select color column", color_options)
        
        # Size options
        use_size = st.checkbox("Add size dimension")
        size_col = None
        
        if use_size:
            size_options = [col for col in num_cols if col not in [x_col, y_col]]
            if size_options:
                size_col = st.selectbox("Select size column", size_options)
            else:
                st.warning("No additional numeric columns available for size.")
                use_size = False
        
        # Trendline option
        add_trendline = st.checkbox("Add trendline")
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Build the scatter plot with various options
            if use_color and color_col:
                if use_size and size_col:
                    # Scatter plot with color and size
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        size=size_col,
                        title=custom_title or f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline="ols" if add_trendline else None
                    )
                else:
                    # Scatter plot with color only
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=custom_title or f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline="ols" if add_trendline else None
                    )
            else:
                if use_size and size_col:
                    # Scatter plot with size only
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        title=custom_title or f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline="ols" if add_trendline else None
                    )
                else:
                    # Simple scatter plot
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        title=custom_title or f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline="ols" if add_trendline else None
                    )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if use_color and color_col else ""
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and show correlation
            correlation = self.df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation Coefficient", f"{correlation:.4f}")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="scatter_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="scatter_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Scatter Plot: {y_col} vs {x_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_heatmap(self):
        """Create heatmap visualization"""
        st.subheader("Heatmap")
        
        # Options for heatmap
        heatmap_type = st.radio("Heatmap Type", ["Correlation", "Crosstab", "Custom"], horizontal=True)
        
        if heatmap_type == "Correlation":
            self._create_correlation_heatmap()
        elif heatmap_type == "Crosstab":
            self._create_crosstab_heatmap()
        else:
            self._create_custom_heatmap()
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap"""
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Correlation heatmaps require at least two numeric columns.")
            return
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns for correlation", 
            num_cols,
            default=num_cols[:min(len(num_cols), 8)]  # Default to first 8 columns or fewer
        )
        
        if not selected_cols or len(selected_cols) < 2:
            st.info("Please select at least two columns.")
            return
        
        # Correlation method
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"]
        )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create heatmap
        st.subheader("Preview")
        
        with st.spinner("Creating heatmap..."):
            # Calculate correlation matrix
            corr_matrix = self.df[selected_cols].corr(method=corr_method)
            
            # Create heatmap with plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title=custom_title or f"{corr_method.capitalize()} Correlation Heatmap",
                template="plotly_white"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="",
                yaxis_title=""
            )
            
            # Display the heatmap
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="correlation_heatmap.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="correlation_heatmap.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"{corr_method.capitalize()} Correlation Heatmap",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_crosstab_heatmap(self):
        """Create crosstab heatmap"""
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(cat_cols) < 2:
            st.info("Crosstab heatmaps require at least two categorical columns.")
            return
        
        # Column selection
        x_col = st.selectbox("Select X-axis", cat_cols)
        y_col = st.selectbox("Select Y-axis", [col for col in cat_cols if col != x_col])
        
        # Aggregation options
        agg_options = ["Count", "Percentage"]
        
        # Add numeric column options if available
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            agg_options.extend(["Mean", "Sum", "Median", "Min", "Max"])
        
        agg_type = st.selectbox("Aggregation type", agg_options)
        
        # Select value column if aggregating numeric values
        value_col = None
        if agg_type not in ["Count", "Percentage"] and num_cols:
            value_col = st.selectbox("Select value column", num_cols)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create heatmap
        st.subheader("Preview")
        
        with st.spinner("Creating heatmap..."):
            # Create crosstab
            if agg_type == "Count":
                # Count values
                crosstab = pd.crosstab(
                    self.df[y_col],
                    self.df[x_col]
                )
                title = custom_title or f"Count of {x_col} by {y_col}"
                z_title = "Count"
            elif agg_type == "Percentage":
                # Percentage values (normalize by columns)
                crosstab = pd.crosstab(
                    self.df[y_col],
                    self.df[x_col],
                    normalize='columns'
                ) * 100
                title = custom_title or f"Percentage of {x_col} by {y_col}"
                z_title = "Percentage (%)"
            else:
                # Aggregated values
                agg_func = agg_type.lower()
                crosstab = self.df.pivot_table(
                    index=y_col,
                    columns=x_col,
                    values=value_col,
                    aggfunc=agg_func
                )
                title = custom_title or f"{agg_type} of {value_col} by {x_col} and {y_col}"
                z_title = f"{agg_type} of {value_col}"
            
            # Create heatmap with plotly
            fig = px.imshow(
                crosstab,
                text_auto=".1f" if agg_type == "Percentage" else ".2f",
                color_continuous_scale="Viridis",
                title=title,
                template="plotly_white",
                aspect="auto"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            
            # Display the heatmap
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="crosstab_heatmap.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="crosstab_heatmap.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_custom_heatmap(self):
        """Create custom heatmap from selected columns"""
        # Get all columns
        all_cols = self.df.columns.tolist()
        
        if len(all_cols) < 3:
            st.info("Custom heatmaps require at least three columns.")
            return
        
        # Column selection
        x_col = st.selectbox("Select X-axis", all_cols)
        y_col = st.selectbox("Select Y-axis", [col for col in all_cols if col != x_col])
        z_col = st.selectbox("Select value column", [col for col in all_cols if col not in [x_col, y_col]])
        
        # Check if value column is numeric
        if self.df[z_col].dtype not in ['int64', 'float64']:
            st.warning("Value column should be numeric for best results.")
        
        # Aggregation function
        agg_func = st.selectbox(
            "Aggregation function",
            ["Mean", "Sum", "Count", "Median", "Min", "Max"]
        )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create heatmap
        st.subheader("Preview")
        
        with st.spinner("Creating heatmap..."):
            try:
                # Create pivot table
                pivot = self.df.pivot_table(
                    index=y_col,
                    columns=x_col,
                    values=z_col,
                    aggfunc=agg_func.lower()
                )
                
                # Create heatmap with plotly
                fig = px.imshow(
                    pivot,
                    text_auto=".2f",
                    color_continuous_scale="Viridis",
                    title=custom_title or f"{agg_func} of {z_col} by {x_col} and {y_col}",
                    template="plotly_white"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                
                # Display the heatmap
                st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    # Create a download link for the plot as HTML
                    buffer = io.StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download as HTML",
                        data=html_bytes,
                        file_name="custom_heatmap.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col2:
                    # Create a download link for the plot as PNG
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    
                    st.download_button(
                        label="Download as PNG",
                        data=img_bytes,
                        file_name="custom_heatmap.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Save visualization button
                if st.button("Save Visualization", use_container_width=True):
                    viz_data = {
                        "title": custom_title or f"{agg_func} of {z_col} by {x_col} and {y_col}",
                        "type": "plotly",
                        "figure": fig
                    }
                    st.session_state.visualizations.append(viz_data)
                    st.success("Visualization saved!")
                    
            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")
                st.info("Try different columns or aggregation function.")
    
    def _create_pie_chart(self):
        """Create pie chart visualization"""
        st.subheader("Pie Chart")
        
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for values
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols:
            st.info("Pie charts require at least one categorical column.")
            return
        
        # User selections
        cat_col = st.selectbox("Select categories", cat_cols)
        
        # Value options
        use_values = st.checkbox("Use custom values (default is count)")
        value_col = None
        if use_values and num_cols:
            value_col = st.selectbox("Select value column", num_cols)
        
        # Limit top categories
        limit_cats = st.slider("Limit to top N categories (0 for all)", 0, 20, 10)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Prepare data
            if use_values and value_col:
                # Sum by category and value
                df_grouped = self.df.groupby(cat_col)[value_col].sum().reset_index()
                value_label = f"Sum of {value_col}"
            else:
                # Count by category
                df_grouped = self.df.groupby(cat_col).size().reset_index(name='count')
                value_col = 'count'
                value_label = "Count"
            
            # Sort and limit categories if requested
            df_grouped = df_grouped.sort_values(value_col, ascending=False)
            
            if limit_cats > 0 and len(df_grouped) > limit_cats:
                # Create 'Other' category for remaining items
                other_sum = df_grouped.iloc[limit_cats:][value_col].sum()
                df_top = df_grouped.iloc[:limit_cats].copy()
                
                other_row = pd.DataFrame({cat_col: ['Other'], value_col: [other_sum]})
                df_grouped = pd.concat([df_top, other_row]).reset_index(drop=True)
            
            # Create plot
            fig = px.pie(
                df_grouped,
                names=cat_col,
                values=value_col,
                title=custom_title or f"Distribution of {cat_col}" + (f" by {value_col}" if use_values and value_col else ""),
                labels={cat_col: cat_col, value_col: value_label},
                template="plotly_white"
            )
            
            # Update layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="pie_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="pie_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Pie Chart: Distribution of {cat_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_histogram(self):
        """Create histogram visualization"""
        st.subheader("Histogram")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Histograms require at least one numeric column.")
            return
        
        # User selections
        col = st.selectbox("Select column", num_cols)
        
        # Histogram options
        n_bins = st.slider("Number of bins", 5, 100, 20)
        density = st.checkbox("Show density (KDE)")
        
        # Grouping options
        use_groups = st.checkbox("Split by category")
        group_col = None
        
        if use_groups:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                group_col = st.selectbox("Select grouping column", cat_cols)
            else:
                st.warning("No categorical columns available for grouping.")
                use_groups = False
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if use_groups and group_col:
                # Create grouped histogram with plotly
                fig = px.histogram(
                    self.df,
                    x=col,
                    color=group_col,
                    marginal="rug" if not density else "kde",
                    nbins=n_bins,
                    opacity=0.7,
                    barmode="overlay",
                    title=custom_title or f"Histogram of {col} by {group_col}",
                    template="plotly_white"
                )
            else:
                # Create simple histogram with plotly
                fig = px.histogram(
                    self.df,
                    x=col,
                    marginal="rug" if not density else "kde",
                    nbins=n_bins,
                    title=custom_title or f"Histogram of {col}",
                    template="plotly_white"
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count",
                bargap=0.1
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic statistics
            st.subheader(f"Statistics for {col}")
            stats = self.df[col].describe()
            
            # Create columns for stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{stats['min']:.2f}")
            with col4:
                st.metric("Max", f"{stats['max']:.2f}")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="histogram.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="histogram.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Histogram of {col}"
                if use_groups and group_col:
                    title += f" by {group_col}"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_box_plot(self):
        """Create box plot visualization"""
        st.subheader("Box Plot")
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Box plots require at least one numeric column.")
            return
        
        # User selections
        y_col = st.selectbox("Select numeric column (Y-axis)", num_cols)
        
        # Grouping options
        use_groups = st.checkbox("Group by category", value=True)
        x_col = None
        
        if use_groups:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                x_col = st.selectbox("Select grouping column (X-axis)", cat_cols)
            else:
                st.warning("No categorical columns available for grouping.")
                use_groups = False
        
        # Additional options
        show_points = st.checkbox("Show all points", value=False)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if use_groups and x_col:
                # Create grouped box plot
                fig = px.box(
                    self.df,
                    x=x_col,
                    y=y_col,
                    points="all" if show_points else "outliers",
                    title=custom_title or f"Box Plot of {y_col} by {x_col}",
                    template="plotly_white"
                )
            else:
                # Create simple box plot
                fig = px.box(
                    self.df,
                    y=y_col,
                    points="all" if show_points else "outliers",
                    title=custom_title or f"Box Plot of {y_col}",
                    template="plotly_white"
                )
            
            # Update layout
            if use_groups and x_col:
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
            else:
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title=y_col
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="box_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="box_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Box Plot of {y_col}"
                if use_groups and x_col:
                    title += f" by {x_col}"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_violin_plot(self):
        """Create violin plot visualization"""
        st.subheader("Violin Plot")
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Violin plots require at least one numeric column.")
            return
        
        # User selections
        y_col = st.selectbox("Select numeric column (Y-axis)", num_cols)
        
        # Grouping options
        use_groups = st.checkbox("Group by category", value=True)
        x_col = None
        
        if use_groups:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                x_col = st.selectbox("Select grouping column (X-axis)", cat_cols)
            else:
                st.warning("No categorical columns available for grouping.")
                use_groups = False
        
        # Additional options
        show_box = st.checkbox("Show box plot inside", value=True)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if use_groups and x_col:
                # Create grouped violin plot
                fig = px.violin(
                    self.df,
                    x=x_col,
                    y=y_col,
                    box=show_box,
                    points="all",
                    title=custom_title or f"Violin Plot of {y_col} by {x_col}",
                    template="plotly_white"
                )
            else:
                # Create simple violin plot
                fig = px.violin(
                    self.df,
                    y=y_col,
                    box=show_box,
                    points="all",
                    title=custom_title or f"Violin Plot of {y_col}",
                    template="plotly_white"
                )
            
            # Update layout
            if use_groups and x_col:
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
            else:
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title=y_col
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="violin_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="violin_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Violin Plot of {y_col}"
                if use_groups and x_col:
                    title += f" by {x_col}"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_area_chart(self):
        """Create area chart visualization"""
        st.subheader("Area Chart")
        
        # Get all columns
        all_cols = self.df.columns.tolist()
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Area charts require at least one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis", all_cols)
        y_cols = st.multiselect("Select Y-axis (can select multiple)", num_cols)
        
        if not y_cols:
            st.info("Please select at least one column for Y-axis.")
            return
        
        # Stacking options
        stack_mode = st.radio("Stacking mode", ["None", "Stack", "Normalize"], horizontal=True)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Sort by x-axis if it's a datetime or numeric column
            if pd.api.types.is_datetime64_any_dtype(self.df[x_col]) or pd.api.types.is_numeric_dtype(self.df[x_col]):
                df_sorted = self.df.sort_values(x_col)
            else:
                df_sorted = self.df.copy()
            
            # Determine the groupby mode
            if stack_mode == "None":
                # Individual area charts
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=custom_title or f"Area Chart: {', '.join(y_cols)} by {x_col}",
                    labels={col: col for col in y_cols},
                    template="plotly_white"
                )
            elif stack_mode == "Stack":
                # Stacked area chart
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=custom_title or f"Stacked Area Chart: {', '.join(y_cols)} by {x_col}",
                    labels={col: col for col in y_cols},
                    template="plotly_white"
                )
            else:
                # Normalized (100%) area chart
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=custom_title or f"Normalized Area Chart: {', '.join(y_cols)} by {x_col}",
                    labels={col: col for col in y_cols},
                    template="plotly_white",
                    groupnorm='fraction'
                )
                # Update y-axis to percentage
                fig.update_layout(yaxis=dict(tickformat='.0%'))
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title="Value" if stack_mode != "Normalize" else "Percentage",
                legend_title="Series"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="area_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="area_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                stack_type = ""
                if stack_mode == "Stack":
                    stack_type = "Stacked "
                elif stack_mode == "Normalize":
                    stack_type = "Normalized "
                    
                viz_data = {
                    "title": custom_title or f"{stack_type}Area Chart: {', '.join(y_cols)} by {x_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_density_plot(self):
        """Create density plot visualization"""
        st.subheader("Density Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Density plots require at least one numeric column.")
            return
        
        # User selections
        cols = st.multiselect("Select columns", num_cols)
        
        if not cols:
            st.info("Please select at least one column.")
            return
        
        # Grouping options
        use_groups = st.checkbox("Group by category")
        group_col = None
        
        if use_groups:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                group_col = st.selectbox("Select grouping column", cat_cols)
            else:
                st.warning("No categorical columns available for grouping.")
                use_groups = False
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if len(cols) == 1 and use_groups and group_col:
                # Single column with grouping
                fig = px.violin(
                    self.df,
                    x=cols[0],
                    color=group_col,
                    box=False,
                    points=False,
                    title=custom_title or f"Density Plot of {cols[0]} by {group_col}",
                    template="plotly_white"
                )
                
                # Remove the yaxis for cleaner look - just show distributions
                fig.update_traces(orientation='h')
                fig.update_layout(yaxis_title="")
                
            elif len(cols) == 1:
                # Single column without grouping (KDE plot)
                fig = px.histogram(
                    self.df,
                    x=cols[0],
                    marginal="kde",
                    histnorm="probability density",
                    title=custom_title or f"Density Plot of {cols[0]}",
                    template="plotly_white"
                )
                
                # Hide the histogram
                fig.update_traces(
                    selector=dict(type="histogram"),
                    visible=False
                )
                
                # Make the KDE the main plot instead of marginal
                for trace in fig.select_traces(selector=dict(name="kde")):
                    trace.update(visible=True)
            else:
                # Multiple columns (overlay density plots)
                fig = go.Figure()
                
                for col in cols:
                    # Using kernel density estimation
                    from scipy import stats
                    kde = self.df[col].dropna()
                    
                    # Skip if no data
                    if len(kde) == 0:
                        continue
                        
                    # Calculate KDE
                    x_range = np.linspace(kde.min(), kde.max(), 300)
                    kde_values = stats.gaussian_kde(kde)(x_range)
                    
                    # Add density trace
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=kde_values,
                            mode='lines',
                            name=col
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=custom_title or f"Density Plot: {', '.join(cols)}",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    template="plotly_white"
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="density_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="density_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Density Plot: {', '.join(cols)}"
                if use_groups and group_col:
                    title += f" by {group_col}"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_pairplot(self):
        """Create pairplot visualization"""
        st.subheader("Pairplot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Pairplots require at least two numeric columns.")
            return
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns to include (recommended 3-5)",
            num_cols,
            default=num_cols[:min(4, len(num_cols))]
        )
        
        if len(selected_cols) < 2:
            st.info("Please select at least two columns.")
            return
        
        # Check if too many columns selected
        if len(selected_cols) > 8:
            st.warning("You've selected many columns. This might make the plot hard to read and slow to render.")
        
        # Color option
        use_color = st.checkbox("Color by category")
        color_col = None
        
        if use_color:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                color_col = st.selectbox("Select color column", cat_cols)
            else:
                st.warning("No categorical columns available for coloring.")
                use_color = False
        
        # Additional options
        diagonal_kind = st.selectbox(
            "Diagonal plot type", 
            ["histogram", "kde", "None"]
        )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Calculate the number of subplots
            n = len(selected_cols)
            fig = make_subplots(
                rows=n, 
                cols=n, 
                shared_xaxes=True, 
                shared_yaxes=True,
                horizontal_spacing=0.02,
                vertical_spacing=0.02
            )
            
            # Create a color map if using colors
            colors = None
            if use_color and color_col:
                # Get unique categories
                categories = self.df[color_col].unique()
                # Generate colors
                import plotly.express as px
                colorscale = px.colors.qualitative.Plotly
                colors = {cat: colorscale[i % len(colorscale)] for i, cat in enumerate(categories)}
            
            # Populate the matrix of plots
            for i, col_i in enumerate(selected_cols):
                for j, col_j in enumerate(selected_cols):
                    if i == j:  # Diagonal
                        if diagonal_kind == "histogram":
                            # Create histogram
                            hist_data = [self.df[col_i]]
                            hist_colors = ['rgba(100, 149, 237, 0.6)']  # Default blue
                            
                            # If coloring by category, create separate histograms
                            if use_color and color_col:
                                hist_data = []
                                hist_colors = []
                                for cat in self.df[color_col].unique():
                                    cat_data = self.df[self.df[color_col] == cat][col_i]
                                    if not cat_data.empty:
                                        hist_data.append(cat_data)
                                        hist_colors.append(colors[cat])
                            
                            for k, data in enumerate(hist_data):
                                fig.add_trace(
                                    go.Histogram(
                                        x=data,
                                        opacity=0.7,
                                        marker_color=hist_colors[k],
                                        showlegend=False
                                    ),
                                    row=i+1, col=j+1
                                )
                        
                        elif diagonal_kind == "kde":
                            # Create KDE plot
                            if use_color and color_col:
                                # Create separate KDEs for each category
                                for cat in self.df[color_col].unique():
                                    cat_data = self.df[self.df[color_col] == cat][col_i].dropna()
                                    if len(cat_data) > 1:  # Need at least 2 points for KDE
                                        from scipy import stats
                                        x_range = np.linspace(cat_data.min(), cat_data.max(), 100)
                                        kde_values = stats.gaussian_kde(cat_data)(x_range)
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=x_range,
                                                y=kde_values,
                                                mode='lines',
                                                name=str(cat),
                                                line=dict(color=colors[cat]),
                                                showlegend=False
                                            ),
                                            row=i+1, col=j+1
                                        )
                            else:
                                # Single KDE for all data
                                data = self.df[col_i].dropna()
                                if len(data) > 1:  # Need at least 2 points for KDE
                                    from scipy import stats
                                    x_range = np.linspace(data.min(), data.max(), 100)
                                    kde_values = stats.gaussian_kde(data)(x_range)
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_range,
                                            y=kde_values,
                                            mode='lines',
                                            line=dict(color='rgba(100, 149, 237, 0.8)'),
                                            showlegend=False
                                        ),
                                        row=i+1, col=j+1
                                    )
                        
                        # For 'None', leave diagonal empty
                        # Set axes titles only for edge plots
                        if i == n-1:
                            fig.update_xaxes(title_text=col_j, row=i+1, col=j+1)
                        if j == 0:
                            fig.update_yaxes(title_text=col_i, row=i+1, col=j+1)
                    
                    elif i > j:  # Lower triangle (typically scatter plots)
                        if use_color and color_col:
                            # Scatter plots colored by category
                            for cat in self.df[color_col].unique():
                                cat_data = self.df[self.df[color_col] == cat]
                                fig.add_trace(
                                    go.Scatter(
                                        x=cat_data[col_j],
                                        y=cat_data[col_i],
                                        mode='markers',
                                        name=str(cat),
                                        marker=dict(color=colors[cat], size=6, opacity=0.7),
                                        showlegend=False
                                    ),
                                    row=i+1, col=j+1
                                )
                        else:
                            # Simple scatter plot
                            fig.add_trace(
                                go.Scatter(
                                    x=self.df[col_j],
                                    y=self.df[col_i],
                                    mode='markers',
                                    marker=dict(color='rgba(100, 149, 237, 0.6)', size=6),
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )
                        
                        # Set axes titles only for edge plots
                        if i == n-1:
                            fig.update_xaxes(title_text=col_j, row=i+1, col=j+1)
                        if j == 0:
                            fig.update_yaxes(title_text=col_i, row=i+1, col=j+1)
                    
                    else:  # Upper triangle (can be left blank or show correlation)
                        # Show correlation coefficient
                        corr = self.df[[col_i, col_j]].corr().iloc[0, 1]
                        fig.add_annotation(
                            x=0.5,
                            y=0.5,
                            text=f"{corr:.2f}",
                            showarrow=False,
                            font=dict(
                                size=14,
                                color="rgba(0, 0, 0, 0.6)"
                            ),
                            row=i+1, col=j+1
                        )
            
            # Update layout
            fig.update_layout(
                title=custom_title or f"Pairplot of Selected Variables",
                showlegend=False,
                height=200 * n,
                width=200 * n,
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="pairplot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="pairplot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or "Pairplot of Selected Variables"
                if use_color and color_col:
                    title += f" (Colored by {color_col})"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_grouped_bar_chart(self):
        """Create grouped bar chart visualization"""
        st.subheader("Grouped Bar Chart")
        
        # Get categorical columns for x-axis
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(cat_cols) < 2 or not num_cols:
            st.info("Grouped bar charts require at least two categorical columns and one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis (primary grouping)", cat_cols)
        color_col = st.selectbox("Select grouping column", [col for col in cat_cols if col != x_col])
        y_col = st.selectbox("Select Y-axis (value)", num_cols)
        
        # Aggregation function
        agg_func = st.selectbox(
            "Select aggregation function", 
            ["Mean", "Sum", "Count", "Median", "Min", "Max"]
        )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Map aggregation function
            agg_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Count": "count",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Prepare data
            df_grouped = self.df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create grouped bar chart with plotly
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                color=color_col,
                barmode="group",
                title=custom_title or f"{agg_func} of {y_col} by {x_col} and {color_col}",
                template="plotly_white"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=f"{agg_func} of {y_col}",
                legend_title=color_col
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="grouped_bar_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="grouped_bar_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Grouped Bar Chart: {agg_func} of {y_col} by {x_col} and {color_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_stacked_bar_chart(self):
        """Create stacked bar chart visualization"""
        st.subheader("Stacked Bar Chart")
        
        # Get categorical columns for x-axis
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for y-axis
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(cat_cols) < 2 or not num_cols:
            st.info("Stacked bar charts require at least two categorical columns and one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis (primary grouping)", cat_cols)
        color_col = st.selectbox("Select stacking column", [col for col in cat_cols if col != x_col])
        y_col = st.selectbox("Select Y-axis (value)", num_cols)
        
        # Aggregation function
        agg_func = st.selectbox(
            "Select aggregation function", 
            ["Sum", "Mean", "Count", "Median", "Min", "Max"]
        )
        
        # Stacking mode
        stack_mode = st.radio(
            "Stacking mode",
            ["Stack", "Percentage"],
            horizontal=True
        )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Map aggregation function
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Count": "count",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Prepare data
            df_grouped = self.df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create stacked bar chart with plotly
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                color=color_col,
                barmode="stack" if stack_mode == "Stack" else "relative",
                title=custom_title or f"{stack_mode.capitalize()} Bar Chart: {agg_func} of {y_col} by {x_col} and {color_col}",
                template="plotly_white"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=f"{agg_func} of {y_col}" if stack_mode == "Stack" else "Percentage",
                legend_title=color_col
            )
            
            # For percentage stack, update y-axis to percentage format
            if stack_mode == "Percentage":
                fig.update_layout(yaxis=dict(tickformat='.0%'))
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="stacked_bar_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="stacked_bar_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"{stack_mode.capitalize()} Bar Chart: {agg_func} of {y_col} by {x_col} and {color_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_bubble_chart(self):
        """Create bubble chart visualization"""
        st.subheader("Bubble Chart")
        
        # Get numeric columns for axes
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 3:
            st.info("Bubble charts require at least three numeric columns (x, y, and size).")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis", num_cols)
        y_col = st.selectbox("Select Y-axis", [col for col in num_cols if col != x_col])
        size_col = st.selectbox("Select size dimension", [col for col in num_cols if col not in [x_col, y_col]])
        
        # Color options
        use_color = st.checkbox("Add color dimension")
        color_col = None
        
        if use_color:
            # Allow categorical or numeric columns for color
            color_options = [col for col in self.df.columns if col not in [x_col, y_col, size_col]]
            if color_options:
                color_col = st.selectbox("Select color column", color_options)
            else:
                st.warning("No additional columns available for color.")
                use_color = False
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Create bubble chart with plotly
            if use_color and color_col:
                fig = px.scatter(
                    self.df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    color=color_col,
                    title=custom_title or f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col}, Color: {color_col})",
                    template="plotly_white",
                    hover_name=self.df.index if self.df.index.name else None,
                    size_max=60
                )
            else:
                fig = px.scatter(
                    self.df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    title=custom_title or f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col})",
                    template="plotly_white",
                    hover_name=self.df.index if self.df.index.name else None,
                    size_max=60
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if use_color and color_col else ""
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="bubble_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="bubble_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col}"
                if use_color and color_col:
                    title += f", Color: {color_col})"
                else:
                    title += ")"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_radar_chart(self):
        """Create radar chart visualization"""
        st.subheader("Radar Chart")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 3:
            st.info("Radar charts require at least three numeric columns.")
            return
        
        # User selections
        value_cols = st.multiselect(
            "Select numeric columns (metrics)",
            num_cols,
            default=num_cols[:min(5, len(num_cols))]
        )
        
        if len(value_cols) < 3:
            st.info("Please select at least three metrics for a meaningful radar chart.")
            return
        
        # Group by options
        use_groups = st.checkbox("Group by category", value=True)
        group_col = None
        
        if use_groups:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                group_col = st.selectbox("Select grouping column", cat_cols)
                # Get top categories if there are many
                top_n = st.slider("Show top N categories", min_value=1, max_value=10, value=min(5, self.df[group_col].nunique()))
                top_categories = self.df[group_col].value_counts().head(top_n).index.tolist()
            else:
                st.warning("No categorical columns available for grouping.")
                use_groups = False
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Create radar chart
            fig = go.Figure()
            
            # Add traces
            if use_groups and group_col:
                # Each group gets its own trace
                for category in top_categories:
                    # Get data for this category
                    cat_data = self.df[self.df[group_col] == category]
                    
                    # Calculate mean for each metric
                    values = [cat_data[col].mean() for col in value_cols]
                    
                    # Add a trace
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=value_cols,
                        fill='toself',
                        name=str(category)
                    ))
            else:
                # Single trace with overall means
                values = [self.df[col].mean() for col in value_cols]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=value_cols,
                    fill='toself',
                    name='Overall'
                ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1.2 * max([self.df[col].max() for col in value_cols])]
                    )
                ),
                showlegend=True,
                title=custom_title or f"Radar Chart of {', '.join(value_cols)}",
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="radar_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="radar_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or "Radar Chart"
                if use_groups and group_col:
                    title += f" by {group_col}"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_distribution_plot(self):
        """Create distribution plot visualization"""
        st.subheader("Distribution Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Distribution plots require at least one numeric column.")
            return
        
        # User selections
        selected_col = st.selectbox("Select column", num_cols)
        
        # Plot options
        show_hist = st.checkbox("Show histogram", value=True)
        show_kde = st.checkbox("Show KDE (density curve)", value=True)
        show_rug = st.checkbox("Show rug plot", value=True)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Create distribution plot with plotly
            fig = go.Figure()
            
            # Add histogram if requested
            if show_hist:
                fig.add_trace(go.Histogram(
                    x=self.df[selected_col],
                    name="Histogram",
                    opacity=0.7,
                    histnorm="probability density"
                ))
            
            # Add KDE if requested
            if show_kde:
                from scipy import stats
                kde = self.df[selected_col].dropna()
                
                # Skip if no data
                if len(kde) > 1:  # Need at least 2 points for KDE
                    x_range = np.linspace(kde.min(), kde.max(), 200)
                    kde_values = stats.gaussian_kde(kde)(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name="Density",
                        line=dict(width=2, color='rgba(255, 100, 102, 0.8)')
                    ))
            
            # Add rug plot if requested
            if show_rug:
                fig.add_trace(go.Scatter(
                    x=self.df[selected_col],
                    y=[0] * len(self.df),
                    mode='markers',
                    marker=dict(
                        symbol='line-ns',
                        color='rgba(0, 0, 0, 0.3)',
                        line=dict(width=1),
                        size=5
                    ),
                    name="Rug Plot"
                ))
            
            # Update layout
            fig.update_layout(
                title=custom_title or f"Distribution of {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Density",
                template="plotly_white",
                barmode='overlay'
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader(f"Statistics for {selected_col}")
            
            # Calculate statistics
            stats_df = self.df[selected_col].describe().reset_index()
            stats_df.columns = ['Statistic', 'Value']
            
            # Add skewness and kurtosis
            from scipy import stats as spstats
            skewness = spstats.skew(self.df[selected_col].dropna())
            kurtosis = spstats.kurtosis(self.df[selected_col].dropna())
            
            # Add to stats dataframe
            stats_df = pd.concat([stats_df, pd.DataFrame({
                'Statistic': ['skewness', 'kurtosis'],
                'Value': [skewness, kurtosis]
            })])
            
            # Display statistics
            st.dataframe(stats_df, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="distribution_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="distribution_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Distribution Plot of {selected_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_qq_plot(self):
        """Create Q-Q plot visualization"""
        st.subheader("Q-Q Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Q-Q plots require at least one numeric column.")
            return
        
        # User selections
        selected_col = st.selectbox("Select column", num_cols)
        
        # Distribution options
        dist_options = ["normal", "t", "chi2", "exponential", "uniform"]
        selected_dist = st.selectbox("Reference distribution", dist_options)
        
        # Distribution parameters (if applicable)
        if selected_dist == "t":
            df_param = st.slider("Degrees of freedom", min_value=1, max_value=30, value=5)
        elif selected_dist == "chi2":
            df_param = st.slider("Degrees of freedom", min_value=1, max_value=30, value=5)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Prepare data
            data = self.df[selected_col].dropna()
            
            if len(data) < 2:
                st.warning("Not enough data points for Q-Q plot.")
                return
            
            # Create Q-Q plot
            from scipy import stats
            
            # Determine the distribution
            if selected_dist == "normal":
                theoretical_dist = stats.norm
                dist_name = "Normal"
                dist_params = stats.norm.fit(data)
                
            elif selected_dist == "t":
                theoretical_dist = stats.t
                dist_name = f"t ({df_param} df)"
                # For t-distribution, fit only location and scale, df is fixed
                dist_params = (df_param, *stats.t.fit(data, df_param, floc=0)[1:])
                
            elif selected_dist == "chi2":
                theoretical_dist = stats.chi2
                dist_name = f"Chi-square ({df_param} df)"
                # For chi2, fit only scale, df is fixed
                dist_params = (df_param, 0, stats.chi2.fit(data, df_param, floc=0)[2])
                
            elif selected_dist == "exponential":
                theoretical_dist = stats.expon
                dist_name = "Exponential"
                dist_params = stats.expon.fit(data)
                
            elif selected_dist == "uniform":
                theoretical_dist = stats.uniform
                dist_name = "Uniform"
                dist_params = stats.uniform.fit(data)
            
            # Calculate quantiles for data and theoretical distribution
            quantiles = np.linspace(0.01, 0.99, 100)
            
            # Calculate ordered values from data
            empirical_quantiles = np.quantile(data, quantiles)
            
            # Calculate theoretical quantiles
            theoretical_quantiles = theoretical_dist.ppf(quantiles, *dist_params)
            
            # Create scatter plot of quantiles
            fig = go.Figure()
            
            # Add Q-Q points
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=empirical_quantiles,
                mode='markers',
                name='Q-Q Points',
                marker=dict(
                    color='rgba(31, 119, 180, 0.8)',
                    size=8
                )
            ))
            
            # Add reference line
            if selected_dist != "chi2" and selected_dist != "exponential":  # Reference line makes most sense for symmetric distributions
                min_val = min(min(theoretical_quantiles), min(empirical_quantiles))
                max_val = max(max(theoretical_quantiles), max(empirical_quantiles))
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Reference Line',
                    line=dict(
                        color='rgba(255, 0, 0, 0.8)',
                        width=2,
                        dash='dash'
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title=custom_title or f"Q-Q Plot: {selected_col} vs {dist_name} Distribution",
                xaxis_title=f"Theoretical Quantiles ({dist_name})",
                yaxis_title=f"Sample Quantiles ({selected_col})",
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Run normality test if normal distribution
            if selected_dist == "normal":
                st.subheader("Normality Tests")
                
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data)
                
                # Anderson-Darling test
                anderson_result = stats.anderson(data, 'norm')
                anderson_stat = anderson_result.statistic
                anderson_crit = anderson_result.critical_values[2]  # 5% significance level
                anderson_p = "< 0.05" if anderson_stat > anderson_crit else "> 0.05"
                
                # Display test results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Shapiro-Wilk Test", f"p = {shapiro_p:.4f}")
                    if shapiro_p < 0.05:
                        st.info("Data likely does not follow a normal distribution (p < 0.05)")
                    else:
                        st.info("No evidence to suggest data does not follow a normal distribution")
                
                with col2:
                    st.metric("Anderson-Darling Test", f"Statistic = {anderson_stat:.4f}")
                    if anderson_stat > anderson_crit:
                        st.info("Data likely does not follow a normal distribution (statistic > critical value)")
                    else:
                        st.info("No evidence to suggest data does not follow a normal distribution")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="qq_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="qq_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Q-Q Plot: {selected_col} vs {dist_name} Distribution",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_ecdf_plot(self):
        """Create Empirical Cumulative Distribution Function (ECDF) plot"""
        st.subheader("ECDF Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("ECDF plots require at least one numeric column.")
            return
        
        # User selections
        selected_cols = st.multiselect(
            "Select columns", 
            num_cols,
            default=[num_cols[0]] if num_cols else []
        )
        
        if not selected_cols:
            st.info("Please select at least one column.")
            return
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Create plot
            fig = go.Figure()
            
            # Define colors for multiple lines
            colors = px.colors.qualitative.Plotly
            
            # Add ECDF for each selected column
            for i, col in enumerate(selected_cols):
                # Get data and sort
                data = self.df[col].dropna().sort_values()
                
                # Calculate ECDF values
                ecdf_y = np.arange(1, len(data) + 1) / len(data)
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=data,
                    y=ecdf_y,
                    mode='lines',
                    name=col,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title=custom_title or f"ECDF Plot for {', '.join(selected_cols)}",
                xaxis_title="Value",
                yaxis_title="Cumulative Probability",
                yaxis=dict(
                    tickformat=".0%",  # Format as percentage
                    range=[0, 1.05]
                ),
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="ecdf_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="ecdf_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"ECDF Plot for {', '.join(selected_cols)}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_residual_plot(self):
        """Create residual plot visualization"""
        st.subheader("Residual Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Residual plots require at least two numeric columns.")
            return
        
        # User selections
        x_col = st.selectbox("Select independent variable (X)", num_cols)
        y_col = st.selectbox("Select dependent variable (Y)", [col for col in num_cols if col != x_col])
        
        # Model type options
        model_type = st.radio(
            "Regression model type",
            ["Linear", "Polynomial", "Lowess (Locally Weighted)"],
            horizontal=True
        )
        
        # Additional options based on model type
        if model_type == "Polynomial":
            degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=2)
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Prepare data
            df_clean = self.df[[x_col, y_col]].dropna()
            
            if len(df_clean) < 3:
                st.warning("Not enough data points for residual plot.")
                return
            
            X = df_clean[x_col].values
            y = df_clean[y_col].values
            
            # Fit model and calculate residuals
            if model_type == "Linear":
                # Fit linear regression
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                
                # Calculate predictions and residuals
                y_pred = slope * X + intercept
                residuals = y - y_pred
                
                # Model info for display
                model_info = f"y = {slope:.4f}x + {intercept:.4f}"
                r_squared = r_value**2
                
            elif model_type == "Polynomial":
                # Fit polynomial regression
                coeffs = np.polyfit(X, y, degree)
                poly_model = np.poly1d(coeffs)
                
                # Calculate predictions and residuals
                y_pred = poly_model(X)
                residuals = y - y_pred
                
                # Model info for display
                model_info = f"Polynomial (degree {degree})"
                r_squared = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
                
            elif model_type == "Lowess":
                # Fit lowess regression
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess(y, X, frac=0.3)
                
                # Extract predictions
                x_lowess, y_pred = lowess[:, 0], lowess[:, 1]
                
                # Interpolate to get predictions for all X values
                from scipy.interpolate import interp1d
                f = interp1d(x_lowess, y_pred, bounds_error=False, fill_value="extrapolate")
                y_pred = f(X)
                
                # Calculate residuals
                residuals = y - y_pred
                
                # Model info for display
                model_info = "Lowess (Locally Weighted Scatterplot Smoothing)"
                r_squared = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=("Regression Plot", "Residual Plot"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Add scatter plot for original data
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        color='rgba(31, 119, 180, 0.6)',
                        size=8
                    )
                ),
                row=1, col=1
            )
            
            # Add regression line
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=y_pred,
                    mode='lines',
                    name='Regression Line',
                    line=dict(
                        color='rgba(255, 0, 0, 0.8)',
                        width=2
                    )
                ),
                row=1, col=1
            )
            
            # Add residual scatter plot
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(
                        color='rgba(44, 160, 44, 0.6)',
                        size=8
                    )
                ),
                row=2, col=1
            )
            
            # Add zero line to residual plot
            fig.add_trace(
                go.Scatter(
                    x=[min(X), max(X)],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(
                        color='rgba(255, 0, 0, 0.8)',
                        width=2,
                        dash='dash'
                    )
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=custom_title or f"Residual Plot: {y_col} vs {x_col} ({model_type} Regression)",
                height=700,
                template="plotly_white",
                showlegend=True
            )
            
            # Update x and y axis labels
            fig.update_xaxes(title_text=x_col, row=1, col=1)
            fig.update_yaxes(title_text=y_col, row=1, col=1)
            fig.update_xaxes(title_text=x_col, row=2, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display regression statistics
            st.subheader("Regression Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R-squared", f"{r_squared:.4f}")
            
            with col2:
                # Mean and standard deviation of residuals
                mean_resid = np.mean(residuals)
                st.metric("Mean of Residuals", f"{mean_resid:.4f}")
            
            with col3:
                std_resid = np.std(residuals)
                st.metric("Std Dev of Residuals", f"{std_resid:.4f}")
            
            # Display model equation or information
            st.info(f"Model: {model_info}")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="residual_plot.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="residual_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                viz_data = {
                    "title": custom_title or f"Residual Plot: {y_col} vs {x_col} ({model_type} Regression)",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
    def _create_correlation_matrix(self):
        """Create correlation matrix visualization"""
        st.subheader("Correlation Matrix")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Correlation matrices require at least two numeric columns.")
            return
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns for correlation", 
            num_cols,
            default=num_cols[:min(len(num_cols), 10)]  # Default to first 10 columns or fewer
        )
        
        if not selected_cols or len(selected_cols) < 2:
            st.info("Please select at least two columns.")
            return
        
        # Correlation method
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"]
        )
        
        # Display options
        viz_type = st.radio(
            "Visualization type",
            ["Heatmap", "Network"],
            horizontal=True
        )
        
        # Additional options
        if viz_type == "Heatmap":
            colorscale = st.selectbox(
                "Color scale",
                ["RdBu_r", "Viridis", "Plasma", "Cividis", "Spectral"]
            )
            show_values = st.checkbox("Show correlation values", value=True)
        else:  # Network
            threshold = st.slider(
                "Correlation threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Only show connections with absolute correlation above this threshold"
            )
        
        # Chart title
        custom_title = st.text_input("Chart title (optional)", "")
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Calculate correlation matrix
            corr_matrix = self.df[selected_cols].corr(method=corr_method)
            
            if viz_type == "Heatmap":
                # Create heatmap with plotly
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f" if show_values else False,
                    color_continuous_scale=colorscale,
                    title=custom_title or f"{corr_method.capitalize()} Correlation Matrix",
                    template="plotly_white",
                    aspect="auto"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title=""
                )
            
            else:  # Network visualization
                # Create network graph from correlation matrix
                G = nx.Graph()
                
                # Add nodes
                for col in selected_cols:
                    G.add_node(col)
                
                # Add edges based on correlations above threshold
                for i, col1 in enumerate(selected_cols):
                    for col2 in selected_cols[i+1:]:
                        corr_val = abs(corr_matrix.loc[col1, col2])
                        if corr_val > threshold:
                            G.add_edge(col1, col2, weight=corr_val)
                
                # Calculate positions using force-directed layout
                pos = nx.spring_layout(G)
                
                # Create edges
                edge_x = []
                edge_y = []
                edge_traces = []
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = G.edges[edge]['weight']
                    
                    # Create edge trace with width proportional to correlation
                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(
                            width=weight * 5,
                            color=f'rgba(0, 0, 255, {weight})'
                        ),
                        hoverinfo='text',
                        text=f"{edge[0]}-{edge[1]}: {weight:.2f}",
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)
                
                # Create nodes
                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=list(G.nodes()),
                    textposition='top center',
                    marker=dict(
                        size=15,
                        color='rgba(255, 0, 0, 0.8)'
                    ),
                    hoverinfo='text',
                    showlegend=False
                )
                
                # Create figure
                fig = go.Figure()
                
                # Add edges
                for edge_trace in edge_traces:
                    fig.add_trace(edge_trace)
                
                # Add nodes
                fig.add_trace(node_trace)
                
                # Update layout
                fig.update_layout(
                    title=custom_title or f"{corr_method.capitalize()} Correlation Network (threshold: {threshold})",
                    showlegend=False,
                    height=600,
                    template="plotly_white"
                )
                
                # Remove axis ticks and labels
                fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            
            # Display the visualization
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Create a download link for the plot as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download as HTML",
                    data=html_bytes,
                    file_name="correlation_matrix.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Create a download link for the plot as PNG
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="correlation_matrix.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Save visualization button
            if st.button("Save Visualization", use_container_width=True):
                title = custom_title or f"{corr_method.capitalize()} Correlation Matrix"
                if viz_type == "Network":
                    title = custom_title or f"{corr_method.capitalize()} Correlation Network"
                    
                viz_data = {
                    "title": title,
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")