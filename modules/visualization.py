import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import math

class Visualizer:
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
        
        # Available visualization types
        viz_types = {
            "Basic": ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart", "Histogram"],
            "Statistical": ["Box Plot", "Violin Plot", "Scatter Plot", "Heatmap", "Density Plot"],
            "Multi-Variable": ["Grouped Bar Chart", "Stacked Bar Chart", "Bubble Chart", "Radar Chart"],
            "Distribution": ["Distribution Plot", "Q-Q Plot", "ECDF Plot", "Residual Plot"]
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
        elif viz_type == "Distribution Plot":
            self._create_distribution_plot()
        elif viz_type == "Q-Q Plot":
            self._create_qq_plot()
        elif viz_type == "ECDF Plot":
            self._create_ecdf_plot()
        elif viz_type == "Residual Plot":
            self._create_residual_plot()
        
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
                        title=f"Count by {x_col} and {color_col}",
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
                        title=f"Count by {x_col}",
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
                        title=f"{agg_func} of {y_col} by {x_col} and {color_col}",
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
                        title=f"{agg_func} of {y_col} by {x_col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Bar Chart: {agg_func} of {y_col} by {x_col}",
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
                    title=f"Violin Plot of {y_col} by {x_col}",
                    template="plotly_white"
                )
            else:
                # Create simple violin plot
                fig = px.violin(
                    self.df,
                    y=y_col,
                    box=show_box,
                    points="all",
                    title=f"Violin Plot of {y_col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Violin Plot of {y_col}" + (f" by {x_col}" if use_groups and x_col else ""),
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
                        title=f"Scatter Plot: {y_col} vs {x_col}",
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
                        title=f"Scatter Plot: {y_col} vs {x_col}",
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
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline="ols" if add_trendline else None
                    )
                else:
                    # Simple scatter plot
                    fig = px.scatter(
                        self.df,
                        x=x_col,
                        y=y_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Scatter Plot: {y_col} vs {x_col}",
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
                title=f"{corr_method.capitalize()} Correlation Heatmap",
                template="plotly_white"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="",
                yaxis_title=""
            )
            
            # Display the heatmap
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"{corr_method.capitalize()} Correlation Heatmap",
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
                title = f"Count of {x_col} by {y_col}"
                z_title = "Count"
            elif agg_type == "Percentage":
                # Percentage values (normalize by columns)
                crosstab = pd.crosstab(
                    self.df[y_col],
                    self.df[x_col],
                    normalize='columns'
                ) * 100
                title = f"Percentage of {x_col} by {y_col}"
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
                title = f"{agg_type} of {value_col} by {x_col} and {y_col}"
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
            
            # Save visualization button
            if st.button("Save Visualization"):
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
                    title=f"{agg_func} of {z_col} by {x_col} and {y_col}",
                    template="plotly_white"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                
                # Display the heatmap
                st.plotly_chart(fig, use_container_width=True)
                
                # Save visualization button
                if st.button("Save Visualization"):
                    viz_data = {
                        "title": f"{agg_func} of {z_col} by {x_col} and {y_col}",
                        "type": "plotly",
                        "figure": fig
                    }
                    st.session_state.visualizations.append(viz_data)
                    st.success("Visualization saved!")
                    
            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")
                st.info("Try different columns or aggregation function.")
    
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
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if len(cols) == 1 and use_groups and group_col:
                # Single column with grouping
                fig = px.violin(
                    self.df,
                    x=group_col,
                    y=cols[0],
                    color=group_col,
                    box=False,
                    points=False,
                    title=f"Density Plot of {cols[0]} by {group_col}",
                    template="plotly_white"
                )
            elif len(cols) == 1:
                # Single column without grouping (KDE plot)
                fig = px.histogram(
                    self.df,
                    x=cols[0],
                    marginal="kde",
                    histnorm="probability density",
                    title=f"Density Plot of {cols[0]}",
                    template="plotly_white"
                )
                
                # Hide the histogram
                fig.update_traces(
                    selector=dict(type="histogram"),
                    visible=False
                )
            else:
                # Multiple columns (overlay density plots)
                fig = go.Figure()
                
                for col in cols:
                    # Using kernel density estimation
                    kde = self.df[col].dropna()
                    
                    # Add density trace
                    fig.add_trace(
                        px.histogram(
                            self.df,
                            x=col,
                            histnorm="probability density",
                            marginal="kde"
                        ).data[1]  # Get just the KDE trace
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Density Plot: {', '.join(cols)}",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    template="plotly_white"
                )
                
                # Update legend
                fig.update_traces(
                    showlegend=True,
                    selector=dict(type="histogram")
                )
                
                # Set legend names
                for i, col in enumerate(cols):
                    fig.data[i].name = col
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Density Plot: {', '.join(cols)}" + 
                             (f" by {group_col}" if use_groups and group_col else ""),
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!").append(viz_data)
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
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Sort by x-axis
            df_sorted = self.df.sort_values(x_col)
            
            # Create plot
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
                title=f"Line Chart: {', '.join(y_cols)} by {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_cols[0] if len(y_cols) == 1 else "Value",
                legend_title="Series",
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Line Chart: {', '.join(y_cols)} by {x_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
    
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
                title=f"Distribution of {cat_col}" + (f" by {value_col}" if use_values and value_col else ""),
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Pie Chart: Distribution of {cat_col}",
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
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Sort by x-axis
            df_sorted = self.df.sort_values(x_col)
            
            # Determine the groupby mode
            if stack_mode == "None":
                # Individual area charts
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=f"Area Chart: {', '.join(y_cols)} by {x_col}",
                    labels={col: col for col in y_cols},
                    template="plotly_white"
                )
            elif stack_mode == "Stack":
                # Stacked area chart
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=f"Stacked Area Chart: {', '.join(y_cols)} by {x_col}",
                    labels={col: col for col in y_cols},
                    template="plotly_white"
                )
            else:
                # Normalized (100%) area chart
                fig = px.area(
                    df_sorted,
                    x=x_col,
                    y=y_cols,
                    title=f"Normalized Area Chart: {', '.join(y_cols)} by {x_col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Area Chart: {', '.join(y_cols)} by {x_col}",
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
                    title=f"Histogram of {col} by {group_col}",
                    template="plotly_white"
                )
            else:
                # Create simple histogram with plotly
                fig = px.histogram(
                    self.df,
                    x=col,
                    marginal="rug" if not density else "kde",
                    nbins=n_bins,
                    title=f"Histogram of {col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Histogram of {col}" + (f" by {group_col}" if use_groups and group_col else ""),
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
                    title=f"Box Plot of {y_col} by {x_col}",
                    template="plotly_white"
                )
            else:
                # Create simple box plot
                fig = px.box(
                    self.df,
                    y=y_col,
                    points="all" if show_points else "outliers",
                    title=f"Box Plot of {y_col}",
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Box Plot of {y_col}" + (f" by {x_col}" if use_groups and x_col else ""),
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations