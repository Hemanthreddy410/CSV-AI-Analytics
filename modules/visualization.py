import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import math

class EnhancedVisualizer:
    """Class for creating data visualizations"""
    
    def __init__(self, df=None):
        """Initialize with dataframe reference (optional)"""
        # We'll rely on session state for the most up-to-date data
        # The df parameter is kept for backward compatibility
        
        # Store visualizations in session state if not already there
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = []
    
    def render_interface(self):
        """Render visualization interface"""
        st.header("Visualization Studio")
        
        # Always use the most current dataframe from session state
        if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
            st.info("No data available. Please upload a dataset or complete data processing first.")
            return
            
        # Get the current dataframe from session state
        self.df = st.session_state.df
        
        # Display basic info about the current dataframe
        with st.expander("Current Data Overview", expanded=False):
            st.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
            st.dataframe(self.df.head(5), use_container_width=True)
            
            # Create a custom dataframe info display
            info_df = pd.DataFrame({
                'Column': self.df.columns,
                'Type': self.df.dtypes.astype(str),
                'Non-Null Count': self.df.count().values,
                'Null Count': self.df.isna().sum().values,
                'Null %': (self.df.isna().sum().values / len(self.df) * 100).round(2),
                'Unique Values': [self.df[col].nunique() for col in self.df.columns]
            })
            st.dataframe(info_df, use_container_width=True)
        
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
        try:
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
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("This could be due to incompatible data types or missing values. Try another visualization type or check your data.")
        
        # Display saved visualizations
        if st.session_state.visualizations:
            st.header("Saved Visualizations")
            
            # Add clear visualizations button
            if st.button("Clear All Saved Visualizations"):
                st.session_state.visualizations = []
                st.success("All visualizations cleared!")
                st.rerun()
            
            # Create columns based on number of visualizations (max 2 per row for better visibility)
            num_cols = min(2, len(st.session_state.visualizations))
            
            # Use st.columns with list comprehension for better clarity
            cols = st.columns(num_cols)
            
            # Display visualizations in columns
            for i, viz in enumerate(st.session_state.visualizations):
                with cols[i % num_cols]:
                    st.subheader(viz["title"])
                    if viz["type"] == "matplotlib":
                        st.pyplot(viz["figure"])
                    elif viz["type"] == "plotly":
                        st.plotly_chart(viz["figure"], use_container_width=True)
                    
                    # Add delete button for individual visualization
                    if st.button(f"Delete Visualization", key=f"delete_viz_{i}"):
                        st.session_state.visualizations.pop(i)
                        st.success(f"Visualization deleted!")
                        st.rerun()
    
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
        
        # Filter out the already selected x column
        y_options = [col for col in num_cols if col != x_col]
        if not y_options:
            st.info("Need at least two distinct numeric columns for scatter plot.")
            return
            
        y_col = st.selectbox("Select Y-axis", y_options)
        
        # Color options
        use_color = st.checkbox("Add color dimension")
        color_col = None
        
        if use_color:
            # Get all columns that aren't already used
            color_options = [col for col in self.df.columns if col not in [x_col, y_col]]
            if color_options:
                color_col = st.selectbox("Select color column", color_options)
            else:
                st.warning("No additional columns available for color.")
                use_color = False
        
        # Size options
        use_size = st.checkbox("Add size dimension")
        size_col = None
        
        if use_size:
            # Only offer numeric columns for size
            size_options = [col for col in num_cols if col not in [x_col, y_col]]
            if size_options:
                size_col = st.selectbox("Select size column", size_options)
            else:
                st.warning("No additional numeric columns available for size.")
                use_size = False
        
        # Trendline option
        add_trendline = st.checkbox("Add trendline")
        trendline_type = None
        if add_trendline:
            trendline_type = st.selectbox(
                "Trendline type", 
                ["ols", "lowess"], 
                help="OLS = linear regression, LOWESS = locally weighted smoothing"
            )
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Handle missing values to prevent visualization errors
            plot_df = self.df.dropna(subset=[x_col, y_col])
            
            if plot_df.empty:
                st.warning("No valid data points after removing missing values.")
                return
                
            # Build the scatter plot with various options
            if use_color and color_col:
                if use_size and size_col:
                    # Scatter plot with color and size
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        size=size_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline=trendline_type if add_trendline else None
                    )
                else:
                    # Scatter plot with color only
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline=trendline_type if add_trendline else None
                    )
            else:
                if use_size and size_col:
                    # Scatter plot with size only
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline=trendline_type if add_trendline else None
                    )
                else:
                    # Simple scatter plot
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        template="plotly_white",
                        trendline=trendline_type if add_trendline else None
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
            correlation = plot_df[[x_col, y_col]].corr().iloc[0, 1]
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
                template="plotly_white",
                aspect="auto"
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
        if not pd.api.types.is_numeric_dtype(self.df[z_col]):
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
            try:
                # Sort by x-axis if possible
                df_sorted = self.df.sort_values(x_col)
            except:
                # If we can't sort (e.g., strings or mixed types), use as-is
                df_sorted = self.df
            
            # Create plot
            fig = go.Figure()
            
            # Add lines for each y column
            for y_col in y_cols:
                # Skip rows with missing values in either x or y
                mask = ~(df_sorted[x_col].isna() | df_sorted[y_col].isna())
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[x_col][mask],
                        y=df_sorted[y_col][mask],
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
        
        # Additional options
        show_values = st.checkbox("Show values and percentages", value=True)
        hole_size = st.slider("Donut hole size (0 for pie chart)", 0.0, 0.8, 0.0, 0.1)
        
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
                template="plotly_white",
                hole=hole_size
            )
            
            # Update text display based on user choice
            if show_values:
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label+value'
                )
            else:
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
            try:
                # Sort by x-axis if possible
                df_sorted = self.df.sort_values(x_col)
            except:
                # If we can't sort (e.g., strings or mixed types), use as-is
                df_sorted = self.df
            
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
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
                
    # Grouped Bar Chart implementation
    def _create_grouped_bar_chart(self):
        """Create grouped bar chart visualization"""
        st.subheader("Grouped Bar Chart")
        
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for values
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or len(cat_cols) < 2:
            st.info("Grouped bar charts require at least two categorical columns.")
            return
            
        if not num_cols:
            st.info("Grouped bar charts require at least one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis (main categories)", cat_cols)
        color_col = st.selectbox("Select grouping column", [col for col in cat_cols if col != x_col])
        y_col = st.selectbox("Select Y-axis (numeric)", num_cols)
        
        # Aggregation function
        agg_func = st.selectbox(
            "Select aggregation function", 
            ["Mean", "Sum", "Count", "Median", "Min", "Max"]
        )
        
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
            
            # Prepare data with aggregation
            df_grouped = self.df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create grouped bar chart
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{agg_func} of {y_col} by {x_col} and {color_col}",
                barmode="group",  # This creates the grouped bars
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
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Grouped Bar Chart: {agg_func} of {y_col} by {x_col} and {color_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Stacked Bar Chart implementation
    def _create_stacked_bar_chart(self):
        """Create stacked bar chart visualization"""
        st.subheader("Stacked Bar Chart")
        
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for values
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or len(cat_cols) < 2:
            st.info("Stacked bar charts require at least two categorical columns.")
            return
            
        if not num_cols:
            st.info("Stacked bar charts require at least one numeric column.")
            return
        
        # User selections
        x_col = st.selectbox("Select X-axis (main categories)", cat_cols)
        color_col = st.selectbox("Select stacking column", [col for col in cat_cols if col != x_col])
        y_col = st.selectbox("Select Y-axis (numeric)", num_cols)
        
        # Aggregation function
        agg_func = st.selectbox(
            "Select aggregation function", 
            ["Sum", "Mean", "Count", "Median", "Min", "Max"]
        )
        
        # Normalize option
        normalize = st.checkbox("Normalize to 100% (percentage)")
        
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
            
            # Prepare data with aggregation
            df_grouped = self.df.groupby([x_col, color_col])[y_col].agg(agg_map[agg_func]).reset_index()
            
            # Create stacked bar chart
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{agg_func} of {y_col} by {x_col} and {color_col}",
                barmode="stack" if not normalize else "relative",  # This creates the stacked bars
                template="plotly_white"
            )
            
            # Update layout
            y_axis_title = f"{agg_func} of {y_col}" if not normalize else "Percentage (%)"
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_axis_title,
                legend_title=color_col
            )
            
            # Update y-axis format for percentage
            if normalize:
                fig.update_layout(yaxis=dict(tickformat='.0%'))
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                title_prefix = "Stacked Bar Chart" if not normalize else "100% Stacked Bar Chart"
                viz_data = {
                    "title": f"{title_prefix}: {agg_func} of {y_col} by {x_col} and {color_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Bubble Chart implementation
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
        
        # Filter out the already selected columns
        y_options = [col for col in num_cols if col != x_col]
        y_col = st.selectbox("Select Y-axis", y_options)
        
        # Options for bubble size
        size_options = [col for col in num_cols if col not in [x_col, y_col]]
        size_col = st.selectbox("Select bubble size", size_options)
        
        # Color options
        use_color = st.checkbox("Add color dimension")
        color_col = None
        
        if use_color:
            # All columns can be used for color
            color_options = [col for col in self.df.columns if col not in [x_col, y_col, size_col]]
            if color_options:
                color_col = st.selectbox("Select color column", color_options)
            else:
                st.warning("No additional columns available for color.")
                use_color = False
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Handle missing values
            plot_df = self.df.dropna(subset=[x_col, y_col, size_col])
            
            if plot_df.empty:
                st.warning("No valid data points after removing missing values.")
                return
                
            # Create bubble chart
            if use_color and color_col:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    color=color_col,
                    title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})",
                    template="plotly_white",
                    size_max=40  # Maximum bubble size
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})",
                    template="plotly_white",
                    size_max=40  # Maximum bubble size
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if use_color and color_col else ""
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Radar Chart implementation
    def _create_radar_chart(self):
        """Create radar chart visualization"""
        st.subheader("Radar Chart")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 3:
            st.info("Radar charts work best with at least three numeric metrics.")
            return
        
        # Get categorical columns for grouping
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            st.info("Radar charts require a categorical column to define the groups.")
            return
        
        # User selections
        group_col = st.selectbox("Select grouping column", cat_cols)
        
        # Select metrics (numeric columns)
        metrics = st.multiselect("Select metrics (3+ recommended)", num_cols, default=num_cols[:min(5, len(num_cols))])
        
        if len(metrics) < 3:
            st.warning("Radar charts work best with at least 3 metrics.")
        
        # Limit number of groups
        top_n_groups = st.slider("Show top N groups", 2, 10, 5)
        
        # Scaling option
        scale_values = st.checkbox("Scale values (0-1)", value=True, 
                               help="Scale each metric between 0 and 1 for better comparison")
        
        # Create the chart
        st.subheader("Preview")
        
        if not metrics:
            st.info("Please select at least one metric.")
            return
        
        with st.spinner("Creating visualization..."):
            # Get top N groups by frequency
            top_groups = self.df[group_col].value_counts().head(top_n_groups).index.tolist()
            
            # Filter data to include only top groups
            filtered_df = self.df[self.df[group_col].isin(top_groups)]
            
            if filtered_df.empty:
                st.warning("No data available for the selected groups.")
                return
            
            # Aggregate data by group
            agg_df = filtered_df.groupby(group_col)[metrics].mean().reset_index()
            
            # Scale values if requested
            if scale_values:
                for metric in metrics:
                    min_val = agg_df[metric].min()
                    max_val = agg_df[metric].max()
                    if max_val > min_val:  # Avoid division by zero
                        agg_df[metric] = (agg_df[metric] - min_val) / (max_val - min_val)
            
            # Create radar chart
            fig = go.Figure()
            
            # Add traces for each group
            for i, group in enumerate(agg_df[group_col]):
                values = agg_df[agg_df[group_col] == group][metrics].values.flatten().tolist()
                # Add the first value again to close the loop
                values.append(values[0])
                
                # Create axis list
                theta = metrics + [metrics[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=theta,
                    fill='toself',
                    name=str(group)
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Radar Chart: Comparison of {group_col} across {len(metrics)} metrics",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1] if scale_values else None
                    )
                ),
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Radar Chart: Comparison of {group_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Distribution Plot implementation
    def _create_distribution_plot(self):
        """Create distribution plot visualization"""
        st.subheader("Distribution Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Distribution plots require at least one numeric column.")
            return
        
        # User selections
        cols = st.multiselect("Select columns to analyze", num_cols)
        
        if not cols:
            st.info("Please select at least one column.")
            return
        
        # Plot type
        plot_type = st.selectbox(
            "Plot type",
            ["Histogram + KDE", "Box Plot", "Violin Plot", "ECDF (Empirical Cumulative Distribution)"]
        )
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            if plot_type == "Histogram + KDE":
                # Create a distribution plot with histogram and KDE
                if len(cols) == 1:
                    # Single column
                    fig = px.histogram(
                        self.df,
                        x=cols[0],
                        marginal="kde",
                        title=f"Distribution of {cols[0]}",
                        template="plotly_white"
                    )
                else:
                    # Multiple columns
                    fig = go.Figure()
                    
                    for col in cols:
                        # Add histogram trace
                        fig.add_trace(
                            go.Histogram(
                                x=self.df[col],
                                name=col,
                                opacity=0.75,
                                histnorm="probability density"
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Distribution of {', '.join(cols)}",
                        xaxis_title="Value",
                        yaxis_title="Density",
                        barmode="overlay",
                        template="plotly_white"
                    )
            
            elif plot_type == "Box Plot":
                # Create box plots for each column
                plot_df = pd.melt(self.df[cols], var_name="Column", value_name="Value")
                
                fig = px.box(
                    plot_df,
                    x="Column",
                    y="Value",
                    title=f"Distribution Comparison for {len(cols)} Columns",
                    template="plotly_white"
                )
            
            elif plot_type == "Violin Plot":
                # Create violin plots for each column
                plot_df = pd.melt(self.df[cols], var_name="Column", value_name="Value")
                
                fig = px.violin(
                    plot_df,
                    x="Column",
                    y="Value",
                    box=True,
                    points="all",
                    title=f"Distribution Comparison for {len(cols)} Columns",
                    template="plotly_white"
                )
            
            elif plot_type == "ECDF (Empirical Cumulative Distribution)":
                # Create ECDF plot
                fig = go.Figure()
                
                for col in cols:
                    # Sort data for ECDF
                    sorted_data = np.sort(self.df[col].dropna())
                    n = len(sorted_data)
                    y = np.arange(1, n+1) / n  # ECDF values (0 to 1)
                    
                    # Add ECDF trace
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_data,
                            y=y,
                            mode='lines',
                            name=col
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Empirical Cumulative Distribution Function",
                    xaxis_title="Value",
                    yaxis_title="Cumulative Probability",
                    template="plotly_white"
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"{plot_type} for {', '.join(cols)}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Q-Q Plot implementation
    def _create_qq_plot(self):
        """Create Q-Q plot visualization"""
        st.subheader("Q-Q Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("Q-Q plots require at least one numeric column.")
            return
        
        # User selections
        col = st.selectbox("Select column", num_cols)
        
        # Distribution to compare against
        distribution = st.selectbox(
            "Compare against distribution",
            ["Normal", "Student's t", "Chi-squared", "Exponential"]
        )
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Import required libraries
            from scipy import stats
            
            # Clean data (remove NaN values)
            clean_data = self.df[col].dropna()
            
            if len(clean_data) < 3:
                st.warning("Not enough valid data points for a Q-Q plot.")
                return
            
            # Standardize the data for comparison
            z_data = (clean_data - clean_data.mean()) / clean_data.std()
            
            # Calculate theoretical quantiles based on selected distribution
            if distribution == "Normal":
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_data)))
                dist_name = "Normal Distribution"
            elif distribution == "Student's t":
                dof = st.slider("Degrees of freedom", 1, 30, 5)
                theoretical_quantiles = stats.t.ppf(np.linspace(0.01, 0.99, len(clean_data)), dof)
                dist_name = f"Student's t Distribution (dof={dof})"
            elif distribution == "Chi-squared":
                dof = st.slider("Degrees of freedom", 1, 30, 2)
                theoretical_quantiles = stats.chi2.ppf(np.linspace(0.01, 0.99, len(clean_data)), dof)
                dist_name = f"Chi-squared Distribution (dof={dof})"
            else:  # Exponential
                theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(clean_data)))
                dist_name = "Exponential Distribution"
            
            # Sort both sets of quantiles
            sorted_data = np.sort(z_data)
            sorted_theory = np.sort(theoretical_quantiles)
            
            # Create QQ plot
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=sorted_theory,
                    y=sorted_data,
                    mode='markers',
                    name='Data Points'
                )
            )
            
            # Add reference line
            min_val = min(sorted_theory.min(), sorted_data.min())
            max_val = max(sorted_theory.max(), sorted_data.max())
            
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
                title=f"Q-Q Plot: {col} vs {dist_name}",
                xaxis_title=f"Theoretical Quantiles ({dist_name})",
                yaxis_title=f"Standardized Sample Quantiles ({col})",
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add normality test results
            st.subheader("Statistical Test")
            
            # Perform statistical test based on selected distribution
            if distribution == "Normal":
                stat, p_value = stats.normaltest(clean_data)
                test_name = "D'Agostino's K^2 Test for Normality"
            elif distribution == "Student's t":
                # For t-distribution, just check skewness as an approximation
                skewness = stats.skew(clean_data)
                stat, p_value = skewness, None
                test_name = "Skewness Test (approximation)"
            elif distribution == "Chi-squared":
                # For chi-squared, test if all values are positive and check skewness
                if (clean_data < 0).any():
                    stat, p_value = np.nan, 0
                else:
                    skewness = stats.skew(clean_data)
                    stat, p_value = skewness, None
                test_name = "Positivity and Skewness Check (approximation)"
            else:  # Exponential
                # For exponential, test if mean ≈ std (property of exponential distribution)
                mean_std_ratio = clean_data.mean() / clean_data.std()
                stat, p_value = mean_std_ratio, None
                test_name = "Mean/Std Ratio Test (approximation)"
            
            # Display test results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test Statistic", f"{stat:.4f}")
            
            with col2:
                if p_value is not None:
                    st.metric("p-value", f"{p_value:.4f}")
                    st.write(f"Null hypothesis: Data follows a {distribution.lower()} distribution")
                    if p_value < 0.05:
                        st.error("Conclusion: The data likely does NOT follow this distribution (p < 0.05)")
                    else:
                        st.success("Conclusion: No evidence against this distribution (p >= 0.05)")
                else:
                    st.write(f"Test: {test_name}")
                    if distribution == "Student's t":
                        st.write(f"Skewness: {stat:.4f} (Should be close to 0 for t-distribution)")
                    elif distribution == "Chi-squared":
                        st.write(f"Skewness: {stat:.4f} (Should be positive for chi-squared)")
                    else:  # Exponential
                        st.write(f"Mean/Std Ratio: {stat:.4f} (Should be close to 1 for exponential)")
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Q-Q Plot: {col} vs {dist_name}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # ECDF Plot implementation  
    def _create_ecdf_plot(self):
        """Create Empirical Cumulative Distribution Function (ECDF) plot"""
        st.subheader("ECDF Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.info("ECDF plots require at least one numeric column.")
            return
        
        # User selections
        cols = st.multiselect("Select columns to analyze", num_cols)
        
        if not cols:
            st.info("Please select at least one column.")
            return
        
        # Plot options
        show_points = st.checkbox("Show individual points", value=False)
        compare_to_normal = st.checkbox("Compare to Normal distribution", value=True)
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Create ECDF plot
            fig = go.Figure()
            
            for col in cols:
                # Remove NaN values
                clean_data = self.df[col].dropna()
                
                if len(clean_data) == 0:
                    continue
                
                # Sort data for ECDF
                sorted_data = np.sort(clean_data)
                n = len(sorted_data)
                y = np.arange(1, n+1) / n  # ECDF values (0 to 1)
                
                # Add ECDF trace
                fig.add_trace(
                    go.Scatter(
                        x=sorted_data,
                        y=y,
                        mode='lines' if not show_points else 'lines+markers',
                        name=col,
                        line=dict(width=2)
                    )
                )
                
                # Add normal distribution comparison if requested
                if compare_to_normal:
                    # Calculate mean and std
                    mean = clean_data.mean()
                    std = clean_data.std()
                    
                    # Generate points for normal CDF
                    x_range = np.linspace(
                        max(sorted_data.min() - 3*std, sorted_data.min()),
                        min(sorted_data.max() + 3*std, sorted_data.max()),
                        100
                    )
                    
                    # Calculate normal CDF
                    from scipy import stats
                    normal_cdf = stats.norm.cdf(x_range, loc=mean, scale=std)
                    
                    # Add normal CDF trace
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=normal_cdf,
                            mode='lines',
                            line=dict(dash='dash'),
                            name=f'{col} (Normal)',
                            opacity=0.7
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=f"Empirical Cumulative Distribution Function",
                xaxis_title="Value",
                yaxis_title="Cumulative Probability",
                template="plotly_white"
            )
            
            # Set y-axis range from 0 to 1
            fig.update_yaxes(range=[0, 1.05])
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"ECDF Plot for {', '.join(cols)}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")
        
    # Residual Plot implementation
    def _create_residual_plot(self):
        """Create residual plot visualization"""
        st.subheader("Residual Plot")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Residual plots require at least two numeric columns (predictor and response).")
            return
        
        # User selections
        x_col = st.selectbox("Select predictor variable (X)", num_cols)
        
        # Remove selected X variable from options for Y
        y_options = [col for col in num_cols if col != x_col]
        y_col = st.selectbox("Select response variable (Y)", y_options)
        
        # Create the chart
        st.subheader("Preview")
        
        with st.spinner("Creating visualization..."):
            # Handle missing values
            plot_df = self.df[[x_col, y_col]].dropna()
            
            if len(plot_df) < 2:
                st.warning("Not enough valid data points after removing missing values.")
                return
            
            # Fit linear regression model
            from scipy import stats
            
            # Extract data
            x_data = plot_df[x_col].values
            y_data = plot_df[y_col].values
            
            # Calculate regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            
            # Calculate fitted values and residuals
            fitted = intercept + slope * x_data
            residuals = y_data - fitted
            
            # Create a figure with subplots (2 rows, 1 column)
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=("Linear Regression", "Residuals"),
                shared_xaxes=True,
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Add scatter plot with regression line to first subplot
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name='Data Points',
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Add regression line
            x_range = np.linspace(min(x_data), max(x_data), 100)
            y_range = intercept + slope * x_range
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'Regression Line (y = {intercept:.2f} + {slope:.2f}x)',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Add residuals to second subplot
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(size=8, color='green')
                ),
                row=2, col=1
            )
            
            # Add zero line to residual plot
            fig.add_trace(
                go.Scatter(
                    x=[min(x_data), max(x_data)],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Residual Plot: {y_col} vs {x_col}",
                template="plotly_white",
                height=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes labels
            fig.update_xaxes(title_text=x_col, row=2, col=1)
            fig.update_yaxes(title_text=y_col, row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display regression statistics
            st.subheader("Regression Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R-squared", f"{r_value**2:.4f}")
                st.write(f"Proportion of variance explained")
            
            with col2:
                st.metric("Slope", f"{slope:.4f}")
                st.write(f"p-value: {p_value:.4f}")
            
            with col3:
                st.metric("Intercept", f"{intercept:.4f}")
                st.write(f"Std Error: {std_err:.4f}")
            
            # Check residual normality
            normality_stat, normality_p = stats.shapiro(residuals)
            
            st.subheader("Residual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean of Residuals", f"{np.mean(residuals):.4f}")
                st.write("Should be close to zero for unbiased model")
            
            with col2:
                st.metric("Shapiro-Wilk p-value", f"{normality_p:.4f}")
                if normality_p < 0.05:
                    st.write("Residuals may not be normally distributed (p < 0.05)")
                else:
                    st.write("Residuals appear normally distributed (p >= 0.05)")
            
            # Save visualization button
            if st.button("Save Visualization"):
                viz_data = {
                    "title": f"Residual Plot: {y_col} vs {x_col}",
                    "type": "plotly",
                    "figure": fig
                }
                st.session_state.visualizations.append(viz_data)
                st.success("Visualization saved!")