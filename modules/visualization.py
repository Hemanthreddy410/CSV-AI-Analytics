import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import math
import datetime

class EnhancedVisualizer:
    """Enhanced class for creating and recommending data visualizations"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Store visualizations in session state if not already there
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = []
            
        # Store recently used visualization settings
        if 'recent_viz_settings' not in st.session_state:
            st.session_state.recent_viz_settings = {}
            
        # Store recommended visualizations
        if 'viz_recommendations' not in st.session_state:
            st.session_state.viz_recommendations = []
    
    def render_interface(self):
        """Render enhanced visualization interface"""
        st.header("Advanced Visualization Studio")
        
        # Check if dataframe is available
        if self.df is None or self.df.empty:
            st.warning("Please upload data to create visualizations.")
            return
        
        # Create tabs for different visualization modes
        viz_tabs = st.tabs([
            "📊 Chart Builder", 
            "🔍 Data Explorer", 
            "🧠 Smart Recommendations",
            "📚 My Visualizations"
        ])
        
        # Chart Builder Tab
        with viz_tabs[0]:
            self._render_chart_builder()
        
        # Data Explorer Tab 
        with viz_tabs[1]:
            self._render_data_explorer()
        
        # Smart Recommendations Tab
        with viz_tabs[2]:
            self._render_recommendations()
        
        # My Visualizations Tab
        with viz_tabs[3]:
            self._render_saved_visualizations()
            
    def _render_chart_builder(self):
        """Render the main chart builder interface"""
        st.subheader("Chart Builder")
        
        # Create two columns for chart selection and options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Available visualization categories
            viz_categories = {
                "Basic": ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart", "Histogram"],
                "Statistical": ["Box Plot", "Violin Plot", "Scatter Plot", "Heatmap", "Density Plot"],
                "Multi-Variable": ["Grouped Bar Chart", "Stacked Bar Chart", "Bubble Chart", "Radar Chart", "Parallel Coordinates"],
                "Distribution": ["Distribution Plot", "Q-Q Plot", "ECDF Plot", "Residual Plot"],
                "Advanced": ["3D Scatter", "Sunburst Chart", "Treemap", "Sankey Diagram", "Network Graph"]
            }
            
            # Select visualization category
            viz_category = st.radio("Chart Category", list(viz_categories.keys()))
            
            # Select visualization type
            viz_type = st.selectbox("Chart Type", viz_categories[viz_category])
            
            # Information about the selected chart type
            chart_info = self._get_chart_info(viz_type)
            
            with st.expander("About this chart", expanded=False):
                st.markdown(chart_info["description"])
                st.markdown("**Best used for:** " + chart_info["best_for"])
                st.markdown("**Required data:** " + chart_info["required_data"])
            
        with col2:
            # Display chart builder for selected visualization type
            self._display_chart_builder(viz_type)
    
    def _render_data_explorer(self):
        """Render interactive data explorer with quick visualizations"""
        st.subheader("Interactive Data Explorer")
        
        # Get column types
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create tabs for different exploration approaches
        explorer_tabs = st.tabs([
            "Single Variable", 
            "Two Variables", 
            "Time Series",
            "Correlations"
        ])
        
        # Single Variable Analysis
        with explorer_tabs[0]:
            st.subheader("Single Variable Analysis")
            
            # Column selection
            col = st.selectbox("Select a column to analyze", self.df.columns)
            
            # Determine column type and show appropriate visualizations
            if col in num_cols:
                # Numeric column analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statistics
                    stats = self.df[col].describe()
                    st.subheader("Statistics")
                    
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                        st.metric("Min", f"{stats['min']:.2f}")
                        st.metric("25%", f"{stats['25%']:.2f}")
                    with col1_2:
                        st.metric("Standard Deviation", f"{stats['std']:.2f}")
                        st.metric("Max", f"{stats['max']:.2f}")
                        st.metric("75%", f"{stats['75%']:.2f}")
                    
                    st.metric("Missing Values", f"{self.df[col].isna().sum()} ({self.df[col].isna().mean()*100:.1f}%)")
                
                with col2:
                    # Visualization options
                    viz_type = st.radio(
                        "Visualization type",
                        ["Histogram", "Box Plot", "Violin Plot"],
                        horizontal=True
                    )
                    
                    if viz_type == "Histogram":
                        fig = px.histogram(
                            self.df, 
                            x=col,
                            marginal="box",
                            template="plotly_white"
                        )
                    elif viz_type == "Box Plot":
                        fig = px.box(
                            self.df,
                            y=col,
                            template="plotly_white"
                        )
                    elif viz_type == "Violin Plot":
                        fig = px.violin(
                            self.df,
                            y=col,
                            box=True,
                            template="plotly_white"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical column analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statistics
                    value_counts = self.df[col].value_counts()
                    
                    st.subheader("Statistics")
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        st.metric("Unique Values", f"{self.df[col].nunique()}")
                        st.metric("Most Common", f"{value_counts.index[0]}")
                    with col1_2:
                        st.metric("Missing Values", f"{self.df[col].isna().sum()} ({self.df[col].isna().mean()*100:.1f}%)")
                        st.metric("Frequency", f"{value_counts.iloc[0]} ({value_counts.iloc[0]/len(self.df)*100:.1f}%)")
                    
                    # Show value counts table
                    st.subheader("Value Counts")
                    value_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(self.df) * 100).round(2)
                    })
                    st.dataframe(value_df, use_container_width=True)
                
                with col2:
                    # Visualization options
                    viz_type = st.radio(
                        "Visualization type",
                        ["Bar Chart", "Pie Chart", "Treemap"],
                        horizontal=True
                    )
                    
                    # Limit categories for visualization
                    max_cats = st.slider("Maximum categories to display", 5, 20, 10)
                    
                    # Prepare data
                    if len(value_counts) > max_cats:
                        top_counts = value_counts.iloc[:max_cats]
                        other_count = value_counts.iloc[max_cats:].sum()
                        
                        # Create new series with "Other" category
                        plot_counts = pd.Series(
                            list(top_counts) + [other_count],
                            index=list(top_counts.index) + ["Other"]
                        )
                    else:
                        plot_counts = value_counts
                    
                    if viz_type == "Bar Chart":
                        fig = px.bar(
                            x=plot_counts.index,
                            y=plot_counts.values,
                            labels={'x': col, 'y': 'Count'},
                            template="plotly_white"
                        )
                    elif viz_type == "Pie Chart":
                        fig = px.pie(
                            values=plot_counts.values,
                            names=plot_counts.index,
                            template="plotly_white"
                        )
                    elif viz_type == "Treemap":
                        fig = px.treemap(
                            names=plot_counts.index,
                            values=plot_counts.values,
                            template="plotly_white"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Two Variable Analysis
        with explorer_tabs[1]:
            st.subheader("Two Variable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("Select X-axis variable", self.df.columns, key="two_var_x")
            
            with col2:
                y_col = st.selectbox("Select Y-axis variable", [c for c in self.df.columns if c != x_col], key="two_var_y")
            
            # Determine column types
            x_is_numeric = x_col in num_cols
            y_is_numeric = y_col in num_cols
            
            # Select appropriate visualization based on column types
            if x_is_numeric and y_is_numeric:
                # Both numeric - scatter plot with options
                viz_type = st.radio(
                    "Visualization type",
                    ["Scatter Plot", "Hex Density", "2D Histogram"],
                    horizontal=True
                )
                
                # Optional color grouping
                use_color = st.checkbox("Group by color")
                color_col = None
                if use_color and cat_cols:
                    color_col = st.selectbox("Color by", cat_cols)
                
                if viz_type == "Scatter Plot":
                    if use_color and color_col:
                        fig = px.scatter(
                            self.df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            template="plotly_white",
                            trendline="ols"
                        )
                    else:
                        fig = px.scatter(
                            self.df,
                            x=x_col,
                            y=y_col,
                            template="plotly_white",
                            trendline="ols"
                        )
                elif viz_type == "Hex Density":
                    fig = px.density_heatmap(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white",
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                elif viz_type == "2D Histogram":
                    fig = px.density_contour(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white",
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                
                # Show correlation
                corr = self.df[[x_col, y_col]].corr().iloc[0, 1]
                st.metric("Correlation", f"{corr:.4f}")
                
            elif x_is_numeric and not y_is_numeric:
                # X numeric, Y categorical - box plot or violin plot
                viz_type = st.radio(
                    "Visualization type",
                    ["Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot"],
                    horizontal=True
                )
                
                if viz_type == "Box Plot":
                    fig = px.box(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white"
                    )
                elif viz_type == "Violin Plot":
                    fig = px.violin(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white",
                        box=True
                    )
                elif viz_type == "Strip Plot":
                    # Use matplotlib for strip plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.stripplot(data=self.df, x=x_col, y=y_col, ax=ax)
                    st.pyplot(fig)
                    # Skip the plotly rendering
                    fig = None
                elif viz_type == "Swarm Plot":
                    # Use matplotlib for swarm plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.swarmplot(data=self.df, x=x_col, y=y_col, ax=ax)
                    st.pyplot(fig)
                    # Skip the plotly rendering
                    fig = None
                    
            elif not x_is_numeric and y_is_numeric:
                # X categorical, Y numeric - bar chart, box plot or violin plot
                viz_type = st.radio(
                    "Visualization type",
                    ["Bar Chart", "Box Plot", "Violin Plot"],
                    horizontal=True
                )
                
                # Group by and aggregation method
                if viz_type == "Bar Chart":
                    agg_method = st.selectbox(
                        "Aggregation method",
                        ["Mean", "Median", "Sum", "Min", "Max", "Count"]
                    )
                    
                    agg_method_map = {
                        "Mean": "mean",
                        "Median": "median",
                        "Sum": "sum",
                        "Min": "min",
                        "Max": "max",
                        "Count": "count"
                    }
                    
                    fig = px.bar(
                        self.df.groupby(x_col)[y_col].agg(agg_method_map[agg_method]).reset_index(),
                        x=x_col,
                        y=y_col,
                        title=f"{agg_method} of {y_col} by {x_col}",
                        template="plotly_white"
                    )
                elif viz_type == "Box Plot":
                    fig = px.box(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white"
                    )
                elif viz_type == "Violin Plot":
                    fig = px.violin(
                        self.df,
                        x=x_col,
                        y=y_col,
                        template="plotly_white",
                        box=True
                    )
            else:
                # Both categorical - heatmap, stacked bar, or grouped bar
                viz_type = st.radio(
                    "Visualization type",
                    ["Heatmap", "Stacked Bar", "Grouped Bar"],
                    horizontal=True
                )
                
                if viz_type == "Heatmap":
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.df[y_col], 
                        self.df[x_col],
                        normalize="all"
                    ) * 100
                    
                    fig = px.imshow(
                        contingency,
                        text_auto='.1f',
                        labels=dict(x=x_col, y=y_col, color="Percentage"),
                        template="plotly_white"
                    )
                elif viz_type == "Stacked Bar":
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.df[x_col], 
                        self.df[y_col]
                    )
                    
                    # Convert to long format
                    contingency_long = contingency.reset_index().melt(
                        id_vars=[x_col],
                        value_vars=contingency.columns,
                        var_name=y_col,
                        value_name="Count"
                    )
                    
                    fig = px.bar(
                        contingency_long,
                        x=x_col,
                        y="Count",
                        color=y_col,
                        template="plotly_white"
                    )
                elif viz_type == "Grouped Bar":
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.df[x_col], 
                        self.df[y_col]
                    )
                    
                    # Convert to long format
                    contingency_long = contingency.reset_index().melt(
                        id_vars=[x_col],
                        value_vars=contingency.columns,
                        var_name=y_col,
                        value_name="Count"
                    )
                    
                    fig = px.bar(
                        contingency_long,
                        x=x_col,
                        y="Count",
                        color=y_col,
                        barmode="group",
                        template="plotly_white"
                    )
            
            # Display the figure if it's a plotly figure
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            
            # Save visualization option
            if st.button("Save this visualization"):
                viz_title = f"Analysis of {x_col} vs {y_col}"
                if fig is not None:
                    viz_data = {
                        "title": viz_title,
                        "type": "plotly",
                        "figure": fig,
                        "timestamp": datetime.datetime.now()
                    }
                    st.session_state.visualizations.append(viz_data)
                    st.success(f"Visualization '{viz_title}' saved successfully!")
        
        # Time Series Analysis
        with explorer_tabs[2]:
            st.subheader("Time Series Analysis")
            
            # Check for potential date columns
            date_cols = []
            
            for col in self.df.columns:
                # Check if column is already datetime
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    date_cols.append(col)
                # Check if column name contains date-related keywords
                elif any(kw in col.lower() for kw in ['date', 'time', 'day', 'month', 'year']):
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(self.df[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if not date_cols:
                st.warning("No date/time columns detected in your data.")
            else:
                # Select date column
                date_col = st.selectbox("Select date/time column", date_cols)
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                    try:
                        date_series = pd.to_datetime(self.df[date_col])
                        st.success(f"Successfully converted '{date_col}' to datetime.")
                    except Exception as e:
                        st.error(f"Error converting to datetime: {str(e)}")
                        date_series = self.df[date_col]
                else:
                    date_series = self.df[date_col]
                
                # Select value column(s)
                value_cols = st.multiselect(
                    "Select value column(s) to plot",
                    [col for col in num_cols if col != date_col],
                    max_selections=5
                )
                
                if not value_cols:
                    st.info("Please select at least one value column to plot.")
                else:
                    # Aggregated values
                    agg_func = agg_type.lower()
                    crosstab = pd.pivot_table(
                        self.df,
                        index=row_col,
                        columns=col_col,
                        values=value_col,
                        aggfunc=agg_func
                    )
                    title = f"{agg_type} of {value_col} by {row_col} and {col_col}"
                    z_title = f"{agg_type} of {value_col}" (Time interval for aggregation)
                    agg_interval = st.selectbox(
                        "Aggregate by",
                        ["None", "Day", "Week", "Month", "Quarter", "Year"]
                    )
                    
                    # Line chart options
                    show_markers = st.checkbox("Show markers")
                    
                    # Create copy of dataframe with proper datetime column
                    plot_df = self.df.copy()
                    plot_df[date_col] = date_series
                    
                    # Aggregate if requested
                    if agg_interval != "None":
                        # Group by time interval
                        interval_map = {
                            "Day": date_series.dt.date,
                            "Week": date_series.dt.isocalendar().week,
                            "Month": date_series.dt.to_period('M').astype(str),
                            "Quarter": date_series.dt.to_period('Q').astype(str),
                            "Year": date_series.dt.year
                        }
                        
                        # Create groupby object
                        grouper = interval_map[agg_interval]
                        
                        # Aggregate each value column
                        agg_dfs = []
                        for col in value_cols:
                            agg_df = plot_df.groupby(grouper)[col].agg(
                                ['mean', 'min', 'max', 'count']
                            ).reset_index()
                            agg_df.columns = [date_col, f"{col}_mean", f"{col}_min", f"{col}_max", f"{col}_count"]
                            agg_dfs.append(agg_df)
                        
                        # Merge all aggregated dataframes
                        if agg_dfs:
                            plot_df = agg_dfs[0]
                            for df in agg_dfs[1:]:
                                plot_df = plot_df.merge(df, on=date_col)
                    
                    # Sort by date
                    plot_df = plot_df.sort_values(date_col)
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Add line for each value column
                    for col in value_cols:
                        # If aggregated, use mean
                        if agg_interval != "None":
                            y_col = f"{col}_mean"
                            
                            # Add range
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[f"{col}_min"],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    name=f"{col} (min)"
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[f"{col}_max"],
                                    mode='lines',
                                    fill='tonexty',
                                    line=dict(width=0),
                                    showlegend=False,
                                    name=f"{col} (max)"
                                )
                            )
                            
                            # Add mean line
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[y_col],
                                    mode='lines+markers' if show_markers else 'lines',
                                    name=f"{col} (mean)",
                                    line=dict(width=2)
                                )
                            )
                        else:
                            # Use raw values
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[col],
                                    mode='lines+markers' if show_markers else 'lines',
                                    name=col
                                )
                            )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Time Series Analysis by {agg_interval if agg_interval != 'None' else 'Raw Data'}",
                        xaxis_title=date_col,
                        yaxis_title="Value",
                        template="plotly_white",
                        height=500
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save visualization option
                    if st.button("Save this time series visualization"):
                        viz_title = f"Time Series of {', '.join(value_cols)} by {date_col}"
                        viz_data = {
                            "title": viz_title,
                            "type": "plotly",
                            "figure": fig,
                            "timestamp": datetime.datetime.now()
                        }
                        st.session_state.visualizations.append(viz_data)
                        st.success(f"Visualization '{viz_title}' saved successfully!")
        
        # Correlation Analysis
        with explorer_tabs[3]:
            st.subheader("Correlation Analysis")
            
            if len(num_cols) < 2:
                st.warning("At least two numeric columns are required for correlation analysis.")
            else:
                # Correlation options
                corr_method = st.radio(
                    "Correlation method",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True
                )
                
                # Select columns for correlation
                selected_cols = st.multiselect(
                    "Select columns for correlation analysis",
                    num_cols,
                    default=num_cols[:min(len(num_cols), 10)]
                )
                
                if not selected_cols or len(selected_cols) < 2:
                    st.info("Please select at least two columns for correlation analysis.")
                else:
                    # Calculate correlation matrix
                    corr_matrix = self.df[selected_cols].corr(method=corr_method.lower())
                    
                    # Visualization options
                    viz_type = st.radio(
                        "Visualization type",
                        ["Heatmap", "Scatter Matrix", "Network Graph"],
                        horizontal=True
                    )
                    
                    if viz_type == "Heatmap":
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale="RdBu_r",
                            title=f"{corr_method} Correlation Matrix",
                            template="plotly_white",
                            color_continuous_midpoint=0
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    elif viz_type == "Scatter Matrix":
                        fig = px.scatter_matrix(
                            self.df[selected_cols],
                            dimensions=selected_cols,
                            title=f"Scatter Matrix with {corr_method} Correlation",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    elif viz_type == "Network Graph":
                        # Create network visualization of correlations
                        fig = self._create_correlation_network(corr_matrix)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strongest correlations
                    st.subheader("Strongest Correlations")
                    
                    # Get correlations in tabular format
                    corr_list = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            corr_value = corr_matrix.iloc[i, j]
                            corr_list.append({
                                "Variable 1": col1,
                                "Variable 2": col2,
                                "Correlation": corr_value,
                                "Abs Correlation": abs(corr_value)
                            })
                    
                    # Create dataframe and sort
                    corr_df = pd.DataFrame(corr_list)
                    corr_df = corr_df.sort_values("Abs Correlation", ascending=False)
                    
                    # Display top correlations
                    st.dataframe(
                        corr_df[["Variable 1", "Variable 2", "Correlation"]],
                        use_container_width=True
                    )
                    
                    # Save visualization option
                    if st.button("Save this correlation visualization"):
                        viz_title = f"{corr_method} Correlation Analysis"
                        viz_data = {
                            "title": viz_title,
                            "type": "plotly",
                            "figure": fig,
                            "timestamp": datetime.datetime.now()
                        }
                        st.session_state.visualizations.append(viz_data)
                        st.success(f"Visualization '{viz_title}' saved successfully!")
    
    def _render_recommendations(self):
        """Render visualization recommendations based on data characteristics"""
        st.subheader("Smart Visualization Recommendations")
        
        # Generate recommendations if not already done
        if not st.session_state.viz_recommendations:
            with st.spinner("Analyzing your data and generating recommendations..."):
                self._generate_recommendations()
        
        # Check if recommendations exist
        if not st.session_state.viz_recommendations:
            st.warning("Could not generate recommendations for this dataset.")
            
            # Refresh button
            if st.button("Refresh Recommendations"):
                st.session_state.viz_recommendations = []
                st.rerun()
            return
        
        # Show recommendations in expandable sections
        for i, rec in enumerate(st.session_state.viz_recommendations):
            with st.expander(f"**{i+1}. {rec['title']}**", expanded=i==0):
                st.markdown(rec["description"])
                
                # Variables involved
                st.markdown(f"**Variables:** {', '.join(rec['variables'])}")
                
                # Show visualization
                if "figure" in rec:
                    st.plotly_chart(rec["figure"], use_container_width=True)
                
                # Create/save button
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("Create/Customize", key=f"create_{i}"):
                        # Set appropriate values in session state for chart builder
                        st.session_state.recommended_chart = rec["chart_type"]
                        st.session_state.recommended_vars = rec["variables"]
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Save Visualization", key=f"save_{i}"):
                        # Save the recommendation to visualizations
                        if "figure" in rec:
                            viz_data = {
                                "title": rec["title"],
                                "type": "plotly",
                                "figure": rec["figure"],
                                "timestamp": datetime.datetime.now()
                            }
                            st.session_state.visualizations.append(viz_data)
                            st.success(f"Visualization '{rec['title']}' saved successfully!")
        
        # Refresh button
        if st.button("Refresh Recommendations"):
            st.session_state.viz_recommendations = []
            st.rerun()