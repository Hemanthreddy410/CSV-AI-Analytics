import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

def render_visualization():
    """Render visualization section"""
    st.header("Data Visualization")
    
    # Create visualization tabs
    viz_tabs = st.tabs([
        "Charts",
        "Interactive Plots",
        "Statistical Plots",
        "Geospatial Plots",
        "Custom Visualization"
    ])
    
    # Charts Tab
    with viz_tabs[0]:
        render_charts_tab()
    
    # Interactive Plots Tab
    with viz_tabs[1]:
        render_interactive_plots_tab()
    
    # Statistical Plots Tab
    with viz_tabs[2]:
        render_statistical_plots_tab()
    
    # Geospatial Plots Tab
    with viz_tabs[3]:
        render_geospatial_plots_tab()
    
    # Custom Visualization Tab
    with viz_tabs[4]:
        render_custom_viz_tab()

def render_charts_tab():
    """Render basic charts tab"""
    st.subheader("Basic Charts")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select chart type:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Area Chart", "Heatmap"],
        key="charts_tab_chart_type"
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Column selectors based on chart type
    if chart_type == "Bar Chart":
        x_axis = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="bar_chart_x_axis")
        y_axis = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist(), key="bar_chart_y_axis")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by another column", key="bar_chart_use_grouping")
        if use_grouping and len(cat_cols) > 1:
            group_col = st.selectbox("Group by:", [col for col in cat_cols if col != x_axis], key="bar_chart_group_col")
            
            # Limit number of categories to prevent overcrowding
            top_n = st.slider("Show top N categories:", 3, 20, 10, key="bar_chart_top_n")
            
            # Count values for each group
            value_counts = st.session_state.df.groupby([x_axis, group_col]).size().reset_index(name='Count')
            
            # Get top N categories for x_axis
            top_categories = st.session_state.df[x_axis].value_counts().nlargest(top_n).index.tolist()
            
            # Filter to only top categories
            value_counts = value_counts[value_counts[x_axis].isin(top_categories)]
            
            # Create grouped bar chart
            fig = px.bar(
                value_counts,
                x=x_axis,
                y='Count',
                color=group_col,
                title=f"Grouped Bar Chart: {x_axis} vs Count, grouped by {group_col}",
                barmode='group'
            )
        else:
            # Simple bar chart
            fig = px.bar(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Bar Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Line Chart":
        if date_cols:
            x_axis = st.selectbox("X-axis (typically date/time):", date_cols + num_cols, key="line_chart_x_axis")
        else:
            x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="line_chart_x_axis")
            
        y_axis = st.selectbox("Y-axis (values):", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols, key="line_chart_y_axis")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="line_chart_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="line_chart_group_col")
            
            # Create line chart with grouping
            fig = px.line(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                color=group_col,
                title=f"Line Chart: {x_axis} vs {y_axis}, grouped by {group_col}",
                labels={x_axis: x_axis, y_axis: y_axis, group_col: group_col}
            )
        else:
            # Simple line chart
            fig = px.line(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Line Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="scatter_plot_x_axis")
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols, key="scatter_plot_y_axis")
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols, key="scatter_plot_color_col")
        size_col = st.selectbox("Size by:", ["None"] + num_cols, key="scatter_plot_size_col")
        
        # Create scatter plot
        scatter_params = {
            "x": x_axis,
            "y": y_axis,
            "title": f"Scatter Plot: {x_axis} vs {y_axis}",
            "labels": {x_axis: x_axis, y_axis: y_axis}
        }
        
        if color_col != "None":
            scatter_params["color"] = color_col
        
        if size_col != "None":
            scatter_params["size"] = size_col
        
        fig = px.scatter(
            st.session_state.df,
            **scatter_params
        )
    
    elif chart_type == "Pie Chart":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="pie_chart_value_col")
        names_col = st.selectbox("Category names:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="pie_chart_names_col")
        
        # Limit number of categories to prevent overcrowding
        top_n = st.slider("Show top N categories:", 3, 20, 10, key="pie_chart_top_n")
        
        # Calculate values for pie chart
        value_counts = st.session_state.df.groupby(names_col)[value_col].sum().reset_index()
        value_counts = value_counts.sort_values(value_col, ascending=False).head(top_n)
        
        # Create pie chart
        fig = px.pie(
            value_counts,
            values=value_col,
            names=names_col,
            title=f"Pie Chart: {value_col} by {names_col} (Top {top_n})"
        )
    
    elif chart_type == "Histogram":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="histogram_value_col")
        
        # Histogram options
        n_bins = st.slider("Number of bins:", 5, 100, 20, key="histogram_n_bins")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="histogram_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="histogram_group_col")
            
            # Create histogram with grouping
            fig = px.histogram(
                st.session_state.df,
                x=value_col,
                color=group_col,
                nbins=n_bins,
                title=f"Histogram of {value_col}, grouped by {group_col}",
                marginal="box"  # Add box plot to the top
            )
        else:
            # Simple histogram
            fig = px.histogram(
                st.session_state.df,
                x=value_col,
                nbins=n_bins,
                title=f"Histogram of {value_col}",
                marginal="box"  # Add box plot to the top
            )
    
    elif chart_type == "Box Plot":
        value_col = st.selectbox("Value:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="box_plot_value_col")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="box_plot_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="box_plot_group_col")
            
            # Create box plot with grouping
            fig = px.box(
                st.session_state.df,
                x=group_col,
                y=value_col,
                title=f"Box Plot of {value_col}, grouped by {group_col}"
            )
        else:
            # Simple box plot
            fig = px.box(
                st.session_state.df,
                y=value_col,
                title=f"Box Plot of {value_col}"
            )
    
    elif chart_type == "Area Chart":
        if date_cols:
            x_axis = st.selectbox("X-axis (typically date/time):", date_cols + num_cols, key="area_chart_x_axis")
        else:
            x_axis = st.selectbox("X-axis:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="area_chart_x_axis")
            
        y_axis = st.selectbox("Y-axis (values):", [col for col in num_cols if col != x_axis] if len(num_cols) > 1 else num_cols, key="area_chart_y_axis")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="area_chart_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="area_chart_group_col")
            
            # Create area chart with grouping
            fig = px.area(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                color=group_col,
                title=f"Area Chart: {x_axis} vs {y_axis}, grouped by {group_col}",
                labels={x_axis: x_axis, y_axis: y_axis, group_col: group_col}
            )
        else:
            # Simple area chart
            fig = px.area(
                st.session_state.df,
                x=x_axis,
                y=y_axis,
                title=f"Area Chart: {x_axis} vs {y_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
    
    elif chart_type == "Heatmap":
        if len(num_cols) < 1:
            st.warning("Need at least one numeric column for heatmap.")
            return
            
        # Column selections
        x_axis = st.selectbox("X-axis:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="heatmap_x_axis")
        y_axis = st.selectbox("Y-axis:", [col for col in cat_cols if col != x_axis] if len(cat_cols) > 1 else cat_cols, key="heatmap_y_axis")
        value_col = st.selectbox("Value:", num_cols, key="heatmap_value_col")
        
        # Limit categories to prevent overcrowding
        top_x = st.slider("Top X categories:", 3, 20, 10, key="heatmap_top_x")
        top_y = st.slider("Top Y categories:", 3, 20, 10, key="heatmap_top_y")
        
        # Get top categories
        top_x_cats = st.session_state.df[x_axis].value_counts().nlargest(top_x).index.tolist()
        top_y_cats = st.session_state.df[y_axis].value_counts().nlargest(top_y).index.tolist()
        
        # Filter data to top categories
        filtered_df = st.session_state.df[
            st.session_state.df[x_axis].isin(top_x_cats) & 
            st.session_state.df[y_axis].isin(top_y_cats)
        ]
        
        # Calculate values for heatmap
        heatmap_data = filtered_df.pivot_table(
            index=y_axis, 
            columns=x_axis, 
            values=value_col,
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x=x_axis, y=y_axis, color=value_col),
            title=f"Heatmap of {value_col} by {x_axis} and {y_axis}"
        )
    
    # Layout settings
    with st.expander("Chart Settings", expanded=False):
        # Title and labels
        chart_title = st.text_input("Chart title:", fig.layout.title.text, key="chart_settings_title")
        
        # Size settings
        width = st.slider("Chart width:", 400, 1200, 800, key="chart_settings_width")
        height = st.slider("Chart height:", 300, 1000, 500, key="chart_settings_height")
        
        # Update chart
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["PNG", "JPEG", "SVG", "HTML"],
            horizontal=True,
            key="charts_tab_export_format"
        )
        
        # Export button
        if st.button("Export Chart", key="charts_tab_export_button", use_container_width=True):
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
                    key="charts_tab_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    key=f"charts_tab_download_{export_format.lower()}"
                )

def render_interactive_plots_tab():
    """Render interactive plots tab"""
    st.subheader("Interactive Plots")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Dynamic Scatter Plot", "Interactive Time Series", "3D Plot", "Multi-axis Plot", "Animated Chart"],
        key="interactive_plots_type"
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
    
    if not num_cols:
        st.warning("Need numeric columns for interactive plots.")
        return
    
    if plot_type == "Dynamic Scatter Plot":
        # Column selectors
        x_axis = st.selectbox("X-axis:", num_cols, key="dynamic_scatter_x_axis")
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis], key="dynamic_scatter_y_axis")
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols, key="dynamic_scatter_color_col")
        size_col = st.selectbox("Size by:", ["None"] + num_cols, key="dynamic_scatter_size_col")
        
        # Animation option
        animate_col = st.selectbox("Animate by:", ["None"] + cat_cols + date_cols, key="dynamic_scatter_animate_col")
        
        # Build plot parameters
        plot_params = {
            "x": x_axis,
            "y": y_axis,
            "title": f"Interactive Scatter Plot: {x_axis} vs {y_axis}",
            "labels": {x_axis: x_axis, y_axis: y_axis}
        }
        
        if color_col != "None":
            plot_params["color"] = color_col
        
        if size_col != "None":
            plot_params["size"] = size_col
            
            # Adjust size range
            plot_params["size_max"] = 30
        
        if animate_col != "None":
            plot_params["animation_frame"] = animate_col
        
        # Create the plot
        fig = px.scatter(
            st.session_state.df,
            **plot_params
        )
        
        # Add hover data for interactivity
        fig.update_traces(
            hovertemplate="<br>".join([
                f"{x_axis}: %{{x}}",
                f"{y_axis}: %{{y}}",
                "Click for more info"
            ])
        )
        
        # Make the plot interactive
        fig.update_layout(
            clickmode='event+select'
        )
    
    elif plot_type == "Interactive Time Series":
        if not date_cols:
            st.warning("No datetime columns found. Consider converting a column to datetime first.")
            time_col = st.selectbox("X-axis (time):", st.session_state.df.columns.tolist(), key="time_series_x_axis")
        else:
            time_col = st.selectbox("X-axis (time):", date_cols, key="time_series_x_axis")
        
        # Select values to plot
        value_cols = st.multiselect(
            "Select values to plot:",
            num_cols,
            default=[num_cols[0]] if num_cols else [],
            key="time_series_value_cols"
        )
        
        if not value_cols:
            st.warning("Please select at least one value column to plot.")
            return
        
        # Create a figure with secondary y-axis if needed
        fig = go.Figure()
        
        # Add traces for each selected value
        for i, col in enumerate(value_cols):
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[time_col],
                    y=st.session_state.df[col],
                    name=col,
                    mode='lines+markers'
                )
            )
        
        # Add range selector for time series
        fig.update_layout(
            title=f"Interactive Time Series Plot",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
    
    elif plot_type == "3D Plot":
        # Ensure we have at least 3 numeric columns
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns for 3D plot.")
            return
            
        # Column selectors
        x_axis = st.selectbox("X-axis:", num_cols, key="3d_plot_x_axis")
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis], key="3d_plot_y_axis")
        z_axis = st.selectbox("Z-axis:", [col for col in num_cols if col not in [x_axis, y_axis]], key="3d_plot_z_axis")
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols + num_cols, key="3d_plot_color_col")
        
        # Plot types
        plot3d_type = st.radio(
            "3D Plot type:",
            ["Scatter", "Surface", "Line"],
            horizontal=True,
            key="3d_plot_type"
        )
        
        if plot3d_type == "Scatter":
            # Create 3D scatter plot
            plot_params = {
                "x": x_axis,
                "y": y_axis,
                "z": z_axis,
                "title": f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}"
            }
            
            if color_col != "None":
                plot_params["color"] = color_col
            
            fig = px.scatter_3d(
                st.session_state.df,
                **plot_params
            )
        
        elif plot3d_type == "Surface":
            # For surface plot, we need to create a grid
            st.info("Surface plot requires data on a regular grid. Attempting to create grid from data...")
            
            # Try to create a grid
            try:
                # Get unique x and y values
                x_vals = st.session_state.df[x_axis].unique()
                y_vals = st.session_state.df[y_axis].unique()
                
                # Create grid
                grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                
                # Create z values grid
                grid_z = np.zeros(grid_x.shape)
                
                # Fill in the z values
                for i, x_val in enumerate(x_vals):
                    for j, y_val in enumerate(y_vals):
                        mask = (st.session_state.df[x_axis] == x_val) & (st.session_state.df[y_axis] == y_val)
                        if mask.any():
                            grid_z[j, i] = st.session_state.df.loc[mask, z_axis].mean()
                
                # Create surface plot
                fig = go.Figure(data=[go.Surface(z=grid_z, x=x_vals, y=y_vals)])
                fig.update_layout(
                    title=f"3D Surface Plot: {z_axis} as a function of {x_axis} and {y_axis}",
                    scene=dict(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        zaxis_title=z_axis
                    )
                )
            except Exception as e:
                st.error(f"Error creating surface plot: {str(e)}")
                st.info("Surface plot requires data on a regular grid. Try using scatter or line 3D plot instead.")
                return
        
        elif plot3d_type == "Line":
            # Create 3D line plot
            plot_params = {
                "x": x_axis,
                "y": y_axis,
                "z": z_axis,
                "title": f"3D Line Plot: {x_axis} vs {y_axis} vs {z_axis}"
            }
            
            if color_col != "None":
                plot_params["color"] = color_col
            
            fig = px.line_3d(
                st.session_state.df,
                **plot_params
            )
    
    elif plot_type == "Multi-axis Plot":
        # Select a common x-axis
        x_axis = st.selectbox("X-axis:", st.session_state.df.columns.tolist(), key="multi_axis_x_axis")
        
        # Select values for left y-axis
        left_y = st.selectbox("Left Y-axis:", num_cols, key="multi_axis_left_y")
        
        # Select values for right y-axis
        right_y = st.selectbox("Right Y-axis:", [col for col in num_cols if col != left_y], key="multi_axis_right_y")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_axis],
                y=st.session_state.df[left_y],
                name=left_y,
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=st.session_state.df[x_axis],
                y=st.session_state.df[right_y],
                name=right_y,
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Multi-axis Plot: {left_y} and {right_y} vs {x_axis}"
        )
        
        # Update axes titles
        fig.update_xaxes(title_text=x_axis)
        fig.update_yaxes(title_text=left_y, secondary_y=False)
        fig.update_yaxes(title_text=right_y, secondary_y=True)
    
    elif plot_type == "Animated Chart":
        # Select columns
        x_axis = st.selectbox("X-axis:", num_cols, key="animated_chart_x_axis")
        y_axis = st.selectbox("Y-axis:", [col for col in num_cols if col != x_axis], key="animated_chart_y_axis")
        
        # Animation frame
        if date_cols:
            frame_col = st.selectbox("Animate by:", date_cols + cat_cols, key="animated_chart_frame_col")
        else:
            frame_col = st.selectbox("Animate by:", cat_cols, key="animated_chart_frame_col")
        
        # Optional parameters
        color_col = st.selectbox("Color by:", ["None"] + cat_cols, key="animated_chart_color_col")
        size_col = st.selectbox("Size by:", ["None"] + num_cols, key="animated_chart_size_col")
        
        # Chart type
        anim_chart_type = st.radio(
            "Chart type:",
            ["Scatter", "Bar", "Line"],
            horizontal=True,
            key="animated_chart_type"
        )
        
        # Build plot parameters
        plot_params = {
            "x": x_axis,
            "y": y_axis,
            "animation_frame": frame_col,
            "title": f"Animated {anim_chart_type} Chart: {x_axis} vs {y_axis} over {frame_col}"
        }
        
        if color_col != "None":
            plot_params["color"] = color_col
        
        if size_col != "None" and anim_chart_type == "Scatter":
            plot_params["size"] = size_col
            plot_params["size_max"] = 30
        
        # Create the plot
        if anim_chart_type == "Scatter":
            fig = px.scatter(st.session_state.df, **plot_params)
        elif anim_chart_type == "Bar":
            fig = px.bar(st.session_state.df, **plot_params)
        elif anim_chart_type == "Line":
            fig = px.line(st.session_state.df, **plot_params)
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text, key="interactive_chart_title")
        width = st.slider("Chart width:", 400, 1200, 800, key="interactive_chart_width")
        height = st.slider("Chart height:", 300, 1000, 600, key="interactive_chart_height")
        
        # Theme selection
        theme = st.selectbox(
            "Color theme:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            key="interactive_chart_theme"
        )
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            template=theme
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True,
            key="interactive_plots_export_format"
        )
        
        # Export button
        if st.button("Export Chart", key="interactive_plots_export_button", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"interactive_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    key="interactive_plots_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"interactive_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    key=f"interactive_plots_download_{export_format.lower()}"
                )

def render_statistical_plots_tab():
    """Render statistical plots tab"""
    st.subheader("Statistical Plots")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Distribution Plot", "Correlation Matrix", "Pair Plot", "ECDF Plot", "Q-Q Plot", "Violin Plot", "Residual Plot"],
        key="statistical_plots_type"
    )
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not num_cols:
        st.warning("Need numeric columns for statistical plots.")
        return
    
    if plot_type == "Distribution Plot":
        # Column selection
        dist_col = st.selectbox("Select column:", num_cols, key="distribution_plot_col")
        
        # Plot options
        show_kde = st.checkbox("Show KDE", value=True, key="distribution_plot_show_kde")
        show_rug = st.checkbox("Show rug plot", value=False, key="distribution_plot_show_rug")
        
        # Number of bins
        n_bins = st.slider("Number of bins:", 5, 100, 30, key="distribution_plot_bins")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="distribution_plot_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="distribution_plot_group_col")
            
            # Create distribution plot with grouping
            fig = px.histogram(
                st.session_state.df,
                x=dist_col,
                color=group_col,
                nbins=n_bins,
                title=f"Distribution of {dist_col}, grouped by {group_col}",
                marginal="box" if not show_kde else "violin",
            )
        else:
            # Simple distribution plot
            fig = px.histogram(
                st.session_state.df,
                x=dist_col,
                nbins=n_bins,
                title=f"Distribution of {dist_col}",
                marginal="box" if not show_kde else "violin",
            )
            
            if show_kde:
                # Add KDE plot
                try:
                    from scipy import stats
                    
                    # Get data for KDE
                    data = st.session_state.df[dist_col].dropna()
                    
                    # Calculate KDE
                    kde_x = np.linspace(data.min(), data.max(), 1000)
                    kde = stats.gaussian_kde(data)
                    kde_y = kde(kde_x)
                    
                    # Scale KDE to match histogram
                    hist, bin_edges = np.histogram(data, bins=n_bins)
                    scaling_factor = max(hist) / max(kde_y)
                    kde_y = kde_y * scaling_factor
                    
                    # Add KDE line
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        )
                    )
                except Exception as e:
                    st.warning(f"Could not calculate KDE: {str(e)}")
            
            if show_rug:
                # Add rug plot
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df[dist_col],
                        y=np.zeros(len(st.session_state.df)),
                        mode='markers',
                        marker=dict(symbol='line-ns', size=5, color='blue'),
                        name='Data points'
                    )
                )
    
    elif plot_type == "Correlation Matrix":
        # Column selection
        corr_cols = st.multiselect(
            "Select columns for correlation:",
            num_cols,
            default=num_cols[:min(5, len(num_cols))],
            key="correlation_matrix_cols"
        )
        
        if not corr_cols:
            st.warning("Please select at least one column for correlation matrix.")
            return
        
        # Correlation method
        corr_method = st.radio(
            "Correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True,
            key="correlation_method_radio"
        )
        
        # Calculate correlation matrix
        corr_matrix = st.session_state.df[corr_cols].corr(method=corr_method.lower())
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f"{corr_method} Correlation Matrix",
            labels=dict(color="Correlation")
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
    
    elif plot_type == "Pair Plot":
        # Column selection
        pair_cols = st.multiselect(
            "Select columns for pair plot:",
            num_cols,
            default=num_cols[:min(3, len(num_cols))],
            key="pair_plot_cols"
        )
        
        if len(pair_cols) < 2:
            st.warning("Please select at least two columns for pair plot.")
            return
        
        # Optional color column
        use_color = st.checkbox("Color by a column", key="pair_plot_use_color")
        color_col = None
        
        if use_color and cat_cols:
            color_col = st.selectbox("Color by:", cat_cols, key="pair_plot_color_col")
        
        # Create pair plot
        if color_col:
            fig = px.scatter_matrix(
                st.session_state.df,
                dimensions=pair_cols,
                color=color_col,
                title="Pair Plot"
            )
        else:
            fig = px.scatter_matrix(
                st.session_state.df,
                dimensions=pair_cols,
                title="Pair Plot"
            )
        
        # Update layout for better appearance
        fig.update_traces(diagonal_visible=False)
    
    elif plot_type == "ECDF Plot":
        # Column selection
        ecdf_col = st.selectbox("Select column for ECDF:", num_cols, key="ecdf_plot_col")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="ecdf_plot_use_grouping")
        group_col = None
        
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="ecdf_plot_group_col")
        
        # Create ECDF plot
        if group_col:
            # Get unique groups
            groups = st.session_state.df[group_col].unique()
            
            # Create figure
            fig = go.Figure()
            
            # Add ECDF for each group
            for group in groups:
                data = st.session_state.df[st.session_state.df[group_col] == group][ecdf_col].dropna().sort_values()
                y = np.arange(1, len(data) + 1) / len(data)
                
                fig.add_trace(
                    go.Scatter(
                        x=data,
                        y=y,
                        mode='lines',
                        name=str(group)
                    )
                )
            
            fig.update_layout(
                title=f"ECDF Plot of {ecdf_col}, grouped by {group_col}",
                xaxis_title=ecdf_col,
                yaxis_title="Cumulative Probability"
            )
        else:
            # Single ECDF
            data = st.session_state.df[ecdf_col].dropna().sort_values()
            y = np.arange(1, len(data) + 1) / len(data)
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data,
                    y=y,
                    mode='lines',
                    name=ecdf_col
                )
            )
            
            fig.update_layout(
                title=f"ECDF Plot of {ecdf_col}",
                xaxis_title=ecdf_col,
                yaxis_title="Cumulative Probability"
            )
    
    elif plot_type == "Q-Q Plot":
        # Column selection
        qq_col = st.selectbox("Select column for Q-Q plot:", num_cols, key="qq_plot_col")
        
        # Distribution selection
        dist = st.selectbox(
            "Theoretical distribution:",
            ["Normal", "Uniform", "Exponential", "Log-normal"],
            key="qq_plot_dist"
        )
        
        # Create Q-Q plot
        try:
            import scipy.stats as stats
            
            # Get data
            data = st.session_state.df[qq_col].dropna()
            
            # Calculate theoretical quantiles
            if dist == "Normal":
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Normal"
            elif dist == "Uniform":
                theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Uniform"
            elif dist == "Exponential":
                theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(data)))
                theoretical_name = "Exponential"
            elif dist == "Log-normal":
                theoretical_quantiles = stats.lognorm.ppf(np.linspace(0.01, 0.99, len(data)), s=1)
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
                title=f"Q-Q Plot: {qq_col} vs {theoretical_name} Distribution",
                xaxis_title=f"Theoretical Quantiles ({theoretical_name})",
                yaxis_title=f"Sample Quantiles ({qq_col})"
            )
            
        except Exception as e:
            st.error(f"Error creating Q-Q plot: {str(e)}")
            return
    
    elif plot_type == "Violin Plot":
        # Column selection
        violin_col = st.selectbox("Select column for values:", num_cols, key="violin_plot_col")
        
        # Optional grouping
        use_grouping = st.checkbox("Group by a column", key="violin_plot_use_grouping")
        if use_grouping and cat_cols:
            group_col = st.selectbox("Group by:", cat_cols, key="violin_plot_group_col")
            
            # Create violin plot with grouping
            fig = px.violin(
                st.session_state.df,
                x=group_col,
                y=violin_col,
                box=True,
                points="all",
                title=f"Violin Plot of {violin_col}, grouped by {group_col}"
            )
        else:
            # Simple violin plot
            fig = px.violin(
                st.session_state.df,
                y=violin_col,
                box=True,
                points="all",
                title=f"Violin Plot of {violin_col}"
            )
    
    elif plot_type == "Residual Plot":
        # Column selection
        x_col = st.selectbox("Independent variable (x):", num_cols, key="residual_plot_x_col")
        y_col = st.selectbox("Dependent variable (y):", [col for col in num_cols if col != x_col], key="residual_plot_y_col")
        
        # Create residual plot
        try:
            # Fit linear regression
            from scipy import stats
            
            # Get data
            x = st.session_state.df[x_col].dropna()
            y = st.session_state.df[y_col].dropna()
            
            # Ensure same length
            df_clean = st.session_state.df[[x_col, y_col]].dropna()
            x = df_clean[x_col]
            y = df_clean[y_col]
            
            # Fit regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate predicted values
            y_pred = intercept + slope * x
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Create figure with subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Linear Regression", "Residuals"))
            
            # Add regression plot
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name='Data',
                    marker=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add regression line
            fig.add_trace(
                go.Scatter(
                    x=[x.min(), x.max()],
                    y=[intercept + slope * x.min(), intercept + slope * x.max()],
                    mode='lines',
                    name=f'Regression Line (r²={r_value**2:.3f})',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Add residual plot
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Add zero line to residual plot
            fig.add_trace(
                go.Scatter(
                    x=[x.min(), x.max()],
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
                height=700
            )
            
            # Update axis labels
            fig.update_xaxes(title_text=x_col, row=2, col=1)
            fig.update_yaxes(title_text=y_col, row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            
        except Exception as e:
            st.error(f"Error creating residual plot: {str(e)}")
            return
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text, key="statistical_plots_title")
        width = st.slider("Chart width:", 400, 1200, 800, key="statistical_plots_width")
        height = st.slider("Chart height:", 300, 1000, 600, key="statistical_plots_height")
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True,
            key="statistical_plots_export_format" 
        )
        
        # Export button
        if st.button("Export Chart", key="statistical_plots_export_button", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"statistical_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    key="statistical_plots_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"statistical_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    key=f"statistical_plots_download_{export_format.lower()}"
                )

def render_geospatial_plots_tab():
    """Render geospatial plots tab"""
    st.subheader("Geospatial Plots")
    
    # Check if we have potential geospatial data
    has_lat_lon = False
    lat_col = None
    lon_col = None
    
    # Look for latitude/longitude columns
    for col in st.session_state.df.columns:
        if col.lower() in ['lat', 'latitude', 'y']:
            lat_col = col
            has_lat_lon = True
        elif col.lower() in ['lon', 'long', 'longitude', 'x']:
            lon_col = col
            has_lat_lon = True
    
    # Get all numeric columns for potential coordinates
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.error("No numeric columns found for geospatial visualization.")
        return
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Scatter Map", "Bubble Map", "Choropleth Map", "Density Map"],
        key="geospatial_plots_type"
    )
    
    # Column selection for coordinates
    if has_lat_lon:
        st.info(f"Found potential latitude/longitude columns: {lat_col}, {lon_col}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lat_col = st.selectbox(
            "Latitude column:",
            num_cols,
            index=num_cols.index(lat_col) if lat_col in num_cols else 0,
            key="geospatial_lat_col"
        )
    
    with col2:
        lon_col = st.selectbox(
            "Longitude column:",
            [col for col in num_cols if col != lat_col],
            index=[col for col in num_cols if col != lat_col].index(lon_col) if lon_col in num_cols and lon_col != lat_col else 0,
            key="geospatial_lon_col"
        )
    
    # Additional parameters based on plot type
    if plot_type == "Scatter Map":
        color_col = st.selectbox(
            "Color by:",
            ["None"] + st.session_state.df.columns.tolist(),
            key="scatter_map_color_col"
        )
        
        hover_data = st.multiselect(
            "Additional hover data:",
            [col for col in st.session_state.df.columns if col not in [lat_col, lon_col]],
            default=[],
            key="scatter_map_hover_data"
        )
        
        # Create map
        map_params = {
            "lat": lat_col,
            "lon": lon_col,
            "hover_name": hover_data[0] if hover_data else None,
            "title": "Scatter Map"
        }
        
        if color_col != "None":
            map_params["color"] = color_col
        
        if hover_data:
            map_params["hover_data"] = hover_data
        
        # Check if coordinates contain valid data
        if st.session_state.df[lat_col].isna().all() or st.session_state.df[lon_col].isna().all():
            st.error(f"Selected columns {lat_col} and/or {lon_col} contain only NaN values.")
            return
            
        # Check if latitude is within range [-90, 90] and longitude is within range [-180, 180]
        lat_values = st.session_state.df[lat_col].dropna()
        lon_values = st.session_state.df[lon_col].dropna()
        
        if len(lat_values) == 0 or len(lon_values) == 0:
            st.error("Selected columns contain no valid coordinate data.")
            return
            
        if lat_values.min() < -90 or lat_values.max() > 90:
            st.warning(f"Latitude values should be between -90 and 90. Current range: [{lat_values.min()}, {lat_values.max()}]")
        
        if lon_values.min() < -180 or lon_values.max() > 180:
            st.warning(f"Longitude values should be between -180 and 180. Current range: [{lon_values.min()}, {lon_values.max()}]")
            
        try:
            fig = px.scatter_mapbox(
                st.session_state.df,
                **map_params,
                zoom=3,
                mapbox_style="carto-positron"
            )
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.info("Try selecting different columns for latitude and longitude, or check if they contain valid numeric data.")
            return
        
    elif plot_type == "Bubble Map":
        size_col = st.selectbox(
            "Size by:",
            num_cols,
            key="bubble_map_size_col"
        )
        
        color_col = st.selectbox(
            "Color by:",
            ["None"] + st.session_state.df.columns.tolist(),
            key="bubble_map_color_col"
        )
        
        hover_data = st.multiselect(
            "Additional hover data:",
            [col for col in st.session_state.df.columns if col not in [lat_col, lon_col, size_col]],
            default=[],
            key="bubble_map_hover_data"
        )
        
        # Create map
        map_params = {
            "lat": lat_col,
            "lon": lon_col,
            "size": size_col,
            "hover_name": hover_data[0] if hover_data else None,
            "title": "Bubble Map"
        }
        
        if color_col != "None":
            map_params["color"] = color_col
        
        if hover_data:
            map_params["hover_data"] = hover_data
        
        # Check if coordinates contain valid data
        if st.session_state.df[lat_col].isna().all() or st.session_state.df[lon_col].isna().all():
            st.error(f"Selected columns {lat_col} and/or {lon_col} contain only NaN values.")
            return
            
        # Check if latitude is within range [-90, 90] and longitude is within range [-180, 180]
        lat_values = st.session_state.df[lat_col].dropna()
        lon_values = st.session_state.df[lon_col].dropna()
        
        if len(lat_values) == 0 or len(lon_values) == 0:
            st.error("Selected columns contain no valid coordinate data.")
            return
            
        if lat_values.min() < -90 or lat_values.max() > 90:
            st.warning(f"Latitude values should be between -90 and 90. Current range: [{lat_values.min()}, {lat_values.max()}]")
        
        if lon_values.min() < -180 or lon_values.max() > 180:
            st.warning(f"Longitude values should be between -180 and 180. Current range: [{lon_values.min()}, {lon_values.max()}]")
        
        try:
            fig = px.scatter_mapbox(
                st.session_state.df,
                **map_params,
                zoom=3,
                mapbox_style="carto-positron"
            )
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.info("Try selecting different columns for latitude and longitude, or check if they contain valid numeric data.")
            return
        
    elif plot_type == "Choropleth Map":
        st.warning("Choropleth maps typically require geographic boundary data (GeoJSON) which is not part of this application. Simplified version shown.")
        
        # Get categorical columns
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            st.error("No categorical columns found for regions in choropleth map.")
            return
            
        # Select region column
        region_col = st.selectbox(
            "Region column (e.g., country, state):",
            cat_cols if cat_cols else st.session_state.df.columns.tolist(),
            key="choropleth_region_col"
        )
        
        # Select value column
        value_col = st.selectbox(
            "Value column:",
            num_cols,
            key="choropleth_value_col"
        )
        
        # Check for valid region data
        if st.session_state.df[region_col].isna().all():
            st.error(f"Selected region column {region_col} contains only NaN values.")
            return
            
        try:
            # Create simplified choropleth
            fig = px.choropleth(
                st.session_state.df,
                locations=region_col,
                locationmode="country names",  # Try to match with country names
                color=value_col,
                title="Choropleth Map",
                labels={value_col: value_col}
            )
        except Exception as e:
            st.error(f"Error creating choropleth map: {str(e)}")
            st.info("Make sure the region column contains valid country or region names.")
            return
        
    elif plot_type == "Density Map":
        # Density maps work best with lots of points
        if len(st.session_state.df) < 100:
            st.warning("Density maps work best with larger datasets (100+ points).")
        
        radius = st.slider("Density radius:", 5, 50, 20, key="density_map_radius")
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color scheme:",
            ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Reds", "Blues", "Greens"],
            key="density_map_color_scheme"
        )
        
        # Check if coordinates contain valid data
        if st.session_state.df[lat_col].isna().all() or st.session_state.df[lon_col].isna().all():
            st.error(f"Selected columns {lat_col} and/or {lon_col} contain only NaN values.")
            return
            
        # Check if latitude is within range [-90, 90] and longitude is within range [-180, 180]
        lat_values = st.session_state.df[lat_col].dropna()
        lon_values = st.session_state.df[lon_col].dropna()
        
        if len(lat_values) == 0 or len(lon_values) == 0:
            st.error("Selected columns contain no valid coordinate data.")
            return
            
        if lat_values.min() < -90 or lat_values.max() > 90:
            st.warning(f"Latitude values should be between -90 and 90. Current range: [{lat_values.min()}, {lat_values.max()}]")
        
        if lon_values.min() < -180 or lon_values.max() > 180:
            st.warning(f"Longitude values should be between -180 and 180. Current range: [{lon_values.min()}, {lon_values.max()}]")
        
        try:
            # Create density map
            fig = px.density_mapbox(
                st.session_state.df,
                lat=lat_col,
                lon=lon_col,
                radius=radius,
                zoom=3,
                mapbox_style="carto-positron",
                title="Density Map",
                color_continuous_scale=color_scheme.lower()
            )
        except Exception as e:
            st.error(f"Error creating density map: {str(e)}")
            st.info("Try selecting different columns for latitude and longitude, or check if they contain valid numeric data.")
            return
    
    # Map style
    map_style = st.selectbox(
        "Map style:",
        ["carto-positron", "open-street-map", "white-bg", "stamen-terrain", "stamen-toner", "carto-darkmatter"],
        key="geospatial_map_style"
    )
    
    # Update map style
    fig.update_layout(mapbox_style=map_style)
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text, key="geospatial_plots_title")
        width = st.slider("Chart width:", 400, 1200, 800, key="geospatial_plots_width")
        height = st.slider("Chart height:", 300, 1000, 600, key="geospatial_plots_height")
        
        # Map center and zoom
        zoom_level = st.slider("Zoom level:", 1, 20, 3, key="geospatial_plots_zoom")
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            mapbox=dict(
                zoom=zoom_level
            )
        )
    
    # Show the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True,
            key="geospatial_plots_export_format"
        )
        
        # Export button
        if st.button("Export Map", key="geospatial_plots_export_button", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name=f"map_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    key="geospatial_plots_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"map_{plot_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    key=f"geospatial_plots_download_{export_format.lower()}"
                )

def render_custom_viz_tab():
    """Render custom visualization tab using raw Plotly"""
    st.subheader("Custom Visualization")
    st.write("Create a custom visualization using Plotly's extensive capabilities.")
    
    # Create subtabs
    custom_tabs = st.tabs(["Template-based", "Chart Gallery", "Advanced JSON"])
    
    # Template-based tab
    with custom_tabs[0]:
        st.markdown("### Template-based Custom Visualization")
        st.write("Start with a template and customize it.")
        
        # Template selection
        template = st.selectbox(
            "Select template:",
            ["Bar Chart with Error Bars", "Multi-Y Axis", "Sunburst Chart", 
             "Radar Chart", "Candlestick Chart", "Waterfall Chart", "Gauge Chart"],
            key="custom_viz_template"
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Template-specific options
        if template == "Bar Chart with Error Bars":
            # Column selections
            x_col = st.selectbox("X-axis (categories):", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="error_bar_x_col")
            y_col = st.selectbox("Y-axis (values):", num_cols if num_cols else st.session_state.df.columns.tolist(), key="error_bar_y_col")
            
            # Error bar column
            error_col = st.selectbox("Error bar column:", ["None"] + num_cols, key="error_bar_error_col")
            
            # Create figure
            fig = go.Figure()
            
            if error_col != "None":
                # With error bars
                fig.add_trace(
                    go.Bar(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        error_y=dict(
                            type='data',
                            array=st.session_state.df[error_col],
                            visible=True
                        ),
                        name=y_col
                    )
                )
            else:
                # Without error bars
                fig.add_trace(
                    go.Bar(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        name=y_col
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"Bar Chart with Error Bars: {y_col} by {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col
            )
        
        elif template == "Multi-Y Axis":
            # Column selections
            x_col = st.selectbox("X-axis:", st.session_state.df.columns.tolist(), key="multi_y_x_col")
            
            y_cols = st.multiselect(
                "Y-axis columns:",
                num_cols,
                default=num_cols[:min(3, len(num_cols))],
                key="multi_y_y_cols"
            )
            
            if len(y_cols) < 1:
                st.warning("Please select at least one Y-axis column.")
                return
            
            # Create figure with multiple y-axes
            fig = go.Figure()
            
            # Colors for different traces
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
            
            # Add first trace on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df[x_col],
                    y=st.session_state.df[y_cols[0]],
                    name=y_cols[0],
                    line=dict(color=colors[0])
                )
            )
            
            # Add additional traces on secondary y-axes
            for i, y_col in enumerate(y_cols[1:]):
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df[x_col],
                        y=st.session_state.df[y_col],
                        name=y_col,
                        yaxis=f"y{i+2}",
                        line=dict(color=colors[(i+1) % len(colors)])
                    )
                )
            
            # Set up the layout with multiple y-axes
            layout = {
                "title": "Multi-Y Axis Chart",
                "xaxis": {"domain": [0.1, 0.9], "title": x_col},
                "yaxis": {"title": y_cols[0], "titlefont": {"color": colors[0]}, "tickfont": {"color": colors[0]}}
            }
            
            # Add secondary y-axes
            for i, y_col in enumerate(y_cols[1:]):
                pos = 0.05 * (i + 1)
                layout[f"yaxis{i+2}"] = {
                    "title": y_col,
                    "titlefont": {"color": colors[(i+1) % len(colors)]},
                    "tickfont": {"color": colors[(i+1) % len(colors)]},
                    "anchor": "x",
                    "overlaying": "y",
                    "side": "right" if i % 2 == 0 else "left",
                    "position": 1 + pos if i % 2 == 0 else 0 - pos
                }
            
            # Update layout
            fig.update_layout(**layout)
        
        elif template == "Sunburst Chart":
            # Column selections for path
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(2, len(cat_cols))],
                key="sunburst_path_cols"
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="sunburst_value_col")
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create sunburst chart
            fig = px.sunburst(
                st.session_state.df,
                path=path_cols,
                values=value_col,
                title=f"Sunburst Chart: {', '.join(path_cols)} with {value_col}"
            )
        
        elif template == "Radar Chart":
            # Column selections
            cat_col = st.selectbox("Category column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="radar_cat_col")
            
            value_cols = st.multiselect(
                "Value columns (radar axes):",
                num_cols,
                default=num_cols[:min(5, len(num_cols))],
                key="radar_value_cols"
            )
            
            if not value_cols:
                st.warning("Please select at least one value column.")
                return
            
            # Number of categories to show
            top_n = st.slider("Show top N categories:", 1, 10, 3, key="radar_top_n")
            
            # Get top categories
            top_categories = st.session_state.df[cat_col].value_counts().nlargest(top_n).index.tolist()
            
            # Create radar chart
            fig = go.Figure()
            
            for category in top_categories:
                # Filter for this category
                df_cat = st.session_state.df[st.session_state.df[cat_col] == category]
                
                # Calculate average for each value column
                values = [df_cat[col].mean() for col in value_cols]
                
                # Add radar trace
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=value_cols,
                        fill='toself',
                        name=str(category)
                    )
                )
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                title=f"Radar Chart for Top {top_n} {cat_col} Categories"
            )
        
        elif template == "Candlestick Chart":
            # Check if we have enough numeric columns
            if len(num_cols) < 4:
                st.warning("Candlestick charts require at least 4 numeric columns (open, high, low, close).")
                return
            
            # Column selections
            x_col = st.selectbox(
                "Date/category column:", 
                date_cols + cat_cols if date_cols + cat_cols else st.session_state.df.columns.tolist(),
                key="candlestick_x_col"
            )
            
            open_col = st.selectbox("Open column:", num_cols, key="candlestick_open_col")
            high_col = st.selectbox("High column:", [col for col in num_cols if col != open_col], key="candlestick_high_col")
            low_col = st.selectbox("Low column:", [col for col in num_cols if col not in [open_col, high_col]], key="candlestick_low_col")
            close_col = st.selectbox("Close column:", [col for col in num_cols if col not in [open_col, high_col, low_col]], key="candlestick_close_col")
            
            # Create candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=st.session_state.df[x_col],
                    open=st.session_state.df[open_col],
                    high=st.session_state.df[high_col],
                    low=st.session_state.df[low_col],
                    close=st.session_state.df[close_col],
                    name="Price"
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f"Candlestick Chart",
                xaxis_title=x_col
            )
        
        elif template == "Waterfall Chart":
            # Column selections
            cat_col = st.selectbox("Category column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="waterfall_cat_col")
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="waterfall_value_col")
            
            # Number of steps to show
            top_n = st.slider("Show top N steps:", 3, 20, 10, key="waterfall_top_n")
            
            # Sort data for waterfall
            sorted_df = st.session_state.df.sort_values(value_col, ascending=False).head(top_n)
            
            # Create a total row
            total_value = sorted_df[value_col].sum()
            
            # Create waterfall chart data
            measure = ["relative"] * len(sorted_df) + ["total"]
            x = sorted_df[cat_col].tolist() + ["Total"]
            y = sorted_df[value_col].tolist() + [total_value]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Waterfall",
                orientation="v",
                measure=measure,
                x=x,
                y=y,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Waterfall Chart: {value_col} by {cat_col}",
                xaxis_title=cat_col,
                yaxis_title=value_col
            )
        
        elif template == "Gauge Chart":
            # Column selection
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="gauge_value_col")
            
            # Gauge range
            min_val = st.number_input("Minimum value:", value=0.0, key="gauge_min_val")
            max_val = st.number_input("Maximum value:", value=100.0, key="gauge_max_val")
            
            # Reference levels
            low_thresh = st.slider("Low threshold:", min_val, max_val, (max_val - min_val) * 0.3 + min_val, key="gauge_low_thresh")
            high_thresh = st.slider("High threshold:", low_thresh, max_val, (max_val - min_val) * 0.7 + min_val, key="gauge_high_thresh")
            
            # Calculate average
            mean_val = st.session_state.df[value_col].mean()
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mean_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Average {value_col}"},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [min_val, low_thresh], 'color': "red"},
                        {'range': [low_thresh, high_thresh], 'color': "yellow"},
                        {'range': [high_thresh, max_val], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': mean_val
                    }
                }
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Gauge Chart: Average {value_col}"
            )
    
    # Chart Gallery tab
    with custom_tabs[1]:
        st.markdown("### Chart Gallery")
        st.write("Select from a gallery of pre-built advanced visualizations.")
        
        # Chart gallery selection
        gallery_chart = st.selectbox(
            "Select chart type:",
            ["Treemap", "Parallel Coordinates", "Sankey Diagram", "Icicle Chart", "Funnel Chart", "Timeline"],
            key="gallery_chart_type"
        )
        
        # Get numeric and categorical columns
        num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        
        if gallery_chart == "Treemap":
            # Column selections
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(2, len(cat_cols))],
                key="treemap_path_cols"
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="treemap_value_col")
            
            color_col = st.selectbox("Color by:", ["None"] + st.session_state.df.columns.tolist(), key="treemap_color_col")
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create treemap
            treemap_params = {
                "path": path_cols,
                "values": value_col,
                "title": f"Treemap: {', '.join(path_cols)} with {value_col}"
            }
            
            if color_col != "None":
                treemap_params["color"] = color_col
            
            fig = px.treemap(
                st.session_state.df,
                **treemap_params
            )
        
        elif gallery_chart == "Parallel Coordinates":
            # Column selections
            dimensions = st.multiselect(
                "Select dimensions:",
                st.session_state.df.columns.tolist(),
                default=st.session_state.df.columns[:min(5, len(st.session_state.df.columns))].tolist(),
                key="parallel_coords_dimensions"
            )
            
            color_col = st.selectbox("Color by:", ["None"] + st.session_state.df.columns.tolist(), key="parallel_coords_color_col")
            
            if not dimensions:
                st.warning("Please select at least one dimension.")
                return
            
            # Create parallel coordinates
            parallel_params = {
                "dimensions": dimensions,
                "title": "Parallel Coordinates Plot"
            }
            
            if color_col != "None":
                parallel_params["color"] = color_col
            
            fig = px.parallel_coordinates(
                st.session_state.df,
                **parallel_params
            )
        
        elif gallery_chart == "Sankey Diagram":
            st.write("Sankey diagrams show flows between nodes.")
            
            # Column selections
            source_col = st.selectbox("Source column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="sankey_source_col")
            target_col = st.selectbox("Target column:", [col for col in cat_cols if col != source_col] if len(cat_cols) > 1 else cat_cols, key="sankey_target_col")
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="sankey_value_col")
            
            # Limit number of nodes for readability
            max_nodes = st.slider("Maximum number of sources/targets:", 5, 50, 20, key="sankey_max_nodes")
            
            try:
                # Prepare Sankey data
                # Get top source nodes
                top_sources = st.session_state.df[source_col].value_counts().nlargest(max_nodes).index.tolist()
                
                # Get top target nodes
                top_targets = st.session_state.df[target_col].value_counts().nlargest(max_nodes).index.tolist()
                
                # Filter data to top nodes
                filtered_df = st.session_state.df[
                    st.session_state.df[source_col].isin(top_sources) & 
                    st.session_state.df[target_col].isin(top_targets)
                ]
                
                # Aggregate values
                sankey_df = filtered_df.groupby([source_col, target_col])[value_col].sum().reset_index()
                
                # Create lists for Sankey diagram
                all_nodes = list(set(sankey_df[source_col].tolist() + sankey_df[target_col].tolist()))
                node_indices = {node: i for i, node in enumerate(all_nodes)}
                
                # Convert source and target to indices
                sources = [node_indices[src] for src in sankey_df[source_col]]
                targets = [node_indices[tgt] for tgt in sankey_df[target_col]]
                values = sankey_df[value_col].tolist()
                
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values
                    )
                )])
                
                # Update layout
                fig.update_layout(
                    title=f"Sankey Diagram: {source_col} → {target_col}"
                )
                
            except Exception as e:
                st.error(f"Error creating Sankey diagram: {str(e)}")
                st.info("Try selecting different columns with categorical data or increase the maximum number of nodes.")
                return
        
        elif gallery_chart == "Icicle Chart":
            # Column selections
            path_cols = st.multiselect(
                "Select columns for hierarchy (in order):",
                cat_cols,
                default=cat_cols[:min(3, len(cat_cols))],
                key="icicle_path_cols"
            )
            
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="icicle_value_col")
            
            if not path_cols:
                st.warning("Please select at least one column for hierarchy.")
                return
            
            # Create icicle chart
            fig = px.icicle(
                st.session_state.df,
                path=path_cols,
                values=value_col,
                title=f"Icicle Chart: {', '.join(path_cols)} with {value_col}"
            )
        
        elif gallery_chart == "Funnel Chart":
            # Column selections
            stage_col = st.selectbox("Stage column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="funnel_stage_col")
            value_col = st.selectbox("Value column:", num_cols if num_cols else st.session_state.df.columns.tolist(), key="funnel_value_col")
            
            # Calculate values for funnel
            funnel_df = st.session_state.df.groupby(stage_col)[value_col].sum().reset_index()
            
            # Sort order
            sort_by = st.radio(
                "Sort stages by:",
                ["Value (descending)", "Value (ascending)", "Custom order"],
                horizontal=True,
                key="funnel_sort_order"
            )
            
            if sort_by == "Value (descending)":
                funnel_df = funnel_df.sort_values(value_col, ascending=False)
            elif sort_by == "Value (ascending)":
                funnel_df = funnel_df.sort_values(value_col, ascending=True)
            else:  # Custom order
                # Allow user to specify order
                stage_order = st.multiselect(
                    "Arrange stages in order (top to bottom):",
                    funnel_df[stage_col].unique().tolist(),
                    default=funnel_df[stage_col].unique().tolist(),
                    key="funnel_stage_order"
                )
                
                if stage_order:
                    # Create a mapping for sorting
                    stage_order_map = {stage: i for i, stage in enumerate(stage_order)}
                    
                    # Apply sorting
                    funnel_df['sort_order'] = funnel_df[stage_col].map(stage_order_map)
                    funnel_df = funnel_df.sort_values('sort_order').drop('sort_order', axis=1)
            
            # Create funnel chart
            fig = go.Figure(go.Funnel(
                y=funnel_df[stage_col],
                x=funnel_df[value_col],
                textinfo="value+percent initial"
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Funnel Chart: {stage_col} by {value_col}"
            )
        
        elif gallery_chart == "Timeline":
            # Check if we have date columns
            if not date_cols:
                st.warning("Timeline requires at least one date/time column. No datetime columns found.")
                return
            
            # Column selections
            date_col = st.selectbox("Date column:", date_cols, key="timeline_date_col")
            event_col = st.selectbox("Event column:", cat_cols if cat_cols else st.session_state.df.columns.tolist(), key="timeline_event_col")
            
            # Optional group column
            use_group = st.checkbox("Group events", key="timeline_use_group")
            if use_group and len(cat_cols) > 1:
                group_col = st.selectbox("Group by:", [col for col in cat_cols if col != event_col], key="timeline_group_col")
                
                # Create timeline with groups
                fig = px.timeline(
                    st.session_state.df,
                    x_start=date_col,
                    y=group_col,
                    color=event_col,
                    hover_name=event_col,
                    title=f"Timeline of {event_col} by {group_col}"
                )
            else:
                # Simple timeline
                # Need to create a fake "y" column for Plotly timeline
                dummy_y = "timeline"
                
                # Create a copy of the dataframe with the dummy column
                timeline_df = st.session_state.df.copy()
                timeline_df[dummy_y] = dummy_y
                
                fig = px.timeline(
                    timeline_df,
                    x_start=date_col,
                    y=dummy_y,
                    color=event_col,
                    hover_name=event_col,
                    title=f"Timeline of {event_col}"
                )
            
            # Update layout
            fig.update_yaxes(autorange="reversed")
    
    # Advanced JSON tab
    with custom_tabs[2]:
        st.markdown("### Advanced JSON Configuration")
        st.write("Use Plotly's JSON configuration for ultimate customization.")
        
        # JSON template examples
        template_options = {
            "None": "{}",
            "Bar Chart": """{
    "data": [
        {
            "type": "bar",
            "x": ["A", "B", "C", "D"],
            "y": [1, 3, 2, 4]
        }
    ],
    "layout": {
        "title": "Bar Chart Example"
    }
}""",
            "Line Chart": """{
    "data": [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": [1, 2, 3, 4, 5],
            "y": [1, 3, 2, 4, 3],
            "name": "Series 1"
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 1, 3, 5],
            "name": "Series 2"
        }
    ],
    "layout": {
        "title": "Line Chart Example"
    }
}""",
            "Pie Chart": """{
    "data": [
        {
            "type": "pie",
            "values": [30, 20, 15, 10, 25],
            "labels": ["Category A", "Category B", "Category C", "Category D", "Category E"]
        }
    ],
    "layout": {
        "title": "Pie Chart Example"
    }
}"""
        }
        
        # Template selection
        template_selection = st.selectbox("Choose a template:", list(template_options.keys()), key="json_template_selection")
        
        # JSON input
        json_input = st.text_area(
            "Enter or modify Plotly JSON:",
            value=template_options[template_selection],
            height=400,
            key="json_input_textarea"
        )
        
        # Help text
        st.markdown("""
        **Tips for JSON Configuration:**
        - The JSON must include both `data` and `layout` properties.
        - Use column names from your dataframe to replace example data.
        - For dynamic data, replace sample arrays with column references.
        - Visit the [Plotly JSON Chart Schema](https://plotly.com/chart-studio-help/json-chart-schema/) for more details.
        """)
        
        # Column reference helper
        with st.expander("Column Reference Helper", expanded=False):
            st.write("Select a column to get a JSON data reference snippet:")
            
            # Column selection
            ref_col = st.selectbox("Select column:", st.session_state.df.columns.tolist(), key="json_ref_col_selection")
            
            # Generate reference code
            st.code(f'st.session_state.df["{ref_col}"].tolist()')
            
            # Show sample of column data
            st.write(f"Sample values from {ref_col}:")
            st.write(st.session_state.df[ref_col].head(5).tolist())
        
        # Apply JSON button
        if st.button("Apply JSON Configuration", key="apply_json_button", use_container_width=True):
            try:
                # Parse JSON
                import json
                config = json.loads(json_input)
                
                # Validate basic structure
                if "data" not in config or "layout" not in config:
                    st.error("JSON must include both 'data' and 'layout' properties.")
                    return
                
                # Evaluate references to dataframe columns
                def process_data(data_item):
                    for key, value in list(data_item.items()):
                        if isinstance(value, str) and value.startswith("st.session_state.df["):
                            # Evaluate the expression to get actual data
                            try:
                                data_item[key] = eval(value)
                            except Exception as e:
                                st.error(f"Error evaluating {value}: {str(e)}")
                    return data_item
                
                # Process each data item
                for i, data_item in enumerate(config["data"]):
                    config["data"][i] = process_data(data_item)
                
                # Create figure from JSON
                fig = go.Figure(config)
                
            except Exception as e:
                st.error(f"Error parsing JSON configuration: {str(e)}")
                return
    
    # Chart settings
    with st.expander("Chart Settings", expanded=False):
        # Title and size
        chart_title = st.text_input("Chart title:", fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else "", key="custom_viz_title")
        width = st.slider("Chart width:", 400, 1200, 800, key="custom_viz_width")
        height = st.slider("Chart height:", 300, 1000, 600, key="custom_viz_height")
        
        # Theme selection
        theme = st.selectbox(
            "Color theme:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            key="custom_viz_theme"
        )
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            width=width,
            height=height,
            template=theme
        )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    with st.expander("Export Options", expanded=False):
        # Format selection
        export_format = st.radio(
            "Export format:",
            ["HTML", "PNG", "JPEG", "SVG"],
            horizontal=True,
            key="custom_viz_export_format"
        )
        
        # Export button
        if st.button("Export Chart", key="custom_viz_export_button", use_container_width=True):
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download HTML",
                    data=html_bytes,
                    file_name="custom_visualization.html",
                    mime="text/html",
                    key="custom_viz_download_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=img_bytes,
                    file_name=f"custom_visualization.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    key=f"custom_viz_download_{export_format.lower()}"
                )