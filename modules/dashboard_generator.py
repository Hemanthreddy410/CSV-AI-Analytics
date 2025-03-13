import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json
import datetime

class DashboardGenerator:
    """Class for generating interactive dashboards and visualizations"""
    
    def __init__(self, df=None):
        """Initialize with optional dataframe"""
        self.df = df
        self.dashboard_components = []
        self.layout = "vertical"  # or "grid"
        
        # Initialize dashboard settings in session state if not present
        if 'dashboard_settings' not in st.session_state:
            st.session_state.dashboard_settings = {
                'title': 'Data Dashboard',
                'description': '',
                'theme': 'light',
                'components': []
            }
    
    def set_dataframe(self, df):
        """Set the dataframe to use for the dashboard"""
        self.df = df
    
    def render_dashboard_builder(self):
        """Render the dashboard builder interface"""
        if self.df is None or self.df.empty:
            st.warning("No data available to build dashboard.")
            return
        
        st.header("Dashboard Builder")
        
        # Dashboard settings
        with st.expander("Dashboard Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.dashboard_settings['title'] = st.text_input(
                    "Dashboard Title", 
                    value=st.session_state.dashboard_settings['title']
                )
                
                st.session_state.dashboard_settings['theme'] = st.selectbox(
                    "Dashboard Theme",
                    ["light", "dark", "blue", "green"],
                    index=["light", "dark", "blue", "green"].index(st.session_state.dashboard_settings['theme'])
                )
            
            with col2:
                st.session_state.dashboard_settings['description'] = st.text_area(
                    "Description",
                    value=st.session_state.dashboard_settings['description']
                )
                
                self.layout = st.selectbox(
                    "Layout", 
                    ["vertical", "grid"], 
                    index=0 if self.layout == "vertical" else 1
                )
        
        # Add components section
        st.subheader("Add Dashboard Components")
        
        component_type = st.selectbox(
            "Component Type",
            ["Chart", "Data Table", "Metric Card", "Text Block", "Filter"]
        )
        
        # Render appropriate component builder based on selection
        if component_type == "Chart":
            self._render_chart_builder()
        elif component_type == "Data Table":
            self._render_table_builder()
        elif component_type == "Metric Card":
            self._render_metric_builder()
        elif component_type == "Text Block":
            self._render_text_builder()
        elif component_type == "Filter":
            self._render_filter_builder()
        
        # Show current dashboard components
        if st.session_state.dashboard_settings['components']:
            st.subheader("Current Dashboard Components")
            
            for i, component in enumerate(st.session_state.dashboard_settings['components']):
                with st.expander(f"{i+1}. {component['type']}: {component.get('title', 'Untitled')}"):
                    st.write(component)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"Remove Component #{i+1}", key=f"remove_{i}"):
                            st.session_state.dashboard_settings['components'].pop(i)
                            st.rerun()
                    
                    with col2:
                        if i > 0 and st.button(f"Move Up #{i+1}", key=f"up_{i}"):
                            component = st.session_state.dashboard_settings['components'].pop(i)
                            st.session_state.dashboard_settings['components'].insert(i-1, component)
                            st.rerun()
        
        # Preview and generate dashboard
        if st.button("Preview Dashboard", use_container_width=True):
            self.render_dashboard()
        
        # Export dashboard
        if st.session_state.dashboard_settings['components']:
            if st.button("Export Dashboard Configuration", use_container_width=True):
                # Export as JSON
                dashboard_config = json.dumps(st.session_state.dashboard_settings, indent=2)
                
                st.download_button(
                    label="Download Dashboard Configuration",
                    data=dashboard_config,
                    file_name=f"dashboard_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def _render_chart_builder(self):
        """Render chart builder interface"""
        st.subheader("Chart Builder")
        
        # Get columns by type
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Histogram", "Heatmap"]
        )
        
        with st.form("chart_form"):
            # Common settings for all chart types
            chart_title = st.text_input("Chart Title", "")
            
            # Chart specific settings
            if chart_type == "Bar Chart":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", cat_cols + date_cols if cat_cols or date_cols else num_cols)
                    orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
                
                with col2:
                    y_col = st.selectbox("Y-axis", num_cols if num_cols else ["No numeric columns available"])
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    color_by = None if color_by == "None" else color_by
            
            elif chart_type == "Line Chart":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", date_cols + num_cols if date_cols else num_cols)
                    line_shape = st.selectbox("Line Shape", ["linear", "spline", "hv", "vh", "hvh", "vhv"])
                
                with col2:
                    y_col = st.selectbox("Y-axis", num_cols if num_cols else ["No numeric columns available"])
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    color_by = None if color_by == "None" else color_by
            
            elif chart_type == "Pie Chart":
                col1, col2 = st.columns(2)
                with col1:
                    names_col = st.selectbox("Names", cat_cols if cat_cols else ["No categorical columns available"])
                    hole = st.slider("Donut Hole Size", 0.0, 0.8, 0.0, 0.1)
                
                with col2:
                    values_col = st.selectbox("Values", num_cols if num_cols else ["No numeric columns available"])
                    sort_values = st.checkbox("Sort Values", value=True)
            
            elif chart_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", num_cols if num_cols else ["No numeric columns available"])
                    size_col = st.selectbox("Size by (optional)", ["None"] + num_cols)
                    size_col = None if size_col == "None" else size_col
                
                with col2:
                    y_col = st.selectbox("Y-axis", [c for c in num_cols if c != x_col] if len(num_cols) > 1 else num_cols)
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols + num_cols)
                    color_by = None if color_by == "None" else color_by
            
            elif chart_type == "Box Plot":
                col1, col2 = st.columns(2)
                with col1:
                    y_col = st.selectbox("Values", num_cols if num_cols else ["No numeric columns available"])
                    notched = st.checkbox("Notched Boxes", value=False)
                
                with col2:
                    x_col = st.selectbox("Group by (optional)", ["None"] + cat_cols)
                    x_col = None if x_col == "None" else x_col
                    
                    points = st.selectbox("Show Points", ["outliers", "all", "suspectedoutliers", False])
            
            elif chart_type == "Histogram":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Column", num_cols if num_cols else ["No numeric columns available"])
                    nbins = st.slider("Number of Bins", 5, 100, 20)
                
                with col2:
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    color_by = None if color_by == "None" else color_by
                    
                    histnorm = st.selectbox(
                        "Normalization", 
                        ["count", "percent", "probability", "density", "probability density"]
                    )
            
            elif chart_type == "Heatmap":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", cat_cols + num_cols)
                    color_scale = st.selectbox(
                        "Color Scale", 
                        ["Viridis", "Plasma", "Inferno", "Blues", "Reds", "Greens", "YlOrRd", "RdBu"]
                    )
                
                with col2:
                    y_col = st.selectbox("Y-axis", [c for c in cat_cols + num_cols if c != x_col])
                    
                    if len(num_cols) > 0:
                        z_col = st.selectbox("Values (optional)", ["Count"] + num_cols)
                        z_col = None if z_col == "Count" else z_col
                    else:
                        z_col = None
            
            # Add chart button
            submit_button = st.form_submit_button("Add Chart", use_container_width=True)
            
            if submit_button:
                try:
                    # Create chart component configuration
                    chart_config = {
                        "type": "chart",
                        "chart_type": chart_type,
                        "title": chart_title,
                        "settings": {}
                    }
                    
                    # Add chart-specific settings
                    if chart_type == "Bar Chart":
                        chart_config["settings"] = {
                            "x_col": x_col,
                            "y_col": y_col,
                            "orientation": orientation.lower(),
                            "color_by": color_by
                        }
                    
                    elif chart_type == "Line Chart":
                        chart_config["settings"] = {
                            "x_col": x_col,
                            "y_col": y_col,
                            "line_shape": line_shape,
                            "color_by": color_by
                        }
                    
                    elif chart_type == "Pie Chart":
                        chart_config["settings"] = {
                            "names_col": names_col,
                            "values_col": values_col,
                            "hole": hole,
                            "sort_values": sort_values
                        }
                    
                    elif chart_type == "Scatter Plot":
                        chart_config["settings"] = {
                            "x_col": x_col,
                            "y_col": y_col,
                            "color_by": color_by,
                            "size_col": size_col
                        }
                    
                    elif chart_type == "Box Plot":
                        chart_config["settings"] = {
                            "y_col": y_col,
                            "x_col": x_col,
                            "notched": notched,
                            "points": points
                        }
                    
                    elif chart_type == "Histogram":
                        chart_config["settings"] = {
                            "x_col": x_col,
                            "nbins": nbins,
                            "color_by": color_by,
                            "histnorm": histnorm
                        }
                    
                    elif chart_type == "Heatmap":
                        chart_config["settings"] = {
                            "x_col": x_col,
                            "y_col": y_col,
                            "z_col": z_col,
                            "color_scale": color_scale.lower()
                        }
                    
                    # Add to dashboard components
                    st.session_state.dashboard_settings['components'].append(chart_config)
                    
                    st.success(f"Added {chart_type} to dashboard!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding chart: {str(e)}")
    
    def _render_table_builder(self):
        """Render table builder interface"""
        st.subheader("Data Table Builder")
        
        with st.form("table_form"):
            table_title = st.text_input("Table Title", "Data Table")
            
            # Column selection
            all_cols = list(self.df.columns)
            selected_cols = st.multiselect("Columns to Display", all_cols, default=all_cols[:5] if len(all_cols) > 5 else all_cols)
            
            # Table options
            col1, col2 = st.columns(2)
            with col1:
                max_rows = st.number_input("Max Rows to Display", 5, 1000, 10)
                sortable = st.checkbox("Sortable", value=True)
            
            with col2:
                filterable = st.checkbox("Filterable", value=True)
                show_index = st.checkbox("Show Index", value=False)
            
            # Add table button
            submit_button = st.form_submit_button("Add Table", use_container_width=True)
            
            if submit_button:
                try:
                    # Create table component configuration
                    table_config = {
                        "type": "data_table",
                        "title": table_title,
                        "settings": {
                            "columns": selected_cols,
                            "max_rows": max_rows,
                            "sortable": sortable,
                            "filterable": filterable,
                            "show_index": show_index
                        }
                    }
                    
                    # Add to dashboard components
                    st.session_state.dashboard_settings['components'].append(table_config)
                    
                    st.success("Added data table to dashboard!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding table: {str(e)}")
    
    def _render_metric_builder(self):
        """Render metric card builder interface"""
        st.subheader("Metric Card Builder")
        
        # Get numeric columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not num_cols:
            st.warning("No numeric columns available for metrics.")
            return
        
        with st.form("metric_form"):
            metric_title = st.text_input("Metric Title", "Key Metric")
            
            # Metric calculation
            calc_method = st.selectbox(
                "Calculation Method", 
                ["Value", "Count", "Sum", "Mean", "Median", "Min", "Max", "First", "Last"]
            )
            
            if calc_method == "Count":
                # Count can be applied to any column
                value_col = st.selectbox("Column", self.df.columns)
                agg_filter = st.text_input("Filter (e.g., >100 or ==1)", "")
            else:
                value_col = st.selectbox("Column", num_cols)
                
                if calc_method == "Value":
                    # For single value, need to specify which row
                    row_index = st.number_input("Row Index", 0, len(self.df)-1, 0)
            
            # Formatting
            col1, col2 = st.columns(2)
            with col1:
                prefix = st.text_input("Prefix", "")
                precision = st.number_input("Decimal Precision", 0, 10, 2)
            
            with col2:
                suffix = st.text_input("Suffix", "")
                color = st.color_picker("Value Color", "#2196F3")
            
            # Delta (change) calculation
            use_delta = st.checkbox("Show Delta (Change)", value=False)
            
            if use_delta:
                delta_method = st.selectbox(
                    "Delta Calculation", 
                    ["Static Value", "Percent Change", "Absolute Change", "vs Previous Period"]
                )
                
                if delta_method == "Static Value":
                    delta_value = st.number_input("Delta Value", value=0.0)
                elif delta_method in ["Percent Change", "Absolute Change"]:
                    # Need reference value
                    delta_reference = st.selectbox(
                        "Reference Value", 
                        ["Previous Row", "First Row", "Custom Value"]
                    )
                    
                    if delta_reference == "Custom Value":
                        delta_custom = st.number_input("Custom Reference Value", value=0.0)
                else:  # vs Previous Period
                    # Need date column for periods
                    date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
                    if not date_cols:
                        st.warning("No datetime columns available for period comparison.")
                        period_col = None
                    else:
                        period_col = st.selectbox("Date/Time Column", date_cols)
                        period_type = st.selectbox("Period Type", ["Day", "Week", "Month", "Quarter", "Year"])
            
            # Add metric button
            submit_button = st.form_submit_button("Add Metric", use_container_width=True)
            
            if submit_button:
                try:
                    # Create metric component configuration
                    metric_config = {
                        "type": "metric_card",
                        "title": metric_title,
                        "settings": {
                            "calc_method": calc_method,
                            "value_col": value_col,
                            "prefix": prefix,
                            "suffix": suffix,
                            "precision": precision,
                            "color": color,
                            "use_delta": use_delta
                        }
                    }
                    
                    # Add method-specific settings
                    if calc_method == "Value":
                        metric_config["settings"]["row_index"] = row_index
                    elif calc_method == "Count":
                        metric_config["settings"]["agg_filter"] = agg_filter
                    
                    # Add delta settings if applicable
                    if use_delta:
                        metric_config["settings"]["delta_method"] = delta_method
                        
                        if delta_method == "Static Value":
                            metric_config["settings"]["delta_value"] = delta_value
                        elif delta_method in ["Percent Change", "Absolute Change"]:
                            metric_config["settings"]["delta_reference"] = delta_reference
                            
                            if delta_reference == "Custom Value":
                                metric_config["settings"]["delta_custom"] = delta_custom
                        else:  # vs Previous Period
                            if period_col:
                                metric_config["settings"]["period_col"] = period_col
                                metric_config["settings"]["period_type"] = period_type
                    
                    # Add to dashboard components
                    st.session_state.dashboard_settings['components'].append(metric_config)
                    
                    st.success("Added metric card to dashboard!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding metric: {str(e)}")
    
    def _render_text_builder(self):
        """Render text block builder interface"""
        st.subheader("Text Block Builder")
        
        with st.form("text_form"):
            text_title = st.text_input("Block Title (optional)", "")
            
            # Text content
            text_content = st.text_area(
                "Content (Markdown supported)", 
                "Enter your text here. You can use **bold**, *italic*, and other [Markdown](https://www.markdownguide.org/basic-syntax/) formatting."
            )
            
            # Text styling
            col1, col2 = st.columns(2)
            with col1:
                text_size = st.selectbox("Text Size", ["Small", "Medium", "Large"])
                text_align = st.selectbox("Text Alignment", ["Left", "Center", "Right"])
            
            with col2:
                text_color = st.color_picker("Text Color", "#000000")
                bg_color = st.color_picker("Background Color", "#FFFFFF")
            
            # Add text block button
            submit_button = st.form_submit_button("Add Text Block", use_container_width=True)
            
            if submit_button:
                try:
                    # Create text block component configuration
                    text_config = {
                        "type": "text_block",
                        "title": text_title,
                        "settings": {
                            "content": text_content,
                            "size": text_size.lower(),
                            "align": text_align.lower(),
                            "text_color": text_color,
                            "bg_color": bg_color
                        }
                    }
                    
                    # Add to dashboard components
                    st.session_state.dashboard_settings['components'].append(text_config)
                    
                    st.success("Added text block to dashboard!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding text block: {str(e)}")
    
    def _render_filter_builder(self):
        """Render filter builder interface"""
        st.subheader("Filter Builder")
        
        with st.form("filter_form"):
            filter_title = st.text_input("Filter Title", "Data Filter")
            
            # Column selection
            all_cols = list(self.df.columns)
            filter_col = st.selectbox("Column to Filter", all_cols)
            
            # Determine filter type based on column data type
            col_type = str(self.df[filter_col].dtype)
            
            if "datetime" in col_type:
                filter_type = "date"
            elif "float" in col_type or "int" in col_type:
                filter_type = "numeric"
            else:
                filter_type = "categorical"
            
            # Filter type specific options
            if filter_type == "numeric":
                col1, col2 = st.columns(2)
                with col1:
                    min_val = float(self.df[filter_col].min())
                    max_val = float(self.df[filter_col].max())
                    
                    filter_control = st.selectbox(
                        "Filter Control", 
                        ["Slider", "Number Input", "Range"]
                    )
                
                with col2:
                    step = (max_val - min_val) / 100
                    step = max(step, 0.01)
                    
                    format_type = st.selectbox(
                        "Format", 
                        ["Float", "Integer", "Percentage"]
                    )
            
            elif filter_type == "categorical":
                filter_control = st.selectbox(
                    "Filter Control", 
                    ["Dropdown", "Multiselect", "Radio", "Checkbox"]
                )
                
                # Limit choices for large categorical columns
                unique_values = self.df[filter_col].nunique()
                if unique_values > 100:
                    st.warning(f"Column has {unique_values} unique values. Only the top 100 will be available in the filter.")
            
            elif filter_type == "date":
                filter_control = st.selectbox(
                    "Filter Control", 
                    ["Date Picker", "Date Range"]
                )
            
            # Default value
            show_all_option = st.checkbox("Include 'All' or 'None' option", value=True)
            
            # Filter layout
            col1, col2 = st.columns(2)
            with col1:
                orientation = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True)
            
            with col2:
                auto_apply = st.checkbox("Auto-apply filter", value=True)
            
            # Add filter button
            submit_button = st.form_submit_button("Add Filter", use_container_width=True)
            
            if submit_button:
                try:
                    # Create filter component configuration
                    filter_config = {
                        "type": "filter",
                        "title": filter_title,
                        "settings": {
                            "column": filter_col,
                            "filter_type": filter_type,
                            "filter_control": filter_control,
                            "show_all_option": show_all_option,
                            "orientation": orientation.lower(),
                            "auto_apply": auto_apply
                        }
                    }
                    
                    # Add type-specific settings
                    if filter_type == "numeric":
                        filter_config["settings"]["min_val"] = min_val
                        filter_config["settings"]["max_val"] = max_val
                        filter_config["settings"]["step"] = step
                        filter_config["settings"]["format_type"] = format_type.lower()
                    
                    # Add to dashboard components
                    st.session_state.dashboard_settings['components'].append(filter_config)
                    
                    st.success("Added filter to dashboard!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding filter: {str(e)}")
    
    def render_dashboard(self):
        """Render the dashboard based on current components"""
        # Ensure we have data
        if self.df is None or self.df.empty:
            st.warning("No data available to render dashboard.")
            return
        
        # Ensure we have components
        if not st.session_state.dashboard_settings['components']:
            st.info("Add components to your dashboard using the options above.")
            return
        
        # Create a container for the dashboard
        dashboard_container = st.container()
        
        with dashboard_container:
            # Dashboard header
            st.title(st.session_state.dashboard_settings['title'])
            
            if st.session_state.dashboard_settings['description']:
                st.write(st.session_state.dashboard_settings['description'])
            
            # Render dashboard components based on layout
            if self.layout == "grid":
                # Use columns for grid layout
                num_components = len(st.session_state.dashboard_settings['components'])
                cols_per_row = 2  # Can be adjusted
                
                # Calculate number of rows needed
                num_rows = (num_components + cols_per_row - 1) // cols_per_row
                
                for row in range(num_rows):
                    # Create columns for this row
                    cols = st.columns(cols_per_row)
                    
                    # Add components to columns
                    for col_idx in range(cols_per_row):
                        component_idx = row * cols_per_row + col_idx
                        
                        if component_idx < num_components:
                            with cols[col_idx]:
                                self._render_component(st.session_state.dashboard_settings['components'][component_idx])
            else:
                # Vertical layout - just render each component sequentially
                for component in st.session_state.dashboard_settings['components']:
                    self._render_component(component)
    
    def _render_component(self, component):
        """Render a single dashboard component"""
        try:
            component_type = component['type']
            
            # Render appropriate component based on type
            if component_type == "chart":
                self._render_chart_component(component)
            elif component_type == "data_table":
                self._render_table_component(component)
            elif component_type == "metric_card":
                self._render_metric_component(component)
            elif component_type == "text_block":
                self._render_text_component(component)
            elif component_type == "filter":
                self._render_filter_component(component)
        except Exception as e:
            st.error(f"Error rendering component: {str(e)}")
    
    def _render_chart_component(self, component):
        """Render a chart component"""
        chart_type = component['chart_type']
        settings = component['settings']
        
        if component.get('title'):
            st.subheader(component['title'])
        
        try:
            # Create appropriate chart based on type
            if chart_type == "Bar Chart":
                # Get settings
                x_col = settings['x_col']
                y_col = settings['y_col']
                orientation = settings.get('orientation', 'vertical')
                color_by = settings.get('color_by')
                
                # Create chart
                if orientation == 'horizontal':
                    fig = px.bar(
                        self.df, 
                        y=x_col, 
                        x=y_col, 
                        color=color_by,
                        orientation='h'
                    )
                else:
                    fig = px.bar(
                        self.df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_by
                    )
            
            elif chart_type == "Line Chart":
                # Get settings
                x_col = settings['x_col']
                y_col = settings['y_col']
                line_shape = settings.get('line_shape', 'linear')
                color_by = settings.get('color_by')
                
                # Create chart
                fig = px.line(
                    self.df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_by,
                    line_shape=line_shape
                )
            
            elif chart_type == "Pie Chart":
                # Get settings
                names_col = settings['names_col']
                values_col = settings['values_col']
                hole = settings.get('hole', 0)
                sort_values = settings.get('sort_values', True)
                
                # For pie charts, often need to aggregate data
                pie_data = self.df.groupby(names_col)[values_col].sum().reset_index()
                
                # Sort if requested
                if sort_values:
                    pie_data = pie_data.sort_values(values_col, ascending=False)
                
                # Create chart
                fig = px.pie(
                    pie_data, 
                    names=names_col, 
                    values=values_col,
                    hole=hole
                )
            
            elif chart_type == "Scatter Plot":
                # Get settings
                x_col = settings['x_col']
                y_col = settings['y_col']
                color_by = settings.get('color_by')
                size_col = settings.get('size_col')
                
                # Create chart
                fig = px.scatter(
                    self.df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_by,
                    size=size_col,
                    opacity=0.7
                )
            
            elif chart_type == "Box Plot":
                # Get settings
                y_col = settings['y_col']
                x_col = settings.get('x_col')
                notched = settings.get('notched', False)
                points = settings.get('points', 'outliers')
                
                # Create chart
                fig = px.box(
                    self.df, 
                    y=y_col, 
                    x=x_col,
                    notched=notched,
                    points=points
                )
            
            elif chart_type == "Histogram":
                # Get settings
                x_col = settings['x_col']
                nbins = settings.get('nbins', 20)
                color_by = settings.get('color_by')
                histnorm = settings.get('histnorm', 'count')
                
                # Create chart
                fig = px.histogram(
                    self.df, 
                    x=x_col, 
                    nbins=nbins,
                    color=color_by,
                    histnorm=histnorm
                )
            
            elif chart_type == "Heatmap":
                # Get settings
                x_col = settings['x_col']
                y_col = settings['y_col']
                z_col = settings.get('z_col')
                color_scale = settings.get('color_scale', 'viridis')
                
                # For heatmaps, need to pivot data
                if z_col:
                    # Use z_col for values
                    pivot_data = self.df.pivot_table(
                        index=y_col, 
                        columns=x_col, 
                        values=z_col,
                        aggfunc='mean'
                    )
                else:
                    # Count occurrences
                    pivot_data = pd.crosstab(self.df[y_col], self.df[x_col])
                
                # Create chart
                fig = px.imshow(
                    pivot_data,
                    color_continuous_scale=color_scale
                )
            
            # Set chart height
            fig.update_layout(height=400)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
    
    def _render_table_component(self, component):
        """Render a data table component"""
        settings = component['settings']
        
        if component.get('title'):
            st.subheader(component['title'])
        
        try:
            # Get settings
            columns = settings.get('columns', list(self.df.columns))
            max_rows = settings.get('max_rows', 10)
            sortable = settings.get('sortable', True)
            filterable = settings.get('filterable', True)
            show_index = settings.get('show_index', False)
            
            # Display the table
            table_df = self.df[columns].head(max_rows)
            
            if filterable:
                st.dataframe(table_df, use_container_width=True, hide_index=not show_index)
            else:
                st.table(table_df)
        
        except Exception as e:
            st.error(f"Error rendering table: {str(e)}")
    
    def _render_metric_component(self, component):
        """Render a metric card component"""
        settings = component['settings']
        
        try:
            # Get settings
            calc_method = settings['calc_method']
            value_col = settings['value_col']
            prefix = settings.get('prefix', '')
            suffix = settings.get('suffix', '')
            precision = settings.get('precision', 2)
            use_delta = settings.get('use_delta', False)
            
            # Calculate the metric value
            if calc_method == "Value":
                row_index = settings.get('row_index', 0)
                value = self.df.iloc[row_index][value_col]
            elif calc_method == "Count":
                agg_filter = settings.get('agg_filter', '')
                if agg_filter:
                    # Apply filter using eval
                    filter_expr = f"{value_col} {agg_filter}"
                    filtered_df = self.df.query(filter_expr)
                    value = len(filtered_df)
                else:
                    value = len(self.df)
            elif calc_method == "Sum":
                value = self.df[value_col].sum()
            elif calc_method == "Mean":
                value = self.df[value_col].mean()
            elif calc_method == "Median":
                value = self.df[value_col].median()
            elif calc_method == "Min":
                value = self.df[value_col].min()
            elif calc_method == "Max":
                value = self.df[value_col].max()
            elif calc_method == "First":
                value = self.df[value_col].iloc[0]
            elif calc_method == "Last":
                value = self.df[value_col].iloc[-1]
            
            # Format the value
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.{precision}f}"
            else:
                formatted_value = str(value)
            
            # Calculate delta if needed
            delta = None
            delta_color = "normal"
            
            if use_delta:
                delta_method = settings.get('delta_method')
                
                if delta_method == "Static Value":
                    delta = settings.get('delta_value', 0)
                elif delta_method in ["Percent Change", "Absolute Change"]:
                    delta_reference = settings.get('delta_reference')
                    
                    if delta_reference == "Previous Row":
                        if len(self.df) > 1:
                            ref_value = self.df[value_col].iloc[-2]
                        else:
                            ref_value = value
                    elif delta_reference == "First Row":
                        ref_value = self.df[value_col].iloc[0]
                    else:  # Custom Value
                        ref_value = settings.get('delta_custom', 0)
                    
                    if delta_method == "Percent Change":
                        if ref_value != 0:
                            delta = ((value - ref_value) / ref_value) * 100
                            delta = f"{delta:.{precision}}%"
                        else:
                            delta = "N/A"
                    else:  # Absolute Change
                        delta = value - ref_value
                        delta = f"{delta:.{precision}f}"
                
                # Set delta color
                if isinstance(delta, (int, float)):
                    delta_color = "normal" if delta >= 0 else "inverse"
            
            # Display the metric
            full_title = component.get('title', '')
            if full_title:
                st.metric(
                    label=full_title,
                    value=f"{prefix}{formatted_value}{suffix}",
                    delta=delta,
                    delta_color=delta_color
                )
            else:
                st.metric(
                    label=f"{value_col} ({calc_method})",
                    value=f"{prefix}{formatted_value}{suffix}",
                    delta=delta,
                    delta_color=delta_color
                )
        
        except Exception as e:
            st.error(f"Error rendering metric: {str(e)}")
    
    def _render_text_component(self, component):
        """Render a text block component"""
        settings = component['settings']
        
        try:
            # Get settings
            content = settings.get('content', '')
            size = settings.get('size', 'medium')
            align = settings.get('align', 'left')
            text_color = settings.get('text_color', '#000000')
            bg_color = settings.get('bg_color', '#FFFFFF')
            
            # Apply title if provided
            if component.get('title'):
                st.subheader(component['title'])
            
            # Set CSS for the text block
            text_style = {
                'small': 'font-size: 0.9em;',
                'medium': 'font-size: 1em;',
                'large': 'font-size: 1.2em;'
            }
            
            align_style = {
                'left': 'text-align: left;',
                'center': 'text-align: center;',
                'right': 'text-align: right;'
            }
            
            css = f"""
            <style>
            .text-block {{
                {text_style[size]}
                {align_style[align]}
                color: {text_color};
                background-color: {bg_color};
                padding: 1rem;
                border-radius: 0.5rem;
            }}
            </style>
            """
            
            # Create the text block
            html = f"""
            {css}
            <div class="text-block">
                {content}
            </div>
            """
            
            # Display the text block
            st.markdown(html, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error rendering text block: {str(e)}")
    
    def _render_filter_component(self, component):
        """Render a filter component"""
        settings = component['settings']
        
        try:
            # Get settings
            column = settings['column']
            filter_type = settings['filter_type']
            filter_control = settings['filter_control']
            show_all_option = settings.get('show_all_option', True)
            orientation = settings.get('orientation', 'horizontal')
            auto_apply = settings.get('auto_apply', True)
            
            # Set component title
            if component.get('title'):
                st.subheader(component['title'])
            
            # Create filter key
            filter_key = f"filter_{column}_{filter_control}"
            
            # Create layout based on orientation
            if orientation == 'horizontal':
                cols = st.columns([3, 1])
                filter_container = cols[0]
                button_container = cols[1]
            else:
                filter_container = st
                button_container = st
            
            # Create filter based on type and control
            with filter_container:
                if filter_type == "numeric":
                    min_val = settings.get('min_val', 0)
                    max_val = settings.get('max_val', 100)
                    step = settings.get('step', 1)
                    format_type = settings.get('format_type', 'float')
                    
                    if filter_control == "Slider":
                        if format_type == "integer":
                            min_val = int(min_val)
                            max_val = int(max_val)
                            step = int(max(step, 1))
                            
                            filter_value = st.slider(
                                f"Select {column}",
                                min_value=min_val,
                                max_value=max_val,
                                step=step,
                                key=filter_key
                            )
                        else:
                            filter_value = st.slider(
                                f"Select {column}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                step=float(step),
                                key=filter_key
                            )
                    
                    elif filter_control == "Number Input":
                        if format_type == "integer":
                            filter_value = st.number_input(
                                f"Enter {column}",
                                min_value=int(min_val),
                                max_value=int(max_val),
                                step=int(max(step, 1)),
                                key=filter_key
                            )
                        else:
                            filter_value = st.number_input(
                                f"Enter {column}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                step=float(step),
                                key=filter_key
                            )
                    
                    elif filter_control == "Range":
                        if format_type == "integer":
                            min_val = int(min_val)
                            max_val = int(max_val)
                            step = int(max(step, 1))
                            
                            filter_value = st.slider(
                                f"Select {column} range",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                step=step,
                                key=filter_key
                            )
                        else:
                            filter_value = st.slider(
                                f"Select {column} range",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=(float(min_val), float(max_val)),
                                step=float(step),
                                key=filter_key
                            )
                
                elif filter_type == "categorical":
                    # Get unique values
                    unique_values = self.df[column].dropna().unique()
                    
                    # Limit to top 100 for performance
                    if len(unique_values) > 100:
                        value_counts = self.df[column].value_counts().nlargest(100)
                        unique_values = value_counts.index.tolist()
                    
                    if filter_control == "Dropdown":
                        options = unique_values.tolist()
                        if show_all_option:
                            options = ["All"] + options
                        
                        filter_value = st.selectbox(
                            f"Select {column}",
                            options,
                            key=filter_key
                        )
                    
                    elif filter_control == "Multiselect":
                        options = unique_values.tolist()
                        if show_all_option:
                            default = ["All"]
                        else:
                            default = []
                        
                        filter_value = st.multiselect(
                            f"Select {column}(s)",
                            options,
                            default=default,
                            key=filter_key
                        )
                    
                    elif filter_control == "Radio":
                        options = unique_values.tolist()
                        if show_all_option:
                            options = ["All"] + options
                        
                        filter_value = st.radio(
                            f"Select {column}",
                            options,
                            key=filter_key
                        )
                    
                    elif filter_control == "Checkbox":
                        # For checkbox, use a limited number of options
                        options = unique_values[:min(5, len(unique_values))].tolist()
                        
                        # Create a checkbox for each option
                        filter_value = []
                        for option in options:
                            if st.checkbox(str(option), key=f"{filter_key}_{option}"):
                                filter_value.append(option)
                
                elif filter_type == "date":
                    # Convert column to datetime if needed
                    if not pd.api.types.is_datetime64_dtype(self.df[column]):
                        datetime_col = pd.to_datetime(self.df[column], errors='coerce')
                    else:
                        datetime_col = self.df[column]
                    
                    min_date = datetime_col.min().date()
                    max_date = datetime_col.max().date()
                    
                    if filter_control == "Date Picker":
                        filter_value = st.date_input(
                            f"Select {column}",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=filter_key
                        )
                    
                    elif filter_control == "Date Range":
                        filter_value = st.date_input(
                            f"Select {column} range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=filter_key
                        )
            
            # Apply button if not auto-apply
            if not auto_apply:
                with button_container:
                    apply_filter = st.button("Apply Filter", key=f"apply_{filter_key}")
            else:
                apply_filter = True
            
            # Apply filter to the dataframe
            if apply_filter:
                # We don't actually filter the dataframe in this demo
                # In a real application, this would update a filtered view of the data
                
                # Show filter info
                if filter_type == "numeric":
                    if filter_control == "Range":
                        min_val, max_val = filter_value
                        st.caption(f"Filtering {column} between {min_val} and {max_val}")
                    else:
                        st.caption(f"Filtering {column} = {filter_value}")
                
                elif filter_type == "categorical":
                    if filter_control == "Multiselect" or filter_control == "Checkbox":
                        if filter_value and filter_value[0] == "All":
                            st.caption(f"Showing all values for {column}")
                        else:
                            st.caption(f"Filtering {column} to {', '.join(str(v) for v in filter_value)}")
                    else:
                        if filter_value == "All":
                            st.caption(f"Showing all values for {column}")
                        else:
                            st.caption(f"Filtering {column} = {filter_value}")
                
                elif filter_type == "date":
                    if filter_control == "Date Range":
                        start_date, end_date = filter_value
                        st.caption(f"Filtering {column} between {start_date} and {end_date}")
                    else:
                        st.caption(f"Filtering {column} = {filter_value}")
        
        except Exception as e:
            st.error(f"Error rendering filter: {str(e)}")
    
    def import_dashboard_config(self, config_json):
        """Import dashboard configuration from JSON"""
        try:
            config = json.loads(config_json)
            
            # Update dashboard settings
            st.session_state.dashboard_settings = config
            
            return True
        except Exception as e:
            st.error(f"Error importing dashboard configuration: {str(e)}")
            return False