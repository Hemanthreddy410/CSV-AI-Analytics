import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import statsmodels.api as sm
from scipy import stats

def render_analysis():
    """Render analysis section"""
    st.header("Data Analysis")
    
    # Create tabs for different analysis types
    analysis_tabs = st.tabs([
        "Summary Statistics",
        "Correlation Analysis", 
        "Distribution Analysis",
        "Trend Analysis",
        "Hypothesis Testing"
    ])
    
    # Summary Statistics Tab
    with analysis_tabs[0]:
        render_summary_statistics()
    
    # Correlation Analysis Tab
    with analysis_tabs[1]:
        render_correlation_analysis()
    
    # Distribution Analysis Tab
    with analysis_tabs[2]:
        render_distribution_analysis()
    
    # Trend Analysis Tab
    with analysis_tabs[3]:
        render_trend_analysis()
    
    # Hypothesis Testing Tab
    with analysis_tabs[4]:
        render_hypothesis_testing()

def render_summary_statistics():
    """Render summary statistics tab"""
    st.subheader("Summary Statistics")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns found for summary statistics.")
        return
    
    # Column selection
    selected_cols = st.multiselect(
        "Select columns for statistics:",
        num_cols,
        default=num_cols,
        key="summary_stats_columns"
    )
    
    if not selected_cols:
        st.warning("Please select at least one column.")
        return
    
    # Calculate statistics
    stats_df = st.session_state.df[selected_cols].describe().T
    
    # Add more statistics
    stats_df['median'] = st.session_state.df[selected_cols].median()
    stats_df['range'] = st.session_state.df[selected_cols].max() - st.session_state.df[selected_cols].min()
    stats_df['missing'] = st.session_state.df[selected_cols].isna().sum()
    stats_df['missing_pct'] = (st.session_state.df[selected_cols].isna().sum() / len(st.session_state.df) * 100).round(2)
    
    try:
        stats_df['skew'] = st.session_state.df[selected_cols].skew().round(3)
        stats_df['kurtosis'] = st.session_state.df[selected_cols].kurtosis().round(3)
    except:
        pass
    
    # Round numeric columns
    stats_df = stats_df.round(3)
    
    # Display the statistics
    st.dataframe(stats_df, use_container_width=True)
    
    # Visualization of statistics
    st.subheader("Statistical Visualization")
    
    # Statistic to visualize
    stat_type = st.selectbox(
        "Statistic to visualize:",
        ["Mean", "Median", "Standard Deviation", "Range", "Missing Values %"],
        key="summary_stat_type"
    )
    
    # Map statistic to column name
    stat_map = {
        "Mean": "mean",
        "Median": "median",
        "Standard Deviation": "std",
        "Range": "range",
        "Missing Values %": "missing_pct"
    }
    
    # Create visualization
    if stat_type in stat_map:
        fig = px.bar(
            stats_df.reset_index(),
            x="index",
            y=stat_map[stat_type],
            title=f"{stat_type} by Column",
            labels={"index": "Column", "y": stat_type}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # Export statistics to CSV
        csv = stats_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download Statistics as CSV",
            data=csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_summary_stats"
        )
    
    with col2:
        # Export chart
        if 'fig' in locals():
            # Format selection
            export_format = st.selectbox(
                "Export chart format:", 
                ["PNG", "SVG", "HTML"],
                key="summary_export_format"
            )
            
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Chart",
                    data=html_bytes,
                    file_name=f"statistics_{stat_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_summary_chart_html"
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Chart as {export_format}",
                    data=img_bytes,
                    file_name=f"statistics_{stat_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True,
                    key=f"download_summary_chart_{export_format.lower()}"
                )

def render_correlation_analysis():
    """Render correlation analysis tab"""
    st.subheader("Correlation Analysis")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for correlation analysis.")
        return
    
    # Column selection
    selected_cols = st.multiselect(
        "Select columns for correlation analysis:",
        num_cols,
        default=num_cols[:min(5, len(num_cols))],
        key="corr_analysis_columns"
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least two columns.")
        return
    
    # Correlation method
    corr_method = st.radio(
        "Correlation method:",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True,
        key="corr_method"
    )
    
    # Calculate correlation matrix
    corr_matrix = st.session_state.df[selected_cols].corr(method=corr_method.lower())
    
    # Round to 3 decimal places
    corr_matrix = corr_matrix.round(3)
    
    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix, use_container_width=True)
    
    # Visualization options
    viz_type = st.radio(
        "Visualization type:",
        ["Heatmap", "Scatter Matrix", "Network Graph"],
        horizontal=True,
        key="viz_type"
    )
    
    if viz_type == "Heatmap":
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f"{corr_method} Correlation Heatmap",
            labels=dict(color="Correlation")
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )
        
    elif viz_type == "Scatter Matrix":
        # Create scatter matrix
        fig = px.scatter_matrix(
            st.session_state.df[selected_cols],
            title=f"Scatter Matrix ({corr_method} Correlation)",
            dimensions=selected_cols
        )
        
        # Update traces
        fig.update_traces(diagonal_visible=False)
        
    elif viz_type == "Network Graph":
        # Network graph visualization of correlations
        try:
            import networkx as nx
            
            # Create a graph from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for col in selected_cols:
                G.add_node(col)
            
            # Add edges with correlation values as weights
            for i, col1 in enumerate(selected_cols):
                for col2 in selected_cols[i+1:]:
                    # Only add edges for correlations above a threshold
                    corr_value = abs(corr_matrix.loc[col1, col2])
                    if corr_value > 0.1:  # Ignore very weak correlations
                        G.add_edge(col1, col2, weight=corr_value)
            
            # Get node positions using force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create network graph
            node_x = []
            node_y = []
            for key, value in pos.items():
                node_x.append(value[0])
                node_y.append(value[1])
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='skyblue',
                    line=dict(width=2, color='darkblue')
                ),
                text=list(pos.keys()),
                textposition='top center',
                hoverinfo='text'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Correlation Network Graph ({corr_method})",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
        except Exception as e:
            st.error(f"Error creating network graph: {str(e)}")
            st.info("Try using the heatmap or scatter matrix visualization instead.")
            return
    
    # Show visualization
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    st.subheader("Top Correlations")
    
    # Convert correlation matrix to a list of pairs
    pairs = []
    for i, col1 in enumerate(selected_cols):
        for j, col2 in enumerate(selected_cols):
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
    
    # Display top correlations
    st.dataframe(pairs_df, use_container_width=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # Export correlation matrix to CSV
        csv = corr_matrix.reset_index().to_csv(index=False)
        st.download_button(
            label="Download Correlation Matrix",
            data=csv,
            file_name=f"correlation_matrix_{corr_method.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_corr_matrix"
        )
    
    with col2:
        # Export visualization
        export_format = st.selectbox(
            "Export chart format:", 
            ["PNG", "SVG", "HTML"],
            key="corr_export_format"
        )
        
        if export_format == "HTML":
            # Export as HTML file
            buffer = StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Visualization",
                data=html_bytes,
                file_name=f"correlation_{viz_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True,
                key="download_corr_viz_html"
            )
        else:
            # Export as image
            img_bytes = fig.to_image(format=export_format.lower())
            
            st.download_button(
                label=f"Download Visualization as {export_format}",
                data=img_bytes,
                file_name=f"correlation_{viz_type.lower().replace(' ', '_')}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True,
                key=f"download_corr_viz_{export_format.lower()}"
            )

def render_distribution_analysis():
    """Render distribution analysis tab"""
    st.subheader("Distribution Analysis")
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        st.info("No numeric columns available for distribution analysis.")
        return
    
    # Column selection
    col = st.selectbox(
        "Select column for analysis:", 
        num_cols,
        key="dist_analysis_column"
    )
    
    # Get data and remove NaNs
    data = st.session_state.df[col].dropna()
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{data.mean():.4f}")
    with col2:
        st.metric("Median", f"{data.median():.4f}")
    with col3:
        st.metric("Std Dev", f"{data.std():.4f}")
    with col4:
        st.metric("IQR", f"{np.percentile(data, 75) - np.percentile(data, 25):.4f}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{data.min():.4f}")
    with col2:
        st.metric("Max", f"{data.max():.4f}")
    with col3:
        try:
            st.metric("Skewness", f"{data.skew():.4f}")
        except:
            st.metric("Skewness", "N/A")
    with col4:
        try:
            st.metric("Kurtosis", f"{data.kurtosis():.4f}")
        except:
            st.metric("Kurtosis", "N/A")
    
    # Visualization options
    viz_type = st.radio(
        "Visualization type:",
        ["Histogram", "Box Plot", "Violin Plot", "KDE Plot", "QQ Plot", "ECDF Plot"],
        horizontal=True,
        key="dist_viz_type"
    )
    
    # Optional transformation
    transform_type = st.selectbox(
        "Apply transformation:",
        ["None", "Log", "Square Root", "Square", "Standardize (Z-score)", "Min-Max Scale"],
        key="dist_transform"
    )
    
    # Apply transformation
    if transform_type == "None":
        transformed_data = data
        transform_label = ""
    elif transform_type == "Log":
        if data.min() <= 0:
            st.warning("Cannot apply log transform to zero or negative values. Using original data.")
            transformed_data = data
            transform_label = ""
        else:
            transformed_data = np.log(data)
            transform_label = "Log "
    elif transform_type == "Square Root":
        if data.min() < 0:
            st.warning("Cannot apply square root to negative values. Using original data.")
            transformed_data = data
            transform_label = ""
        else:
            transformed_data = np.sqrt(data)
            transform_label = "√"
    elif transform_type == "Square":
        transformed_data = data ** 2
        transform_label = "Squared "
    elif transform_type == "Standardize (Z-score)":
        transformed_data = (data - data.mean()) / data.std()
        transform_label = "Standardized "
    elif transform_type == "Min-Max Scale":
        transformed_data = (data - data.min()) / (data.max() - data.min())
        transform_label = "Scaled "
    
    # Create visualization
    if viz_type == "Histogram":
        # Histogram options
        col1, col2 = st.columns(2)
        with col1:
            hist_bins = st.slider(
                "Number of bins:", 
                5, 100, 20,
                key="hist_bins"
            )
        with col2:
            hist_kde = st.checkbox(
                "Show KDE", 
                value=True,
                key="hist_kde"
            )
        
        # Create histogram
        fig = px.histogram(
            transformed_data,
            nbins=hist_bins,
            histnorm='probability density' if hist_kde else None,
            title=f"Histogram of {transform_label}{col}",
            labels={"value": f"{transform_label}{col}", "count": "Frequency"}
        )
        
        # Add KDE overlay if requested
        if hist_kde:
            kde = stats.gaussian_kde(transformed_data)
            x_range = np.linspace(transformed_data.min(), transformed_data.max(), 1000)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                )
            )
        
        # Add vertical line for mean and median
        fig.add_vline(
            x=transformed_data.mean(),
            line_dash="solid",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=transformed_data.median(),
            line_dash="dash",
            line_color="orange",
            annotation_text="Median",
            annotation_position="top left"
        )
        
    elif viz_type == "Box Plot":
        # Create box plot
        fig = px.box(
            transformed_data, 
            title=f"Box Plot of {transform_label}{col}",
            labels={"value": f"{transform_label}{col}"}
        )
        
        # Show all points
        fig.update_traces(boxmean=True, jitter=0.3, pointpos=0, boxpoints='all')
        
    elif viz_type == "Violin Plot":
        # Create violin plot
        fig = px.violin(
            transformed_data,
            title=f"Violin Plot of {transform_label}{col}",
            labels={"value": f"{transform_label}{col}"},
            box=True,
            points="all"
        )
        
    elif viz_type == "KDE Plot":
        # Create KDE plot
        kde = stats.gaussian_kde(transformed_data)
        x_range = np.linspace(transformed_data.min(), transformed_data.max(), 1000)
        y_values = kde(x_range)
        
        fig = go.Figure()
        
        # Add KDE curve
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_values,
                mode='lines',
                name='KDE',
                fill='tozeroy',
                line=dict(width=2)
            )
        )
        
        # Add rug plot
        fig.add_trace(
            go.Scatter(
                x=transformed_data,
                y=np.zeros_like(transformed_data),
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=8,
                    color='rgba(0, 0, 0, 0.5)'
                ),
                name='Data Points'
            )
        )
        
        # Add vertical line for mean and median
        fig.add_vline(
            x=transformed_data.mean(),
            line_dash="solid",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=transformed_data.median(),
            line_dash="dash",
            line_color="orange",
            annotation_text="Median",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            title=f"KDE Plot of {transform_label}{col}",
            xaxis_title=f"{transform_label}{col}",
            yaxis_title="Density"
        )
        
    elif viz_type == "QQ Plot":
        # Create QQ plot
        fig = go.Figure()
        
        # Get theoretical quantiles (normal distribution)
        theoretical_quantiles = np.sort(np.random.normal(0, 1, len(transformed_data)))
        
        # Get sample quantiles
        sample_quantiles = np.sort(transformed_data)
        if transform_type != "Standardize (Z-score)":
            # Standardize if not already standardized
            sample_quantiles = (sample_quantiles - np.mean(sample_quantiles)) / np.std(sample_quantiles)
        
        # Create QQ plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.7
                ),
                name='QQ Points'
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
                line=dict(color='red', dash='dash'),
                name='Reference Line'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"QQ Plot of {transform_label}{col} (vs Normal Distribution)",
            xaxis_title="Theoretical Quantiles (Normal Distribution)",
            yaxis_title="Sample Quantiles"
        )
        
    elif viz_type == "ECDF Plot":
        # Create ECDF plot
        sorted_data = np.sort(transformed_data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig = go.Figure()
        
        # Add ECDF curve
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=ecdf,
                mode='lines',
                name='ECDF'
            )
        )
        
        # If data is standardized, add normal CDF for comparison
        if transform_type == "Standardize (Z-score)":
            x_range = np.linspace(sorted_data.min(), sorted_data.max(), 1000)
            normal_cdf = stats.norm.cdf(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_cdf,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Normal CDF'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Empirical Cumulative Distribution Function of {transform_label}{col}",
            xaxis_title=f"{transform_label}{col}",
            yaxis_title="Cumulative Probability"
        )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add distribution fitting if applicable
    if viz_type in ["Histogram", "KDE Plot"]:
        # Distribution fitting options
        st.subheader("Distribution Fitting")
        
        dist_options = [
            "Normal", "Log-Normal", "Exponential", "Gamma", "Weibull", 
            "Beta", "Uniform", "Student's t"
        ]
        
        dist_type = st.selectbox(
            "Fit distribution:",
            dist_options,
            key="fit_dist_type"
        )
        
        if st.button("Fit Distribution", key="fit_dist_button", use_container_width=True):
            try:
                # Fit distribution
                if dist_type == "Normal":
                    params = stats.norm.fit(transformed_data)
                    dist = stats.norm(*params)
                    param_names = ["μ (mean)", "σ (std dev)"]
                elif dist_type == "Log-Normal":
                    if transformed_data.min() <= 0:
                        st.error("Cannot fit log-normal to zero or negative values.")
                        raise ValueError("Data contains zero or negative values")
                    params = stats.lognorm.fit(transformed_data)
                    dist = stats.lognorm(*params)
                    param_names = ["s", "loc", "scale"]
                elif dist_type == "Exponential":
                    params = stats.expon.fit(transformed_data)
                    dist = stats.expon(*params)
                    param_names = ["loc", "scale"]
                elif dist_type == "Gamma":
                    if transformed_data.min() <= 0:
                        st.error("Cannot fit gamma to zero or negative values.")
                        raise ValueError("Data contains zero or negative values")
                    params = stats.gamma.fit(transformed_data)
                    dist = stats.gamma(*params)
                    param_names = ["a (shape)", "loc", "scale"]
                elif dist_type == "Weibull":
                    if transformed_data.min() <= 0:
                        st.error("Cannot fit Weibull to zero or negative values.")
                        raise ValueError("Data contains zero or negative values")
                    params = stats.weibull_min.fit(transformed_data)
                    dist = stats.weibull_min(*params)
                    param_names = ["c (shape)", "loc", "scale"]
                elif dist_type == "Beta":
                    # Beta requires data between 0 and 1
                    if transformed_data.min() < 0 or transformed_data.max() > 1:
                        st.error("Beta distribution requires data between 0 and 1.")
                        raise ValueError("Data not between 0 and 1")
                    params = stats.beta.fit(transformed_data)
                    dist = stats.beta(*params)
                    param_names = ["a", "b", "loc", "scale"]
                elif dist_type == "Uniform":
                    params = stats.uniform.fit(transformed_data)
                    dist = stats.uniform(*params)
                    param_names = ["loc", "scale"]
                elif dist_type == "Student's t":
                    params = stats.t.fit(transformed_data)
                    dist = stats.t(*params)
                    param_names = ["df", "loc", "scale"]
                
                # Display parameters
                st.subheader(f"Fitted {dist_type} Distribution Parameters")
                
                # Create parameter display in columns
                param_cols = st.columns(len(params))
                for i, (name, value) in enumerate(zip(param_names, params)):
                    with param_cols[i]:
                        st.metric(name, f"{value:.4f}")
                
                # Perform goodness-of-fit test
                st.subheader("Goodness of Fit")
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.kstest(transformed_data, dist.cdf)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("K-S Statistic", f"{ks_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{ks_pvalue:.4f}")
                
                if ks_pvalue < 0.05:
                    st.warning(f"The data does not follow a {dist_type} distribution (p < 0.05).")
                else:
                    st.success(f"The data appears to follow a {dist_type} distribution (p ≥ 0.05).")
                
                # Plot the fitted distribution
                st.subheader("Fitted Distribution Visualization")
                
                # Create visualization
                fig = go.Figure()
                
                # Add histogram of actual data
                fig.add_trace(
                    go.Histogram(
                        x=transformed_data,
                        histnorm='probability density',
                        name='Observed Data',
                        opacity=0.7,
                        nbinsx=30
                    )
                )
                
                # Add KDE curve of the data
                kde = stats.gaussian_kde(transformed_data)
                x_range = np.linspace(transformed_data.min(), transformed_data.max(), 1000)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        mode='lines',
                        name='KDE',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Add PDF of fitted distribution
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=dist.pdf(x_range),
                        mode='lines',
                        name=f'Fitted {dist_type} PDF',
                        line=dict(color='red', width=2)
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Fitted {dist_type} Distribution for {transform_label}{col}",
                    xaxis_title=f"{transform_label}{col}",
                    yaxis_title="Density"
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="dist_fit_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name=f"distribution_fit_{dist_type.lower()}.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_dist_fit_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"distribution_fit_{dist_type.lower()}.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_dist_fit_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error fitting distribution: {str(e)}")
    
    # Export visualization
    st.subheader("Export Visualization")
    
    export_format = st.selectbox(
        "Export chart format:", 
        ["PNG", "SVG", "HTML"],
        key="dist_export_format"
    )
    
    if export_format == "HTML":
        # Export as HTML file
        buffer = StringIO()
        fig.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        
        st.download_button(
            label="Download Visualization",
            data=html_bytes,
            file_name=f"distribution_{viz_type.lower().replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True,
            key="download_dist_viz_html"
        )
    else:
        # Export as image
        img_bytes = fig.to_image(format=export_format.lower())
        
        st.download_button(
            label=f"Download Visualization as {export_format}",
            data=img_bytes,
            file_name=f"distribution_{viz_type.lower().replace(' ', '_')}.{export_format.lower()}",
            mime=f"image/{export_format.lower()}",
            use_container_width=True,
            key=f"download_dist_viz_{export_format.lower()}"
        )

def render_trend_analysis():
    """Render trend analysis tab"""
    st.subheader("Trend Analysis")
    
    # Check if there are any date/datetime columns
    date_cols = []
    for col in st.session_state.df.columns:
        try:
            # Check if column can be converted to datetime
            pd.to_datetime(st.session_state.df[col])
            date_cols.append(col)
        except:
            pass
    
    # Get numeric columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    
    # Check if we have necessary columns for trend analysis
    if not date_cols:
        st.warning("No date columns found. Please ensure you have at least one column that can be converted to a date/datetime format.")
        
        # Allow creating a date index from another column
        if st.checkbox("Create date/time index from another column", key="create_date_index"):
            # Get all columns
            all_cols = st.session_state.df.columns.tolist()
            
            # Column to convert
            date_col = st.selectbox(
                "Select column to convert to date:", 
                all_cols,
                key="convert_date_col"
            )
            
            # Date format
            date_format = st.text_input(
                "Enter date format (e.g., '%Y-%m-%d' or '%d/%m/%Y'):",
                value="%Y-%m-%d",
                key="date_format"
            )
            
            # Try to convert
            if st.button("Convert to Date", key="convert_date_button"):
                try:
                    st.session_state.df['__date_index__'] = pd.to_datetime(st.session_state.df[date_col], format=date_format)
                    date_cols = ['__date_index__']
                    st.success(f"Successfully converted {date_col} to date format.")
                except Exception as e:
                    st.error(f"Error converting to date: {str(e)}")
        return
    
    if not num_cols:
        st.warning("No numeric columns found for trend analysis.")
        return
    
    # # Select date column
    # date_col = st.selectbox(
    #     "Select date/time column:", 
    #     date_cols,
    #     key="trend_date_col"
    # )

    # Add a "Select..." option to date columns
    date_cols_with_default = ["Select a date column..."] + date_cols

    # Select date column with a default null-like option
    date_col = st.selectbox(
        "Select date/time column:", 
        date_cols_with_default,
        index=0,  # Default to the "Select..." option
        key="trend_date_col"
    )
    
    # # Select value column
    # value_col = st.selectbox(
    #     "Select value column:", 
    #     num_cols,
    #     key="trend_value_col"
    # )

    # Add a "Select..." option to numeric columns
    num_cols_with_default = ["Select a value column..."] + num_cols

    # Select value column with a default null-like option
    value_col = st.selectbox(
        "Select value column:", 
        num_cols_with_default,
        index=0,  # Default to the "Select..." option
        key="trend_value_col"
    )

    # Check if the user has made selections for both fields
    if date_col == "Select a date column..." or value_col == "Select a value column...":
        st.info("Please select both a date/time column and a value column to view the trend analysis.")
        return
    
    # Ensure the date column is in datetime format
    df_trend = st.session_state.df.copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col])
    
    # Sort by date
    df_trend = df_trend.sort_values(date_col)
    
    # Check for missing values
    missing_values = df_trend[value_col].isna().sum()
    if missing_values > 0:
        st.warning(f"Found {missing_values} missing values in {value_col}.")
        
        # Handling missing values
        missing_method = st.radio(
            "How to handle missing values:",
            ["Drop", "Fill with mean", "Fill with median", "Forward fill", "Backward fill", "Linear interpolation"],
            horizontal=True,
            key="missing_method"
        )
        
        if missing_method == "Drop":
            df_trend = df_trend.dropna(subset=[value_col])
        elif missing_method == "Fill with mean":
            df_trend[value_col] = df_trend[value_col].fillna(df_trend[value_col].mean())
        elif missing_method == "Fill with median":
            df_trend[value_col] = df_trend[value_col].fillna(df_trend[value_col].median())
        elif missing_method == "Forward fill":
            df_trend[value_col] = df_trend[value_col].fillna(method='ffill')
        elif missing_method == "Backward fill":
            df_trend[value_col] = df_trend[value_col].fillna(method='bfill')
        elif missing_method == "Linear interpolation":
            df_trend[value_col] = df_trend[value_col].interpolate(method='linear')
    
    # Set the date column as index for time series analysis
    df_trend = df_trend.set_index(date_col)
    
    # Create time series visualization
    st.subheader("Time Series Visualization")
    
    # Basic time series plot
    fig = px.line(
        df_trend, 
        y=value_col,
        title=f"Time Series Plot of {value_col}",
        labels={"index": "Date", "value": value_col}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        showlegend=True
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add advanced analysis options
    st.subheader("Advanced Time Series Analysis")
    
    # Create tabs for different analyses
    ts_tabs = st.tabs([
        "Trend Analysis",
        "Seasonal Decomposition",
        "Moving Averages",
        "Autocorrelation"
    ])
    
    # Trend Analysis Tab
    with ts_tabs[0]:
        st.subheader("Trend Analysis")
        
        # Trend options
        trend_type = st.radio(
            "Trend type:",
            ["Linear", "Polynomial", "Exponential", "Loess (Lowess)"],
            horizontal=True,
            key="trend_type"
        )
        
        if trend_type == "Polynomial":
            # Degree of polynomial
            poly_degree = st.slider(
                "Polynomial degree:",
                1, 10, 2,
                key="poly_degree"
            )
        
        # Create trend plot
        fig_trend = px.scatter(
            df_trend,
            y=value_col,
            trendline=("lowess" if trend_type == "Loess (Lowess)" else 
                      ("ols" if trend_type == "Linear" else None)),
            trendline_options=dict(frac=0.1) if trend_type == "Loess (Lowess)" else None,
            title=f"{trend_type} Trend of {value_col}",
            labels={"index": "Date", "value": value_col}
        )
        
        # Add custom trend lines if needed
        if trend_type == "Polynomial":
            # X values as numeric
            x_numeric = np.arange(len(df_trend))
            
            # Fit polynomial
            coeffs = np.polyfit(x_numeric, df_trend[value_col].values, poly_degree)
            poly_func = np.poly1d(coeffs)
            
            # Add to plot
            fig_trend.add_trace(
                go.Scatter(
                    x=df_trend.index,
                    y=poly_func(x_numeric),
                    mode='lines',
                    name=f'Polynomial (degree={poly_degree})',
                    line=dict(color='red', width=2)
                )
            )
        
        elif trend_type == "Exponential":
            # X values as numeric
            x_numeric = np.arange(len(df_trend))
            
            # Only positive values can be used for exponential fit
            if df_trend[value_col].min() <= 0:
                st.warning("Exponential trend requires positive values. Showing linear trend instead.")
                
                # Fallback to linear
                coeffs = np.polyfit(x_numeric, df_trend[value_col].values, 1)
                linear_func = np.poly1d(coeffs)
                
                # Add to plot
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=linear_func(x_numeric),
                        mode='lines',
                        name='Linear Trend',
                        line=dict(color='red', width=2)
                    )
                )
            else:
                # Log transform for exponential fit
                log_y = np.log(df_trend[value_col].values)
                
                # Fit linear to log-transformed data
                coeffs = np.polyfit(x_numeric, log_y, 1)
                
                # Convert back to exponential
                exponential = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_numeric)
                
                # Add to plot
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=exponential,
                        mode='lines',
                        name='Exponential Trend',
                        line=dict(color='red', width=2)
                    )
                )
        
        # Update layout
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col,
            showlegend=True
        )
        
        # Show the plot
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Seasonal Decomposition Tab
    with ts_tabs[1]:
        st.subheader("Seasonal Decomposition")
        
        # Calculate frequency from the data
        try:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(df_trend.index)
            if inferred_freq is None:
                # Check common frequencies
                if len(df_trend) >= 365:
                    suggested_freq = 365  # Daily data for a year
                elif len(df_trend) >= 52:
                    suggested_freq = 52  # Weekly data for a year
                elif len(df_trend) >= 12:
                    suggested_freq = 12  # Monthly data for a year
                elif len(df_trend) >= 4:
                    suggested_freq = 4  # Quarterly data for a year
                else:
                    suggested_freq = 2  # Minimum for seasonal decomposition
            else:
                # Map pandas frequency to integer
                if "D" in inferred_freq:
                    suggested_freq = 365  # Daily
                elif "W" in inferred_freq:
                    suggested_freq = 52  # Weekly
                elif "M" in inferred_freq:
                    suggested_freq = 12  # Monthly
                elif "Q" in inferred_freq:
                    suggested_freq = 4  # Quarterly
                elif "Y" in inferred_freq or "A" in inferred_freq:
                    suggested_freq = 1  # Annual
                else:
                    suggested_freq = 12  # Default to monthly
            
            # Allow user to override
            freq = st.number_input(
                "Seasonal frequency (e.g., 12 for monthly with yearly seasonality):",
                min_value=2,
                value=suggested_freq,
                key="seasonal_freq"
            )
            
            # Decomposition model
            model = st.radio(
                "Decomposition model:",
                ["Additive", "Multiplicative"],
                horizontal=True,
                key="decomp_model"
            )
            
            # Need enough data points for decomposition
            if len(df_trend) < 2 * freq:
                st.warning(f"Need at least {2 * freq} data points for seasonal decomposition with frequency {freq}. You have {len(df_trend)} points.")
            else:
                # Perform decomposition
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Perform decomposition
                    decomposition = seasonal_decompose(
                        df_trend[value_col],
                        model=model.lower(),
                        period=int(freq)
                    )
                    
                    # Create figure with subplots
                    fig_decomp = make_subplots(
                        rows=4, 
                        cols=1,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                        vertical_spacing=0.1
                    )
                    
                    # Add observed data
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=df_trend.index,
                            y=decomposition.observed,
                            mode='lines',
                            name='Observed'
                        ),
                        row=1, col=1
                    )
                    
                    # Add trend
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=df_trend.index,
                            y=decomposition.trend,
                            mode='lines',
                            name='Trend',
                            line=dict(color='green')
                        ),
                        row=2, col=1
                    )
                    
                    # Add seasonal
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=df_trend.index,
                            y=decomposition.seasonal,
                            mode='lines',
                            name='Seasonal',
                            line=dict(color='red')
                        ),
                        row=3, col=1
                    )
                    
                    # Add residual
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=df_trend.index,
                            y=decomposition.resid,
                            mode='lines',
                            name='Residual',
                            line=dict(color='purple')
                        ),
                        row=4, col=1
                    )
                    
                    # Update layout
                    fig_decomp.update_layout(
                        height=800,
                        title=f"{model} Seasonal Decomposition of {value_col}",
                        showlegend=False
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # Export options
                    export_format = st.selectbox(
                        "Export chart format:", 
                        ["PNG", "SVG", "HTML"],
                        key="decomp_export_format"
                    )
                    
                    if export_format == "HTML":
                        # Export as HTML file
                        buffer = StringIO()
                        fig_decomp.write_html(buffer)
                        html_bytes = buffer.getvalue().encode()
                        
                        st.download_button(
                            label="Download Decomposition",
                            data=html_bytes,
                            file_name=f"seasonal_decomposition_{model.lower()}.html",
                            mime="text/html",
                            use_container_width=True,
                            key="download_decomp_html"
                        )
                    else:
                        # Export as image
                        img_bytes = fig_decomp.to_image(format=export_format.lower())
                        
                        st.download_button(
                            label=f"Download Decomposition as {export_format}",
                            data=img_bytes,
                            file_name=f"seasonal_decomposition_{model.lower()}.{export_format.lower()}",
                            mime=f"image/{export_format.lower()}",
                            use_container_width=True,
                            key=f"download_decomp_{export_format.lower()}"
                        )
                    
                except Exception as e:
                    st.error(f"Error performing seasonal decomposition: {str(e)}")
                    st.info("Try adjusting the frequency or using a different model.")
            
        except Exception as e:
            st.error(f"Error inferring frequency: {str(e)}")
    
    # Moving Averages Tab
    with ts_tabs[2]:
        st.subheader("Moving Averages")
        
        # Moving average types
        ma_types = st.multiselect(
            "Select moving average types:",
            ["Simple Moving Average (SMA)", "Exponential Moving Average (EMA)"],
            default=["Simple Moving Average (SMA)"],
            key="ma_types"
        )
        
        if not ma_types:
            st.warning("Please select at least one moving average type.")
        else:
            # Window sizes
            col1, col2 = st.columns(2)
            
            with col1:
                window_short = st.slider(
                    "Short window size:",
                    3, 100, 7,
                    key="window_short"
                )
            
            with col2:
                window_long = st.slider(
                    "Long window size:",
                    5, 200, 30,
                    key="window_long",
                    help=f"Must be greater than short window ({window_short})"
                )
            
            # Ensure long window is greater than short window
            if window_long <= window_short:
                st.error(f"Long window ({window_long}) must be greater than short window ({window_short})")
                window_long = window_short + 1
            
            # Create figure
            fig_ma = go.Figure()
            
            # Add original data
            fig_ma.add_trace(
                go.Scatter(
                    x=df_trend.index,
                    y=df_trend[value_col],
                    mode='lines',
                    name='Original'
                )
            )
            
            # Add moving averages
            if "Simple Moving Average (SMA)" in ma_types:
                # Calculate short SMA
                df_trend[f'SMA_{window_short}'] = df_trend[value_col].rolling(window=window_short).mean()
                
                # Add to plot
                fig_ma.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=df_trend[f'SMA_{window_short}'],
                        mode='lines',
                        name=f'SMA ({window_short})',
                        line=dict(color='orange')
                    )
                )
                
                # Calculate long SMA
                df_trend[f'SMA_{window_long}'] = df_trend[value_col].rolling(window=window_long).mean()
                
                # Add to plot
                fig_ma.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=df_trend[f'SMA_{window_long}'],
                        mode='lines',
                        name=f'SMA ({window_long})',
                        line=dict(color='red')
                    )
                )
                
                # Calculate and plot crossovers
                df_trend['Signal'] = 0
                df_trend.loc[df_trend[f'SMA_{window_short}'] > df_trend[f'SMA_{window_long}'], 'Signal'] = 1
                df_trend.loc[df_trend[f'SMA_{window_short}'] < df_trend[f'SMA_{window_long}'], 'Signal'] = -1
                
                # Find crossing points (signal changes)
                signal_changes = df_trend['Signal'].diff().fillna(0)
                buy_signals = df_trend[signal_changes == 2].index  # -1 to 1 (buy)
                sell_signals = df_trend[signal_changes == -2].index  # 1 to -1 (sell)
                
                # Plot buy signals
                fig_ma.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=df_trend.loc[buy_signals, value_col],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        ),
                        name='Buy Signal (SMA Crossover)'
                    )
                )
                
                # Plot sell signals
                fig_ma.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=df_trend.loc[sell_signals, value_col],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        ),
                        name='Sell Signal (SMA Crossover)'
                    )
                )
            
            if "Exponential Moving Average (EMA)" in ma_types:
                # Calculate short EMA
                df_trend[f'EMA_{window_short}'] = df_trend[value_col].ewm(span=window_short, adjust=False).mean()
                
                # Add to plot
                fig_ma.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=df_trend[f'EMA_{window_short}'],
                        mode='lines',
                        name=f'EMA ({window_short})',
                        line=dict(color='blue')
                    )
                )
                
                # Calculate long EMA
                df_trend[f'EMA_{window_long}'] = df_trend[value_col].ewm(span=window_long, adjust=False).mean()
                
                # Add to plot
                fig_ma.add_trace(
                    go.Scatter(
                        x=df_trend.index,
                        y=df_trend[f'EMA_{window_long}'],
                        mode='lines',
                        name=f'EMA ({window_long})',
                        line=dict(color='purple')
                    )
                )
                
                # Calculate and plot crossovers
                df_trend['EMA_Signal'] = 0
                df_trend.loc[df_trend[f'EMA_{window_short}'] > df_trend[f'EMA_{window_long}'], 'EMA_Signal'] = 1
                df_trend.loc[df_trend[f'EMA_{window_short}'] < df_trend[f'EMA_{window_long}'], 'EMA_Signal'] = -1
                
                # Find crossing points (signal changes)
                ema_signal_changes = df_trend['EMA_Signal'].diff().fillna(0)
                ema_buy_signals = df_trend[ema_signal_changes == 2].index  # -1 to 1 (buy)
                ema_sell_signals = df_trend[ema_signal_changes == -2].index  # 1 to -1 (sell)
                
                # Plot buy signals
                fig_ma.add_trace(
                    go.Scatter(
                        x=ema_buy_signals,
                        y=df_trend.loc[ema_buy_signals, value_col],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='green'
                        ),
                        name='Buy Signal (EMA Crossover)'
                    )
                )
                
                # Plot sell signals
                fig_ma.add_trace(
                    go.Scatter(
                        x=ema_sell_signals,
                        y=df_trend.loc[ema_sell_signals, value_col],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='red'
                        ),
                        name='Sell Signal (EMA Crossover)'
                    )
                )
            
            # Update layout
            fig_ma.update_layout(
                title=f"Moving Averages for {value_col}",
                xaxis_title="Date",
                yaxis_title=value_col,
                showlegend=True
            )
            
            # Show the plot
            st.plotly_chart(fig_ma, use_container_width=True)
            
            # Export options
            export_format = st.selectbox(
                "Export chart format:", 
                ["PNG", "SVG", "HTML"],
                key="ma_export_format"
            )
            
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig_ma.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Moving Averages",
                    data=html_bytes,
                    file_name="moving_averages.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_ma_html"
                )
            else:
                # Export as image
                img_bytes = fig_ma.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Moving Averages as {export_format}",
                    data=img_bytes,
                    file_name=f"moving_averages.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True,
                    key=f"download_ma_{export_format.lower()}"
                )
    
    # Autocorrelation Tab
    with ts_tabs[3]:
        st.subheader("Autocorrelation Analysis")
        
        # Max lags
        max_lags = st.slider(
            "Maximum lags:",
            5, 50, 20,
            key="max_lags"
        )
        
        # Create figure with ACF and PACF
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            import io
            import matplotlib.pyplot as plt
            
            # Calculate ACF and PACF values
            from statsmodels.tsa.stattools import acf, pacf
            
            acf_values = acf(df_trend[value_col].dropna(), nlags=max_lags)
            pacf_values = pacf(df_trend[value_col].dropna(), nlags=max_lags)
            
            # Create figure
            fig_corr = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=["Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)"],
                vertical_spacing=0.15
            )
            
            # Add ACF
            lags = list(range(len(acf_values)))
            fig_corr.add_trace(
                go.Bar(
                    x=lags,
                    y=acf_values,
                    name='ACF'
                ),
                row=1, col=1
            )
            
            # Add confidence intervals for ACF
            # Approximate 95% confidence interval: +/- 1.96/sqrt(n)
            n = len(df_trend[value_col].dropna())
            ci = 1.96 / np.sqrt(n)
            
            fig_corr.add_shape(
                type="line",
                x0=0, x1=max_lags,
                y0=ci, y1=ci,
                line=dict(color="red", dash="dash"),
                row=1, col=1
            )
            
            fig_corr.add_shape(
                type="line",
                x0=0, x1=max_lags,
                y0=-ci, y1=-ci,
                line=dict(color="red", dash="dash"),
                row=1, col=1
            )
            
            # Add PACF
            fig_corr.add_trace(
                go.Bar(
                    x=lags,
                    y=pacf_values,
                    name='PACF'
                ),
                row=2, col=1
            )
            
            # Add confidence intervals for PACF
            fig_corr.add_shape(
                type="line",
                x0=0, x1=max_lags,
                y0=ci, y1=ci,
                line=dict(color="red", dash="dash"),
                row=2, col=1
            )
            
            fig_corr.add_shape(
                type="line",
                x0=0, x1=max_lags,
                y0=-ci, y1=-ci,
                line=dict(color="red", dash="dash"),
                row=2, col=1
            )
            
            # Update layout
            fig_corr.update_layout(
                height=600,
                title=f"Autocorrelation Analysis for {value_col}",
                showlegend=False
            )
            
            fig_corr.update_yaxes(range=[-1, 1])
            
            # Show the plot
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Stationarity testing
            st.subheader("Stationarity Test")
            
            # Perform ADF test
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(df_trend[value_col].dropna())
            
            adf_output = pd.Series(
                adf_result[0:4],
                index=['Test Statistic', 'p-value', '# Lags Used', '# Observations']
            )
            
            for key, value in adf_result[4].items():
                adf_output[f'Critical Value ({key})'] = value
            
            st.write(adf_output)
            
            # Interpret results
            alpha = 0.05
            if adf_result[1] < alpha:
                st.success(f"Series is stationary (p < {alpha})")
            else:
                st.warning(f"Series is not stationary (p ≥ {alpha})")
                
                # Suggest differencing
                st.write("Consider applying differencing to make the series stationary:")
                
                # Differencing options
                diff_order = st.number_input(
                    "Differencing order:",
                    min_value=1,
                    max_value=3,
                    value=1,
                    key="diff_order"
                )
                
                if st.button("Apply Differencing", key="apply_diff"):
                    # Apply differencing
                    df_diff = df_trend[[value_col]].diff(diff_order).dropna()
                    
                    # Perform ADF test on differenced series
                    adf_diff_result = adfuller(df_diff[value_col].dropna())
                    
                    # Create plot of differenced series
                    fig_diff = px.line(
                        df_diff,
                        y=value_col,
                        title=f"Differenced Series (d={diff_order}) of {value_col}",
                        labels={"index": "Date", "value": f"Diff({value_col}, {diff_order})"}
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # Display ADF test results for differenced series
                    adf_diff_output = pd.Series(
                        adf_diff_result[0:4],
                        index=['Test Statistic', 'p-value', '# Lags Used', '# Observations']
                    )
                    
                    for key, value in adf_diff_result[4].items():
                        adf_diff_output[f'Critical Value ({key})'] = value
                    
                    st.write(adf_diff_output)
                    
                    # Interpret results
                    if adf_diff_result[1] < alpha:
                        st.success(f"Differenced series is stationary (p < {alpha})")
                    else:
                        st.warning(f"Differenced series is still not stationary (p ≥ {alpha})")
                        st.write("Consider using a higher differencing order or transforming the data.")
            
            # Export options
            export_format = st.selectbox(
                "Export chart format:", 
                ["PNG", "SVG", "HTML"],
                key="acf_export_format"
            )
            
            if export_format == "HTML":
                # Export as HTML file
                buffer = StringIO()
                fig_corr.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Autocorrelation Analysis",
                    data=html_bytes,
                    file_name="autocorrelation.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_acf_html"
                )
            else:
                # Export as image
                img_bytes = fig_corr.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Autocorrelation Analysis as {export_format}",
                    data=img_bytes,
                    file_name=f"autocorrelation.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True,
                    key=f"download_acf_{export_format.lower()}"
                )
            
        except Exception as e:
            st.error(f"Error calculating autocorrelation: {str(e)}")
    
    # Export time series visualization
    st.subheader("Export Time Series Visualization")
    
    export_format = st.selectbox(
        "Export chart format:", 
        ["PNG", "SVG", "HTML"],
        key="ts_export_format"
    )
    
    if export_format == "HTML":
        # Export as HTML file
        buffer = StringIO()
        fig.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        
        st.download_button(
            label="Download Time Series",
            data=html_bytes,
            file_name="time_series.html",
            mime="text/html",
            use_container_width=True,
            key="download_ts_html"
        )
    else:
        # Export as image
        img_bytes = fig.to_image(format=export_format.lower())
        
        st.download_button(
            label=f"Download Time Series as {export_format}",
            data=img_bytes,
            file_name=f"time_series.{export_format.lower()}",
            mime=f"image/{export_format.lower()}",
            use_container_width=True,
            key=f"download_ts_{export_format.lower()}"
        )

def render_hypothesis_testing():
    """Render hypothesis testing tab"""
    st.subheader("Hypothesis Testing")
    
    # Select test type
    test_type = st.selectbox(
        "Select test type:",
        [
            "One Sample t-test",
            "Two Sample t-test",
            "Paired t-test",
            "One-way ANOVA",
            "Chi-Square Test",
            "Correlation Test"
        ],
        key="test_type"
    )
    
    # Get numeric and categorical columns
    num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if test_type == "One Sample t-test":
        st.write("Test if the mean of a sample is significantly different from a hypothesized value.")
        
        if not num_cols:
            st.warning("No numeric columns available for t-test.")
            return
        
        # Column selection
        col = st.selectbox(
            "Select column:", 
            num_cols,
            key="ttest_one_column"
        )
        
        # Hypothesized mean
        pop_mean = st.number_input(
            "Hypothesized population mean:", 
            value=0.0,
            key="ttest_pop_mean"
        )
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05,
            key="ttest_one_alpha"
        )
        
        # Run the test
        if st.button("Run One Sample t-test", use_container_width=True, key="run_ttest_one"):
            try:
                from scipy import stats
                
                # Get data
                data = st.session_state.df[col].dropna()
                
                # Run t-test
                t_stat, p_value = stats.ttest_1samp(data, pop_mean)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sample Mean", f"{data.mean():.4f}")
                with col2:
                    st.metric("Hypothesized Mean", f"{pop_mean:.4f}")
                with col3:
                    st.metric("t-statistic", f"{t_stat:.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean of {col} is significantly different from {pop_mean}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean of {col} is significantly different from {pop_mean}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create distribution plot
                fig = go.Figure()
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name="Sample Distribution",
                        opacity=0.7,
                        histnorm="probability density"
                    )
                )
                
                # Add normal curve
                x = np.linspace(data.min(), data.max(), 1000)
                y = stats.norm.pdf(x, data.mean(), data.std())
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name="Normal Distribution",
                        line=dict(color="red")
                    )
                )
                
                # Add vertical lines for means
                fig.add_vline(
                    x=data.mean(),
                    line_dash="solid",
                    line_color="blue",
                    annotation_text="Sample Mean",
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=pop_mean,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Hypothesized Mean",
                    annotation_position="top left"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Distribution of {col} vs. Hypothesized Mean",
                    xaxis_title=col,
                    yaxis_title="Density"
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="ttest_one_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name=f"ttest_one_sample.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_ttest_one_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"ttest_one_sample.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_ttest_one_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running t-test: {str(e)}")
    
    elif test_type == "Two Sample t-test":
        st.write("Test if the means of two independent samples are significantly different.")
        
        if not num_cols:
            st.warning("No numeric columns available for t-test.")
            return
        
        if not cat_cols:
            st.warning("No categorical columns available for grouping.")
            return
        
        # Column selections
        value_col = st.selectbox(
            "Value column:", 
            num_cols,
            key="ttest_two_value_col"
        )
        group_col = st.selectbox(
            "Group column:", 
            cat_cols,
            key="ttest_two_group_col"
        )
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for two-sample t-test.")
            return
        
        # Group selection
        col1, col2 = st.columns(2)
        
        with col1:
            group1 = st.selectbox(
                "First group:", 
                groups,
                key="ttest_two_group1"
            )
        
        with col2:
            group2 = st.selectbox(
                "Second group:", 
                [g for g in groups if g != group1],
                key="ttest_two_group2"
            )
        
        # Variance assumption
        equal_var = st.checkbox(
            "Assume equal variances", 
            value=False,
            key="ttest_two_equal_var"
        )
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05,
            key="ttest_two_alpha"
        )
        
        # Run the test
        if st.button("Run Two Sample t-test", use_container_width=True, key="run_ttest_two"):
            try:
                from scipy import stats
                
                # Get data for each group
                data1 = st.session_state.df[st.session_state.df[group_col] == group1][value_col].dropna()
                data2 = st.session_state.df[st.session_state.df[group_col] == group2][value_col].dropna()
                
                # Run t-test
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Mean ({group1})", f"{data1.mean():.4f}")
                with col2:
                    st.metric(f"Mean ({group2})", f"{data2.mean():.4f}")
                with col3:
                    st.metric("t-statistic", f"{t_stat:.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean {value_col} is significantly different between {group1} and {group2}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean {value_col} is significantly different between {group1} and {group2}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create box plot
                fig = go.Figure()
                
                # Add box plot for each group
                fig.add_trace(
                    go.Box(
                        y=data1,
                        name=str(group1),
                        boxmean=True
                    )
                )
                
                fig.add_trace(
                    go.Box(
                        y=data2,
                        name=str(group2),
                        boxmean=True
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Comparison of {value_col} between {group1} and {group2}",
                    yaxis_title=value_col
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="ttest_two_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name=f"ttest_two_sample.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_ttest_two_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"ttest_two_sample.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_ttest_two_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running t-test: {str(e)}")
    
    elif test_type == "Paired t-test":
        st.write("Test if the mean difference between paired observations is significantly different from zero.")
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for paired t-test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox(
                "First column:", 
                num_cols,
                key="paired_first_col"
            )
        
        with col2:
            second_col = st.selectbox(
                "Second column:", 
                [col for col in num_cols if col != first_col],
                key="paired_second_col"
            )
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05, 
            key="paired_alpha"
        )
        
        # Run the test
        if st.button("Run Paired t-test", use_container_width=True, key="run_paired_ttest"):
            try:
                from scipy import stats
                
                # Get data
                # Remove rows with NaN in either column
                valid_data = st.session_state.df[[first_col, second_col]].dropna()
                
                data1 = valid_data[first_col]
                data2 = valid_data[second_col]
                
                # Run paired t-test
                t_stat, p_value = stats.ttest_rel(data1, data2)
                
                # Calculate differences
                differences = data1 - data2
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Mean ({first_col})", f"{data1.mean():.4f}")
                with col2:
                    st.metric(f"Mean ({second_col})", f"{data2.mean():.4f}")
                with col3:
                    st.metric("Mean Difference", f"{differences.mean():.4f}")
                with col4:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that there is a significant difference between {first_col} and {second_col}.")
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant difference between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create visualization of paired differences
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Paired Values", "Differences"))
                
                # Add scatter plot with paired values
                fig.add_trace(
                    go.Scatter(
                        x=data1,
                        y=data2,
                        mode="markers",
                        marker=dict(color="blue"),
                        name="Paired Values"
                    ),
                    row=1, col=1
                )
                
                # Add diagonal line y=x
                min_val = min(data1.min(), data2.min())
                max_val = max(data1.max(), data2.max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="y=x"
                    ),
                    row=1, col=1
                )
                
                # Add histogram of differences
                fig.add_trace(
                    go.Histogram(
                        x=differences,
                        marker=dict(color="green"),
                        name="Differences"
                    ),
                    row=1, col=2
                )
                
                # Add vertical line at zero
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="black",
                    row=1, col=2
                )
                
                # Add vertical line at mean difference
                fig.add_vline(
                    x=differences.mean(),
                    line_dash="solid",
                    line_color="red",
                    annotation_text="Mean Difference",
                    annotation_position="top right",
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Paired Comparison: {first_col} vs {second_col}",
                    height=500
                )
                
                # Update axes titles
                fig.update_xaxes(title_text=first_col, row=1, col=1)
                fig.update_yaxes(title_text=second_col, row=1, col=1)
                fig.update_xaxes(title_text="Difference", row=1, col=2)
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="paired_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name="paired_ttest.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_paired_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"paired_ttest.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_paired_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running paired t-test: {str(e)}")
    
    elif test_type == "One-way ANOVA":
        st.write("Test if the means of multiple groups are significantly different.")
        
        if not num_cols:
            st.warning("No numeric columns available for ANOVA.")
            return
        
        if not cat_cols:
            st.warning("No categorical columns available for grouping.")
            return
        
        # Column selections
        value_col = st.selectbox(
            "Value column:", 
            num_cols,
            key="anova_value_col"
        )
        group_col = st.selectbox(
            "Group column:", 
            cat_cols,
            key="anova_group_col"
        )
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for ANOVA.")
            return
        
        # Select groups to include
        selected_groups = st.multiselect(
            "Select groups to compare:",
            groups,
            default=groups[:min(5, len(groups))],
            key="anova_groups"
        )
        
        if len(selected_groups) < 2:
            st.warning("Please select at least 2 groups for comparison.")
            return
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05, 
            key="anova_alpha"
        )
        
        # Run the test
        if st.button("Run One-way ANOVA", use_container_width=True, key="run_anova"):
            try:
                from scipy import stats
                
                # Get data for each group
                group_data = []
                group_means = []
                group_counts = []
                
                for group in selected_groups:
                    data = st.session_state.df[st.session_state.df[group_col] == group][value_col].dropna()
                    group_data.append(data)
                    group_means.append(data.mean())
                    group_counts.append(len(data))
                
                # Run ANOVA
                f_stat, p_value = stats.f_oneway(*group_data)
                
                # Show results
                st.subheader("Test Results")
                
                # Group statistics
                group_stats = pd.DataFrame({
                    "Group": selected_groups,
                    "Count": group_counts,
                    "Mean": group_means,
                    "Std": [data.std() for data in group_data],
                    "Min": [data.min() for data in group_data],
                    "Max": [data.max() for data in group_data]
                })
                
                st.dataframe(group_stats.round(4), use_container_width=True)
                
                # ANOVA results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F-statistic", f"{f_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    st.metric("# of Groups", len(selected_groups))
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that the mean {value_col} is significantly different among the groups.")
                    
                    # Post-hoc test (Tukey's HSD)
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        import numpy as np
                        
                        # Prepare data for Tukey's test
                        all_data = np.concatenate(group_data)
                        all_groups = np.concatenate([[group] * len(data) for group, data in zip(selected_groups, group_data)])
                        
                        # Run Tukey's test
                        tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=alpha)
                        
                        # Display results
                        st.subheader("Post-hoc: Tukey's HSD Test")
                        st.write("Pairwise comparisons:")
                        
                        # Convert Tukey results to dataframe
                        tukey_df = pd.DataFrame(
                            data=np.column_stack([tukey_result.groupsunique[tukey_result.pairindices[:,0]], 
                                              tukey_result.groupsunique[tukey_result.pairindices[:,1]], 
                                              tukey_result.meandiffs, 
                                              tukey_result.confint, 
                                              tukey_result.pvalues, 
                                              tukey_result.reject]),
                            columns=['group1', 'group2', 'mean_diff', 'lower_bound', 'upper_bound', 'p_value', 'reject']
                        )
                        # Continuing the ANOVA section where we left off
                        
                        # Convert numeric columns
                        for col in ['mean_diff', 'lower_bound', 'upper_bound', 'p_value']:
                            tukey_df[col] = pd.to_numeric(tukey_df[col])
                        
                        # Format reject column
                        tukey_df['significant'] = tukey_df['reject'].map({True: 'Yes', False: 'No'})
                        
                        # Round numeric columns
                        tukey_df = tukey_df.round(4)
                        
                        # Display dataframe
                        st.dataframe(
                            tukey_df[['group1', 'group2', 'mean_diff', 'p_value', 'significant']],
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.warning(f"Could not perform post-hoc test: {str(e)}")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that the mean {value_col} is significantly different among the groups.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create box plot
                fig = go.Figure()
                
                # Add box plot for each group
                for i, (group, data) in enumerate(zip(selected_groups, group_data)):
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=str(group),
                            boxmean=True
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Comparison of {value_col} across groups",
                    yaxis_title=value_col
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="anova_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name="anova_results.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_anova_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"anova_results.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_anova_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running ANOVA: {str(e)}")
    
    elif test_type == "Chi-Square Test":
        st.write("Test if there is a significant association between two categorical variables.")
        
        if len(cat_cols) < 2:
            st.warning("Need at least two categorical columns for Chi-Square test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox(
                "First categorical column:", 
                cat_cols,
                key="chi2_first_col"
            )
        
        with col2:
            second_col = st.selectbox(
                "Second categorical column:", 
                [col for col in cat_cols if col != first_col],
                key="chi2_second_col"
            )
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05, 
            key="chi2_alpha"
        )
        
        # Run the test
        if st.button("Run Chi-Square Test", use_container_width=True, key="run_chi2"):
            try:
                from scipy import stats
                
                # Create contingency table
                contingency = pd.crosstab(
                    st.session_state.df[first_col],
                    st.session_state.df[second_col]
                )
                
                # Run chi-square test
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                # Check if expected frequencies are too small
                expected_df = pd.DataFrame(
                    expected,
                    index=contingency.index,
                    columns=contingency.columns
                )
                
                # Check for small expected frequencies
                small_expected = (expected_df < 5).values.sum()
                total_cells = expected_df.size
                
                # Convert expected to dataframe for display
                expected_df_display = expected_df.round(2)
                
                # Show results
                st.subheader("Contingency Table (Observed)")
                st.dataframe(contingency, use_container_width=True)
                
                st.subheader("Expected Frequencies")
                st.dataframe(expected_df_display, use_container_width=True)
                
                # Chi-square results
                st.subheader("Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chi-Square", f"{chi2:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    st.metric("Degrees of Freedom", dof)
                with col4:
                    # Calculate Cramer's V (effect size)
                    n = contingency.values.sum()
                    cramers_v = np.sqrt(chi2 / (n * min(contingency.shape[0]-1, contingency.shape[1]-1)))
                    st.metric("Cramer's V", f"{cramers_v:.4f}")
                
                # Warning for small expected frequencies
                if small_expected > 0:
                    pct_small = (small_expected / total_cells) * 100
                    st.warning(f"{small_expected} cells ({pct_small:.1f}%) have expected frequencies less than 5. Chi-square results may not be reliable.")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    st.write(f"There is enough evidence to suggest that there is a significant association between {first_col} and {second_col}.")
                    
                    # Interpret Cramer's V
                    if cramers_v < 0.1:
                        effect_size = "negligible"
                    elif cramers_v < 0.3:
                        effect_size = "small"
                    elif cramers_v < 0.5:
                        effect_size = "medium"
                    else:
                        effect_size = "large"
                    
                    st.write(f"The strength of the association (Cramer's V = {cramers_v:.4f}) is {effect_size}.")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant association between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create heatmap of observed frequencies
                fig = px.imshow(
                    contingency,
                    labels=dict(
                        x=second_col,
                        y=first_col,
                        color="Frequency"
                    ),
                    text_auto=True,
                    title=f"Contingency Table: {first_col} vs {second_col}"
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Create stacked bar chart
                # Normalize contingency table by rows
                prop_table = contingency.div(contingency.sum(axis=1), axis=0)
                
                # Create stacked bar chart of proportions
                fig_bar = px.bar(
                    prop_table.reset_index().melt(id_vars=first_col),
                    x=first_col,
                    y="value",
                    color="variable",
                    labels={"value": "Proportion", "variable": second_col},
                    title=f"Proportions of {second_col} within each {first_col}",
                    text_auto='.2f'
                )
                
                # Show plot
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="chi2_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Heatmap",
                        data=html_bytes,
                        file_name="chi2_heatmap.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_chi2_heatmap_html"
                    )
                    
                    # Export bar chart
                    buffer = StringIO()
                    fig_bar.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Bar Chart",
                        data=html_bytes,
                        file_name="chi2_barchart.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_chi2_bar_html"
                    )
                else:
                    # Export heatmap
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Heatmap as {export_format}",
                        data=img_bytes,
                        file_name=f"chi2_heatmap.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_chi2_heatmap_{export_format.lower()}"
                    )
                    
                    # Export bar chart
                    img_bytes = fig_bar.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Bar Chart as {export_format}",
                        data=img_bytes,
                        file_name=f"chi2_barchart.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_chi2_bar_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running Chi-Square test: {str(e)}")
    
    elif test_type == "Correlation Test":
        st.write("Test if there is a significant correlation between two numeric variables.")
        
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for correlation test.")
            return
        
        # Column selections
        col1, col2 = st.columns(2)
        
        with col1:
            first_col = st.selectbox(
                "First numeric column:", 
                num_cols,
                key="corr_test_first_col"
            )
        
        with col2:
            second_col = st.selectbox(
                "Second numeric column:", 
                [col for col in num_cols if col != first_col],
                key="corr_test_second_col"
            )
        
        # Correlation method
        method = st.radio(
            "Correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True,
            key="corr_test_method"
        )
        
        # Alpha level
        alpha = st.slider(
            "Significance level (alpha):", 
            0.01, 0.10, 0.05, 
            key="corr_test_alpha"
        )
        
        # Run the test
        if st.button("Run Correlation Test", use_container_width=True, key="run_corr_test"):
            try:
                from scipy import stats
                
                # Get data
                # Remove rows with NaN in either column
                valid_data = st.session_state.df[[first_col, second_col]].dropna()
                
                data1 = valid_data[first_col]
                data2 = valid_data[second_col]
                
                # Run correlation test
                if method == "Pearson":
                    corr, p_value = stats.pearsonr(data1, data2)
                elif method == "Spearman":
                    corr, p_value = stats.spearmanr(data1, data2)
                else:  # Kendall
                    corr, p_value = stats.kendalltau(data1, data2)
                
                # Show results
                st.subheader("Test Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{method} Correlation", f"{corr:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                with col3:
                    # Calculate coefficient of determination (r-squared)
                    r_squared = corr ** 2
                    st.metric("R²", f"{r_squared:.4f}")
                
                # Test interpretation
                st.subheader("Interpretation")
                
                if p_value < alpha:
                    st.success(f"Result: Reject the null hypothesis (p-value = {p_value:.4f} < {alpha})")
                    
                    if corr > 0:
                        direction = "positive"
                    else:
                        direction = "negative"
                    
                    # Interpret correlation strength
                    corr_abs = abs(corr)
                    if corr_abs < 0.3:
                        strength = "weak"
                    elif corr_abs < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    st.write(f"There is enough evidence to suggest that there is a significant {direction} correlation ({strength}) between {first_col} and {second_col}.")
                    st.write(f"The coefficient of determination (R²) indicates that {r_squared:.2%} of the variance in one variable can be explained by the other variable.")
                    
                else:
                    st.info(f"Result: Fail to reject the null hypothesis (p-value = {p_value:.4f} ≥ {alpha})")
                    st.write(f"There is not enough evidence to suggest that there is a significant correlation between {first_col} and {second_col}.")
                
                # Add visualization
                st.subheader("Visualization")
                
                # Create scatter plot
                fig = px.scatter(
                    valid_data,
                    x=first_col,
                    y=second_col,
                    trendline="ols",
                    labels={
                        first_col: first_col,
                        second_col: second_col
                    },
                    title=f"Scatter Plot: {first_col} vs {second_col}"
                )
                
                # Add annotation with correlation coefficient
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    text=f"{method} correlation: {corr:.4f}<br>p-value: {p_value:.4f}<br>R²: {r_squared:.4f}",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export chart format:", 
                    ["PNG", "SVG", "HTML"],
                    key="corr_test_export_format"
                )
                
                if export_format == "HTML":
                    # Export as HTML file
                    buffer = StringIO()
                    fig.write_html(buffer)
                    html_bytes = buffer.getvalue().encode()
                    
                    st.download_button(
                        label="Download Visualization",
                        data=html_bytes,
                        file_name="correlation_test.html",
                        mime="text/html",
                        use_container_width=True,
                        key="download_corr_test_html"
                    )
                else:
                    # Export as image
                    img_bytes = fig.to_image(format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download Visualization as {export_format}",
                        data=img_bytes,
                        file_name=f"correlation_test.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}",
                        use_container_width=True,
                        key=f"download_corr_test_{export_format.lower()}"
                    )
                
            except Exception as e:
                st.error(f"Error running correlation test: {str(e)}")