import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

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
        default=num_cols
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
        ["Mean", "Median", "Standard Deviation", "Range", "Missing Values %"]
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
            use_container_width=True
        )
    
    with col2:
        # Export chart
        if 'fig' in locals():
            # Format selection
            export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
            
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
                    use_container_width=True
                )
            else:
                # Export as image
                img_bytes = fig.to_image(format=export_format.lower())
                
                st.download_button(
                    label=f"Download Chart as {export_format}",
                    data=img_bytes,
                    file_name=f"statistics_{stat_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True
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
        default=num_cols[:min(5, len(num_cols))]
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least two columns.")
        return
    
    # Correlation method
    corr_method = st.radio(
        "Correlation method:",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True
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
        horizontal=True
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
            use_container_width=True
        )
    
    with col2:
        # Export visualization
        export_format = st.selectbox("Export chart format:", ["PNG", "SVG", "HTML"])
        
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
                use_container_width=True
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
        ]
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
        col = st.selectbox("Select column:", num_cols)
        
        # Hypothesized mean
        pop_mean = st.number_input("Hypothesized population mean:", value=0.0)
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05)
        
        # Run the test
        if st.button("Run One Sample t-test", use_container_width=True):
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
        value_col = st.selectbox("Value column:", num_cols)
        group_col = st.selectbox("Group column:", cat_cols)
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for two-sample t-test.")
            return
        
        # Group selection
        col1, col2 = st.columns(2)
        
        with col1:
            group1 = st.selectbox("First group:", groups)
        
        with col2:
            group2 = st.selectbox("Second group:", [g for g in groups if g != group1])
        
        # Variance assumption
        equal_var = st.checkbox("Assume equal variances", value=False)
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05)
        
        # Run the test
        if st.button("Run Two Sample t-test", use_container_width=True):
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
            first_col = st.selectbox("First column:", num_cols)
        
        with col2:
            second_col = st.selectbox("Second column:", [col for col in num_cols if col != first_col])
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="paired_alpha")
        
        # Run the test
        if st.button("Run Paired t-test", use_container_width=True):
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
        value_col = st.selectbox("Value column:", num_cols)
        group_col = st.selectbox("Group column:", cat_cols)
        
        # Get unique groups
        groups = st.session_state.df[group_col].unique().tolist()
        
        if len(groups) < 2:
            st.warning(f"Need at least 2 groups in '{group_col}' for ANOVA.")
            return
        
        # Select groups to include
        selected_groups = st.multiselect(
            "Select groups to compare:",
            groups,
            default=groups[:min(5, len(groups))]
        )
        
        if len(selected_groups) < 2:
            st.warning("Please select at least 2 groups for comparison.")
            return
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="anova_alpha")
        
        # Run the test
        if st.button("Run One-way ANOVA", use_container_width=True):
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
            first_col = st.selectbox("First categorical column:", cat_cols)
        
        with col2:
            second_col = st.selectbox("Second categorical column:", [col for col in cat_cols if col != first_col])
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="chi2_alpha")
        
        # Run the test
        if st.button("Run Chi-Square Test", use_container_width=True):
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
            first_col = st.selectbox("First numeric column:", num_cols)
        
        with col2:
            second_col = st.selectbox("Second numeric column:", [col for col in num_cols if col != first_col])
        
        # Correlation method
        method = st.radio(
            "Correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True
        )
        
        # Alpha level
        alpha = st.slider("Significance level (alpha):", 0.01, 0.10, 0.05, key="corr_alpha")
        
        # Run the test
        if st.button("Run Correlation Test", use_container_width=True):
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
                
            except Exception as e:
                st.error(f"Error running correlation test: {str(e)}")