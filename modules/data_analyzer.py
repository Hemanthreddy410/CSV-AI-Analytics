import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

class DataAnalyzer:
    """Class for performing data analysis on datasets"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Store analysis results in session state if not already there
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def generate_summary_statistics(self):
        """Generate summary statistics for numeric columns"""
        if self.df is None:
            return None
            
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return None
            
        # Calculate statistics
        summary = {}
        for col in numeric_cols:
            summary[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'skewness': float(stats.skew(self.df[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.df[col].dropna()))
            }
            
        return summary
    
    def analyze_correlation(self):
        """Generate correlation matrix for numeric columns"""
        if self.df is None:
            return None
            
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
            
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        return corr_matrix
    
    def analyze_distributions(self):
        """Analyze distributions of numeric columns"""
        if self.df is None:
            return None
            
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return None
            
        # Calculate distribution statistics
        distributions = {}
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Basic statistics
            distributions[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                # Normality test
                'shapiro_test': stats.shapiro(data.sample(min(len(data), 1000))),
                # Percentiles
                'percentiles': {
                    'p25': float(np.percentile(data, 25)),
                    'p50': float(np.percentile(data, 50)),
                    'p75': float(np.percentile(data, 75)),
                    'p95': float(np.percentile(data, 95)),
                    'p99': float(np.percentile(data, 99))
                }
            }
            
        return distributions
    
    def analyze_categorical(self):
        """Analyze categorical columns"""
        if self.df is None:
            return None
            
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            return None
            
        # Calculate categorical statistics
        categorical_stats = {}
        for col in cat_cols:
            data = self.df[col].dropna()
            
            value_counts = data.value_counts()
            top_categories = value_counts.head(10).index.tolist()
            top_counts = value_counts.head(10).values.tolist()
            
            # Calculate frequency and percentage
            total = len(data)
            frequencies = value_counts.head(10).tolist()
            percentages = [(count / total) * 100 for count in frequencies]
            
            categorical_stats[col] = {
                'unique_count': data.nunique(),
                'top_value': value_counts.index[0] if not value_counts.empty else None,
                'top_freq': value_counts.iloc[0] if not value_counts.empty else 0,
                'top_categories': top_categories,
                'top_counts': top_counts,
                'percentages': percentages
            }
            
        return categorical_stats
    
    def analyze_outliers(self):
        """Detect outliers in numeric columns using IQR method"""
        if self.df is None:
            return None
            
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return None
            
        # Calculate outliers using IQR method
        outliers = {}
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Calculate Q1, Q3 and IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outlier_indices = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index
            outlier_count = len(outlier_indices)
            outlier_percentage = (outlier_count / len(data)) * 100
            
            outliers[col] = {
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': int(outlier_count),
                'outlier_percentage': float(outlier_percentage)
            }
            
        return outliers
    
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        if self.df is None:
            return None
            
        # Calculate missing value statistics
        missing = {}
        
        # Overall missing
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = self.df.isna().sum().sum()
        missing['overall'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(total_missing),
            'percentage': float((total_missing / total_cells) * 100)
        }
        
        # Missing by column
        missing['columns'] = {}
        for col in self.df.columns:
            col_missing = self.df[col].isna().sum()
            missing['columns'][col] = {
                'count': int(col_missing),
                'percentage': float((col_missing / len(self.df)) * 100)
            }
            
        return missing
    
    def perform_group_analysis(self, group_col, agg_col, agg_func='mean'):
        """Perform groupby analysis on the data"""
        if self.df is None or group_col not in self.df.columns or agg_col not in self.df.columns:
            return None
            
        # Available aggregation functions
        agg_functions = {
            'mean': 'mean',
            'median': 'median',
            'sum': 'sum',
            'count': 'count',
            'std': 'std',
            'min': 'min',
            'max': 'max'
        }
        
        # Perform groupby
        if agg_func in agg_functions:
            grouped = self.df.groupby(group_col)[agg_col].agg(agg_functions[agg_func])
            return grouped.reset_index()
            
        return None
    
    def render_interface(self):
        """Render the analysis interface"""
        st.header("Analysis Hub")
        
        # Analysis types
        analysis_types = [
            "Statistical Summary",
            "Correlation Analysis",
            "Distribution Analysis",
            "Categorical Analysis",
            "Outlier Detection",
            "Missing Values Analysis",
            "Group Analysis"
        ]
        
        selected_analysis = st.selectbox("Select Analysis Type", analysis_types)
        
        # Perform selected analysis
        if selected_analysis == "Statistical Summary":
            self._render_summary_statistics()
            
        elif selected_analysis == "Correlation Analysis":
            self._render_correlation_analysis()
            
        elif selected_analysis == "Distribution Analysis":
            self._render_distribution_analysis()
            
        elif selected_analysis == "Categorical Analysis":
            self._render_categorical_analysis()
            
        elif selected_analysis == "Outlier Detection":
            self._render_outlier_analysis()
            
        elif selected_analysis == "Missing Values Analysis":
            self._render_missing_values_analysis()
            
        elif selected_analysis == "Group Analysis":
            self._render_group_analysis()
    
    def _render_summary_statistics(self):
        """Render summary statistics interface"""
        st.subheader("Statistical Summary")
        
        # Generate summary statistics
        summary = self.generate_summary_statistics()
        
        if summary is None or not summary:
            st.info("No numeric columns found for statistical analysis.")
            return
        
        # Display summary as table
        summary_df = pd.DataFrame.from_dict({col: stats for col, stats in summary.items()}, 
                                           orient='index')
        
        # Format table
        st.dataframe(summary_df, use_container_width=True)
        
        # Visualization
        st.subheader("Visualize Distributions")
        selected_cols = st.multiselect("Select columns to visualize", list(summary.keys()))
        
        if selected_cols:
            fig = plt.figure(figsize=(10, 6))
            
            for col in selected_cols:
                sns.kdeplot(self.df[col].dropna(), label=col)
                
            plt.title("Distribution of Selected Columns")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            
            st.pyplot(fig)
            
            # Download plot button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label="Download Plot",
                data=buf,
                file_name="distribution_plot.png",
                mime="image/png",
                use_container_width=True
            )
    
    def _render_correlation_analysis(self):
        """Render correlation analysis interface"""
        st.subheader("Correlation Analysis")
        
        # Get correlation matrix
        corr_matrix = self.analyze_correlation()
        
        if corr_matrix is None or corr_matrix.empty:
            st.info("Insufficient numeric columns for correlation analysis.")
            return
            
        # Display correlation matrix
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)
        
        # Visualization options
        viz_type = st.radio("Visualization Type", ["Heatmap", "Pairplot", "Scatter Matrix"], horizontal=True)
        
        # Create correlation heatmap
        if viz_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, 
                mask=mask,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                ax=ax
            )
            plt.title("Correlation Heatmap")
            st.pyplot(fig)
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label="Download Heatmap",
                data=buf,
                file_name="correlation_heatmap.png",
                mime="image/png",
                use_container_width=True
            )
        
        # Create pairplot (limit to max 5 columns)
        elif viz_type == "Pairplot":
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 5:
                st.warning("Too many numeric columns. Select up to 5 columns for pairplot.")
                selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:5])
            else:
                selected_cols = numeric_cols
                
            if selected_cols:
                with st.spinner("Generating pairplot..."):
                    fig = sns.pairplot(self.df[selected_cols], diag_kind="kde", height=2.5)
                    plt.suptitle("Pairwise Relationships", y=1.02)
                    st.pyplot(fig)
                    
                    # Download button
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=200)
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Pairplot",
                        data=buf,
                        file_name="pairplot.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        # Create scatter matrix (interactive with plotly)
        elif viz_type == "Scatter Matrix":
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 5:
                st.warning("Too many numeric columns. Select up to 5 columns for scatter matrix.")
                selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:5])
            else:
                selected_cols = numeric_cols
                
            if selected_cols:
                with st.spinner("Generating scatter matrix..."):
                    # Choose color column if there's a categorical column
                    cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
                    color_col = None
                    
                    if cat_cols:
                        color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                        
                    if color_col and color_col != "None":
                        fig = px.scatter_matrix(
                            self.df, 
                            dimensions=selected_cols,
                            color=color_col,
                            title="Interactive Scatter Matrix"
                        )
                    else:
                        fig = px.scatter_matrix(
                            self.df, 
                            dimensions=selected_cols,
                            title="Interactive Scatter Matrix"
                        )
                        
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_distribution_analysis(self):
        """Render distribution analysis interface"""
        st.subheader("Distribution Analysis")
        
        # Get distribution analysis
        distributions = self.analyze_distributions()
        
        if distributions is None or not distributions:
            st.info("No numeric columns found for distribution analysis.")
            return
        
        # Column selection
        selected_col = st.selectbox("Select column for detailed analysis", list(distributions.keys()))
        
        if selected_col:
            # Display distribution statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Statistics")
                st.write(f"**Mean:** {distributions[selected_col]['mean']:.2f}")
                st.write(f"**Median:** {distributions[selected_col]['median']:.2f}")
                st.write(f"**Standard Deviation:** {distributions[selected_col]['std']:.2f}")
                st.write(f"**Skewness:** {distributions[selected_col]['skewness']:.2f}")
                st.write(f"**Kurtosis:** {distributions[selected_col]['kurtosis']:.2f}")
                
                # Normality test
                shapiro_stat, shapiro_p = distributions[selected_col]['shapiro_test']
                st.write("**Normality Test (Shapiro-Wilk):**")
                st.write(f"- Statistic: {shapiro_stat:.4f}")
                st.write(f"- p-value: {shapiro_p:.4f}")
                
                if shapiro_p < 0.05:
                    st.write("- Result: Data is **not normally distributed**")
                else:
                    st.write("- Result: Data is **normally distributed**")
                    
            with col2:
                st.markdown("### Percentiles")
                percentiles = distributions[selected_col]['percentiles']
                
                st.write(f"**25th Percentile:** {percentiles['p25']:.2f}")
                st.write(f"**50th Percentile (Median):** {percentiles['p50']:.2f}")
                st.write(f"**75th Percentile:** {percentiles['p75']:.2f}")
                st.write(f"**95th Percentile:** {percentiles['p95']:.2f}")
                st.write(f"**99th Percentile:** {percentiles['p99']:.2f}")
                
                # IQR
                iqr = percentiles['p75'] - percentiles['p25']
                st.write(f"**Interquartile Range (IQR):** {iqr:.2f}")
            
            # Distribution visualizations
            st.subheader(f"Distribution Visualization for {selected_col}")
            
            viz_type = st.radio(
                "Select visualization type:", 
                ["Histogram with KDE", "Box Plot", "Violin Plot", "ECDF"], 
                horizontal=True
            )
            
            # Create selected visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "Histogram with KDE":
                sns.histplot(self.df[selected_col].dropna(), kde=True, ax=ax)
                plt.title(f"Histogram of {selected_col}")
                plt.xlabel(selected_col)
                plt.ylabel("Frequency")
                
            elif viz_type == "Box Plot":
                sns.boxplot(y=self.df[selected_col].dropna(), ax=ax)
                plt.title(f"Box Plot of {selected_col}")
                plt.ylabel(selected_col)
                
            elif viz_type == "Violin Plot":
                sns.violinplot(y=self.df[selected_col].dropna(), ax=ax)
                plt.title(f"Violin Plot of {selected_col}")
                plt.ylabel(selected_col)
                
            elif viz_type == "ECDF":
                # Empirical Cumulative Distribution Function
                sns.ecdfplot(self.df[selected_col].dropna(), ax=ax)
                plt.title(f"ECDF of {selected_col}")
                plt.xlabel(selected_col)
                plt.ylabel("Proportion")
                
            st.pyplot(fig)
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label=f"Download {viz_type}",
                data=buf,
                file_name=f"{selected_col}_{viz_type.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
    
    def _render_categorical_analysis(self):
        """Render categorical analysis interface"""
        st.subheader("Categorical Analysis")
        
        # Get categorical analysis
        categorical_stats = self.analyze_categorical()
        
        if categorical_stats is None or not categorical_stats:
            st.info("No categorical columns found for analysis.")
            return
        
        # Column selection
        selected_col = st.selectbox("Select categorical column for analysis", list(categorical_stats.keys()))
        
        if selected_col:
            stats = categorical_stats[selected_col]
            
            # Display basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Statistics")
                st.write(f"**Unique Values:** {stats['unique_count']}")
                st.write(f"**Most Common Value:** {stats['top_value']}")
                st.write(f"**Frequency of Most Common:** {stats['top_freq']}")
                
            with col2:
                # Create a pie chart for top categories
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(
                    stats['top_counts'][:5], 
                    labels=stats['top_categories'][:5], 
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')
                plt.title(f"Top 5 Categories in {selected_col}")
                st.pyplot(fig)
            
            # Detailed value counts
            st.subheader("Value Distribution")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # If too many categories, limit to top 20
            top_n = min(20, len(stats['top_categories']))
            
            bars = ax.bar(
                stats['top_categories'][:top_n],
                stats['top_counts'][:top_n]
            )
            
            # Add percentage labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f"{stats['percentages'][i]:.1f}%",
                    ha='center', 
                    va='bottom',
                    rotation=0
                )
            
            plt.title(f"Distribution of {selected_col}")
            plt.xlabel(selected_col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label="Download Bar Chart",
                data=buf,
                file_name=f"{selected_col}_distribution.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Show value counts table
            value_counts_df = pd.DataFrame({
                'Category': stats['top_categories'],
                'Count': stats['top_counts'],
                'Percentage': [f"{p:.2f}%" for p in stats['percentages']]
            })
            
            st.dataframe(value_counts_df, use_container_width=True)
    
    def _render_outlier_analysis(self):
        """Render outlier analysis interface"""
        st.subheader("Outlier Detection")
        
        # Get outlier analysis
        outliers = self.analyze_outliers()
        
        if outliers is None or not outliers:
            st.info("No numeric columns found for outlier analysis.")
            return
        
        # Overview of all columns
        st.markdown("### Outlier Summary")
        
        # Create summary table
        summary_data = []
        for col, stats in outliers.items():
            summary_data.append({
                'Column': col,
                'Q1': f"{stats['Q1']:.2f}",
                'Q3': f"{stats['Q3']:.2f}",
                'IQR': f"{stats['IQR']:.2f}",
                'Lower Bound': f"{stats['lower_bound']:.2f}",
                'Upper Bound': f"{stats['upper_bound']:.2f}",
                'Outlier Count': stats['outlier_count'],
                'Outlier %': f"{stats['outlier_percentage']:.2f}%"
            })
            
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Column selection for detailed analysis
        selected_col = st.selectbox("Select column for detailed outlier analysis", list(outliers.keys()))
        
        if selected_col:
            st.markdown(f"### Outlier Analysis: {selected_col}")
            
            # Get outlier stats for selected column
            stats = outliers[selected_col]
            
            # Show boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=self.df[selected_col].dropna(), ax=ax)
            
            # Add vertical lines for bounds
            plt.axvline(x=stats['lower_bound'], color='r', linestyle='--', label='Lower Bound')
            plt.axvline(x=stats['upper_bound'], color='r', linestyle='--', label='Upper Bound')
            
            plt.title(f"Box Plot with Outlier Bounds for {selected_col}")
            plt.xlabel(selected_col)
            plt.legend()
            
            st.pyplot(fig)
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label="Download Box Plot",
                data=buf,
                file_name=f"{selected_col}_outliers_boxplot.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Display outlier values if not too many
            if stats['outlier_count'] > 0:
                st.markdown(f"### Outlier Values ({stats['outlier_count']} outliers found)")
                
                # Get outlier rows
                outlier_mask = (self.df[selected_col] < stats['lower_bound']) | (self.df[selected_col] > stats['upper_bound'])
                outlier_df = self.df[outlier_mask]
                
                if len(outlier_df) <= 100:
                    st.dataframe(outlier_df, use_container_width=True)
                else:
                    st.warning(f"Too many outliers to display ({len(outlier_df)}). Showing first 100.")
                    st.dataframe(outlier_df.head(100), use_container_width=True)
    
    def _render_missing_values_analysis(self):
        """Render missing values analysis interface"""
        st.subheader("Missing Values Analysis")
        
        # Get missing values analysis
        missing = self.analyze_missing_values()
        
        if missing is None:
            st.info("No data available for missing values analysis.")
            return
        
        # Overall stats
        st.markdown("### Overall Missing Values")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cells", missing['overall']['total_cells'])
        with col2:
            st.metric("Missing Cells", missing['overall']['missing_cells'])
        with col3:
            st.metric("Missing Percentage", f"{missing['overall']['percentage']:.2f}%")
        
        # Missing values by column
        st.markdown("### Missing Values by Column")
        
        # Create dataframe for visualization
        missing_df = pd.DataFrame([
            {
                'Column': col,
                'Missing Count': stats['count'],
                'Missing Percentage': stats['percentage']
            }
            for col, stats in missing['columns'].items()
        ])
        
        # Sort by missing percentage
        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
        
        # Only show columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if missing_df.empty:
            st.success("No missing values found in any column!")
        else:
            # Display as table
            st.dataframe(missing_df, use_container_width=True)
            
            # Create bar chart for visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot
            bars = ax.bar(
                missing_df['Column'],
                missing_df['Missing Percentage']
            )
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.3,
                    f"{height:.1f}%",
                    ha='center', 
                    va='bottom',
                    rotation=0
                )
            
            plt.title("Missing Values by Column")
            plt.xlabel("Column")
            plt.ylabel("Missing Percentage (%)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            
            st.download_button(
                label="Download Missing Values Chart",
                data=buf,
                file_name="missing_values.png",
                mime="image/png",
                use_container_width=True
            )
    
    def _render_group_analysis(self):
        """Render group analysis interface"""
        st.subheader("Group Analysis")
        
        # Get columns for grouping and aggregation
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or not num_cols:
            st.info("Dataset must have both categorical and numeric columns for group analysis.")
            return
        
        # User selections
        group_col = st.selectbox("Select grouping column", cat_cols)
        agg_col = st.selectbox("Select column to aggregate", num_cols)
        
        agg_functions = {
            'Mean': 'mean',
            'Median': 'median',
            'Sum': 'sum',
            'Count': 'count',
            'Standard Deviation': 'std',
            'Minimum': 'min',
            'Maximum': 'max'
        }
        
        agg_func = st.selectbox("Select aggregation function", list(agg_functions.keys()))
        
        # Perform groupby
        grouped_df = self.perform_group_analysis(
            group_col, 
            agg_col, 
            agg_functions[agg_func]
        )
        
        if grouped_df is None or grouped_df.empty:
            st.error("Error performing group analysis.")
            return
        
        # Display results
        st.markdown(f"### {agg_func} of {agg_col} by {group_col}")
        
        # Show data table
        st.dataframe(grouped_df, use_container_width=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort values for better visualization
        grouped_df = grouped_df.sort_values(agg_col, ascending=False)
        
        # Plot
        bars = ax.bar(
            grouped_df[group_col].astype(str),
            grouped_df[agg_col]
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (height * 0.01),
                f"{height:.2f}",
                ha='center', 
                va='bottom',
                rotation=0
            )
        
        plt.title(f"{agg_func} of {agg_col} by {group_col}")
        plt.xlabel(group_col)
        plt.ylabel(f"{agg_func} of {agg_col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Download button
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        
        st.download_button(
            label="Download Group Analysis Chart",
            data=buf,
            file_name=f"{agg_func}_{agg_col}_by_{group_col}.png",
            mime="image/png",
            use_container_width=True
        )