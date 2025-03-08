insights.append("🔍 Categorical variables:")
                for insight in cat_insights:
                    insights.append(insight)
        
        # Outlier insights (if there are numeric columns)
        if numeric_cols:
            outlier_insights = []
            for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
                # Calculate Q1, Q3, and IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                if outlier_percentage > 5:
                    outlier_insights.append(f"  - {col}: {outlier_count} outliers ({outlier_percentage:.1f}% of data)")
            
            if outlier_insights:
                insights.append("⚠️ Significant outliers detected:")
                for insight in outlier_insights:
                    insights.append(insight)
        
        # Add insights to response
        for insight in insights:
            insights_text += f"\n- {insight}"
        
        # Create visualization
        if numeric_cols and len(numeric_cols) >= 2:
            # Create scatter plot of two key numeric variables
            col1 = numeric_cols[0]
            col2 = numeric_cols[1]
            
            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                title=f"Relationship: {col2} vs {col1}",
                template="plotly_white"
            )
            
            # Add trendline
            fig.update_layout(
                xaxis_title=col1,
                yaxis_title=col2
            )
        elif categorical_cols:
            # Create bar chart of categorical distribution
            col = categorical_cols[0]
            value_counts = self.df[col].value_counts()
            
            # Limit to top 10 categories if there are too many
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count"
            )
        else:
            # Create a simple data overview
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=["Column", "Type", "Non-Null", "Unique Values"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        self.df.columns,
                        self.df.dtypes.astype(str),
                        self.df.count().values,
                        [self.df[col].nunique() for col in self.df.columns]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )])
            
            fig.update_layout(
                title="Dataset Overview",
                template="plotly_white"
            )
        
        return {
            "text": insights_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _default_response(self, query):
        """Generate a default response when query isn't recognized"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        return {
            "text": f"""
            I'm not sure how to answer that specific question about your data. Here are some things you can ask me:
            
            - Summarize this dataset
            - Show me correlations between variables
            - Analyze the distribution of a specific column
            - Compare different groups in the data
            - Create visualizations like bar charts, scatter plots, etc.
            - Find outliers in the data
            - Provide statistics for specific columns
            
            Feel free to try one of these questions or rephrase your query.
            """
        }
            ### Bar Chart: Count by {x_col}
            
            I've created a bar chart showing the count of items in each {x_col} category.
            
            **Key observations:**
            - Most common {x_col}: {df_grouped.iloc[0][x_col]} ({df_grouped.iloc[0]['count']} instances)
            - Least common {x_col} (shown): {df_grouped.iloc[-1][x_col]} ({df_grouped.iloc[-1]['count']} instances)
            """
        else:
            # Aggregate by category and value
            df_grouped = self.df.groupby(x_col)[y_col].mean().reset_index()
            
            # Sort by the numeric column for better visualization
            df_grouped = df_grouped.sort_values(y_col, ascending=False)
            
            # Limit to top 15 categories if there are too many
            if len(df_grouped) > 15:
                df_grouped = df_grouped.head(15)
            
            fig = px.bar(
                df_grouped,
                x=x_col,
                y=y_col,
                title=f"Average {y_col} by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Bar Chart: Average {y_col} by {x_col}
            
            I've created a bar chart showing the average {y_col} for each {x_col} category.
            
            **Key observations:**
            - Highest average {y_col}: {df_grouped.iloc[0][x_col]} ({df_grouped.iloc[0][y_col]:.2f})
            - Lowest average {y_col} (shown): {df_grouped.iloc[-1][x_col]} ({df_grouped.iloc[-1][y_col]:.2f})
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_line_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a line chart visualization"""
        # Need at least one column to use as x-axis and one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a line chart because I need at least one numeric column."
            }
        
        # Look for potential time/date columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also look for columns that might be dates but not detected as such
        potential_date_cols = []
        for col in self.df.columns:
            if col in date_cols:
                continue
                
            # Look for columns with "date", "time", "year", "month", "day" in the name
            if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                potential_date_cols.append(col)
        
        # Combine confirmed and potential date columns
        time_cols = date_cols + potential_date_cols
        
        # Select columns for the line chart
        x_col = None
        y_cols = []
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for a time column first
            for col in mentioned_cols:
                if col in time_cols:
                    x_col = col
                    break
            
            # Look for numeric columns
            for col in mentioned_cols:
                if col in numeric_cols and col != x_col:
                    y_cols.append(col)
        
        # If no time column found, look for a column that's sortable
        if not x_col:
            # Use the first time column if available
            if time_cols:
                x_col = time_cols[0]
            else:
                # Otherwise, use the first column as x-axis
                x_col = self.df.columns[0]
        
        # If no y columns found, use all numeric columns (up to 3)
        if not y_cols:
            y_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        # Limit to 3 y columns for readability
        y_cols = y_cols[:3]
        
        # Sort by x column
        df_sorted = self.df.sort_values(x_col)
        
        # Create the line chart
        fig = go.Figure()
        
        for col in y_cols:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{', '.join(y_cols)} by {x_col}",
            xaxis_title=x_col,
            yaxis_title="Value",
            template="plotly_white"
        )
        
        response_text = f"""
        ### Line Chart: {', '.join(y_cols)} by {x_col}
        
        I've created a line chart showing how {', '.join(y_cols)} change across {x_col}.
        
        **Key observations:**
        """
        
        # Add observations for each y column
        for col in y_cols:
            try:
                min_val = df_sorted[col].min()
                max_val = df_sorted[col].max()
                min_x = df_sorted[df_sorted[col] == min_val].iloc[0][x_col]
                max_x = df_sorted[df_sorted[col] == max_val].iloc[0][x_col]
                
                response_text += f"""
                - **{col}**:
                  - Maximum: {max_val:.2f} at {max_x}
                  - Minimum: {min_val:.2f} at {min_x}
                """
            except:
                # Skip if there's an error (e.g., with datetime values)
                pass
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_scatter_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a scatter plot visualization"""
        # Need at least two numeric columns
        if len(numeric_cols) < 2:
            return {
                "text": "I couldn't create a scatter plot because I need at least two numeric columns."
            }
        
        # Select columns for the scatter plot
        x_col = None
        y_col = None
        color_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Find numeric columns
            numeric_mentions = [col for col in mentioned_cols if col in numeric_cols]
            
            if len(numeric_mentions) >= 2:
                x_col = numeric_mentions[0]
                y_col = numeric_mentions[1]
            
            # Look for a color column
            if len(mentioned_cols) > 2:
                for col in mentioned_cols:
                    if col != x_col and col != y_col:
                        color_col = col
                        break
        
        # If no suitable columns found from mentions, use defaults
        if not x_col or not y_col:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        
        # Create the scatter plot
        if color_col:
            fig = px.scatter(
                self.df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col} (colored by {color_col})",
                template="plotly_white"
            )
        else:
            fig = px.scatter(
                self.df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                template="plotly_white"
            )
        
        # Add trendline
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        # Calculate correlation
        correlation = self.df[[x_col, y_col]].corr().iloc[0, 1]
        
        response_text = f"""
        ### Scatter Plot: {y_col} vs {x_col}
        
        I've created a scatter plot showing the relationship between {x_col} and {y_col}.
        
        **Key observations:**
        - Correlation coefficient: {correlation:.4f}
        - Relationship: """
        
        if correlation > 0.7:
            response_text += "Strong positive correlation"
        elif correlation > 0.3:
            response_text += "Moderate positive correlation"
        elif correlation > 0:
            response_text += "Weak positive correlation"
        elif correlation > -0.3:
            response_text += "Weak negative correlation"
        elif correlation > -0.7:
            response_text += "Moderate negative correlation"
        else:
            response_text += "Strong negative correlation"
        
        if color_col:
            response_text += f"\n- Points are colored by {color_col}"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_histogram_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a histogram visualization"""
        # Need at least one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a histogram because I need at least one numeric column."
            }
        
        # Select column for the histogram
        col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            for mentioned_col in mentioned_cols:
                if mentioned_col in numeric_cols:
                    col = mentioned_col
                    break
        
        # If no suitable column found from mentions, use default
        if not col:
            col = numeric_cols[0]
        
        # Create the histogram
        fig = px.histogram(
            self.df,
            x=col,
            marginal="box",
            title=f"Distribution of {col}",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Count"
        )
        
        # Calculate basic statistics
        mean = self.df[col].mean()
        median = self.df[col].median()
        std = self.df[col].std()
        skewness = self.df[col].skew()
        
        response_text = f"""
        ### Histogram: Distribution of {col}
        
        I've created a histogram showing the distribution of {col}.
        
        **Key statistics:**
        - Mean: {mean:.2f}
        - Median: {median:.2f}
        - Standard Deviation: {std:.2f}
        - Skewness: {skewness:.2f} ("""
        
        if skewness > 1:
            response_text += "highly right-skewed"
        elif skewness > 0.5:
            response_text += "moderately right-skewed"
        elif skewness > -0.5:
            response_text += "approximately symmetric"
        elif skewness > -1:
            response_text += "moderately left-skewed"
        else:
            response_text += "highly left-skewed"
        
        response_text += ")"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_pie_visualization(self, query, mentioned_cols, categorical_cols):
        """Create a pie chart visualization"""
        # Need at least one categorical column
        if not categorical_cols:
            return {
                "text": "I couldn't create a pie chart because I need at least one categorical column."
            }
        
        # Select column for the pie chart
        col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            for mentioned_col in mentioned_cols:
                if mentioned_col in categorical_cols:
                    col = mentioned_col
                    break
        
        # If no suitable column found from mentions, use default
        if not col:
            col = categorical_cols[0]
        
        # Count values in the categorical column
        value_counts = self.df[col].value_counts()
        
        # Limit to top 10 categories if there are too many
        if len(value_counts) > 10:
            others_sum = value_counts[10:].sum()
            value_counts = value_counts[:10]
            value_counts["Other"] = others_sum
        
        # Create the pie chart
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {col}",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        response_text = f"""
        ### Pie Chart: Distribution of {col}
        
        I've created a pie chart showing the distribution of {col}.
        
        **Key observations:**
        - Most common category: {value_counts.index[0]} ({value_counts.values[0]} instances, {(value_counts.values[0] / value_counts.sum()) * 100:.1f}%)
        """
        
        if len(value_counts) > 5:
            response_text += f"- There are {len(value_counts)} different categories shown"
        
        if "Other" in value_counts:
            response_text += f"\n- The 'Other' category combines {len(self.df[col].unique()) - 9} less common categories"
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_box_visualization(self, query, mentioned_cols, numeric_cols, categorical_cols):
        """Create a box plot visualization"""
        # Need at least one numeric column
        if not numeric_cols:
            return {
                "text": "I couldn't create a box plot because I need at least one numeric column."
            }
        
        # Select columns for the box plot
        y_col = None
        x_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for numeric column first
            for col in mentioned_cols:
                if col in numeric_cols:
                    y_col = col
                    break
            
            # Then look for categorical column
            for col in mentioned_cols:
                if col in categorical_cols:
                    x_col = col
                    break
        
        # If no suitable columns found from mentions, use defaults
        if not y_col:
            y_col = numeric_cols[0]
        
        # Create the box plot
        if x_col:
            fig = px.box(
                self.df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Box Plot: Distribution of {y_col} by {x_col}
            
            I've created a box plot showing how the distribution of {y_col} varies across different {x_col} categories.
            
            **Key observations:**
            """
            
            # Calculate statistics for each group
            grouped_stats = self.df.groupby(x_col)[y_col].agg(['median', 'mean', 'std']).reset_index()
            grouped_stats = grouped_stats.sort_values('median', ascending=False)
            
            response_text += f"""
            - Highest median {y_col}: {grouped_stats.iloc[0][x_col]} ({grouped_stats.iloc[0]['median']:.2f})
            - Lowest median {y_col}: {grouped_stats.iloc[-1][x_col]} ({grouped_stats.iloc[-1]['median']:.2f})
            """
        else:
            fig = px.box(
                self.df,
                y=y_col,
                title=f"Distribution of {y_col}",
                template="plotly_white"
            )
            
            # Calculate basic statistics
            median = self.df[y_col].median()
            mean = self.df[y_col].mean()
            q1 = self.df[y_col].quantile(0.25)
            q3 = self.df[y_col].quantile(0.75)
            iqr = q3 - q1
            
            response_text = f"""
            ### Box Plot: Distribution of {y_col}
            
            I've created a box plot showing the distribution of {y_col}.
            
            **Key statistics:**
            - Median: {median:.2f}
            - Mean: {mean:.2f}
            - Interquartile Range (IQR): {iqr:.2f}
            - Q1 (25th percentile): {q1:.2f}
            - Q3 (75th percentile): {q3:.2f}
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_heatmap_visualization(self, query, mentioned_cols, numeric_cols):
        """Create a heatmap visualization"""
        # Need at least two numeric columns
        if len(numeric_cols) < 2:
            return {
                "text": "I couldn't create a heatmap because I need at least two numeric columns."
            }
        
        # Select columns for the heatmap
        cols_to_use = []
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            cols_to_use = [col for col in mentioned_cols if col in numeric_cols]
        
        # If no suitable columns found from mentions, use defaults
        if not cols_to_use or len(cols_to_use) < 2:
            # Use all numeric columns (up to 10 for readability)
            cols_to_use = numeric_cols[:min(10, len(numeric_cols))]
        
        # Calculate correlation matrix
        corr_matrix = self.df[cols_to_use].corr()
        
        # Create the heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
            template="plotly_white"
        )
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(cols_to_use)):
            for j in range(i+1, len(cols_to_use)):
                col1 = cols_to_use[i]
                col2 = cols_to_use[j]
                corr_value = corr_matrix.loc[col1, col2]
                corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        response_text = """
        ### Correlation Heatmap
        
        I've created a heatmap showing the correlations between numeric variables.
        
        **Key correlations:**
        """
        
        # Add top 3 correlations (or fewer if there aren't that many)
        for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
            strength = "strong positive" if corr > 0.7 else \
                      "moderate positive" if corr > 0.3 else \
                      "weak positive" if corr > 0 else \
                      "strong negative" if corr < -0.7 else \
                      "moderate negative" if corr < -0.3 else \
                      "weak negative"
            
            response_text += f"""
            - {col1} & {col2}: {corr:.2f} ({strength})
            """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _compare_groups(self, query):
        """Compare groups in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get categorical columns for grouping
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numeric columns for metrics
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not cat_cols or not numeric_cols:
            return {
                "text": "Group comparison requires at least one categorical column and one numeric column."
            }
        
        # Try to identify the grouping column and the metrics from the query
        group_col = None
        metric_cols = []
        
        # Check which columns are mentioned in the query
        mentioned_cols = [col for col in self.df.columns if col.lower() in query.lower()]
        
        if mentioned_cols:
            # Look for a categorical column first
            for col in mentioned_cols:
                if col in cat_cols:
                    group_col = col
                    break
            
            # Then look for numeric columns
            for col in mentioned_cols:
                if col in numeric_cols and col != group_col:
                    metric_cols.append(col)
        
        # If no suitable columns found from mentions, use defaults
        if not group_col:
            group_col = cat_cols[0]
        
        if not metric_cols:
            # Use the first 3 numeric columns as metrics
            metric_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        # Generate comparison
        response_text = f"""
        ### Group Comparison: by {group_col}
        
        I've compared different groups based on **{group_col}**, analyzing these metrics:
        {', '.join([f"**{col}**" for col in metric_cols])}
        
        **Key findings:**
        """
        
        # Calculate group statistics
        group_stats = {}
        
        for metric in metric_cols:
            # Calculate mean, median, std for each group
            stats = self.df.groupby(group_col)[metric].agg(['mean', 'median', 'std']).reset_index()
            
            # Sort by mean
            stats = stats.sort_values('mean', ascending=False)
            
            group_stats[metric] = stats
            
            response_text += f"""
            
            **{metric}**:
            - Highest average: {stats.iloc[0][group_col]} ({stats.iloc[0]['mean']:.2f})
            - Lowest average: {stats.iloc[-1][group_col]} ({stats.iloc[-1]['mean']:.2f})
            """
            
            # Add statistical significance info if there are only 2 major groups
            unique_groups = self.df[group_col].nunique()
            if unique_groups == 2:
                group_values = self.df[group_col].unique().tolist()
                group1_data = self.df[self.df[group_col] == group_values[0]][metric].dropna()
                group2_data = self.df[self.df[group_col] == group_values[1]][metric].dropna()
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    try:
                        # Perform t-test
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        
                        response_text += f"""
                        - Statistical significance: {"Significant" if p_value < 0.05 else "Not significant"} (p-value: {p_value:.4f})
                        """
                    except:
                        # Skip if t-test fails
                        pass
        
        # Create visualization
        if len(metric_cols) == 1:
            # Single metric: bar chart
            metric = metric_cols[0]
            stats = group_stats[metric]
            
            fig = px.bar(
                stats,
                x=group_col,
                y='mean',
                error_y=stats['std'],
                title=f"Comparison of {metric} by {group_col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=group_col,
                yaxis_title=f"Average {metric}"
            )
        else:
            # Multiple metrics: radar chart or multi-bar chart
            if len(self.df[group_col].unique()) <= 5:
                # Use radar chart for few groups
                fig = go.Figure()
                
                categories = metric_cols
                
                for group in self.df[group_col].unique():
                    group_means = [self.df[self.df[group_col] == group][metric].mean() for metric in metric_cols]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=group_means,
                        theta=categories,
                        fill='toself',
                        name=str(group)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True
                        )
                    ),
                    title=f"Comparison by {group_col}",
                    template="plotly_white"
                )
            else:
                # Use bar chart for many groups
                # Create comparison of first metric
                metric = metric_cols[0]
                stats = group_stats[metric]
                
                # Limit to top 10 groups if there are too many
                if len(stats) > 10:
                    stats = stats.head(10)
                
                fig = px.bar(
                    stats,
                    x=group_col,
                    y='mean',
                    error_y=stats['std'],
                    title=f"Comparison of {metric} by {group_col} (Top 10)",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    xaxis_title=group_col,
                    yaxis_title=f"Average {metric}"
                )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _provide_insights(self):
        """Provide general insights about the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Basic dataset info
        rows, cols = self.df.shape
        
        # Get column types
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate insights text
        insights_text = f"""
        ### Key Insights from the Dataset
        
        I've analyzed your dataset with {rows:,} rows and {cols} columns, and here are the main insights:
        """
        
        insights = []
        
        # Missing values insight
        missing_values = self.df.isna().sum().sum()
        missing_percentage = (missing_values / (rows * cols)) * 100
        if missing_percentage > 0:
            insights.append(f"📊 Your dataset has {missing_values:,} missing values ({missing_percentage:.2f}% of all cells).")
            
            # Add columns with most missing values
            cols_with_missing = self.df.columns[self.df.isna().any()].tolist()
            if cols_with_missing:
                missing_by_col = self.df[cols_with_missing].isna().sum().sort_values(ascending=False)
                top_missing = missing_by_col.head(3)
                insights.append(f"🔍 Columns with most missing values: " + 
                               ", ".join([f"{col} ({missing:,}, {(missing/rows)*100:.1f}%)" 
                                         for col, missing in top_missing.items()]))
        
        # Correlations insight (if there are numeric columns)
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find highest absolute correlations (excluding self-correlations)
            high_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Only include strong correlations
                        high_corrs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation value
            high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            if high_corrs:
                insights.append("📈 Strong correlations found:")
                for col1, col2, corr in high_corrs[:3]:  # Show top 3
                    corr_type = "positive" if corr > 0 else "negative"
                    insights.append(f"  - {col1} and {col2}: {corr:.2f} ({corr_type})")
        
        # Distribution insights (if there are numeric columns)
        if numeric_cols:
            # Look for columns with skewed distributions
            skewed_cols = []
            for col in numeric_cols:
                skewness = self.df[col].skew()
                if abs(skewness) > 1:  # Significantly skewed
                    skew_direction = "right" if skewness > 0 else "left"
                    skewed_cols.append((col, skewness, skew_direction))
            
            if skewed_cols:
                skewed_cols.sort(key=lambda x: abs(x[1]), reverse=True)
                insights.append("📊 Skewed distributions:")
                for col, skewness, direction in skewed_cols[:3]:  # Show top 3
                    insights.append(f"  - {col}: {direction}-skewed (skewness = {skewness:.2f})")
        
        # Categorical insights (if there are categorical columns)
        if categorical_cols:
            cat_insights = []
            for col in categorical_cols[:3]:  # Analyze top 3 categorical columns
                value_counts = self.df[col].value_counts()
                unique_count = len(value_counts)
                top_value = value_counts.index[0] if not value_counts.empty else "N/A"
                top_percentage = (value_counts.iloc[0] / len(self.df)) * 100 if not value_counts.empty else 0
                
                if unique_count == 1:
                    cat_insights.append(f"  - {col}: Only one value ({top_value})")
                elif unique_count <= 5:
                    cat_insights.append(f"  - {col}: {unique_count} unique values, most common is {top_value} ({top_percentage:.1f}%)")
                else:
                    cat_insights.append(f"  - {col}: {unique_count} unique values, most common is {top_value} ({top_percentage:.1f}%)")
            
            if cat_insights:
                insights.append("🔍 Categorical variables:")            unique_count = self.df[col].nunique()
            summary += f"""
            - **{col}**: 
                - Unique values: {unique_count}
                - Most common: {top_value} ({top_count} occurrences)
                - Missing: {self.df[col].isna().sum()}
            """
        
        # Create a visualization showing column data types
        fig = go.Figure()
        
        # Count column types
        type_counts = {
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'Date/Time': len(date_cols),
            'Other': cols - len(numeric_cols) - len(categorical_cols) - len(date_cols)
        }
        
        # Filter out zero counts
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        # Create pie chart
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Data Types",
            template="plotly_white"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return {
            "text": summary,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_correlations(self, query):
        """Analyze correlations in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns for correlation analysis
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {
                "text": "Correlation analysis requires at least two numeric columns, but I could only find " +
                      f"{len(numeric_cols)} numeric column in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use all numeric columns
        if len(mentioned_cols) >= 2:
            cols_to_analyze = mentioned_cols
        else:
            # Use all numeric columns (limit to 10 for readability)
            cols_to_analyze = numeric_cols[:min(10, len(numeric_cols))]
        
        # Calculate correlation matrix
        corr_matrix = self.df[cols_to_analyze].corr()
        
        # Find highest correlations (excluding self-correlations)
        correlations = []
        for i in range(len(cols_to_analyze)):
            for j in range(i+1, len(cols_to_analyze)):
                col1 = cols_to_analyze[i]
                col2 = cols_to_analyze[j]
                corr_value = corr_matrix.loc[col1, col2]
                correlations.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate response text
        response_text = f"""
        ### Correlation Analysis
        
        I've analyzed the relationships between numeric variables in your dataset. 
        Here are the key findings:
        
        **Top Correlations:**
        """
        
        # Add top correlations to response
        for col1, col2, corr in correlations[:5]:
            strength = "strong positive" if corr > 0.7 else \
                      "moderate positive" if corr > 0.3 else \
                      "weak positive" if corr > 0 else \
                      "strong negative" if corr < -0.7 else \
                      "moderate negative" if corr < -0.3 else \
                      "weak negative"
            
            response_text += f"""
            - **{col1}** and **{col2}**: {corr:.3f} ({strength} correlation)
            """
        
        # Create heatmap visualization
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            template="plotly_white",
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )
        
        # Add interpretation if requested
        if "interpret" in query.lower() or "explain" in query.lower() or "insights" in query.lower():
            response_text += """
            
            **Interpretation:**
            
            - **Positive correlation** means that as one variable increases, the other tends to increase as well.
            - **Negative correlation** means that as one variable increases, the other tends to decrease.
            - **Correlation values** range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
            - A value close to 0 suggests little to no linear relationship.
            
            Remember that correlation does not imply causation. Just because two variables are correlated doesn't mean that one causes the other.
            """
            
            # If we have highly correlated variables, add specific insights
            if correlations and abs(correlations[0][2]) > 0.7:
                col1, col2, corr = correlations[0]
                direction = "positively" if corr > 0 else "negatively"
                response_text += f"""
                
                **Key Insight:**
                
                The strongest relationship in your data is between **{col1}** and **{col2}** with a correlation of {corr:.3f}. These variables are strongly {direction} correlated, meaning they tend to {
                "increase together" if corr > 0 else "move in opposite directions"}.
                """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_distributions(self, query):
        """Analyze distributions of variables in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Distribution analysis requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use a subset of numeric columns
        if mentioned_cols:
            cols_to_analyze = mentioned_cols[:4]  # Limit to 4 for visualization
        else:
            # Use a subset of numeric columns (up to 4 for readability)
            cols_to_analyze = numeric_cols[:min(4, len(numeric_cols))]
        
        # Generate response text
        response_text = f"""
        ### Distribution Analysis
        
        I've analyzed the distributions of the numeric variables in your dataset.
        Here are the key findings:
        
        """
        
        # Create visualization
        if len(cols_to_analyze) == 1:
            # Single column distribution
            col = cols_to_analyze[0]
            
            # Check for skewness
            skewness = self.df[col].skew()
            skew_description = "highly right-skewed" if skewness > 1 else \
                              "moderately right-skewed" if skewness > 0.5 else \
                              "approximately symmetric" if abs(skewness) <= 0.5 else \
                              "moderately left-skewed" if skewness > -1 else \
                              "highly left-skewed"
            
            # Calculate basic statistics
            col_stats = self.df[col].describe()
            iqr = col_stats['75%'] - col_stats['25%']
            
            response_text += f"""
            **{col}**:
            - Mean: {col_stats['mean']:.2f}
            - Median: {col_stats['50%']:.2f}
            - Standard Deviation: {col_stats['std']:.2f}
            - Minimum: {col_stats['min']:.2f}
            - Maximum: {col_stats['max']:.2f}
            - Interquartile Range (IQR): {iqr:.2f}
            - Distribution shape: {skew_description} (skewness = {skewness:.2f})
            """
            
            # Generate distribution plot
            fig = px.histogram(
                self.df,
                x=col,
                marginal="box",
                title=f"Distribution of {col}",
                template="plotly_white"
            )
        else:
            # Multiple columns distribution
            response_text += "**Summary Statistics:**\n\n"
            
            for col in cols_to_analyze:
                col_stats = self.df[col].describe()
                skewness = self.df[col].skew()
                skew_description = "right-skewed" if skewness > 0.5 else \
                                  "approximately symmetric" if abs(skewness) <= 0.5 else \
                                  "left-skewed"
                
                response_text += f"""
                **{col}**:
                - Mean: {col_stats['mean']:.2f}
                - Median: {col_stats['50%']:.2f}
                - Distribution: {skew_description}
                """
            
            # Create subplots for multiple distributions
            fig = make_subplots(
                rows=2, 
                cols=2, 
                subplot_titles=[f"Distribution of {col}" for col in cols_to_analyze[:4]]
            )
            
            # Add histograms
            for i, col in enumerate(cols_to_analyze[:4]):
                row = i // 2 + 1
                col_idx = i % 2 + 1
                
                # Add histogram trace
                fig.add_trace(
                    go.Histogram(
                        x=self.df[col],
                        name=col,
                        showlegend=False,
                        opacity=0.7
                    ),
                    row=row, 
                    col=col_idx
                )
                
                # Add KDE
                kde_x, kde_y = self._compute_kde(self.df[col])
                fig.add_trace(
                    go.Scatter(
                        x=kde_x, 
                        y=kde_y,
                        mode='lines',
                        name=f"{col} KDE",
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=row, 
                    col=col_idx
                )
            
            # Update layout
            fig.update_layout(
                title="Distribution Analysis",
                template="plotly_white",
                height=600
            )
        
        # Add interpretation
        response_text += """
        
        **Interpretation:**
        - A **symmetric distribution** means values are evenly distributed around the mean.
        - A **right-skewed distribution** has a long tail to the right, with most values concentrated on the left.
        - A **left-skewed distribution** has a long tail to the left, with most values concentrated on the right.
        - **Outliers** are points that lie far from the majority of the data and may need special attention.
        """
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _compute_kde(self, data, points=100):
        """Helper method to compute KDE for a data series"""
        import numpy as np
        from scipy import stats
        
        # Remove NaN values
        data = data.dropna()
        
        # Define the range for the KDE
        min_val = data.min()
        max_val = data.max()
        x = np.linspace(min_val, max_val, points)
        
        # Compute KDE
        kde = stats.gaussian_kde(data)
        y = kde(x)
        
        return x, y
    
    def _provide_statistics(self, query):
        """Provide statistical analysis based on the query"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get all columns
        all_cols = self.df.columns.tolist()
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Statistical analysis requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check which statistic is requested
        stat_type = None
        if "mean" in query.lower() or "average" in query.lower():
            stat_type = "mean"
            stat_function = np.mean
            stat_name = "Mean"
        elif "median" in query.lower():
            stat_type = "median"
            stat_function = np.median
            stat_name = "Median"
        elif "min" in query.lower() or "minimum" in query.lower():
            stat_type = "min"
            stat_function = np.min
            stat_name = "Minimum"
        elif "max" in query.lower() or "maximum" in query.lower():
            stat_type = "max"
            stat_function = np.max
            stat_name = "Maximum"
        elif "sum" in query.lower() or "total" in query.lower():
            stat_type = "sum"
            stat_function = np.sum
            stat_name = "Sum"
        elif "std" in query.lower() or "standard deviation" in query.lower():
            stat_type = "std"
            stat_function = np.std
            stat_name = "Standard Deviation"
        elif "var" in query.lower() or "variance" in query.lower():
            stat_type = "var"
            stat_function = np.var
            stat_name = "Variance"
        else:
            # Default to providing a complete statistical summary
            stat_type = "summary"
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in all_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use numeric columns
        if mentioned_cols:
            cols_to_analyze = [col for col in mentioned_cols if col in numeric_cols]
            if not cols_to_analyze:
                return {
                    "text": f"I found the columns you mentioned ({', '.join(mentioned_cols)}), but they don't appear to be numeric columns that I can analyze statistically."
                }
        else:
            # Use all numeric columns
            cols_to_analyze = numeric_cols
        
        # Generate response based on the requested statistic
        if stat_type == "summary":
            # Full statistical summary
            stats_df = self.df[cols_to_analyze].describe().T
            
            # Round to 2 decimal places for readability
            stats_df = stats_df.round(2)
            
            # Convert to HTML for display
            stats_html = stats_df.to_html()
            
            response_text = f"""
            ### Statistical Summary
            
            Here's a statistical summary of the numeric columns in your dataset:
            
            {stats_html}
            
            **Interpretation:**
            - **count**: Number of non-missing values
            - **mean**: Average value
            - **std**: Standard deviation (measure of spread)
            - **min**: Minimum value
            - **25%**: First quartile (25th percentile)
            - **50%**: Median (50th percentile)
            - **75%**: Third quartile (75th percentile)
            - **max**: Maximum value
            """
            
            # Create visualization (box plots)
            fig = px.box(
                self.df,
                y=cols_to_analyze[:10],  # Limit to 10 columns for readability
                title="Statistical Distribution",
                template="plotly_white"
            )
        else:
            # Specific statistic for each column
            results = {}
            for col in cols_to_analyze:
                results[col] = stat_function(self.df[col].dropna())
            
            # Create response text
            response_text = f"""
            ### {stat_name} Analysis
            
            I've calculated the {stat_name.lower()} for the following columns:
            
            """
            
            for col, value in results.items():
                response_text += f"- **{col}**: {value:.2f}\n"
            
            # Create bar chart visualization
            fig = px.bar(
                x=list(results.keys()),
                y=list(results.values()),
                title=f"{stat_name} by Column",
                labels={"x": "Column", "y": stat_name},
                template="plotly_white"
            )
            
            # Add value labels on top of bars
            fig.update_traces(
                text=[f"{val:.2f}" for val in results.values()],
                textposition="outside"
            )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _analyze_trends(self, query):
        """Analyze trends in the data"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Look for potential time/date columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also look for columns that might be dates but not detected as such
        potential_date_cols = []
        for col in self.df.columns:
            if col in date_cols:
                continue
                
            # Look for columns with "date", "time", "year", "month", "day" in the name
            if any(term in col.lower() for term in ["date", "time", "year", "month", "day"]):
                potential_date_cols.append(col)
        
        # Combine confirmed and potential date columns
        time_cols = date_cols + potential_date_cols
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in self.df.columns if col.lower() in query.lower()]
        time_col = None
        
        # Try to identify the time column and value column from the query
        if mentioned_cols:
            for col in mentioned_cols:
                if col in time_cols:
                    time_col = col
                    break
            
            # If no time column found among mentioned columns, try to find a numeric column
            value_cols = [col for col in mentioned_cols if col in self.df.select_dtypes(include=['number']).columns.tolist()]
            if not value_cols and not time_col:
                return {
                    "text": "I couldn't identify suitable columns for trend analysis from your query. Please specify a time/date column and a numeric column to analyze trends."
                }
        
        # If no time column identified yet, use the first available time column
        if not time_col and time_cols:
            time_col = time_cols[0]
        
        # If still no time column, look for any ordered numeric column that could serve as a time index
        if not time_col:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                # Check if values are mostly sequential (could be a time index)
                if len(self.df) > 1 and self.df[col].is_monotonic_increasing:
                    time_col = col
                    break
        
        # If still no time column, return an error message
        if not time_col:
            return {
                "text": "I couldn't identify a suitable time or date column for trend analysis. Please specify a column that represents time or a sequential order."
            }
        
        # Get numeric columns for analysis
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != time_col]  # Exclude time column if it's numeric
        
        if not numeric_cols:
            return {
                "text": "Trend analysis requires numeric columns to analyze, but I couldn't find any suitable columns in this dataset."
            }
        
        # If value columns were identified from the query, use them; otherwise, use a subset of numeric columns
        if 'value_cols' in locals() and value_cols:
            cols_to_analyze = value_cols[:3]  # Limit to 3 for visualization
        else:
            # Use a subset of numeric columns (up to 3 for readability)
            cols_to_analyze = numeric_cols[:min(3, len(numeric_cols))]
        
        # Sort by time column
        df_sorted = self.df.sort_values(time_col)
        
        # Generate response text
        response_text = f"""
        ### Trend Analysis
        
        I've analyzed trends over time using **{time_col}** as the time dimension.
        Here are the key findings:
        """
        
        # Create trend visualization
        fig = go.Figure()
        
        for col in cols_to_analyze:
            # Add line trace
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[time_col],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                )
            )
            
            # Calculate trend (simple linear regression)
            if len(df_sorted) >= 2:
                try:
                    # Convert to numeric index for regression
                    x = np.arange(len(df_sorted))
                    y = df_sorted[col].values
                    
                    # Remove NaN values
                    mask = ~np.isnan(y)
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) >= 2:
                        slope, intercept = np.polyfit(x_clean, y_clean, 1)
                        
                        # Interpret trend
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        trend_strength = "strongly" if abs(slope) > 0.1 else "moderately" if abs(slope) > 0.01 else "slightly"
                        
                        response_text += f"""
                        
                        **{col}**:
                        - Overall trend: {trend_strength} {trend_direction}
                        - Change rate: {slope:.4f} units per step
                        """
                        
                        # Add trendline to plot
                        trend_y = intercept + slope * x
                        fig.add_trace(
                            go.Scatter(
                                x=df_sorted[time_col],
                                y=trend_y,
                                mode='lines',
                                line=dict(dash='dash'),
                                name=f'{col} Trend',
                                showlegend=True
                            )
                        )
                except Exception as e:
                    # Skip trendline if there's an error (e.g., non-numeric data)
                    pass
        
        # Update layout
        fig.update_layout(
            title=f"Trends over {time_col}",
            xaxis_title=time_col,
            yaxis_title="Value",
            template="plotly_white"
        )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _detect_outliers(self, query):
        """Detect outliers in the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                "text": "Outlier detection requires numeric columns, but I couldn't find any in this dataset."
            }
        
        # Check if specific columns are mentioned in the query
        mentioned_cols = [col for col in numeric_cols if col.lower() in query.lower()]
        
        # If specific columns are mentioned, use them; otherwise, use a subset of numeric columns
        if mentioned_cols:
            cols_to_analyze = mentioned_cols[:4]  # Limit to 4 for visualization
        else:
            # Use a subset of numeric columns (up to 4 for readability)
            cols_to_analyze = numeric_cols[:min(4, len(numeric_cols))]
        
        # Generate response text
        response_text = f"""
        ### Outlier Detection
        
        I've analyzed the following columns for outliers:
        """
        
        # Find outliers using IQR method
        outlier_results = {}
        
        for col in cols_to_analyze:
            # Calculate Q1, Q3, and IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_results[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage
            }
            
            response_text += f"""
            
            **{col}**:
            - Outlier count: {outlier_count} ({outlier_percentage:.2f}% of data)
            - Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]
            """
        
        # Create box plot visualization to show outliers
        fig = px.box(
            self.df,
            y=cols_to_analyze,
            title="Outlier Analysis",
            template="plotly_white"
        )
        
        return {
            "text": response_text,
            "visualization": fig,
            "viz_type": "plotly"
        }
    
    def _create_visualization(self, query):
        """Create a visualization based on the query"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to visualize. Please upload a dataset first."
            }
        
        # Determine visualization type from the query
        viz_type = None
        if any(term in query.lower() for term in ["bar", "column"]):
            viz_type = "bar"
        elif any(term in query.lower() for term in ["line", "trend", "time series"]):
            viz_type = "line"
        elif any(term in query.lower() for term in ["scatter", "relationship", "correlation"]):
            viz_type = "scatter"
        elif any(term in query.lower() for term in ["histogram", "distribution"]):
            viz_type = "histogram"
        elif any(term in query.lower() for term in ["pie", "proportion", "percentage"]):
            viz_type = "pie"
        elif any(term in query.lower() for term in ["box", "boxplot", "range"]):
            viz_type = "box"
        elif any(term in query.lower() for term in ["heatmap", "correlation matrix"]):
            viz_type = "heatmap"
        else:
            # Default to bar chart
            viz_type = "bar"
        
        # Extract columns from query
        all_cols = self.df.columns.tolist()
        mentioned_cols = [col for col in all_cols if col.lower() in query.lower()]
        
        # Get numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create the visualization based on type
        if viz_type == "bar":
            return self._create_bar_visualization(query, mentioned_cols, numeric_cols, categorical_cols)
        elif viz_type == "line":
            return self._create_line_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "scatter":
            return self._create_scatter_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "histogram":
            return self._create_histogram_visualization(query, mentioned_cols, numeric_cols)
        elif viz_type == "pie":
            return self._create_pie_visualization(query, mentioned_cols, categorical_cols)
        elif viz_type == "box":
            return self._create_box_visualization(query, mentioned_cols, numeric_cols, categorical_cols)
        elif viz_type == "heatmap":
            return self._create_heatmap_visualization(query, mentioned_cols, numeric_cols)
    
    def _create_bar_visualization(self, query, mentioned_cols, numeric_cols, categorical_cols):
        """Create a bar chart visualization"""
        # Need at least one categorical and one numeric column for a bar chart
        if not categorical_cols or not numeric_cols:
            return {
                "text": "I couldn't create a bar chart because I need at least one categorical column and one numeric column."
            }
        
        # Select columns for the bar chart
        x_col = None
        y_col = None
        
        # If columns are mentioned, try to use them
        if mentioned_cols:
            # Look for a categorical column first
            for col in mentioned_cols:
                if col in categorical_cols:
                    x_col = col
                    break
            
            # Then look for a numeric column
            for col in mentioned_cols:
                if col in numeric_cols and col != x_col:
                    y_col = col
                    break
        
        # If no suitable columns found from mentions, use defaults
        if not x_col:
            x_col = categorical_cols[0]
        
        if not y_col:
            # Default to count if no specific numeric column
            y_col = "count"
        
        # Create the bar chart
        if y_col == "count":
            # Count by category
            df_grouped = self.df.groupby(x_col).size().reset_index(name='count')
            
            # Sort by count for better visualization
            df_grouped = df_grouped.sort_values('count', ascending=False)
            
            # Limit to top 15 categories if there are too many
            if len(df_grouped) > 15:
                df_grouped = df_grouped.head(15)
            
            fig = px.bar(
                df_grouped,
                x=x_col,
                y='count',
                title=f"Count by {x_col}",
                template="plotly_white"
            )
            
            response_text = f"""
            ### Bar Chart: Count by {x_col}
            
            I've created a bar chart showing the count of items in each {x_col} category.
            
            **Key observations:**
            - Most common {x_col}: {import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import json
import time

class AIAssistant:
    """AI-powered data analysis assistant"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Store chat history in session state if not already there
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        # Store analysis cache
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
    
    def render_interface(self):
        """Render the AI assistant interface"""
        st.header("Data Assistant")
        
        # Features section
        with st.expander("✨ What can the Data Assistant do?", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Ask questions about your data:**
                - 📊 "Summarize this dataset"
                - 🔍 "Find correlations between columns X and Y"
                - 📈 "What insights can you provide from this data?"
                - 🧮 "Calculate the average of column Z"
                - 📉 "Show me trends in this data"
                """)
            
            with col2:
                st.markdown("""
                **Generate visualizations:**
                - 📊 "Create a bar chart of column X"
                - 📈 "Show me a scatter plot of X vs Y"
                - 🔄 "Visualize the distribution of column Z"
                - 🧩 "Create a correlation heatmap"
                - 📉 "Make a line chart showing trends in column X"
                """)
        
        # Chat interface
        st.subheader("Ask me about your data")
        
        # Display chat history
        self._render_chat_history()
        
        # Input for new message
        user_input = st.text_area("Type your question here:", key="user_query", height=100)
        
        # Suggested questions
        st.markdown("**⚡ Suggested questions:**")
        suggestion_cols = st.columns(3)
        
        suggestions = [
            "Summarize this dataset",
            "What are the main insights?",
            "Show correlations between columns",
            "Identify outliers in the data",
            "Visualize the distribution of numeric columns",
            "What trends can you identify?"
        ]
        
        # Add column-specific suggestions if we have data
        if self.df is not None:
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if num_cols and len(num_cols) >= 2:
                suggestions.append(f"Show a scatter plot of {num_cols[0]} vs {num_cols[1]}")
            
            if cat_cols and num_cols:
                suggestions.append(f"Compare {num_cols[0]} across different {cat_cols[0]} values")
        
        # Display suggestion buttons
        for i, suggestion in enumerate(suggestions):
            col_idx = i % 3
            with suggestion_cols[col_idx]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    # Use the suggestion as input
                    user_input = suggestion
        
        # Submit button
        if st.button("Submit", use_container_width=True, type="primary") or user_input:
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Process the query and generate a response
                with st.spinner("Thinking..."):
                    response = self._generate_response(user_input)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["text"],
                    "visualization": response.get("visualization", None),
                    "viz_type": response.get("viz_type", None)
                })
                
                # Clear input
                st.experimental_rerun()
    
    def _render_chat_history(self):
        """Render the chat history"""
        if not st.session_state.chat_history:
            return
        
        # Create chat container
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <div class='chat-message-content'>
                        <p>{message["content"]}</p>
                    </div>
                    <div class='chat-message-avatar'>👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div class='chat-message-avatar'>🤖</div>
                    <div class='chat-message-content'>
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display visualization if available
                if "visualization" in message and message["visualization"] is not None:
                    if message["viz_type"] == "plotly":
                        st.plotly_chart(message["visualization"], use_container_width=True)
                    elif message["viz_type"] == "matplotlib":
                        st.pyplot(message["visualization"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _generate_response(self, query):
        """Generate a response to the user query"""
        query_lower = query.lower()
        
        # Check if we have a cached response for this query
        cache_key = query_lower.strip()
        if cache_key in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[cache_key]
        
        # Basic dataset summary
        if any(phrase in query_lower for phrase in ["summarize", "summary", "describe", "overview"]):
            response = self._summarize_dataset()
        
        # Correlation analysis
        elif "correlation" in query_lower or "correlate" in query_lower or "relationship between" in query_lower:
            response = self._analyze_correlations(query)
        
        # Distribution analysis
        elif any(phrase in query_lower for phrase in ["distribution", "histogram", "spread"]):
            response = self._analyze_distributions(query)
        
        # Statistical insights
        elif any(phrase in query_lower for phrase in ["statistics", "average", "mean", "median", "min", "max"]):
            response = self._provide_statistics(query)
        
        # Trend analysis
        elif any(phrase in query_lower for phrase in ["trend", "change over", "time series"]):
            response = self._analyze_trends(query)
        
        # Outlier detection
        elif any(phrase in query_lower for phrase in ["outlier", "anomaly", "unusual"]):
            response = self._detect_outliers(query)
        
        # Visualization requests
        elif any(phrase in query_lower for phrase in ["plot", "chart", "graph", "visualize", "visualization", "figure"]):
            response = self._create_visualization(query)
        
        # Compare groups
        elif any(phrase in query_lower for phrase in ["compare", "comparison", "versus", "vs"]):
            response = self._compare_groups(query)
        
        # General insights
        elif any(phrase in query_lower for phrase in ["insight", "pattern", "discover", "interesting", "important"]):
            response = self._provide_insights()
        
        # Default response for other queries
        else:
            response = self._default_response(query)
        
        # Cache the response
        st.session_state.analysis_cache[cache_key] = response
        
        return response
    
    def _summarize_dataset(self):
        """Provide a summary of the dataset"""
        if self.df is None or self.df.empty:
            return {
                "text": "I don't have any data to analyze. Please upload a dataset first."
            }
        
        # Basic dataset info
        rows, cols = self.df.shape
        missing_values = self.df.isna().sum().sum()
        missing_percentage = (missing_values / (rows * cols)) * 100
        
        # Column types
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Generate summary text
        summary = f"""
        ### Dataset Summary
        
        This dataset contains **{rows:,} rows** and **{cols} columns**. 
        
        **Data Quality:**
        - Missing values: {missing_values:,} ({missing_percentage:.2f}% of all cells)
        - Duplicate rows: {self.df.duplicated().sum():,}
        
        **Column Types:**
        - Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{"..." if len(numeric_cols) > 5 else ""}
        - Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{"..." if len(categorical_cols) > 5 else ""}
        - Date columns ({len(date_cols)}): {', '.join(date_cols)}
        
        **Key Statistics:**
        """
        
        # Add statistics for key numeric columns (up to 3)
        for col in numeric_cols[:3]:
            col_stats = self.df[col].describe()
            summary += f"""
            - **{col}**: 
                - Mean: {col_stats['mean']:.2f}
                - Median: {col_stats['50%']:.2f}
                - Min: {col_stats['min']:.2f}
                - Max: {col_stats['max']:.2f}
            """
        
        # Add info for key categorical columns (up to 3)
        for col in categorical_cols[:3]:
            value_counts = self.df[col].value_counts()
            top_value = value_counts.index[0] if not value_counts.empty else "N/A"
            top_count = value_counts.iloc[0] if not value_counts.empty else 0
            unique_count = self.df[col