import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from modules.utils import fix_arrow_dtypes
import plotly.express as px

class DataExplorer:
    """Class for exploring and examining data"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
    
    def render_interface(self):
        """Render the data explorer interface"""
        st.header("Data Explorer")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin exploring.")
            return
        
        # Create tabs for different data exploration views
        tabs = st.tabs([
            "Overview", 
            "Column Explorer", 
            "Enhanced Analysis", 
            "Data Quality"
        ])
        
        # Overview Tab - Basic dataset information
        with tabs[0]:
            # Display dataset info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", self.df.shape[0])
            with col2:
                st.metric("Columns", self.df.shape[1])
            with col3:
                st.metric("Missing Values", self.df.isna().sum().sum())
            with col4:
                st.metric("Duplicates", self.df.duplicated().sum())
            
            # Data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(fix_arrow_dtypes(self.df.head(10)), use_container_width=True)
            
            # Column information
            with st.expander("Column Information"):
                col_info = pd.DataFrame({
                    'Type': self.df.dtypes,
                    'Non-Null Count': self.df.count(),
                    'Null Count': self.df.isna().sum(),
                    'Unique Values': [self.df[col].nunique() for col in self.df.columns]
                })
                st.dataframe(fix_arrow_dtypes(col_info), use_container_width=True)
            
            # Data statistics
            with st.expander("Data Statistics"):
                if self.df.select_dtypes(include=['number']).columns.tolist():
                    st.dataframe(self.df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistics")
        
        # Column Explorer Tab - Details about specific columns
        with tabs[1]:
            # Column details
            st.subheader("Column Explorer")
            selected_column = st.selectbox("Select a column for details:", self.df.columns)
            
            if selected_column:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Column:** {selected_column}")
                    st.write(f"**Type:** {self.df[selected_column].dtype}")
                    st.write(f"**Missing Values:** {self.df[selected_column].isna().sum()}")
                    st.write(f"**Unique Values:** {self.df[selected_column].nunique()}")
                    
                    if self.df[selected_column].dtype in ['int64', 'float64']:
                        st.write(f"**Min:** {self.df[selected_column].min()}")
                        st.write(f"**Max:** {self.df[selected_column].max()}")
                        st.write(f"**Mean:** {self.df[selected_column].mean()}")
                        st.write(f"**Median:** {self.df[selected_column].median()}")
                
                with col2:
                    # Show value counts or histogram depending on data type
                    if self.df[selected_column].dtype in ['int64', 'float64']:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(self.df[selected_column].dropna(), kde=True, ax=ax)
                        plt.title(f'Distribution of {selected_column}')
                        st.pyplot(fig)
                    else:
                        # Show top 10 value counts for non-numeric columns
                        value_counts = self.df[selected_column].value_counts().head(10)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        value_counts.plot(kind='bar', ax=ax)
                        plt.title(f'Top 10 Values in {selected_column}')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
        
        # Enhanced Analysis Tab - More detailed analysis with the new methods
        with tabs[2]:
            self.render_enhanced_column_analysis()
            
        # Data Quality Tab - Focus on data quality issues
        with tabs[3]:
            st.subheader("Data Quality Assessment")
            
            # Missing values analysis
            st.markdown("### Missing Values")
            
            # Calculate missing values by column
            missing_data = pd.DataFrame({
                'Column': self.df.columns,
                'Missing Values': self.df.isna().sum().values,
                'Percentage': (self.df.isna().sum().values / len(self.df) * 100).round(2)
            })
            
            # Sort by missing percentage
            missing_data = missing_data.sort_values('Percentage', ascending=False)
            
            # Display missing values
            st.dataframe(missing_data, use_container_width=True)
            
            # Visualize missing values
            if missing_data['Missing Values'].sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Column', y='Percentage', data=missing_data)
                plt.title('Missing Values by Column')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.success("No missing values found in the dataset!")
                
            # Duplicate rows analysis
            st.markdown("### Duplicate Rows")
            duplicate_count = self.df.duplicated().sum()
            
            if duplicate_count > 0:
                st.warning(f"Found {duplicate_count} duplicate rows ({duplicate_count/len(self.df)*100:.2f}% of data)")
                
                # Show sample of duplicates
                if st.checkbox("Show sample of duplicates"):
                    duplicates = self.df[self.df.duplicated(keep='first')]
                    st.dataframe(duplicates.head(5), use_container_width=True)
            else:
                st.success("No duplicate rows found in the dataset!")
                
    def render_enhanced_column_analysis(self):
        """Render enhanced column analysis section"""
        st.subheader("Enhanced Column Analysis")
        
        # Let user select column
        selected_column = st.selectbox(
            "Select column for enhanced analysis:",
            self.df.columns,
            key="enhanced_column"
        )
        
        if not selected_column:
            return
        
        # Get column data
        col_data = self.df[selected_column]
        
        # Create tabs for different analyses
        col_tabs = st.tabs(["Summary", "Distribution", "Patterns", "Export"])
        
        # Summary tab
        with col_tabs[0]:
            self._render_column_summary(selected_column, col_data)
        
        # Distribution tab
        with col_tabs[1]:
            self._render_column_distribution(selected_column, col_data)
        
        # Patterns tab
        with col_tabs[2]:
            self._render_column_patterns(selected_column, col_data)
        
        # Export tab
        with col_tabs[3]:
            self._render_column_export(selected_column, col_data)

    def _render_column_summary(self, column_name, col_data):
        """Render detailed column summary"""
        st.subheader(f"Summary for {column_name}")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Type", str(col_data.dtype))
        with col2:
            st.metric("Non-Null Values", int(col_data.count()))
        with col3:
            st.metric("Null Values", int(col_data.isna().sum()))
        
        # Type-specific stats
        if pd.api.types.is_numeric_dtype(col_data):
            self._render_numeric_summary(col_data)
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            self._render_categorical_summary(col_data)
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            self._render_datetime_summary(col_data)

    def _render_numeric_summary(self, col_data):
        """Render summary for numeric column"""
        # Display descriptive statistics
        stats = col_data.describe()
        
        # Additional metrics
        skewness = col_data.skew()
        kurtosis = col_data.kurtosis()
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
            st.metric("Standard Deviation", f"{stats['std']:.2f}")
        with col2:
            st.metric("Minimum", f"{stats['min']:.2f}")
            st.metric("Maximum", f"{stats['max']:.2f}")
        with col3:
            st.metric("Skewness", f"{skewness:.2f}")
            st.metric("Kurtosis", f"{kurtosis:.2f}")
        
        # Show percentiles
        st.subheader("Percentiles")
        percentiles = {
            "25%": stats["25%"],
            "50% (Median)": stats["50%"],
            "75%": stats["75%"],
            "90%": col_data.quantile(0.9),
            "95%": col_data.quantile(0.95),
            "99%": col_data.quantile(0.99)
        }
        
        # Display percentiles
        percentile_df = pd.DataFrame({
            "Percentile": percentiles.keys(),
            "Value": percentiles.values()
        })
        st.dataframe(percentile_df, use_container_width=True)

    def _render_categorical_summary(self, col_data):
        """Render summary for categorical column"""
        # Count values
        value_counts = col_data.value_counts(dropna=False)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", int(col_data.nunique()))
        with col2:
            st.metric("Most Common", str(value_counts.index[0]) if not value_counts.empty else "None")
        with col3:
            st.metric("Frequency", int(value_counts.iloc[0]) if not value_counts.empty else 0)
        
        # Top categories
        st.subheader("Top Categories")
        top_n = min(10, len(value_counts))
        
        # Calculate percentages
        top_df = pd.DataFrame({
            "Value": value_counts.index[:top_n],
            "Count": value_counts.values[:top_n],
            "Percentage": (value_counts.values[:top_n] / len(col_data) * 100).round(2)
        })
        
        st.dataframe(top_df, use_container_width=True)
        
        # Show all categories button
        if len(value_counts) > 10:
            with st.expander("Show all categories"):
                all_df = pd.DataFrame({
                    "Value": value_counts.index,
                    "Count": value_counts.values,
                    "Percentage": (value_counts.values / len(col_data) * 100).round(2)
                })
                st.dataframe(all_df, use_container_width=True)

    def _render_datetime_summary(self, col_data):
        """Render summary for datetime column"""
        # Basic datetime metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Earliest Date", col_data.min())
        with col2:
            st.metric("Latest Date", col_data.max())
        
        # Range
        date_range = (col_data.max() - col_data.min()).days
        st.metric("Date Range", f"{date_range} days")
        
        # Distribution by year, month, day
        st.subheader("Date Distribution")
        
        # Year distribution
        year_counts = col_data.dt.year.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        year_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        st.pyplot(fig)
        
        # Month distribution
        month_counts = col_data.dt.month.value_counts().sort_index()
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        month_counts.index = month_counts.index.map(month_names)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        month_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Count')
        st.pyplot(fig)

    def _render_column_distribution(self, column_name, col_data):
        """Render column distribution visualizations"""
        st.subheader(f"Distribution of {column_name}")
        
        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric columns
            viz_type = st.radio(
                "Visualization type:",
                ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"],
                horizontal=True
            )
            
            # Visualization parameters
            use_log = st.checkbox("Use log scale")
            
            if viz_type == "Histogram":
                bins = st.slider("Number of bins:", 5, 100, 30)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(col_data.dropna(), bins=bins, kde=True, ax=ax)
                
                if use_log:
                    plt.yscale('log')
                    plt.title(f'Histogram of {column_name} (Log Scale)')
                else:
                    plt.title(f'Histogram of {column_name}')
                    
                plt.xlabel(column_name)
                plt.ylabel('Count')
                st.pyplot(fig)
                
            elif viz_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=col_data.dropna(), ax=ax)
                plt.title(f'Box Plot of {column_name}')
                plt.xlabel(column_name)
                st.pyplot(fig)
                
            elif viz_type == "Violin Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(x=col_data.dropna(), ax=ax)
                plt.title(f'Violin Plot of {column_name}')
                plt.xlabel(column_name)
                st.pyplot(fig)
                
            elif viz_type == "KDE Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(col_data.dropna(), fill=True, ax=ax)
                plt.title(f'KDE Plot of {column_name}')
                plt.xlabel(column_name)
                st.pyplot(fig)
                
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            # For categorical columns
            viz_type = st.radio(
                "Visualization type:",
                ["Bar Chart", "Pie Chart", "Treemap"],
                horizontal=True
            )
            
            # Limit categories
            max_cats = st.slider("Max categories to show:", 5, 30, 10)
            
            # Get value counts
            value_counts = col_data.value_counts()
            
            # Limit categories if needed
            if len(value_counts) > max_cats:
                top_counts = value_counts.iloc[:max_cats]
                other_count = value_counts.iloc[max_cats:].sum()
                
                # Create new Series with "Other" category
                values = list(top_counts.values) + [other_count]
                labels = list(top_counts.index) + ["Other"]
            else:
                values = value_counts.values
                labels = value_counts.index
                
            if viz_type == "Bar Chart":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=labels, y=values, ax=ax)
                plt.title(f'Bar Chart of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Pie Chart":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title(f'Pie Chart of {column_name}')
                st.pyplot(fig)
                
            elif viz_type == "Treemap":
                # Use plotly for treemap
                fig = px.treemap(
                    names=labels,
                    values=values,
                    title=f'Treemap of {column_name}'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # For datetime columns
            time_unit = st.selectbox(
                "Aggregate by:",
                ["Year", "Month", "Day", "Hour"]
            )
            
            # Extract time component
            if time_unit == "Year":
                time_values = col_data.dt.year
            elif time_unit == "Month":
                time_values = col_data.dt.month
            elif time_unit == "Day":
                time_values = col_data.dt.day
            elif time_unit == "Hour":
                time_values = col_data.dt.hour
                
            # Count values
            value_counts = time_values.value_counts().sort_index()
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='bar', ax=ax)
            plt.title(f'Distribution by {time_unit}')
            plt.xlabel(time_unit)
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)

    def _render_column_patterns(self, column_name, col_data):
        """Render patterns and relationships with other columns"""
        st.subheader(f"Patterns and Relationships for {column_name}")
        
        # Get potential columns to compare with
        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric columns, find correlations
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col]) and col != column_name]
            
            if not numeric_cols:
                st.info("No other numeric columns found for correlation analysis.")
                return
                
            st.subheader("Correlations with Other Numeric Columns")
            
            # Calculate correlations
            correlations = [
                (col, self.df[column_name].corr(self.df[col]))
                for col in numeric_cols
            ]
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Display correlations
            corr_df = pd.DataFrame(correlations, columns=["Column", "Correlation"])
            st.dataframe(corr_df, use_container_width=True)
            
            # Plot top correlations
            top_corr_col = correlations[0][0] if correlations else None
            
            if top_corr_col:
                st.subheader(f"Scatter Plot: {column_name} vs {top_corr_col}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=self.df[column_name], y=self.df[top_corr_col], ax=ax)
                plt.title(f'{column_name} vs {top_corr_col} (corr: {correlations[0][1]:.2f})')
                plt.xlabel(column_name)
                plt.ylabel(top_corr_col)
                st.pyplot(fig)
        
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            # For categorical columns, find relationships with numeric columns
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not numeric_cols:
                st.info("No numeric columns found for relationship analysis.")
                return
                
            # Select numeric column to analyze
            num_col = st.selectbox("Select numeric column for relationship analysis:", numeric_cols)
            
            if num_col:
                st.subheader(f"Relationship Between {column_name} and {num_col}")
                
                # Calculate grouped statistics
                grouped = self.df.groupby(column_name)[num_col].agg(['mean', 'median', 'std', 'count']).reset_index()
                grouped = grouped.sort_values('mean', ascending=False)
                
                # Limit to top categories
                max_cats = st.slider("Maximum categories to display:", 5, 20, 10)
                if len(grouped) > max_cats:
                    grouped = grouped.head(max_cats)
                    
                # Display statistics
                st.dataframe(grouped, use_container_width=True)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=column_name, y='mean', data=grouped, ax=ax)
                plt.title(f'Mean {num_col} by {column_name}')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel(f'Mean {num_col}')
                plt.tight_layout()
                st.pyplot(fig)
                
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # For datetime columns, find trends over time
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not numeric_cols:
                st.info("No numeric columns found for trend analysis.")
                return
                
            # Select numeric column to analyze
            num_col = st.selectbox("Select numeric column for trend analysis:", numeric_cols)
            
            if num_col:
                st.subheader(f"Trend of {num_col} Over Time")
                
                # Select time aggregation
                time_unit = st.selectbox(
                    "Aggregate by time unit:",
                    ["Day", "Week", "Month", "Quarter", "Year"]
                )
                
                # Create time aggregation
                if time_unit == "Day":
                    self.df['time_agg'] = col_data.dt.date
                elif time_unit == "Week":
                    self.df['time_agg'] = col_data.dt.isocalendar().week
                elif time_unit == "Month":
                    self.df['time_agg'] = col_data.dt.to_period('M').astype(str)
                elif time_unit == "Quarter":
                    self.df['time_agg'] = col_data.dt.to_period('Q').astype(str)
                elif time_unit == "Year":
                    self.df['time_agg'] = col_data.dt.year
                    
                # Aggregate data
                aggregated = self.df.groupby('time_agg')[num_col].agg(['mean', 'min', 'max', 'count']).reset_index()
                
                # Display aggregated data
                st.dataframe(aggregated, use_container_width=True)
                
                # Create line chart
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(aggregated['time_agg'], aggregated['mean'], marker='o')
                plt.title(f'Trend of {num_col} by {time_unit}')
                plt.xlabel(time_unit)
                plt.ylabel(f'Mean {num_col}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Clean up temporary column
                if 'time_agg' in self.df.columns:
                    self.df.drop('time_agg', axis=1, inplace=True)

    def _render_column_export(self, column_name, col_data):
        """Render export options for column data"""
        st.subheader(f"Export Options for {column_name}")
        
        # Create summary table for export
        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric summary
            summary = col_data.describe().to_frame().reset_index()
            summary.columns = ["Statistic", "Value"]
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                "Statistic": ["skewness", "kurtosis", "null_count", "non_null_count"],
                "Value": [
                    col_data.skew(),
                    col_data.kurtosis(),
                    col_data.isna().sum(),
                    col_data.count()
                ]
            })
            
            summary = pd.concat([summary, additional_stats])
            
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            # Categorical summary
            value_counts = col_data.value_counts(dropna=False).head(20)
            summary = pd.DataFrame({
                "Value": value_counts.index,
                "Count": value_counts.values,
                "Percentage": (value_counts.values / len(col_data) * 100).round(2)
            })
            
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Datetime summary
            summary = pd.DataFrame({
                "Statistic": ["min_date", "max_date", "range_days", "null_count", "non_null_count"],
                "Value": [
                    col_data.min(),
                    col_data.max(),
                    (col_data.max() - col_data.min()).days,
                    col_data.isna().sum(),
                    col_data.count()
                ]
            })
        
        # Display summary
        st.dataframe(summary, use_container_width=True)
        
        # Export options
        export_format = st.radio(
            "Export format:",
            ["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        # Export button
        if st.button("Export Column Summary"):
            if export_format == "CSV":
                csv = summary.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{column_name}_summary.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                excel_data = BytesIO()
                summary.to_excel(excel_data, index=False)
                excel_data.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{column_name}_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = summary.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{column_name}_summary.json",
                    mime="application/json"
                )
        
        # Export raw column data option
        st.subheader("Export Raw Column Data")
        
        if st.button("Export Raw Column Data"):
            if export_format == "CSV":
                csv = col_data.to_csv(index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{column_name}_data.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                excel_data = BytesIO()
                col_data.to_excel(excel_data)
                excel_data.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{column_name}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = col_data.to_json(orient="index")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{column_name}_data.json",
                    mime="application/json"
                )
                
    def get_column_summary(self, column_name):
        """Get summary statistics for a specific column"""
        if column_name not in self.df.columns:
            return None
            
        summary = {
            "name": column_name,
            "type": str(self.df[column_name].dtype),
            "non_null_count": int(self.df[column_name].count()),
            "null_count": int(self.df[column_name].isna().sum()),
            "unique_values": int(self.df[column_name].nunique())
        }
        
        # Add numeric-specific stats if applicable
        if self.df[column_name].dtype in ['int64', 'float64']:
            summary.update({
                "min": float(self.df[column_name].min()),
                "max": float(self.df[column_name].max()),
                "mean": float(self.df[column_name].mean()),
                "median": float(self.df[column_name].median()),
                "std": float(self.df[column_name].std())
            })
            
        # Add categorical-specific stats if applicable
        if self.df[column_name].dtype == 'object' or self.df[column_name].dtype.name == 'category':
            top_value = self.df[column_name].value_counts().index[0] if not self.df[column_name].value_counts().empty else None
            top_count = int(self.df[column_name].value_counts().iloc[0]) if not self.df[column_name].value_counts().empty else 0
            
            summary.update({
                "top_value": top_value,
                "top_count": top_count
            })
            
        return summary
        
    def get_dataset_summary(self):
        """Get summary statistics for the entire dataset"""
        if self.df is None or self.df.empty:
            return None
            
        summary = {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "missing_values": int(self.df.isna().sum().sum()),
            "missing_percentage": float((self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100),
            "duplicates": int(self.df.duplicated().sum()),
            "memory_usage": int(self.df.memory_usage(deep=True).sum())
        }
        
        # Count column types
        summary["numeric_columns"] = len(self.df.select_dtypes(include=['number']).columns)
        summary["categorical_columns"] = len(self.df.select_dtypes(include=['object', 'category']).columns)
        summary["datetime_columns"] = len(self.df.select_dtypes(include=['datetime']).columns)
        
        return summary