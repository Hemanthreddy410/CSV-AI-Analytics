import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import time
import uuid
import base64
from datetime import datetime
from scipy import stats
import re

class AIAssistant:
    """AI-powered data analysis chat assistant"""
    
    def __init__(self, df=None):
        """Initialize with optional dataframe"""
        self.df = df
        self.session_id = str(uuid.uuid4())  # Generate a unique session ID
        
        # Initialize session state variables if they don't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
            
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
    
    def render_interface(self):
        """Render the chat assistant interface"""
        st.header("Smart Data Chat Assistant")
        
        # Render the chat interface
        self._render_chat_interface()
    
    # Callback for suggestion buttons
    def _set_suggestion(self, text):
        st.session_state.user_input = text
        
    # Callback for submitting the chat input
    def _submit_chat(self):
        # Only process if there's input text
        if st.session_state.user_input.strip():
            # Get the input
            input_to_process = st.session_state.user_input
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": input_to_process
            })
            
            # Process the query and generate a response
            response = self._generate_response(input_to_process)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["text"],
                "visualization": response.get("visualization", None),
                "viz_type": response.get("viz_type", None)
            })
            
            # Clear the input field
            st.session_state.user_input = ""
    
    def _render_chat_interface(self):
        """Render the chat interface for data analysis"""
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
        
        # Create suggestions
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
                st.button(
                    suggestion, 
                    key=f"suggestion_{i}_{self.session_id}",
                    on_click=self._set_suggestion,
                    args=(suggestion,),
                    use_container_width=True
                )
        
        # Input area
        st.text_area(
            "Type your question here:", 
            height=100,
            key="user_input"
        )
        
        # Submit button - directly calls the callback
        submit_button = st.button(
            "Submit", 
            on_click=self._submit_chat,
            key=f"submit_{self.session_id}",
            use_container_width=True,
            type="primary"
        )
    
    def _render_chat_history(self):
        """Render the chat history"""
        if not st.session_state.chat_history:
            return
        
        # Add some custom CSS for chat bubbles
        st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }
        .chat-message {
            display: flex;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 10px;
            position: relative;
        }
        .user-message {
            background-color: #e6f7ff;
            margin-left: 20px;
            margin-right: 60px;
        }
        .assistant-message {
            background-color: #f0f2f6;
            margin-right: 20px;
            margin-left: 60px;
        }
        .chat-message-content {
            flex: 1;
            padding: 0.5rem;
        }
        .chat-message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .user-message .chat-message-avatar {
            background-color: #2b8eff;
            color: white;
        }
        .assistant-message .chat-message-avatar {
            background-color: #565656;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create chat container
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for idx, message in enumerate(st.session_state.chat_history):
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
                        st.plotly_chart(
                            message["visualization"], 
                            use_container_width=True, 
                            key=f"plot_{idx}_{self.session_id}"
                        )
                    elif message["viz_type"] == "matplotlib":
                        st.pyplot(
                            message["visualization"], 
                            key=f"mpl_{idx}_{self.session_id}"
                        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _generate_response(self, query):
        """Generate a response to the user query"""
        query_lower = query.lower()
        
        # Check if we have a cached response for this query
        cache_key = query_lower.strip()
        if cache_key in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[cache_key]
        
        # If no dataframe is loaded
        if self.df is None or self.df.empty:
            response = {
                "text": "I don't have any data to analyze yet. Please upload a dataset first."
            }
            # Cache the response
            st.session_state.analysis_cache[cache_key] = response
            return response
        
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
            unique_count = self.df[col].nunique()
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
                # Simple KDE calculation for display
                from scipy import stats
                if not self.df[col].dropna().empty:
                    kde_x = np.linspace(
                        self.df[col].min(), 
                        self.df[col].max(), 
                        100
                    )
                    kde = stats.gaussian_kde(self.df[col].dropna())
                    kde_y = kde(kde_x)
                    
                    # Scale KDE to match histogram height
                    hist_values, bin_edges = np.histogram(
                        self.df[col].dropna(), 
                        bins=20, 
                        density=True
                    )
                    scale_factor = 1
                    kde_y = kde_y * scale_factor
                    
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
        
        # Logic for creating basic visualization
        if viz_type == "bar":
            # Need at least one categorical and one numeric column
            if not categorical_cols:
                return {"text": "I need categorical columns to create a bar chart. None were found in the dataset."}
                
            # Select columns
            x_col = categorical_cols[0]
            if numeric_cols:
                y_col = numeric_cols[0]
                
                # Create simple bar chart
                df_grouped = self.df.groupby(x_col)[y_col].mean().reset_index()
                df_grouped = df_grouped.sort_values(y_col, ascending=False).head(10)
                
                fig = px.bar(
                    df_grouped,
                    x=x_col,
                    y=y_col,
                    title=f"Average {y_col} by {x_col}",
                    template="plotly_white"
                )
                
                return {
                    "text": f"Here's a bar chart showing average {y_col} by {x_col}:",
                    "visualization": fig,
                    "viz_type": "plotly"
                }
            else:
                # Create count-based bar chart
                value_counts = self.df[x_col].value_counts().head(10).reset_index()
                value_counts.columns = [x_col, 'count']
                
                fig = px.bar(
                    value_counts,
                    x=x_col,
                    y='count',
                    title=f"Count by {x_col}",
                    template="plotly_white"
                )
                
                return {
                    "text": f"Here's a bar chart showing counts for {x_col}:",
                    "visualization": fig,
                    "viz_type": "plotly"
                }
                
        elif viz_type == "scatter":
            # Need at least two numeric columns
            if len(numeric_cols) < 2:
                return {"text": "I need at least two numeric columns to create a scatter plot. Not enough were found in the dataset."}
                
            # Create simple scatter plot
            fig = px.scatter(
                self.df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                template="plotly_white"
            )
            
            return {
                "text": f"Here's a scatter plot showing the relationship between {numeric_cols[0]} and {numeric_cols[1]}:",
                "visualization": fig,
                "viz_type": "plotly"
            }
        
        elif viz_type == "histogram":
            # Need at least one numeric column
            if not numeric_cols:
                return {"text": "I need numeric columns to create a histogram. None were found in the dataset."}
                
            # Create histogram
            fig = px.histogram(
                self.df,
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                template="plotly_white"
            )
            
            return {
                "text": f"Here's a histogram showing the distribution of {numeric_cols[0]}:",
                "visualization": fig,
                "viz_type": "plotly"
            }
            
        # Default response with a general visualization  
        return {
            "text": "I'm not sure what specific visualization you're looking for. Here's a summary of the data types in your dataset:",
            "visualization": self._summarize_dataset()["visualization"],
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
        
        # Choose the first categorical column and numeric column
        group_col = cat_cols[0]
        metric_col = numeric_cols[0]
        
        # Generate comparison
        response_text = f"""
        ### Group Comparison: by {group_col}
        
        I've compared different groups based on **{group_col}**, analyzing the metric **{metric_col}**.
        """
        
        # Calculate group statistics
        stats = self.df.groupby(group_col)[metric_col].agg(['mean', 'median', 'std']).reset_index()
        stats = stats.sort_values('mean', ascending=False).head(10)  # Top 10 groups
        
        # Create visualization
        fig = px.bar(
            stats,
            x=group_col,
            y='mean',
            error_y=stats['std'],
            title=f"Comparison of {metric_col} by {group_col}",
            template="plotly_white"
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
        
        # Return summary as insights
        return self._summarize_dataset()
    
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