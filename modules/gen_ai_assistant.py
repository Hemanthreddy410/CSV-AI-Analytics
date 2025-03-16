import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import traceback
import base64

class GenAIAssistant:
    """Streamlined Generative AI powered data analysis assistant with Claude API integration"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Initialize session state variables
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'claude_api_key' not in st.session_state:
            st.session_state.claude_api_key = ""
            
        if 'claude_model' not in st.session_state:
            st.session_state.claude_model = "claude-3-sonnet-20240229"
    
    def render_interface(self):
        """Render the streamlined GenAI assistant interface"""
        st.header("💡 Claude Data Chat Assistant")
        
        # Two columns layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self._render_chat_interface()
        
        with col2:
            self._render_compact_settings()
    
    def _render_compact_settings(self):
        """Render a compact settings panel"""
        st.subheader("Settings")
        
        # API Key input
        api_key = st.text_input(
            "Claude API Key", 
            value=st.session_state.claude_api_key,
            type="password",
            key="claude_api_key_input",
            help="Enter your Anthropic Claude API key"
        )
        
        if api_key != st.session_state.claude_api_key:
            st.session_state.claude_api_key = api_key
            if api_key:
                st.success("API key saved!")
        
        # Model selection
        model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(st.session_state.claude_model) if st.session_state.claude_model in model_options else 1,
            key="model_select"
        )
        
        if model != st.session_state.claude_model:
            st.session_state.claude_model = model
        
        # Clear chat history button
        if st.button("Clear Chat History", key="clear_history_button"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Dataset info
        if self.df is not None and not self.df.empty:
            st.subheader("Dataset Info")
            st.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
            
            # Show column types
            st.write("Column Types:")
            col_types = pd.DataFrame({
                'Column': self.df.columns,
                'Type': self.df.dtypes.astype(str)
            })
            st.dataframe(col_types, hide_index=True)
    
    def _render_chat_interface(self):
        """Render enhanced chat interface with visualization support"""
        st.subheader("Chat with your Data")
        
        # Check if API key is set
        if not st.session_state.claude_api_key:
            st.warning("Please set your Claude API key in the Settings panel to use this feature.")
            return
        
        # Check if data is available
        if self.df is None or self.df.empty:
            st.warning("Please upload a dataset first to use this feature.")
            return
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display visualizations for assistant messages if they exist
                if message["role"] == "assistant" and "visualization_code" in message:
                    try:
                        st.write("---")  # Add a separator before visualization
                        
                        # Create a container for the visualization
                        viz_container = st.container()
                        
                        with viz_container:
                            # Create a namespace for executing the code
                            local_namespace = {
                                'self': self,
                                'df': self.df,
                                'pd': pd,
                                'np': np,
                                'plt': plt,
                                'px': px,
                                'go': go,
                                'sns': sns,
                                'make_subplots': make_subplots,
                                'st': st
                            }
                            
                            # Add explicit display code if not already present
                            viz_code = message["visualization_code"]
                            
                            # Make sure the code has explicit st.plotly_chart or st.pyplot calls
                            if "st.plotly_chart" not in viz_code and "st.pyplot" not in viz_code:
                                # Check if it's using plotly
                                if "px." in viz_code or "go." in viz_code:
                                    viz_code += "\n\n# Explicitly display the figure\nif 'fig' in locals():\n    st.plotly_chart(fig, use_container_width=True)"
                                # Check if it's using matplotlib/seaborn
                                elif "plt." in viz_code or "sns." in viz_code:
                                    viz_code += "\n\n# Explicitly display the figure\nif 'plt' in locals():\n    st.pyplot(plt.gcf())"
                            
                            # Execute the modified code
                            exec(viz_code, globals(), local_namespace)
                            
                            # Additional fallback to ensure plot is displayed
                            if 'fig' in local_namespace and 'st.plotly_chart' not in viz_code:
                                if hasattr(local_namespace['fig'], 'update_layout'):  # It's a plotly figure
                                    st.plotly_chart(local_namespace['fig'], use_container_width=True)
                            
                            # Add download options
                            col1, col2 = st.columns(2)
                            
                            # Download code
                            with col1:
                                st.download_button(
                                    "Download Code",
                                    viz_code,
                                    file_name=f"visualization_code_{i}.py",
                                    mime="text/plain",
                                    key=f"download_viz_code_{i}"
                                )
                            
                            # Download visualization as PNG if available
                            with col2:
                                if "fig" in local_namespace:
                                    if hasattr(local_namespace["fig"], "to_image"):
                                        # Plotly figure
                                        img_bytes = local_namespace["fig"].to_image(format="png")
                                        st.download_button(
                                            "Download PNG",
                                            img_bytes,
                                            file_name=f"visualization_{i}.png",
                                            mime="image/png",
                                            key=f"download_viz_png_{i}"
                                        )
                                    elif "plt" in local_namespace:
                                        # Matplotlib figure - save to buffer
                                        buf = BytesIO()
                                        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                        buf.seek(0)
                                        st.download_button(
                                            "Download PNG",
                                            buf.getvalue(),
                                            file_name=f"visualization_{i}.png",
                                            mime="image/png",
                                            key=f"download_viz_png_{i}"
                                        )
                                        # Close the figure to prevent display duplicates
                                        plt.close()
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        st.code(traceback.format_exc(), language="python")
                        
                        # Show the raw code on error to help with debugging
                        with st.expander("View Visualization Code"):
                            st.code(message["visualization_code"], language="python")
        
        # Chat input
        user_input = st.chat_input("Ask about your data or request visualizations")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display the latest user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Process and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    # Detect if this is a visualization request
                    is_viz_request = self._is_visualization_request(user_input)
                    
                    if is_viz_request:
                        # Handle as visualization request
                        response, viz_code = self._get_visualization_response(user_input)
                    else:
                        # Handle as regular question
                        response, viz_code = self._get_claude_response(user_input)
                    
                    # Display text response
                    st.markdown(response)
                    
                    # If visualization code was generated, execute and display it
                    if viz_code:
                        try:
                            st.write("---")  # Add a separator before visualization
                            
                            # Create a container for the visualization 
                            viz_container = st.container()
                            
                            with viz_container:
                                # Create namespace for executing the code
                                local_namespace = {
                                    'self': self,
                                    'df': self.df,
                                    'pd': pd,
                                    'np': np, 
                                    'plt': plt,
                                    'px': px,
                                    'go': go,
                                    'sns': sns,
                                    'make_subplots': make_subplots,
                                    'st': st
                                }
                                
                                # Add explicit display code if not already present
                                if "st.plotly_chart" not in viz_code and "st.pyplot" not in viz_code:
                                    # Check if it's using plotly
                                    if "px." in viz_code or "go." in viz_code:
                                        viz_code += "\n\n# Explicitly display the figure\nif 'fig' in locals():\n    st.plotly_chart(fig, use_container_width=True)"
                                    # Check if it's using matplotlib/seaborn
                                    elif "plt." in viz_code or "sns." in viz_code:
                                        viz_code += "\n\n# Explicitly display the figure\nif 'plt' in locals():\n    st.pyplot(plt.gcf())"
                                
                                # Show the visualization code in an expander for debugging
                                with st.expander("View Visualization Code"):
                                    st.code(viz_code, language="python")
                                
                                # Execute the modified code
                                exec(viz_code, globals(), local_namespace)
                                
                                # Additional fallback to ensure plot is displayed
                                if 'fig' in local_namespace and 'st.plotly_chart' not in viz_code:
                                    if hasattr(local_namespace['fig'], 'update_layout'):  # It's a plotly figure
                                        st.plotly_chart(local_namespace['fig'], use_container_width=True)
                                
                                # Add download options
                                col1, col2 = st.columns(2)
                                
                                # Download code
                                with col1:
                                    st.download_button(
                                        "Download Code",
                                        viz_code,
                                        file_name="visualization_code.py",
                                        mime="text/plain",
                                        key="download_latest_viz_code"
                                    )
                                
                                # Download as image (if matplotlib or plotly)
                                with col2:
                                    if "fig" in local_namespace:
                                        if hasattr(local_namespace["fig"], "to_image"):
                                            # Plotly figure
                                            img_bytes = local_namespace["fig"].to_image(format="png")
                                            st.download_button(
                                                "Download PNG",
                                                img_bytes,
                                                file_name="visualization.png",
                                                mime="image/png",
                                                key="download_latest_viz_png"
                                            )
                                        elif "plt" in local_namespace:
                                            # Matplotlib figure - save to buffer
                                            buf = BytesIO()
                                            plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                            buf.seek(0)
                                            st.download_button(
                                                "Download PNG",
                                                buf.getvalue(),
                                                file_name="visualization.png",
                                                mime="image/png",
                                                key="download_latest_viz_png"
                                            )
                                            # Close the figure to prevent display duplicates
                                            plt.close()
                        except Exception as e:
                            st.error(f"Error displaying visualization: {str(e)}")
                            st.code(traceback.format_exc(), language="python")
            
            # Add AI response to chat history
            response_dict = {"role": "assistant", "content": response}
            if viz_code:
                response_dict["visualization_code"] = viz_code
            
            st.session_state.chat_history.append(response_dict)
    
    def _is_visualization_request(self, query):
        """Detect if user is requesting a visualization"""
        visualization_keywords = [
            "visualize", "visualization", "visualise", "visualisation",
            "plot", "chart", "graph", "figure", "diagram", "display",
            "show", "draw", "create", "generate", "make", "image",
            # Specific visualization types
            "histogram", "bar chart", "scatter plot", "line chart", "pie chart",
            "heatmap", "correlation", "box plot", "violin plot", "density plot",
            "area chart", "bubble chart", "radar chart", "contour plot",
            "3d plot", "surface plot", "map", "geo map", "choropleth",
            "network graph", "tree map", "sunburst", "sankey", "parallel coordinates"
        ]
        
        # Check if query contains visualization keywords
        query_lower = query.lower()
        for keyword in visualization_keywords:
            if keyword in query_lower:
                return True
                
        return False
    
    def _get_visualization_response(self, query):
        """Process a visualization request and generate appropriate visualization code"""
        try:
            # Get dataframe info for context
            df_info = self._get_dataframe_info()
            
            # Get column types for context
            column_types = self._get_column_types_info()
            
            # Create system prompt
            system_prompt = f"""You are an expert data visualization assistant specialized in creating informative and attractive visualizations using Python.

The user has uploaded a dataset with the following characteristics:
{df_info}

Column types information:
{column_types}

They have requested a visualization. Your task is to:

1. Identify which type of visualization would best suit their request
2. Generate clear, well-documented Python code that creates this visualization
3. Provide a brief explanation of what the visualization shows and why it's appropriate

IMPORTANT INSTRUCTIONS FOR CODE GENERATION:
- Use Plotly Express (px) or Plotly Graph Objects (go) for interactive visualizations when appropriate
- Use seaborn (sns) or matplotlib (plt) for statistical visualizations when appropriate
- Store the final plot object as 'fig' (important for download functionality)
- Add proper titles, labels, and colors to make the visualization informative and attractive
- Access the dataframe as 'self.df'
- Make the code standalone and ready to execute in a Streamlit environment
- ALWAYS include explicit display commands like st.plotly_chart(fig, use_container_width=True) or st.pyplot() at the end of your code
- Include appropriate code comments to explain key steps
- Always check column data types before using them - add explicit error handling

IMPORTANT DATA HANDLING INSTRUCTIONS:
- CAREFULLY CHECK the data type of each column before using it in a visualization
- DO NOT use categorical columns (strings/objects) directly in numeric operations
- For categorical columns in numeric contexts, use encoding (like get_dummies) or mapping
- Add proper error handling for all operations that could fail with mixed data types
- If categorical data is detected, ensure it's handled appropriately for the visualization type
- Use df.select_dtypes() to filter columns by type when necessary
- For scatter plots, ensure both axes use numeric data
- For categorical plots, ensure categorical columns are used appropriately
- For all visualizations involving numeric operations, verify columns are numeric first

Your response should have two parts:
1. A brief explanation of the visualization (2-4 sentences)
2. The complete Python code that generates the visualization

The code will be executed automatically, so ensure it is correct and ready to run."""
            
            # Make API call
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": st.session_state.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Create messages array with context
            messages = []
            
            # Add last few messages for context (up to 3 for visualization requests)
            if len(st.session_state.chat_history) > 0:
                for message in st.session_state.chat_history[-3:]:
                    role = "user" if message["role"] == "user" else "assistant"
                    messages.append({"role": role, "content": message["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            payload = {
                "model": st.session_state.claude_model,
                "messages": messages,
                "system": system_prompt,
                "max_tokens": 3000
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                text_response = response.json()["content"][0]["text"]
                
                # Extract code if present
                code_blocks = re.findall(r'```python\n(.*?)```', text_response, re.DOTALL)
                
                if code_blocks:
                    # Extract the first code block for execution
                    viz_code = code_blocks[0]
                    
                    # Ensure there are display commands in the code
                    if "st.plotly_chart" not in viz_code and "st.pyplot" not in viz_code:
                        # Check if it's using plotly
                        if "px." in viz_code or "go." in viz_code:
                            viz_code += "\n\n# Explicitly display the figure\nif 'fig' in locals():\n    st.plotly_chart(fig, use_container_width=True)"
                        # Check if it's using matplotlib/seaborn
                        elif "plt." in viz_code or "sns." in viz_code:
                            viz_code += "\n\n# Explicitly display the figure\nif 'plt' in locals():\n    st.pyplot(plt.gcf())"
                    
                    # Remove code blocks from the text response
                    clean_response = re.sub(r'```python\n.*?```', '', text_response, flags=re.DOTALL)
                    clean_response = re.sub(r'```\n', '', clean_response)
                    clean_response = clean_response.replace('```', '')
                    
                    # Add visualization success note
                    clean_response += "\n\n*The visualization has been automatically generated below.*"
                    
                    return clean_response.strip(), viz_code
                else:
                    # If no code block was found but it's a visualization request,
                    # generate a default visualization
                    return text_response, self._generate_fallback_visualization()
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}", None
                
        except Exception as e:
            return f"Sorry, an error occurred while generating the visualization: {str(e)}", None
    
    def _get_column_types_info(self):
        """Get detailed information about column types for visualization context"""
        try:
            # Identify column types
            numeric_cols = list(self.df.select_dtypes(include=['number']).columns)
            categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = list(self.df.select_dtypes(include=['datetime']).columns)
            bool_cols = list(self.df.select_dtypes(include=['bool']).columns)
            
            # Create information string
            info = []
            info.append("COLUMN TYPES:")
            
            if numeric_cols:
                info.append(f"- Numeric columns: {', '.join(numeric_cols)}")
            else:
                info.append("- No numeric columns detected")
                
            if categorical_cols:
                info.append(f"- Categorical columns: {', '.join(categorical_cols)}")
                # Add sample values for categorical columns
                for col in categorical_cols[:3]:  # Limit to first 3 columns
                    unique_values = self.df[col].unique()
                    if len(unique_values) <= 10:  # Only include if small number of unique values
                        values_str = ', '.join([f"'{str(val)}'" for val in unique_values[:5]])
                        if len(unique_values) > 5:
                            values_str += f", ... ({len(unique_values)} unique values)"
                        info.append(f"  - {col} values: {values_str}")
                    else:
                        info.append(f"  - {col}: {len(unique_values)} unique values")
            else:
                info.append("- No categorical columns detected")
                
            if datetime_cols:
                info.append(f"- Datetime columns: {', '.join(datetime_cols)}")
            else:
                info.append("- No datetime columns detected")
                
            if bool_cols:
                info.append(f"- Boolean columns: {', '.join(bool_cols)}")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting column types: {str(e)}"
    
    def _generate_fallback_visualization(self):
        """Generate a safe fallback visualization that handles categorical data properly"""
        try:
            # Get column types
            numeric_cols = list(self.df.select_dtypes(include=['number']).columns)
            categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
            
            # Choose visualization based on available data
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                # Create a box plot of numeric data grouped by categorical
                code = f"""
import plotly.express as px
import pandas as pd

# Create a box plot of numeric data grouped by categorical
# This visualization safely handles both numeric and categorical data
try:
    numeric_col = "{numeric_cols[0]}"
    categorical_col = "{categorical_cols[0]}"
    
    # Create figure - box plots work well with mixed data types
    fig = px.box(
        self.df, 
        x=categorical_col, 
        y=numeric_col, 
        title=f"Distribution of {{numeric_col}} by {{categorical_col}}",
        points="all"  # Show all points for small datasets
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=categorical_col,
        yaxis_title=numeric_col,
        height=500
    )
    
    # Display the figure - IMPORTANT for visualization to appear
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error creating visualization: {{str(e)}}")
    
    # Fallback to safe table view
    st.write("### Dataset Preview")
    st.write("Showing a sample of the data instead:")
    st.dataframe(self.df.head(10))
"""
            elif len(numeric_cols) >= 2:
                # Create a scatter plot of two numeric columns
                code = f"""
import plotly.express as px
import pandas as pd

# Create a scatter plot of two numeric columns
try:
    # Select two numeric columns
    x_col = "{numeric_cols[0]}"
    y_col = "{numeric_cols[1]}"
    
    # Create scatter plot
    fig = px.scatter(
        self.df, 
        x=x_col, 
        y=y_col, 
        title=f"{{y_col}} vs {{x_col}}"
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500
    )
    
    # Display the figure - IMPORTANT for visualization to appear
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error creating visualization: {{str(e)}}")
    
    # Fallback to safe table view
    st.write("### Dataset Preview")
    st.write("Showing a sample of the data instead:")
    st.dataframe(self.df.head(10))
"""
            elif len(numeric_cols) == 1:
                # Create a histogram of the numeric column
                code = f"""
import plotly.express as px
import pandas as pd

# Create a histogram of the numeric column
try:
    numeric_col = "{numeric_cols[0]}"
    
    # Create histogram
    fig = px.histogram(
        self.df, 
        x=numeric_col, 
        title=f"Distribution of {{numeric_col}}"
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=numeric_col,
        yaxis_title="Count",
        height=500
    )
    
    # Display the figure - IMPORTANT for visualization to appear
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error creating visualization: {{str(e)}}")
    
    # Fallback to safe table view
    st.write("### Dataset Preview")
    st.write("Showing a sample of the data instead:")
    st.dataframe(self.df.head(10))
"""
            elif len(categorical_cols) >= 1:
                # Create a bar chart of categorical value counts
                code = f"""
import plotly.express as px
import pandas as pd

# Create a bar chart of categorical value counts
try:
    categorical_col = "{categorical_cols[0]}"
    
    # Count values
    value_counts = self.df[categorical_col].value_counts().reset_index()
    value_counts.columns = [categorical_col, "Count"]
    
    # Create bar chart
    fig = px.bar(
        value_counts, 
        x=categorical_col, 
        y="Count", 
        title=f"Counts of {{categorical_col}}"
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=categorical_col,
        yaxis_title="Count",
        height=500
    )
    
    # Display the figure - IMPORTANT for visualization to appear
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error creating visualization: {{str(e)}}")
    
    # Fallback to safe table view
    st.write("### Dataset Preview")
    st.write("Showing a sample of the data instead:")
    st.dataframe(self.df.head(10))
"""
            else:
                # Fallback to table view
                code = """
# Create a table view of the data
st.write("### Dataset Preview")
st.dataframe(self.df.head(10))
"""
            
            return code
        except Exception as e:
            # Ultra-safe fallback
            return f"""
# Error generating visualization code: {str(e)}
st.error("Could not generate visualization due to data type issues.")
st.write("### Dataset Preview")
st.dataframe(self.df.head(10))
"""
    
    def _get_claude_response(self, user_query):
        """Get response from Claude API with potential visualization code"""
        try:
            # Get dataframe info for context
            df_info = self._get_dataframe_info()
            
            # Create messages array
            messages = []
            
            # Add past messages (up to most recent 6)
            for message in st.session_state.chat_history[-6:]:
                role = "user" if message["role"] == "user" else "assistant"
                messages.append({"role": role, "content": message["content"]})
            
            # Create system prompt
            system_prompt = f"""You are an advanced data analysis assistant that helps users analyze and understand their data.
            
The user has uploaded a dataset with the following characteristics:
{df_info}

Your task is to provide insightful analysis and answer questions about this data.
Make your responses clear, informative and actionable. Use markdown formatting for clarity.

If the user's question could benefit from a visualization, include Python code that generates an appropriate visualization.
When including visualization code, follow these rules:
- Store the final plot object as 'fig' (important for download functionality)
- Use Plotly Express (px) or Plotly Graph Objects (go) for interactive visualizations
- For Matplotlib plots, use plt.figure() to create new figure objects
- Add proper titles, labels, and colors
- Access the dataframe as 'self.df'
- Include appropriate code comments
- Make the code complete and ready to execute
- ALWAYS check data types before visualizing to prevent errors with categorical data
- ALWAYS include explicit display commands (st.plotly_chart or st.pyplot) at the end of your code

The code will be executed automatically if provided."""
            
            # Make API call
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": st.session_state.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": st.session_state.claude_model,
                "messages": messages + [{"role": "user", "content": user_query}],
                "system": system_prompt,
                "max_tokens": 2000
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                text_response = response.json()["content"][0]["text"]
                
                # Extract code if present
                code_blocks = re.findall(r'```python\n(.*?)```', text_response, re.DOTALL)
                
                if code_blocks:
                    # Extract the first code block that might contain visualization code
                    potential_viz_code = code_blocks[0]
                    
                    # Check if it's likely visualization code
                    viz_keywords = ['plt.', 'px.', 'sns.', 'fig.', 'go.Figure', '.plot(', 'st.plotly_chart', 'st.pyplot']
                    is_viz_code = any(keyword in potential_viz_code for keyword in viz_keywords)
                    
                    if is_viz_code:
                        # Make sure there are display commands
                        if "st.plotly_chart" not in potential_viz_code and "st.pyplot" not in potential_viz_code:
                            # Check if it's using plotly
                            if "px." in potential_viz_code or "go." in potential_viz_code:
                                potential_viz_code += "\n\n# Explicitly display the figure\nif 'fig' in locals():\n    st.plotly_chart(fig, use_container_width=True)"
                            # Check if it's using matplotlib/seaborn
                            elif "plt." in potential_viz_code or "sns." in potential_viz_code:
                                potential_viz_code += "\n\n# Explicitly display the figure\nif 'plt' in locals():\n    st.pyplot(plt.gcf())"
                        
                        # Clean up response text
                        clean_response = text_response.replace(f"```python\n{code_blocks[0]}```", "*Visualization code has been extracted and executed below.*")
                        
                        return clean_response, potential_viz_code
                    else:
                        # It's likely just example code, not visualization code
                        return text_response, None
                else:
                    # No code blocks found
                    return text_response, None
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}", None
                
        except Exception as e:
            return f"Sorry, an error occurred: {str(e)}", None
    
    def _get_dataframe_info(self):
        """Get information about the dataframe for context"""
        try:
            # Basic stats
            info = []
            info.append(f"- Dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Column names and types
            info.append("- Columns:")
            for col in self.df.columns:
                dtype = self.df[col].dtype
                info.append(f"  - {col}: {dtype}")
            
            # Sample data
            info.append("\n- Sample Data (first 5 rows):")
            sample_data = self.df.head(5).to_csv(index=False)
            info.append("```")
            info.append(sample_data)
            info.append("```")
            
            # Basic statistics for numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                info.append("\n- Numeric Column Statistics:")
                stats = self.df[numeric_cols].describe().transpose()
                stats_str = stats.to_string()
                info.append("```")
                info.append(stats_str)
                info.append("```")
            
            # Missing values information
            missing_values = self.df.isnull().sum()
            if missing_values.sum() > 0:
                info.append("\n- Missing Values:")
                for col, count in missing_values.items():
                    if count > 0:
                        percentage = (count / len(self.df)) * 100
                        info.append(f"  - {col}: {count} ({percentage:.2f}%)")
            
            return "\n".join(info)
        except Exception as e:
            return f"Error getting dataframe info: {str(e)}"

# Main app code
def main():
    st.set_page_config(page_title="Claude Data Assistant", layout="wide")
    
    st.title("Claude Data Assistant")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display success message and data preview
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df.head(10))
            
            # Initialize and render GenAI assistant
            assistant = GenAIAssistant(df)
            assistant.render_interface()
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.code(traceback.format_exc())
            
    else:
        st.info("Please upload a CSV or Excel file to begin.")
        
        # Demo mode with sample data
        if st.button("Use Sample Data (Iris Dataset)"):
            # Load Iris dataset
            from sklearn.datasets import load_iris
            iris = load_iris()
            iris_df = pd.DataFrame(
                data=np.c_[iris['data'], iris['target']],
                columns=iris['feature_names'] + ['species']
            )
            
            # Convert numeric species to string labels
            species_names = iris.target_names
            iris_df['species'] = iris_df['species'].astype(int).map({
                0: species_names[0],
                1: species_names[1],
                2: species_names[2]
            })
            
            # Rename columns to remove spaces
            iris_df.columns = [col.replace(' ', '_') for col in iris_df.columns]
            
            st.success("Loaded Iris Dataset")
            
            with st.expander("Data Preview", expanded=False):
                st.dataframe(iris_df.head(10))
            
            # Initialize and render GenAI assistant
            assistant = GenAIAssistant(iris_df)
            assistant.render_interface()
