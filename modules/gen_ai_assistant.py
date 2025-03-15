# import streamlit as st
# import pandas as pd
# import numpy as np
# import json
# import os
# import requests
# import time
# import re
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from io import StringIO

# class GenAIAssistant:
#     """Generative AI powered data analysis assistant"""
    
#     def __init__(self, df):
#         """Initialize with dataframe"""
#         self.df = df
        
#         # Store chat history in session state if not already there
#         if 'genai_chat_history' not in st.session_state:
#             st.session_state.genai_chat_history = []
        
#         # Store API key
#         if 'openai_api_key' not in st.session_state:
#             st.session_state.openai_api_key = ""
            
#         # Store latest generated code
#         if 'latest_code' not in st.session_state:
#             st.session_state.latest_code = None

#     def render_interface(self):
#         """Render the GenAI assistant interface"""
#         st.header("💡 GenAI Data Assistant")
        
#         # Setup tab or direct interface
#         tab1, tab2, tab3, tab4 = st.tabs([
#             "💬 Chat Analysis", 
#             "📊 Smart Insights", 
#             "👨‍💻 Code Generation",
#             "⚙️ Settings"
#         ])
        
#         # Chat with Data tab
#         with tab1:
#             self._render_chat_interface()
        
#         # Smart Insights tab
#         with tab2:
#             self._render_insights_interface()
            
#         # Code Generation tab
#         with tab3:
#             self._render_code_generation()
            
#         # Settings tab
#         with tab4:
#             self._render_settings()
    
#     def _render_settings(self):
#         """Render settings interface"""
#         st.subheader("API Settings")
        
#         with st.form("api_settings"):
#             api_key = st.text_input(
#                 "OpenAI API Key", 
#                 value=st.session_state.openai_api_key,
#                 type="password",
#                 help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
#             )
            
#             model = st.selectbox(
#                 "Model",
#                 options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
#                 index=2,
#                 help="Select the OpenAI model to use"
#             )
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
#                                      help="Higher values make output more creative, lower values more deterministic")
#             with col2:
#                 max_tokens = st.slider("Max Response Length", 250, 4000, 1500, 50,
#                                        help="Maximum length of AI responses")
                
#             # Submit button
#             if st.form_submit_button("Save Settings"):
#                 st.session_state.openai_api_key = api_key
#                 st.session_state.openai_model = model
#                 st.session_state.openai_temperature = temperature
#                 st.session_state.openai_max_tokens = max_tokens
#                 st.success("Settings saved successfully!")
        
#         # Clear chat history button
#         if st.button("Clear Chat History"):
#             st.session_state.genai_chat_history = []
#             st.success("Chat history cleared!")
#             st.rerun()
    
#     def _render_chat_interface(self):
#         """Render chat interface for natural language data analysis"""
        
#         # Check if API key is set
#         if not st.session_state.openai_api_key:
#             st.warning("Please set your OpenAI API key in the Settings tab to use this feature.")
#             return
        
#         # Check if data is available
#         if self.df is None or self.df.empty:
#             st.warning("Please upload a dataset first to use this feature.")
#             return
            
#         st.subheader("Chat with your Data")
#         st.markdown("Ask questions about your data in natural language")
        
#         # Display chat history
#         self._display_chat_history()
        
#         # Chat input
#         with st.form("chat_input_form", clear_on_submit=True):
#             user_input = st.text_area("Ask a question about your data:", key="genai_chat_input", height=100)
#             col1, col2 = st.columns([4, 1])
            
#             with col1:
#                 sample_questions = [
#                     "What are the key patterns in this dataset?",
#                     "Analyze the relationship between [column X] and [column Y]",
#                     "What interesting insights can you find in this data?",
#                     "Suggest visualizations that would work well for this data",
#                     "What potential data quality issues do you see?",
#                 ]
#                 selected_sample = st.selectbox("Example questions:", [""] + sample_questions)
                
#             with col2:
#                 submit_button = st.form_submit_button("Submit", use_container_width=True)
        
#         # Process selected sample question or user input
#         if selected_sample and selected_sample != "":
#             user_input = selected_sample
        
#         if submit_button and user_input:
#             # Add user message to chat history
#             st.session_state.genai_chat_history.append({
#                 "role": "user",
#                 "content": user_input
#             })
            
#             # Get AI response
#             with st.spinner("Analyzing data and generating response..."):
#                 response = self._generate_chat_response(user_input)
                
#             # Add AI response to chat history
#             st.session_state.genai_chat_history.append({
#                 "role": "assistant",
#                 "content": response
#             })
            
#             # Refresh the UI
#             st.rerun()
    
#     def _display_chat_history(self):
#         """Display the chat conversation history"""
#         if not st.session_state.genai_chat_history:
#             st.info("Ask me questions about your data in natural language. I'll analyze and provide insights!")
#             return
        
#         # Container for the entire chat history
#         chat_container = st.container()
        
#         with chat_container:
#             for message in st.session_state.genai_chat_history:
#                 if message["role"] == "user":
#                     with st.chat_message("user", avatar="👤"):
#                         st.write(message["content"])
#                 else:
#                     with st.chat_message("assistant", avatar="🤖"):
#                         # Check if this is a markdown response with potential code blocks
#                         if "```python" in message["content"] or "```" in message["content"]:
#                             # Split by code blocks and render appropriately
#                             parts = re.split(r'(```(?:python)?\n[\s\S]*?\n```)', message["content"])
#                             for part in parts:
#                                 if part.startswith("```"):
#                                     # This is a code block - extract the code
#                                     code = re.search(r'```(?:python)?\n([\s\S]*?)\n```', part).group(1)
#                                     st.code(code, language="python")
#                                 else:
#                                     # This is regular text
#                                     st.write(part)
#                         else:
#                             # Regular text
#                             st.write(message["content"])
    
#     def _generate_chat_response(self, user_query):
#         """Generate AI response to user query about the data"""
#         # Verify API key
#         if not st.session_state.openai_api_key:
#             return "Please set your OpenAI API key in the Settings tab."
        
#         # Get dataframe info
#         df_info = self._get_dataframe_info()
        
#         # Create system prompt
#         system_prompt = f"""You are an advanced data analysis assistant that helps users analyze and understand their data.
        
# The user has uploaded a dataset with the following characteristics:
# {df_info}

# Your task is to provide insightful analysis and answer questions about this data.
# Make your responses clear, informative and actionable. Use markdown formatting for clarity.
# If appropriate, suggest Python code that could be used to further analyze or visualize the data.
# """

#         try:
#             # Prepare the messages
#             messages = [
#                 {"role": "system", "content": system_prompt},
#             ]
            
#             # Add chat history (up to last 6 messages to stay within token limits)
#             for message in st.session_state.genai_chat_history[-6:]:
#                 messages.append({"role": message["role"], "content": message["content"]})
            
#             # Add the current query if it's not already in the history
#             if not st.session_state.genai_chat_history or st.session_state.genai_chat_history[-1]["content"] != user_query:
#                 messages.append({"role": "user", "content": user_query})
            
#             # Get model parameters
#             model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
#             temperature = getattr(st.session_state, 'openai_temperature', 0.7)
#             max_tokens = getattr(st.session_state, 'openai_max_tokens', 1500)
            
#             # Make API call
#             url = "https://api.openai.com/v1/chat/completions"
#             headers = {
#                 "Authorization": f"Bearer {st.session_state.openai_api_key}",
#                 "Content-Type": "application/json"
#             }
#             payload = {
#                 "model": model,
#                 "messages": messages,
#                 "temperature": temperature,
#                 "max_tokens": max_tokens
#             }
            
#             response = requests.post(url, headers=headers, json=payload)
            
#             if response.status_code == 200:
#                 return response.json()["choices"][0]["message"]["content"]
#             else:
#                 error_message = f"Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
#                 return error_message
                
#         except Exception as e:
#             return f"Sorry, an error occurred while generating the response: {str(e)}"
    
#     def _get_dataframe_info(self):
#         """Get information about the dataframe for context"""
#         # Basic stats
#         info = []
#         info.append(f"- Dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
#         # Column names and types
#         info.append("- Columns:")
#         for col in self.df.columns:
#             dtype = self.df[col].dtype
#             unique = self.df[col].nunique()
#             missing = self.df[col].isna().sum()
#             missing_pct = (missing / len(self.df)) * 100
            
#             # Add column info
#             if missing_pct > 0:
#                 info.append(f"  - {col}: {dtype} ({unique} unique values, {missing_pct:.1f}% missing)")
#             else:
#                 info.append(f"  - {col}: {dtype} ({unique} unique values)")
        
#         # Sample data (first 5 rows in CSV format)
#         sample_data = self.df.head(5).to_csv(index=False)
#         info.append("\n- Sample Data (first 5 rows):")
#         info.append("```")
#         info.append(sample_data)
#         info.append("```")
        
#         # Add summary statistics for numeric columns
#         numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
#         if numeric_cols:
#             info.append("\n- Summary Statistics (numeric columns):")
#             stats_df = self.df[numeric_cols].describe().to_string()
#             info.append("```")
#             info.append(stats_df)
#             info.append("```")
            
#         return "\n".join(info)
    
#     def _render_insights_interface(self):
#         """Render automated insights generation interface"""
        
#         # Check if API key is set
#         if not st.session_state.openai_api_key:
#             st.warning("Please set your OpenAI API key in the Settings tab to use this feature.")
#             return
        
#         # Check if data is available
#         if self.df is None or self.df.empty:
#             st.warning("Please upload a dataset first to use this feature.")
#             return
            
#         st.subheader("Smart Insights Generator")
#         st.markdown("Automatically generate key insights from your data")
        
#         # Insight generation options
#         insight_type = st.selectbox(
#             "Select insight type:",
#             options=[
#                 "General Overview",
#                 "Data Quality Assessment",
#                 "Patterns & Correlations",
#                 "Key Metrics Summary",
#                 "Actionable Recommendations"
#             ]
#         )
        
#         # Extra context
#         extra_context = st.text_area(
#             "Additional context (optional):",
#             placeholder="Add any specific information or focus areas you'd like insights on...",
#             help="Provide additional context about your data or specific aspects you're interested in"
#         )
        
#         # Generate insights button
#         if st.button("Generate Insights", type="primary", use_container_width=True):
#             with st.spinner("Analyzing data and generating insights..."):
#                 insights = self._generate_insights(insight_type, extra_context)
                
#             # Display insights in a nice format
#             st.success("✅ Insights generated successfully!")
            
#             with st.expander("Generated Insights", expanded=True):
#                 st.markdown(insights)
                
#                 # Add download option
#                 timestamp = int(time.time())
#                 filename = f"{insight_type.lower().replace(' ', '_')}_{timestamp}.md"
                
#                 st.download_button(
#                     label="Download Insights",
#                     data=insights,
#                     file_name=filename,
#                     mime="text/markdown",
#                     use_container_width=True
#                 )
    
#     def _generate_insights(self, insight_type, extra_context=""):
#         """Generate automated insights based on the data"""
#         # Verify API key
#         if not st.session_state.openai_api_key:
#             return "Please set your OpenAI API key in the Settings tab."
        
#         # Get dataframe info
#         df_info = self._get_dataframe_info()
        
#         # Create prompt based on insight type
#         if insight_type == "General Overview":
#             prompt = "Provide a comprehensive overview of this dataset. Include key statistics, notable patterns, and overall insights."
#         elif insight_type == "Data Quality Assessment":
#             prompt = "Assess the quality of this dataset. Identify missing values, outliers, inconsistencies, and other data quality issues."
#         elif insight_type == "Patterns & Correlations":
#             prompt = "Identify key patterns, correlations, and relationships between variables in this dataset."
#         elif insight_type == "Key Metrics Summary":
#             prompt = "Summarize the key metrics and KPIs from this dataset. Highlight the most important numbers and trends."
#         elif insight_type == "Actionable Recommendations":
#             prompt = "Based on this data, provide actionable recommendations and next steps for analysis or business decisions."
#         else:
#             prompt = "Analyze this dataset and provide key insights."
            
#         # Add extra context if provided
#         if extra_context:
#             prompt += f"\n\nAdditional context: {extra_context}"
        
#         # Create system prompt
#         system_prompt = f"""You are an advanced data analysis assistant that generates insightful reports.

# The user has uploaded a dataset with the following characteristics:
# {df_info}

# Your task is to create a well-structured, comprehensive {insight_type} report.
# Format your response in Markdown with clear sections, bullet points, and emphasis where appropriate.
# Include specific details and numbers from the dataset to support your insights.
# Be thorough yet concise, focusing on actionable information.
# """

#         try:
#             # Make API call
#             url = "https://api.openai.com/v1/chat/completions"
#             headers = {
#                 "Authorization": f"Bearer {st.session_state.openai_api_key}",
#                 "Content-Type": "application/json"
#             }
            
#             # Get model parameters
#             model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
#             temperature = getattr(st.session_state, 'openai_temperature', 0.7)
#             max_tokens = getattr(st.session_state, 'openai_max_tokens', 1500)
            
#             payload = {
#                 "model": model,
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": temperature,
#                 "max_tokens": max_tokens
#             }
            
#             response = requests.post(url, headers=headers, json=payload)
            
#             if response.status_code == 200:
#                 return response.json()["choices"][0]["message"]["content"]
#             else:
#                 error_message = f"Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
#                 return error_message
                
#         except Exception as e:
#             return f"Sorry, an error occurred while generating insights: {str(e)}"
    
#     def _render_code_generation(self):
#         """Render code generation interface"""
        
#         # Check if API key is set
#         if not st.session_state.openai_api_key:
#             st.warning("Please set your OpenAI API key in the Settings tab to use this feature.")
#             return
        
#         # Check if data is available
#         if self.df is None or self.df.empty:
#             st.warning("Please upload a dataset first to use this feature.")
#             return
            
#         st.subheader("Code Generation")
#         st.markdown("Generate Python code for data analysis and visualization")
        
#         # Code generation options
#         code_type = st.selectbox(
#             "What type of code do you need?",
#             options=[
#                 "Data Cleaning & Preprocessing",
#                 "Exploratory Data Analysis",
#                 "Statistical Analysis",
#                 "Visualization",
#                 "Machine Learning Model",
#                 "Custom Request"
#             ]
#         )
        
#         # Custom request input
#         if code_type == "Custom Request":
#             code_request = st.text_area(
#                 "Describe what you want the code to do:",
#                 placeholder="Example: Generate code to find outliers in the numeric columns and visualize them using boxplots",
#                 height=100
#             )
#         else:
#             code_request = ""
        
#         # Target columns
#         all_columns = self.df.columns.tolist()
#         selected_columns = st.multiselect(
#             "Select columns to focus on (optional):",
#             options=all_columns
#         )
        
#         # Generate code button
#         if st.button("Generate Code", type="primary", use_container_width=True):
#             if code_type == "Custom Request" and not code_request:
#                 st.error("Please describe what you want the code to do.")
#             else:
#                 with st.spinner("Generating code..."):
#                     generated_code = self._generate_code(code_type, code_request, selected_columns)
                    
#                     # Store in session state
#                     st.session_state.latest_code = generated_code
                
#                 # Display code
#                 st.code(generated_code, language="python")
                
#                 # Execute code button
#                 if st.button("Execute Code", type="secondary", use_container_width=True):
#                     with st.spinner("Executing code..."):
#                         self._execute_code(generated_code)
    
#     def _generate_code(self, code_type, code_request="", selected_columns=None):
#         """Generate Python code for data analysis based on requirements"""
#         # Verify API key
#         if not st.session_state.openai_api_key:
#             return "# Please set your OpenAI API key in the Settings tab."
        
#         # Get dataframe info
#         df_info = self._get_dataframe_info()
        
#         # Prepare column info
#         if selected_columns and len(selected_columns) > 0:
#             column_focus = "Focus on these columns: " + ", ".join(selected_columns)
#         else:
#             column_focus = "Consider all columns in the dataset, but prioritize the most relevant ones."
            
#         # Create prompt based on code type
#         if code_type == "Data Cleaning & Preprocessing":
#             prompt = "Generate Python code to clean and preprocess this dataset. Include handling missing values, outliers, and data type conversions as needed."
#         elif code_type == "Exploratory Data Analysis":
#             prompt = "Generate Python code for exploratory data analysis of this dataset. Include summary statistics, distribution analysis, and basic visualizations."
#         elif code_type == "Statistical Analysis":
#             prompt = "Generate Python code to perform statistical analysis on this dataset. Include hypothesis testing, correlation analysis, and other relevant statistical methods."
#         elif code_type == "Visualization":
#             prompt = "Generate Python code to create informative visualizations for this dataset. Use matplotlib, seaborn, or plotly to create charts that reveal insights."
#         elif code_type == "Machine Learning Model":
#             prompt = "Generate Python code to build a machine learning model for this dataset. Include preprocessing, model selection, training, and evaluation."
#         elif code_type == "Custom Request":
#             prompt = code_request
#         else:
#             prompt = "Generate Python code to analyze this dataset."
            
#         # Add column focus
#         prompt += f"\n\n{column_focus}"
        
#         # Create system prompt
#         system_prompt = f"""You are an advanced Python code generator for data analysis.

# The user is working with a dataset that has the following characteristics:
# {df_info}

# Your task is to generate clear, well-documented Python code that meets their requirements.
# The user is working in a Streamlit environment and has pandas, numpy, matplotlib, seaborn, and plotly available.
# The dataframe is available as 'self.df' in the code.

# Important guidelines for your code:
# 1. Make the code complete and executable
# 2. Include helpful comments to explain key steps
# 3. Use best practices for data analysis
# 4. Include appropriate error handling
# 5. Generate visualizations that provide meaningful insights
# 6. Use pandas efficiently for data manipulation

# Return ONLY the Python code without any additional explanation.
# """

#         try:
#             # Make API call
#             url = "https://api.openai.com/v1/chat/completions"
#             headers = {
#                 "Authorization": f"Bearer {st.session_state.openai_api_key}",
#                 "Content-Type": "application/json"
#             }
            
#             # Get model parameters
#             model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
#             temperature = getattr(st.session_state, 'openai_temperature', 0.7)
            
#             payload = {
#                 "model": model,
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": temperature,
#                 "max_tokens": 2500  # Higher token limit for code generation
#             }
            
#             response = requests.post(url, headers=headers, json=payload)
            
#             if response.status_code == 200:
#                 code = response.json()["choices"][0]["message"]["content"]
#                 # Strip markdown code blocks if present
#                 code = re.sub(r'```python\n', '', code)
#                 code = re.sub(r'```\n?', '', code)
#                 return code
#             else:
#                 error_message = f"# Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
#                 return error_message
                
#         except Exception as e:
#             return f"# Sorry, an error occurred while generating code: {str(e)}"
    
#     def _execute_code(self, code):
#         """Execute the generated Python code and display results"""
#         try:
#             # Create a local namespace with access to the dataframe
#             local_namespace = {
#                 'self': self,
#                 'df': self.df,
#                 'pd': pd,
#                 'np': np,
#                 'plt': plt,
#                 'px': px,
#                 'go': go,
#                 'st': st
#             }
            
#             # Execute the code
#             exec(code, globals(), local_namespace)
            
#             # Success message
#             st.success("Code executed successfully!")
            
#         except Exception as e:
#             st.error(f"Error executing code: {str(e)}")
            
#             # Display traceback for debugging
#             import traceback
#             st.code(traceback.format_exc(), language="python")
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
from io import StringIO
import uuid
import traceback

class GenAIAssistant:
    """Generative AI powered data analysis assistant with multi-provider support"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        
        # Store chat history in session state if not already there
        if 'genai_chat_history' not in st.session_state:
            st.session_state.genai_chat_history = []
        
        # Store API keys
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = ""
            
        if 'claude_api_key' not in st.session_state:
            st.session_state.claude_api_key = ""
            
        # Store AI provider selection
        if 'ai_provider' not in st.session_state:
            st.session_state.ai_provider = "OpenAI"
            
        # Store latest generated code
        if 'latest_code' not in st.session_state:
            st.session_state.latest_code = None
            
        # Generate unique instance ID
        self.instance_id = str(uuid.uuid4())[:8]
    
    def _get_unique_key(self, base_name):
        """Generate a unique key for a Streamlit element"""
        return f"genai_{base_name}_{self.instance_id}"

    def render_interface(self):
        """Render the GenAI assistant interface"""
        st.header("💡 GenAI Data Assistant")
        
        # Setup tab or direct interface
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Chat Analysis", 
            "📊 Smart Insights", 
            "👨‍💻 Code Generation",
            "⚙️ Settings"
        ])
        
        # Chat with Data tab
        with tab1:
            self._render_chat_interface()
        
        # Smart Insights tab
        with tab2:
            self._render_insights_interface()
            
        # Code Generation tab
        with tab3:
            self._render_code_generation()
            
        # Settings tab
        with tab4:
            self._render_settings()
    
    def _render_settings(self):
        """Render settings interface"""
        st.subheader("API Settings")
        
        # Provider selection
        ai_provider = st.radio(
            "Select AI Provider:",
            ["OpenAI", "Claude"],
            horizontal=True,
            key=self._get_unique_key("ai_provider")
        )
        
        # Store the selected provider
        st.session_state.ai_provider = ai_provider
        
        # Show appropriate API settings based on the selected provider
        if ai_provider == "OpenAI":
            with st.form(key=self._get_unique_key("openai_settings_form")):
                st.subheader("OpenAI Settings")
                
                api_key = st.text_input(
                    "OpenAI API Key", 
                    value=st.session_state.openai_api_key,
                    type="password",
                    key=self._get_unique_key("openai_api_key_input"),
                    help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
                )
                
                model = st.selectbox(
                    "Model",
                    options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=2,
                    key=self._get_unique_key("openai_model_select"),
                    help="Select the OpenAI model to use"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    temperature = st.slider(
                        "Temperature", 
                        0.0, 1.0, 0.7, 0.1, 
                        key=self._get_unique_key("openai_temperature_slider"),
                        help="Higher values make output more creative, lower values more deterministic"
                    )
                with col2:
                    max_tokens = st.slider(
                        "Max Response Length", 
                        250, 4000, 1500, 50,
                        key=self._get_unique_key("openai_max_tokens_slider"),
                        help="Maximum length of AI responses"
                    )
                    
                # Submit button - regular form submit button
                submit_button = st.form_submit_button("Save OpenAI Settings")
                
                # Process the form submission
                if submit_button:
                    st.session_state.openai_api_key = api_key
                    st.session_state.openai_model = model
                    st.session_state.openai_temperature = temperature
                    st.session_state.openai_max_tokens = max_tokens
                    st.success("OpenAI settings saved successfully!")
        
        else:  # Claude settings
            with st.form(key=self._get_unique_key("claude_settings_form")):
                st.subheader("Claude Settings")
                
                api_key = st.text_input(
                    "Claude API Key", 
                    value=st.session_state.claude_api_key,
                    type="password",
                    key=self._get_unique_key("claude_api_key_input"),
                    help="Enter your Anthropic Claude API key. Get one at https://console.anthropic.com/"
                )
                
                model = st.selectbox(
                    "Model",
                    options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                    index=1,
                    key=self._get_unique_key("claude_model_select"),
                    help="Select the Claude model to use"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    temperature = st.slider(
                        "Temperature", 
                        0.0, 1.0, 0.7, 0.1, 
                        key=self._get_unique_key("claude_temperature_slider"),
                        help="Higher values make output more creative, lower values more deterministic"
                    )
                with col2:
                    max_tokens = st.slider(
                        "Max Response Length", 
                        250, 4000, 1500, 50,
                        key=self._get_unique_key("claude_max_tokens_slider"),
                        help="Maximum length of AI responses"
                    )
                    
                # Submit button - regular form submit button
                submit_button = st.form_submit_button("Save Claude Settings")
                
                # Process the form submission
                if submit_button:
                    st.session_state.claude_api_key = api_key
                    st.session_state.claude_model = model
                    st.session_state.claude_temperature = temperature
                    st.session_state.claude_max_tokens = max_tokens
                    st.success("Claude settings saved successfully!")
        
        # Clear chat history button
        if st.button("Clear Chat History", key=self._get_unique_key("clear_history")):
            st.session_state.genai_chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    def _render_chat_interface(self):
        """Render chat interface for natural language data analysis"""
        
        # Check if API key is set
        if (st.session_state.ai_provider == "OpenAI" and not st.session_state.openai_api_key) or \
           (st.session_state.ai_provider == "Claude" and not st.session_state.claude_api_key):
            st.warning(f"Please set your {st.session_state.ai_provider} API key in the Settings tab to use this feature.")
            return
        
        # Check if data is available
        if self.df is None or self.df.empty:
            st.warning("Please upload a dataset first to use this feature.")
            return
            
        st.subheader("Chat with your Data")
        st.markdown(f"Ask questions about your data in natural language (using {st.session_state.ai_provider})")
        
        # Display chat history
        self._display_chat_history()
        
        # Chat input
        with st.form(key=self._get_unique_key("chat_input_form"), clear_on_submit=True):
            user_input = st.text_area(
                "Ask a question about your data:", 
                key=self._get_unique_key("chat_input"), 
                height=100
            )
            col1, col2 = st.columns([4, 1])
            
            with col1:
                sample_questions = [
                    "What are the key patterns in this dataset?",
                    "Analyze the relationship between [column X] and [column Y]",
                    "What interesting insights can you find in this data?",
                    "Suggest visualizations that would work well for this data",
                    "What potential data quality issues do you see?",
                ]
                selected_sample = st.selectbox(
                    "Example questions:", 
                    [""] + sample_questions,
                    key=self._get_unique_key("sample_questions")
                )
                
            with col2:
                # Submit button - regular form submit button
                submit_button = st.form_submit_button("Submit", use_container_width=True)
        
        # Process selected sample question or user input
        if selected_sample and selected_sample != "":
            user_input = selected_sample
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.genai_chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get AI response
            with st.spinner(f"Analyzing data and generating response with {st.session_state.ai_provider}..."):
                response = self._generate_chat_response(user_input)
                
            # Add AI response to chat history
            st.session_state.genai_chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Refresh the UI
            st.rerun()
    
    def _display_chat_history(self):
        """Display the chat conversation history"""
        if not st.session_state.genai_chat_history:
            st.info("Ask me questions about your data in natural language. I'll analyze and provide insights!")
            return
        
        # Container for the entire chat history
        chat_container = st.container()
        
        with chat_container:
            for idx, message in enumerate(st.session_state.genai_chat_history):
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        # Check if this is a markdown response with potential code blocks
                        if "```python" in message["content"] or "```" in message["content"]:
                            # Split by code blocks and render appropriately
                            parts = re.split(r'(```(?:python)?\n[\s\S]*?\n```)', message["content"])
                            for part_idx, part in enumerate(parts):
                                if part.startswith("```"):
                                    # This is a code block - extract the code
                                    code_match = re.search(r'```(?:python)?\n([\s\S]*?)\n```', part)
                                    if code_match:
                                        code = code_match.group(1)
                                        st.code(
                                            code, 
                                            language="python", 
                                            key=self._get_unique_key(f"code_{idx}_{part_idx}")
                                        )
                                else:
                                    # This is regular text
                                    if part.strip():  # Only show if not empty
                                        st.write(part)
                        else:
                            # Regular text
                            st.write(message["content"])
    
    def _generate_chat_response(self, user_query):
        """Generate AI response to user query about the data"""
        # Check which provider to use
        if st.session_state.ai_provider == "OpenAI":
            return self._generate_openai_response(user_query)
        else:
            return self._generate_claude_response(user_query)
    
    def _generate_openai_response(self, user_query):
        """Generate response using OpenAI API"""
        # Verify API key
        if not st.session_state.openai_api_key:
            return "Please set your OpenAI API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Create system prompt
        system_prompt = f"""You are an advanced data analysis assistant that helps users analyze and understand their data.
        
The user has uploaded a dataset with the following characteristics:
{df_info}

Your task is to provide insightful analysis and answer questions about this data.
Make your responses clear, informative and actionable. Use markdown formatting for clarity.
If appropriate, suggest Python code that could be used to further analyze or visualize the data.
"""

        try:
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add chat history (up to last 6 messages to stay within token limits)
            for message in st.session_state.genai_chat_history[-6:]:
                messages.append({"role": message["role"], "content": message["content"]})
            
            # Add the current query if it's not already in the history
            if not st.session_state.genai_chat_history or st.session_state.genai_chat_history[-1]["content"] != user_query:
                messages.append({"role": "user", "content": user_query})
            
            # Get model parameters
            model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
            temperature = getattr(st.session_state, 'openai_temperature', 0.7)
            max_tokens = getattr(st.session_state, 'openai_max_tokens', 1500)
            
            # Make API call
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {st.session_state.openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_message = f"Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"Sorry, an error occurred while generating the response: {str(e)}"
    
    def _generate_claude_response(self, user_query):
        """Generate response using Claude API"""
        # Verify API key
        if not st.session_state.claude_api_key:
            return "Please set your Claude API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Create system prompt
        system_prompt = f"""You are an advanced data analysis assistant that helps users analyze and understand their data.
        
The user has uploaded a dataset with the following characteristics:
{df_info}

Your task is to provide insightful analysis and answer questions about this data.
Make your responses clear, informative and actionable. Use markdown formatting for clarity.
If appropriate, suggest Python code that could be used to further analyze or visualize the data.
"""

        try:
            # Prepare messages in Anthropic format
            messages = []
            
            # Add chat history (up to last 6 messages to stay within token limits)
            for message in st.session_state.genai_chat_history[-6:]:
                role = "user" if message["role"] == "user" else "assistant"
                messages.append({"role": role, "content": message["content"]})
            
            # Add the current query if it's not already in the history
            if not st.session_state.genai_chat_history or st.session_state.genai_chat_history[-1]["content"] != user_query:
                messages.append({"role": "user", "content": user_query})
            
            # Get model parameters
            model = getattr(st.session_state, 'claude_model', "claude-3-sonnet-20240229")
            temperature = getattr(st.session_state, 'claude_temperature', 0.7)
            max_tokens = getattr(st.session_state, 'claude_max_tokens', 1500)
            
            # Prepare payload
            payload = {
                "model": model,
                "messages": messages,
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make API call
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": st.session_state.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                error_message = f"Error from Claude API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"Sorry, an error occurred while generating the response with Claude: {str(e)}"
    
    def _get_dataframe_info(self):
        """Get information about the dataframe for context"""
        # Basic stats
        info = []
        info.append(f"- Dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Column names and types
        info.append("- Columns:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique = self.df[col].nunique()
            missing = self.df[col].isna().sum()
            missing_pct = (missing / len(self.df)) * 100
            
            # Add column info
            if missing_pct > 0:
                info.append(f"  - {col}: {dtype} ({unique} unique values, {missing_pct:.1f}% missing)")
            else:
                info.append(f"  - {col}: {dtype} ({unique} unique values)")
        
        # Sample data (first 5 rows in CSV format)
        sample_data = self.df.head(5).to_csv(index=False)
        info.append("\n- Sample Data (first 5 rows):")
        info.append("```")
        info.append(sample_data)
        info.append("```")
        
        # Add summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            info.append("\n- Summary Statistics (numeric columns):")
            stats_df = self.df[numeric_cols].describe().to_string()
            info.append("```")
            info.append(stats_df)
            info.append("```")
            
        return "\n".join(info)
    
    def _render_insights_interface(self):
        """Render automated insights generation interface"""
        
        # Check if API key is set
        if (st.session_state.ai_provider == "OpenAI" and not st.session_state.openai_api_key) or \
           (st.session_state.ai_provider == "Claude" and not st.session_state.claude_api_key):
            st.warning(f"Please set your {st.session_state.ai_provider} API key in the Settings tab to use this feature.")
            return
        
        # Check if data is available
        if self.df is None or self.df.empty:
            st.warning("Please upload a dataset first to use this feature.")
            return
            
        st.subheader("Smart Insights Generator")
        st.markdown(f"Automatically generate key insights from your data (using {st.session_state.ai_provider})")
        
        # Insight generation options
        insight_type = st.selectbox(
            "Select insight type:",
            options=[
                "General Overview",
                "Data Quality Assessment",
                "Patterns & Correlations",
                "Key Metrics Summary",
                "Actionable Recommendations"
            ],
            key=self._get_unique_key("insight_type")
        )
        
        # Extra context
        extra_context = st.text_area(
            "Additional context (optional):",
            placeholder="Add any specific information or focus areas you'd like insights on...",
            key=self._get_unique_key("extra_context"),
            help="Provide additional context about your data or specific aspects you're interested in"
        )
        
        # Generate insights button
        if st.button(
            "Generate Insights", 
            key=self._get_unique_key("generate_insights"), 
            type="primary", 
            use_container_width=True
        ):
            with st.spinner(f"Analyzing data and generating insights with {st.session_state.ai_provider}..."):
                insights = self._generate_insights(insight_type, extra_context)
                
            # Display insights in a nice format
            st.success("✅ Insights generated successfully!")
            
            with st.expander("Generated Insights", expanded=True):
                st.markdown(insights)
                
                # Add download option
                timestamp = int(time.time())
                filename = f"{insight_type.lower().replace(' ', '_')}_{timestamp}.md"
                
                st.download_button(
                    label="Download Insights",
                    data=insights,
                    file_name=filename,
                    mime="text/markdown",
                    key=self._get_unique_key("download_insights"),
                    use_container_width=True
                )
    
    def _generate_insights(self, insight_type, extra_context=""):
        """Generate automated insights based on the data"""
        # Check which provider to use
        if st.session_state.ai_provider == "OpenAI":
            return self._generate_openai_insights(insight_type, extra_context)
        else:
            return self._generate_claude_insights(insight_type, extra_context)
    
    def _generate_openai_insights(self, insight_type, extra_context=""):
        """Generate insights using OpenAI API"""
        # Verify API key
        if not st.session_state.openai_api_key:
            return "Please set your OpenAI API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Create prompt based on insight type
        if insight_type == "General Overview":
            prompt = "Provide a comprehensive overview of this dataset. Include key statistics, notable patterns, and overall insights."
        elif insight_type == "Data Quality Assessment":
            prompt = "Assess the quality of this dataset. Identify missing values, outliers, inconsistencies, and other data quality issues."
        elif insight_type == "Patterns & Correlations":
            prompt = "Identify key patterns, correlations, and relationships between variables in this dataset."
        elif insight_type == "Key Metrics Summary":
            prompt = "Summarize the key metrics and KPIs from this dataset. Highlight the most important numbers and trends."
        elif insight_type == "Actionable Recommendations":
            prompt = "Based on this data, provide actionable recommendations and next steps for analysis or business decisions."
        else:
            prompt = "Analyze this dataset and provide key insights."
            
        # Add extra context if provided
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        
        # Create system prompt
        system_prompt = f"""You are an advanced data analysis assistant that generates insightful reports.

The user has uploaded a dataset with the following characteristics:
{df_info}

Your task is to create a well-structured, comprehensive {insight_type} report.
Format your response in Markdown with clear sections, bullet points, and emphasis where appropriate.
Include specific details and numbers from the dataset to support your insights.
Be thorough yet concise, focusing on actionable information.
"""

        try:
            # Make API call
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {st.session_state.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Get model parameters
            model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
            temperature = getattr(st.session_state, 'openai_temperature', 0.7)
            max_tokens = getattr(st.session_state, 'openai_max_tokens', 1500)
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_message = f"Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"Sorry, an error occurred while generating insights: {str(e)}"
    
    def _generate_claude_insights(self, insight_type, extra_context=""):
        """Generate insights using Claude API"""
        # Verify API key
        if not st.session_state.claude_api_key:
            return "Please set your Claude API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Create prompt based on insight type
        if insight_type == "General Overview":
            prompt = "Provide a comprehensive overview of this dataset. Include key statistics, notable patterns, and overall insights."
        elif insight_type == "Data Quality Assessment":
            prompt = "Assess the quality of this dataset. Identify missing values, outliers, inconsistencies, and other data quality issues."
        elif insight_type == "Patterns & Correlations":
            prompt = "Identify key patterns, correlations, and relationships between variables in this dataset."
        elif insight_type == "Key Metrics Summary":
            prompt = "Summarize the key metrics and KPIs from this dataset. Highlight the most important numbers and trends."
        elif insight_type == "Actionable Recommendations":
            prompt = "Based on this data, provide actionable recommendations and next steps for analysis or business decisions."
        else:
            prompt = "Analyze this dataset and provide key insights."
            
        # Add extra context if provided
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        
        # Create system prompt
        system_prompt = f"""You are an advanced data analysis assistant that generates insightful reports.

The user has uploaded a dataset with the following characteristics:
{df_info}

Your task is to create a well-structured, comprehensive {insight_type} report.
Format your response in Markdown with clear sections, bullet points, and emphasis where appropriate.
Include specific details and numbers from the dataset to support your insights.
Be thorough yet concise, focusing on actionable information.
"""

        try:
            # Prepare payload
            payload = {
                "model": getattr(st.session_state, 'claude_model', "claude-3-sonnet-20240229"),
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt,
                "temperature": getattr(st.session_state, 'claude_temperature', 0.7),
                "max_tokens": getattr(st.session_state, 'claude_max_tokens', 1500)
            }
            
            # Make API call
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": st.session_state.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                error_message = f"Error from Claude API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"Sorry, an error occurred while generating insights with Claude: {str(e)}"
    
    def _render_code_generation(self):
        """Render code generation interface"""
        
        # Check if API key is set
        if (st.session_state.ai_provider == "OpenAI" and not st.session_state.openai_api_key) or \
           (st.session_state.ai_provider == "Claude" and not st.session_state.claude_api_key):
            st.warning(f"Please set your {st.session_state.ai_provider} API key in the Settings tab to use this feature.")
            return
        
        # Check if data is available
        if self.df is None or self.df.empty:
            st.warning("Please upload a dataset first to use this feature.")
            return
            
        st.subheader("Code Generation")
        st.markdown(f"Generate Python code for data analysis and visualization (using {st.session_state.ai_provider})")
        
        # Code generation options
        code_type = st.selectbox(
            "What type of code do you need?",
            options=[
                "Data Cleaning & Preprocessing",
                "Exploratory Data Analysis",
                "Statistical Analysis",
                "Visualization",
                "Machine Learning Model",
                "Custom Request"
            ],
            key=self._get_unique_key("code_type")
        )
        
        # Custom request input
        if code_type == "Custom Request":
            code_request = st.text_area(
                "Describe what you want the code to do:",
                placeholder="Example: Generate code to find outliers in the numeric columns and visualize them using boxplots",
                height=100,
                key=self._get_unique_key("code_request")
            )
        else:
            code_request = ""
        
        # Target columns
        all_columns = self.df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to focus on (optional):",
            options=all_columns,
            key=self._get_unique_key("selected_columns")
        )
        
        # Generate code button
        if st.button(
            "Generate Code", 
            key=self._get_unique_key("generate_code"),
            type="primary", 
            use_container_width=True
        ):
            if code_type == "Custom Request" and not code_request:
                st.error("Please describe what you want the code to do.")
            else:
                with st.spinner(f"Generating code with {st.session_state.ai_provider}..."):
                    generated_code = self._generate_code(code_type, code_request, selected_columns)
                    
                    # Store in session state
                    st.session_state.latest_code = generated_code
                
                # Display code
                st.code(generated_code, language="python")
                
                # Execute code button
                if st.button(
                    "Execute Code", 
                    key=self._get_unique_key("execute_code"),
                    type="secondary", 
                    use_container_width=True
                ):
                    with st.spinner("Executing code..."):
                        self._execute_code(generated_code)
    
    def _generate_code(self, code_type, code_request="", selected_columns=None):
        """Generate Python code for data analysis based on requirements"""
        # Check which provider to use
        if st.session_state.ai_provider == "OpenAI":
            return self._generate_openai_code(code_type, code_request, selected_columns)
        else:
            return self._generate_claude_code(code_type, code_request, selected_columns)
    
    def _generate_openai_code(self, code_type, code_request="", selected_columns=None):
        """Generate code using OpenAI API"""
        # Verify API key
        if not st.session_state.openai_api_key:
            return "# Please set your OpenAI API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Prepare column info
        if selected_columns and len(selected_columns) > 0:
            column_focus = "Focus on these columns: " + ", ".join(selected_columns)
        else:
            column_focus = "Consider all columns in the dataset, but prioritize the most relevant ones."
            
        # Create prompt based on code type
        if code_type == "Data Cleaning & Preprocessing":
            prompt = "Generate Python code to clean and preprocess this dataset. Include handling missing values, outliers, and data type conversions as needed."
        elif code_type == "Exploratory Data Analysis":
            prompt = "Generate Python code for exploratory data analysis of this dataset. Include summary statistics, distribution analysis, and basic visualizations."
        elif code_type == "Statistical Analysis":
            prompt = "Generate Python code to perform statistical analysis on this dataset. Include hypothesis testing, correlation analysis, and other relevant statistical methods."
        elif code_type == "Visualization":
            prompt = "Generate Python code to create informative visualizations for this dataset. Use matplotlib, seaborn, or plotly to create charts that reveal insights."
        elif code_type == "Machine Learning Model":
            prompt = "Generate Python code to build a machine learning model for this dataset. Include preprocessing, model selection, training, and evaluation."
        elif code_type == "Custom Request":
            prompt = code_request
        else:
            prompt = "Generate Python code to analyze this dataset."
            
        # Add column focus
        prompt += f"\n\n{column_focus}"
        
        # Create system prompt
        system_prompt = f"""You are an advanced Python code generator for data analysis.

The user is working with a dataset that has the following characteristics:
{df_info}

Your task is to generate clear, well-documented Python code that meets their requirements.
The user is working in a Streamlit environment and has pandas, numpy, matplotlib, seaborn, and plotly available.
The dataframe is available as 'self.df' in the code.

Important guidelines for your code:
1. Make the code complete and executable
2. Include helpful comments to explain key steps
3. Use best practices for data analysis
4. Include appropriate error handling
5. Generate visualizations that provide meaningful insights
6. Use pandas efficiently for data manipulation

Return ONLY the Python code without any additional explanation.
"""

        try:
            # Make API call
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {st.session_state.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Get model parameters
            model = getattr(st.session_state, 'openai_model', "gpt-3.5-turbo")
            temperature = getattr(st.session_state, 'openai_temperature', 0.7)
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2500  # Higher token limit for code generation
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                code = response.json()["choices"][0]["message"]["content"]
                # Strip markdown code blocks if present
                code = re.sub(r'```python\n', '', code)
                code = re.sub(r'```\n?', '', code)
                return code
            else:
                error_message = f"# Error from OpenAI API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"# Sorry, an error occurred while generating code: {str(e)}"
    
    def _generate_claude_code(self, code_type, code_request="", selected_columns=None):
        """Generate code using Claude API"""
        # Verify API key
        if not st.session_state.claude_api_key:
            return "# Please set your Claude API key in the Settings tab."
        
        # Get dataframe info
        df_info = self._get_dataframe_info()
        
        # Prepare column info
        if selected_columns and len(selected_columns) > 0:
            column_focus = "Focus on these columns: " + ", ".join(selected_columns)
        else:
            column_focus = "Consider all columns in the dataset, but prioritize the most relevant ones."
            
        # Create prompt based on code type
        if code_type == "Data Cleaning & Preprocessing":
            prompt = "Generate Python code to clean and preprocess this dataset. Include handling missing values, outliers, and data type conversions as needed."
        elif code_type == "Exploratory Data Analysis":
            prompt = "Generate Python code for exploratory data analysis of this dataset. Include summary statistics, distribution analysis, and basic visualizations."
        elif code_type == "Statistical Analysis":
            prompt = "Generate Python code to perform statistical analysis on this dataset. Include hypothesis testing, correlation analysis, and other relevant statistical methods."
        elif code_type == "Visualization":
            prompt = "Generate Python code to create informative visualizations for this dataset. Use matplotlib, seaborn, or plotly to create charts that reveal insights."
        elif code_type == "Machine Learning Model":
            prompt = "Generate Python code to build a machine learning model for this dataset. Include preprocessing, model selection, training, and evaluation."
        elif code_type == "Custom Request":
            prompt = code_request
        else:
            prompt = "Generate Python code to analyze this dataset."
            
        # Add column focus
        prompt += f"\n\n{column_focus}"
        
        # Create system prompt
        system_prompt = f"""You are an advanced Python code generator for data analysis.

The user is working with a dataset that has the following characteristics:
{df_info}

Your task is to generate clear, well-documented Python code that meets their requirements.
The user is working in a Streamlit environment and has pandas, numpy, matplotlib, seaborn, and plotly available.
The dataframe is available as 'self.df' in the code.

Important guidelines for your code:
1. Make the code complete and executable
2. Include helpful comments to explain key steps
3. Use best practices for data analysis
4. Include appropriate error handling
5. Generate visualizations that provide meaningful insights
6. Use pandas efficiently for data manipulation

Return ONLY the Python code without any additional explanation.
"""

        try:
            # Prepare payload
            payload = {
                "model": getattr(st.session_state, 'claude_model', "claude-3-sonnet-20240229"),
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt,
                "temperature": getattr(st.session_state, 'claude_temperature', 0.7),
                "max_tokens": 2500  # Higher token limit for code generation
            }
            
            # Make API call
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": st.session_state.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                code = response.json()["content"][0]["text"]
                # Strip markdown code blocks if present
                code = re.sub(r'```python\n', '', code)
                code = re.sub(r'```\n?', '', code)
                return code
            else:
                error_message = f"# Error from Claude API: {response.json().get('error', {}).get('message', 'Unknown error')}"
                return error_message
                
        except Exception as e:
            return f"# Sorry, an error occurred while generating code with Claude: {str(e)}"
    
    def _execute_code(self, code):
        """Execute the generated Python code and display results"""
        try:
            # Create a local namespace with access to the dataframe
            local_namespace = {
                'self': self,
                'df': self.df,
                'pd': pd,
                'np': np,
                'plt': plt,
                'px': px,
                'go': go,
                'st': st
            }
            
            # Execute the code
            exec(code, globals(), local_namespace)
            
            # Success message
            st.success("Code executed successfully!")
            
        except Exception as e:
            st.error(f"Error executing code: {str(e)}")
            
            # Display traceback for debugging
            st.code(traceback.format_exc(), language="python")