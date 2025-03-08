import streamlit as st
import pandas as pd
import datetime
from io import BytesIO
import base64

def create_header():
    """Create application header"""
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">DataInsightHub</h1>
            <p class="header-subtitle">Advanced Data Analysis & Visualization Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_footer():
    """Create application footer"""
    st.markdown(
        """
        <div class="footer-container">
            <p class="footer-text">© 2025 DataInsightHub. Powered by AI.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_info_card(title, content, icon="ℹ️"):
    """Create an info card with title and content"""
    st.markdown(
        f"""
        <div style="padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div>
                <h3 style="margin: 0; font-size: 1.2rem; color: #2E4172;">{title}</h3>
            </div>
            <p style="margin: 0; color: #6E6E6E;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_metric_card(title, value, delta=None, delta_description=None, help_text=None, icon=None):
    """Create a custom metric card"""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_icon = "↑" if delta >= 0 else "↓"
        delta_html = f"""
        <div style="display: flex; align-items: center; color: {delta_color};">
            <span style="font-size: 1rem; margin-right: 0.25rem;">{delta_icon}</span>
            <span style="font-size: 0.9rem;">{abs(delta):.2f}%</span>
            {f'<span style="font-size: 0.8rem; margin-left: 0.25rem; color: #6E6E6E;">{delta_description}</span>' if delta_description else ''}
        </div>
        """
    
    icon_html = f'<div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div>' if icon else ''
    
    help_icon = '❓' if help_text else ''
    
    st.markdown(
        f"""
        <div style="padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <div style="display: flex; align-items: center;">
                    {icon_html}
                    <div style="font-size: 0.9rem; color: #6E6E6E;">{title}</div>
                </div>
                <div title="{help_text}" style="cursor: {'help' if help_text else 'default'}; color: #6E6E6E;">{help_icon}</div>
            </div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #4A6FE3;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def file_uploader_ui():
    """Create a styled file uploader UI"""
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, Excel, JSON, Text)",
        type=["csv", "xlsx", "xls", "json", "txt"],
        help="Upload a data file to begin analysis"
    )
    
    return uploaded_file

def download_button(object_to_download, download_filename, button_text):
    """Create a download button for various types of data"""
    if isinstance(object_to_download, pd.DataFrame):
        # Download dataframe as CSV
        csv = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    elif isinstance(object_to_download, BytesIO):
        # Download bytes (e.g., plot image)
        b64 = base64.b64encode(object_to_download.getvalue()).decode()
        download_link = f'<a href="data:image/png;base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    elif isinstance(object_to_download, str):
        # Download text data
        b64 = base64.b64encode(object_to_download.encode()).decode()
        download_link = f'<a href="data:text/plain;base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    else:
        raise TypeError("Object type not supported for download")
    
    # Create a styled download button
    button_uuid = str(hash(download_filename))
    custom_css = f"""
        <style>
            #{button_uuid} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: #4A6FE3;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                text-decoration: none;
                font-weight: 500;
                border: none;
                cursor: pointer;
                transition: all 0.3s;
            }}
            #{button_uuid}:hover {{
                background-color: #2E4172;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }}
            #{button_uuid}::before {{
                content: "📥";
                margin-right: 0.5rem;
            }}
        </style>
    """
    
    # Create download button with custom ID
    download_button_html = f'<a id="{button_uuid}" href="data:file/csv;base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    # Combine CSS and button HTML
    html = f"{custom_css}{download_button_html}"
    
    return html

def create_data_preview(df, max_rows=5, max_cols=None):
    """Create a styled data preview"""
    if df is None or df.empty:
        st.info("No data to display")
        return
    
    # Handle large dataframes
    preview_df = df
    
    if len(df) > max_rows:
        preview_df = df.head(max_rows)
        st.info(f"Showing first {max_rows} rows of {len(df)} total rows")
    
    if max_cols and len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        st.info(f"Showing first {max_cols} columns of {len(df.columns)} total columns")
    
    # Display the dataframe with custom styling
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(preview_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", len(df))
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum()
        memory_str = format_bytes(memory_usage)
        st.metric("Memory Usage", memory_str)
    
    with col4:
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / (len(df) * len(df.columns))) * 100
        st.metric("Missing Values", f"{missing_cells} ({missing_pct:.1f}%)")

def format_bytes(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes == 0:
        return "0B"
        
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"