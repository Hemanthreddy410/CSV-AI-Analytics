import streamlit as st
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_custom_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    /* Main color palette */
    :root {
        --primary-color: #4A6FE3;
        --secondary-color: #2E4172;
        --accent-color: #6C63FF;
        --background-color: #F8F9FA;
        --card-bg-color: #FFFFFF;
        --text-color: #333333;
        --light-text-color: #6E6E6E;
        --border-color: #E0E0E0;
        --hover-color: #EEF2FF;
        --positive-color: #28A745;
        --warning-color: #FFC107;
        --negative-color: #DC3545;
    }
    
    /* Global styles */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--secondary-color);
        font-weight: 600;
    }
    
    /* Header styles */
    .header-container {
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .welcome-title {
        font-size: 3rem;
        color: var(--secondary-color);
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.5rem;
        color: var(--light-text-color);
        margin-bottom: 3rem;
    }
    
    .features-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1.5rem;
        width: 220px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        font-size: 0.9rem;
        color: var(--light-text-color);
    }
    
    .welcome-instruction {
        font-size: 1.2rem;
        color: var(--primary-color);
        margin-top: 2rem;
    }
    
    /* Project title */
    .project-title {
        font-size: 2rem;
        color: var(--secondary-color);
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Chat styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: var(--card-bg-color);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .chat-message {
        display: flex;
        margin-bottom: 1rem;
    }
    
    .user-message {
        flex-direction: row-reverse;
    }
    
    .assistant-message {
        flex-direction: row;
    }
    
    .chat-message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: var(--primary-color);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin: 0 0.5rem;
    }
    
    .user-message .chat-message-avatar {
        background-color: var(--secondary-color);
    }
    
    .chat-message-content {
        max-width: 80%;
        padding: 1rem;
        border-radius: 10px;
        background-color: #F0F2F6;
    }
    
    .user-message .chat-message-content {
        background-color: #E6F3FF;
        text-align: right;
    }
    
    .chat-message-content p {
        margin: 0;
    }
    
    /* Footer styles */
    .footer-container {
        margin-top: 4rem;
        padding: 1rem 0;
        text-align: center;
        border-top: 1px solid var(--border-color);
    }
    
    .footer-text {
        color: var(--light-text-color);
        font-size: 0.9rem;
    }
    
    /* Improved Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: var(--background-color);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.75rem 1rem;
        background-color: var(--card-bg-color);
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        font-weight: 500;
        white-space: pre-wrap;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 8px rgba(74, 111, 227, 0.3) !important;
    }
    
    /* Card styling */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stExpander"] > details {
        background-color: var(--card-bg-color);
    }
    
    div[data-testid="stExpander"] > details > summary {
        padding: 1rem;
        font-weight: 600;
        color: var(--secondary-color);
        background-color: rgba(74, 111, 227, 0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: rgba(74, 111, 227, 0.1);
    }
    
    div[data-testid="stExpander"] > details[open] > summary {
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stMetric"] > div:first-child {
        color: var(--secondary-color);
    }
    
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color);
    }
    
    /* Dataframe styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    /* Success/Warning/Error message styling */
    div[data-testid="stSuccessMessage"] {
        background-color: rgba(40, 167, 69, 0.1);
        border-color: var(--positive-color);
    }
    
    div[data-testid="stWarningMessage"] {
        background-color: rgba(255, 193, 7, 0.1);
        border-color: var(--warning-color);
    }
    
    div[data-testid="stErrorMessage"] {
        background-color: rgba(220, 53, 69, 0.1);
        border-color: var(--negative-color);
    }
    
    /* Data Processing Workflow Guide */
    .workflow-step {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        padding: 0.75rem;
        background-color: rgba(74, 111, 227, 0.05);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .workflow-step:hover {
        background-color: rgba(74, 111, 227, 0.1);
    }
    
    .workflow-step-number {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 30px;
        height: 30px;
        background-color: var(--primary-color);
        color: white;
        border-radius: 50%;
        margin-right: 1rem;
        font-weight: bold;
    }
    
    .workflow-step-content {
        flex: 1;
    }
    
    /* Processing History */
    .history-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.75rem;
        background-color: var(--card-bg-color);
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 3px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background-color: var(--hover-color);
        transform: translateX(3px);
    }
    
    .history-icon {
        margin-right: 0.75rem;
        color: var(--primary-color);
    }
    
    .history-description {
        flex: 1;
    }
    
    .history-timestamp {
        color: var(--light-text-color);
        font-size: 0.8rem;
    }
    
    /* Form controls styling */
    div[data-baseweb="select"] {
        border-radius: 6px;
        transition: all 0.3s;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: var(--primary-color);
    }
    
    div[data-baseweb="input"] {
        border-radius: 6px;
        transition: all 0.3s;
    }
    
    div[data-baseweb="input"]:hover {
        border-color: var(--primary-color);
    }
    
    /* Slider styling */
    div[data-testid="stSlider"] > div {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    
    div[data-testid="stSlider"] > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Checkbox styling */
    label[data-baseweb="checkbox"] {
        gap: 0.5rem;
    }
    
    label[data-baseweb="checkbox"] > div {
        background-color: var(--card-bg-color);
        border-color: var(--border-color);
        transition: all 0.3s;
    }
    
    label[data-baseweb="checkbox"]:hover > div {
        border-color: var(--primary-color);
    }
    
    label[data-baseweb="checkbox"][aria-checked="true"] > div {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* Radio button styling */
    label[data-baseweb="radio"] {
        gap: 0.5rem;
    }
    
    label[data-baseweb="radio"] > div {
        background-color: var(--card-bg-color);
        border-color: var(--border-color);
        transition: all 0.3s;
    }
    
    label[data-baseweb="radio"]:hover > div {
        border-color: var(--primary-color);
    }
    
    label[data-baseweb="radio"][aria-checked="true"] > div > div {
        background-color: var(--primary-color) !important;
    }
    
    /* Progress bar */
    div[role="progressbar"] > div {
        background-color: var(--primary-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--background-color);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1rem;
    }
    
    /* Data table styling */
    div[data-testid="stTable"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    /* Make tooltips more visible */
    span[data-baseweb="tooltip"] {
        background-color: var(--secondary-color) !important;
        border-radius: 4px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Graph container styling */
    [data-testid="stDecoration"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
    }
    
    [data-testid="stDecoration"]:hover {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        border-radius: 8px;
        overflow: hidden;
        border: 2px dashed var(--border-color);
        padding: 1rem;
        transition: all 0.3s;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
        background-color: rgba(74, 111, 227, 0.05);
    }
    
    div[data-testid="stFileUploadDropzone"] > span {
        color: var(--primary-color);
        font-weight: 500;
    }
    
    /* Download button styling */
    a.download-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: var(--primary-color);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        margin-top: 0.5rem;
    }
    
    a.download-button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    a.download-button::before {
        content: "📥";
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def add_logo():
    """Add app logo to the sidebar"""
    # Generate a simple SVG logo
    logo_svg = '''
    <svg width="150" height="150" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <rect x="20" y="20" width="160" height="160" rx="20" fill="#4A6FE3" />
        <circle cx="100" cy="75" r="30" fill="white" />
        <rect x="60" y="115" width="80" height="40" rx="10" fill="white" />
        <circle cx="70" cy="135" r="10" fill="#4A6FE3" />
        <circle cx="100" cy="135" r="10" fill="#4A6FE3" />
        <circle cx="130" cy="135" r="10" fill="#4A6FE3" />
    </svg>
    '''
    
    # Render the logo in the sidebar
    st.sidebar.markdown(
        f'<div style="display: flex; justify-content: center; margin-bottom: 20px;">{logo_svg}</div>',
        unsafe_allow_html=True
    )


def encode_image(image):
    """Encode a PIL image as base64 for embedding in HTML/CSS"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def download_dataframe(df, filename="data.csv"):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


def handle_uploaded_file(uploaded_file):
    """Process an uploaded file and return a dataframe"""
    if uploaded_file is None:
        return None
        
    try:
        # Check file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        elif file_extension == "txt":
            # Try to detect delimiter
            content = uploaded_file.getvalue().decode("utf-8")
            if "," in content:
                df = pd.read_csv(uploaded_file, sep=",")
            elif "\t" in content:
                df = pd.read_csv(uploaded_file, sep="\t")
            elif ";" in content:
                df = pd.read_csv(uploaded_file, sep=";")
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


def format_bytes(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes == 0:
        return "0B"
        
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"
def fix_arrow_dtypes(df):
    """
    Fix data types that may cause problems with PyArrow conversion.
    This is particularly helpful for Int64DType and other pandas extension types.
    """
    if df is None:
        return None
        
    # Make a copy to avoid modifying the original
    df_fixed = df.copy()
    
    # Check for problematic Int64 pandas extension type
    for col in df_fixed.columns:
        if hasattr(df_fixed[col].dtype, 'name'):
            # Check for pandas extension integer types
            if 'Int' in df_fixed[col].dtype.name:
                # Convert to standard numpy int type
                df_fixed[col] = df_fixed[col].astype('int64')
            
            # Check for other potentially problematic extension types
            elif df_fixed[col].dtype.name == 'boolean':
                df_fixed[col] = df_fixed[col].astype('bool')
    
    # For any object columns that should be string type
    for col in df_fixed.select_dtypes(include=['object']).columns:
        try:
            # Check if the column has string values
            if df_fixed[col].apply(lambda x: isinstance(x, str)).all():
                df_fixed[col] = df_fixed[col].astype('string')
        except:
            # If any error, skip this column
            pass
    
    return df_fixed


def apply_theme_to_plot(fig):
    """Apply custom theme to matplotlib figure"""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Set colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#4A6FE3", "#6C63FF", "#2E4172", "#F87060", "#81B29A", "#F2CC8F"]
    )
    
    # Set font properties
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
    
    # Set figure properties
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.labelcolor"] = "#333333"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    
    # Set grid properties
    plt.rcParams["grid.color"] = "#EEEEEE"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5
    
    # Set tick properties
    plt.rcParams["xtick.color"] = "#666666"
    plt.rcParams["ytick.color"] = "#666666"
    
    return fig