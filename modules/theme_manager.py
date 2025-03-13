import streamlit as st

class ThemeManager:
    """Class for managing application themes and styling"""
    
    def __init__(self):
        """Initialize theme manager"""
        # Initialize theme settings in session state if not present
        if 'theme' not in st.session_state:
            st.session_state.theme = "Light"
    
    def apply_theme(self):
        """Apply the current theme to the application"""
        if st.session_state.theme == "Dark":
            self._apply_dark_theme()
        else:
            self._apply_light_theme()
    
    def _apply_light_theme(self):
        """Apply light theme styling"""
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #31333F;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 15px;
            border-radius: 4px 4px 0px 0px;
            background-color: #f8f9fa;
        }
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #ffffff;
        }
        .data-info {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_dark_theme(self):
        """Apply dark theme styling"""
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #F0F2F6;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1E1E1E;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 15px;
            border-radius: 4px 4px 0px 0px;
            background-color: #262730;
        }
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #0E1117;
        }
        .data-info {
            background-color: #262730;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_theme_selector(self):
        """Render a theme selector widget"""
        theme = st.selectbox(
            "Theme:",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key="theme_selector"
        )
        
        # Update theme if changed
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()