import streamlit as st
import pandas as pd
import uuid
import datetime

class ProjectManager:
    """Class to manage data analysis projects"""
    
    def __init__(self):
        """Initialize project manager"""
        # Ensure projects state exists
        if 'projects' not in st.session_state:
            st.session_state.projects = {}
            
        if 'current_project' not in st.session_state:
            st.session_state.current_project = None
    
    def create_project(self, project_name):
        """Create a new project with given name"""
        if not project_name:
            st.sidebar.error("Please enter a project name!")
            return False
            
        if project_name in st.session_state.projects:
            st.sidebar.error("A project with this name already exists!")
            return False
            
        # Create new project
        st.session_state.projects[project_name] = {
            'id': str(uuid.uuid4()),
            'name': project_name,
            'data': None,
            'created_at': datetime.datetime.now(),
            'last_modified': datetime.datetime.now(),
            'visualizations': [],
            'analyses': [],
            'chat_history': []
        }
        
        # Set as current project
        st.session_state.current_project = project_name
        
        # Reset dataframe
        st.session_state.df = None
        
        return True
    
    def delete_project(self, project_name):
        """Delete a project by name"""
        if project_name in st.session_state.projects:
            # Remove from projects dict
            del st.session_state.projects[project_name]
            
            # If current project was deleted, set to None
            if st.session_state.current_project == project_name:
                st.session_state.current_project = None
                st.session_state.df = None
                
            return True
        return False
    
    def get_project(self, project_name):
        """Get project data by name"""
        if project_name in st.session_state.projects:
            return st.session_state.projects[project_name]
        return None
    
    def get_all_projects(self):
        """Get all projects"""
        return st.session_state.projects
    
    def switch_project(self, project_name):
        """Switch to another project"""
        if project_name in st.session_state.projects:
            st.session_state.current_project = project_name
            
            # Load project data if it exists
            if st.session_state.projects[project_name]['data'] is not None:
                st.session_state.df = st.session_state.projects[project_name]['data']
            else:
                st.session_state.df = None
                
            return True
        return False
    
    def render_sidebar(self):
        """Render project management UI in sidebar"""
        st.sidebar.header("🗂️ Project Management")
        
        # Project creation
        with st.sidebar.expander("Create New Project", expanded=False):
            new_project = st.text_input("Project Name", key="new_project_name")
            create_btn = st.button("Create Project", use_container_width=True, type="primary")
            
            if create_btn and new_project:
                success = self.create_project(new_project)
                if success:
                    st.success(f"Project '{new_project}' created successfully!")
                    st.experimental_rerun()
        
        # Project selection
        if st.session_state.projects:
            st.sidebar.subheader("Your Projects")
            
            for project_name in st.session_state.projects:
                col1, col2, col3 = st.sidebar.columns([5, 1, 1])
                
                # Project is active
                is_active = project_name == st.session_state.current_project
                
                with col1:
                    if is_active:
                        st.markdown(f"**📂 {project_name}**")
                    else:
                        if st.button(f"📁 {project_name}", key=f"select_{project_name}", use_container_width=True):
                            self.switch_project(project_name)
                            st.experimental_rerun()
                
                with col2:
                    if not is_active and st.button("🔄", key=f"switch_{project_name}", help=f"Switch to {project_name}"):
                        self.switch_project(project_name)
                        st.experimental_rerun()
                
                with col3:
                    if st.button("🗑️", key=f"delete_{project_name}", help=f"Delete {project_name}"):
                        self.delete_project(project_name)
                        st.experimental_rerun()