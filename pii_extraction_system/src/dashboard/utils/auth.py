"""
Authentication utilities for PII Extraction Dashboard

This module provides user authentication and role-based access control
for the dashboard application.
"""

import streamlit as st
import hashlib
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Simple user database (in production, use proper database)
USERS_DB = {
    "admin": {
        "password_hash": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
        "role": "admin",
        "permissions": ["read", "write", "delete", "configure"]
    },
    "analyst": {
        "password_hash": "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4",  # hello
        "role": "analyst", 
        "permissions": ["read", "write"]
    },
    "viewer": {
        "password_hash": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",  # viewer
        "role": "viewer",
        "permissions": ["read"]
    }
}

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict]]:
    """Authenticate user credentials"""
    if username not in USERS_DB:
        return False, None
    
    user_data = USERS_DB[username]
    if verify_password(password, user_data["password_hash"]):
        return True, {
            "username": username,
            "role": user_data["role"],
            "permissions": user_data["permissions"],
            "login_time": datetime.now()
        }
    
    return False, None

def check_authentication() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def has_permission(permission: str) -> bool:
    """Check if current user has specific permission"""
    if not check_authentication():
        return False
    
    user_permissions = st.session_state.get('user_permissions', [])
    return permission in user_permissions

def require_permission(permission: str) -> bool:
    """Decorator-like function to require specific permission"""
    if not has_permission(permission):
        st.error(f"Access denied. Required permission: {permission}")
        st.stop()
    return True

def show_login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🔒 PII Extraction System</h1>
        <h3>Please log in to continue</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show available credentials for development
    st.info("""
    **Available Login Credentials:**
    - **Admin**: `admin` / `admin` (Full access)
    - **Analyst**: `analyst` / `hello` (Read/Write access)  
    - **Viewer**: `viewer` / `viewer` (Read-only access)
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    success, user_data = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = user_data["username"]
                        st.session_state.user_role = user_data["role"]
                        st.session_state.user_permissions = user_data["permissions"]
                        st.session_state.login_time = user_data["login_time"]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        # Demo credentials info
        with st.expander("Demo Credentials"):
            st.markdown("""
            **Admin User:**
            - Username: admin
            - Password: admin
            
            **Analyst User:**
            - Username: analyst  
            - Password: hello
            
            **Viewer User:**
            - Username: viewer
            - Password: viewer
            """)

def logout():
    """Log out current user"""
    # Clear authentication state
    auth_keys = ['authenticated', 'username', 'user_role', 'user_permissions', 'login_time']
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("Logged out successfully!")

def get_current_user() -> Optional[Dict]:
    """Get current user information"""
    if not check_authentication():
        return None
    
    return {
        "username": st.session_state.get('username'),
        "role": st.session_state.get('user_role'),
        "permissions": st.session_state.get('user_permissions', []),
        "login_time": st.session_state.get('login_time')
    }

def check_session_timeout(timeout_hours: int = 8) -> bool:
    """Check if user session has timed out"""
    if not check_authentication():
        return True
    
    login_time = st.session_state.get('login_time')
    if not login_time:
        return True
    
    time_since_login = datetime.now() - login_time
    return time_since_login > timedelta(hours=timeout_hours)

def role_required(required_role: str):
    """Check if user has required role"""
    if not check_authentication():
        st.error("Authentication required")
        st.stop()
    
    user_role = st.session_state.get('user_role')
    role_hierarchy = {'viewer': 1, 'analyst': 2, 'admin': 3}
    
    if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
        st.error(f"Access denied. Required role: {required_role}")
        st.stop()
    
    return True