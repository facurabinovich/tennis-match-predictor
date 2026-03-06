import mysql.connector
import streamlit as st

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            port=int(st.secrets["DB_PORT"]),
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"]
        )
        return connection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None