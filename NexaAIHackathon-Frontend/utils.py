# utils.py

import pandas as pd
from io import BytesIO
import streamlit as st
import pandas as pd
from io import BytesIO

def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV format."""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_json(df):
    """Converts a DataFrame to a JSON format."""
    return df.to_json(orient='records').encode('utf-8')

def load_css(file_name):
    """Loads a CSS file into the Streamlit application."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def convert_df_to_excel(df):
    """Converts a DataFrame to an Excel format."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        # The writer is automatically saved and closed when the with block is exited

    processed_data = output.getvalue()
    return processed_data
