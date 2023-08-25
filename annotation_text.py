import streamlit as st
from annotated_text import annotated_text
import pickle


with open('display_text.pickle', 'rb') as f:
    display_text = pickle.load(f)
    
annotated_text(*display_text)