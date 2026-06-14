# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:30:15 2026

@author: KIIT
"""

import streamlit as st
from PIL import Image
logo=Image.open("Logo.png")
st.sidebar.image(logo,width=120)
st.sidebar.markdown("**Hello world app**")

st.title("Hello world!!")
st.write("welcome to my first streamlit App")