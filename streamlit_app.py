import streamlit as st
st.title("EarlyDetect")
pages = {
    "Models": [
        st.Page("diabetes.py", title="Diabetes Model"),
        st.Page("heart.py", title="Cardiovascular Disease Model"),
    ],
    "About": [st.Page("about.py", title="About")],
}
pg = st.navigation(pages)
pg.run()
