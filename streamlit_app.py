import streamlit as st
pages = {
    "Models":[
        st.Page("diabetes.py",title='Diabetes Model'),
        st.Page("heart.py",title='Heart Model')
    ],
    'About':[
        st.Page("about.py",title='About')
    ]
}
pg = st.navigation(pages)
pg.run()