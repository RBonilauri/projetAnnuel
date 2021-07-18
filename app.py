import streamlit as st
import random

def main():
    st.markdown("")
    st.title("PROJET ANNUEL")

    st.text("The purpose of this application is to define the type of a given planet.\n"
            "The answer can be 'telluric', 'gaseous' or 'other'.\n"
            "\nTo do so, copy and paste the link of the desired image in the text bar below.\n")

    form = st.form(key='planet_name')
    url = form.text_input('Paste image url')
    form.form_submit_button('Submit')

    if url != "":
        st.image(url, width=400)
        rand_int = random.Random()
        st.title("Result :")
        if rand_int == 1:
            st.text("It's a telluric planet !")
        else:
            st.text("It's a gaseous planet !")


if __name__ == "__main__":
    main()
