import json

import streamlit as st
import random
import requests

def affichage():
    serveur = "http://127.0.0.1:5000/"
    st.set_page_config(layout="wide")
    st.title("Projet Machine Learning 3IABD")

    # La bare d'information
    extend_bar = st.beta_expander("Information")
    extend_bar.markdown("""
    - **Description du projet** : Dans le cadre du Projet Annuel, nous avions réaliser une application permettant de classer les planètes selons leur types.
   Pour cela, nous avions développer des modèles d'IA ayant pour but de classer des images selon le type de la planète affiché.
   
    - **Développeurs** : Rafaël Bonilauri, Tsashua Bowe Tumawe, Toky Cedric Andriamahefa
    - **Technologie** : C++, Python, Flask, Streamlit
    - **Base de données** : MongoDB
    - **Source** : Cryptonaute, GoogleNews, CoinMarketCap
    """)

    # Création de la colonne de paramétrage à gauche
    side = st.sidebar

    side.header("Paramètres : ")
    mode = ["Planète ?", "Tellurique ou Gazeuse ?"]

    type_of_model = side.selectbox("Choisissez le type de classification à faire", mode)

    if type_of_model == "Planète ?":
        side.write("Vous avez choisis de Classer l'image en : Planète ou non")
    else:
        side.write("Vous avez choisis de Classer l'image en : Planète tellurique ou gazeuse ?")

    list_model = ["testPlaneteModel2","autres model"]

    choix_model = side.selectbox("Choisissez le modele à utiliser : ", list_model)

    # st.text("The purpose of this application is to define the type of a given planet.\n"
    #         "The answer can be 'telluric', 'gaseous' or 'other'.\n"
    #         "\nTo do so, copy and paste the link of the desired image in the text bar below.\n")


    form = st.form(key='planet_name')
    url = form.text_input('Paste image url')
    form.form_submit_button('Submit')
    st.markdown("""---""")
    middle_page, right_side = st.beta_columns((1, 1))
    if url != "":
        middle_page.markdown("""
        ## L'image que vous avez ajouté : 
        """)
        middle_page.image(url, width=400)
        rand_int = random.Random()
        middle_page.markdown("""
        ### Resultat :
        """)
        request = {
            "model": json.dumps(choix_model),
            "url": json.dumps(url)
        }

        req_path = serveur + "predict"

        response = requests.get(req_path, params=request)
        result = response.json()

        if response.status_code == 200:
            result = result["value"]
            result = result[1:len(result)-1]
            result = float(result)

            if result > 0:
                resultat = "une image de planète"
            else:
                resultat = "tout sauf une image de planète"
            middle_page.markdown(f"""
            Votre image est  **{resultat}** \n
            Le modèle donne un score de : **{result}**
            """)
            print(response.status_code)
        else:
            print(f"problem, status code : {response.status_code}")
    else:
        middle_page.write("Insert an URL please")


    right_side.markdown(f"""
            ## Information sur le modèle choisi
            Nom du modèle : {choix_model}
        """)


if __name__ == '__main__':
    affichage()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
