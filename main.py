import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Charger le modèle
model = load_model('./model/cassava.h5')

class_names = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

st.title("Prediction de l'etat des feuilles de manioc cassava")

# Charger l'image depuis l'ordinateur local
file_to_predict = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
if file_to_predict is not None:
    image_bytes = file_to_predict.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_to_predict = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_to_predict is not None:
        st.image(image_to_predict, caption="Image chargée", use_column_width=True)
        
        # Redimensionnez l'image à la taille attendue par le modèle (224x224)
        target_size = (224, 224)
        image_to_predict = cv2.resize(image_to_predict, target_size)
        
        # Normalisez l'image si nécessaire
        image_to_predict = image_to_predict / 255.0
        
        # Utilisez le modèle pour obtenir les prédictions
        img_to_predict = np.expand_dims(image_to_predict, axis=0)
        predictions = model.predict(img_to_predict)
        
        # Obtenez la classe prédite (indice de la classe)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names.get(predicted_class, "Classe inconnue")
        
        # Affichez la classe prédite
        st.write(f"Predicted Class: {predicted_class} - {predicted_class_name}")
        
        # Affichez les probabilités pour chaque classe
        st.write("Probabilités :", predictions)
    else:
        st.write("Erreur de chargement de l'image.")

