import streamlit as st
import io
import numpy as np
from PIL import Image
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

# Fonction pour prédire l'état des feuilles de manioc
def predict(image_file):
    # Charger l'image en utilisant Pillow (PIL)
    image = Image.open(image_file)
    image = np.array(image)

    # Redimensionner l'image à la taille attendue par le modèle (224x224)
    image = np.array(Image.fromarray(image).resize((224, 224)))

    # Prétraiter l'image (par exemple, normalisation)
    image = image / 255.0  # Normalisation (vous pouvez adapter le prétraitement en fonction de votre modèle)

    # Effectuer la prédiction
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Obtenir la classe prédite
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class]

    return predicted_class, predicted_class_name

# Titre de l'application
st.title("Prédiction de l'état des feuilles de manioc")

# Description de l'application
st.markdown("Cette application utilise un modèle d'apprentissage profond pour prédire l'état des feuilles de manioc.")

# Charger l'image depuis l'ordinateur local
file_to_predict = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
if file_to_predict is not None:
    # Obtenez les prédictions
    predicted_class, predicted_class_name = predict(file_to_predict)

    if predicted_class is not None:
        st.image(file_to_predict, caption="Image chargée", use_column_width=True)

        # Affichez la classe prédite
        st.write(f"Classe prédite : {predicted_class} - {predicted_class_name}")
    else:
        st.write("Erreur de chargement de l'image.")
else:
    st.write("Veuillez télécharger une image.")
