import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load the model
model = tf.keras.models.load_model("mobilenetv2_oculus.h5")

# Load class labels (you must save them during training)
# Example: ['normal', 'glaucoma', 'cataract', 'retina_disease']
class_labels = ['cataract', 'Diabetic Retenopathy', 'Glucouma', 'Normal']  # Replace with your actual labels

def predict(image):
    try:
        # Preprocess
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        print("RAW PREDICTION:", prediction)

        # Get index of highest probability
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]
        confidence = prediction[0][predicted_index]

        return f"Prediction: {predicted_label} ({confidence*100:.2f}%)"

    except Exception as e:
        return f"Error: {str(e)}"
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; font-size: 36px;'>OCULUS AI</h1>")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Fundus Image")
        output_label = gr.Textbox(label="Predicted Disease")

    flag_button = gr.Button("Flag")
    image_input.change(fn=predict, inputs=image_input, outputs=output_label)

# Launch
gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text").launch()
