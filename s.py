import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model (.h5)
model = tf.keras.models.load_model("mobilenetv2_oculus.h5")

# Make sure to match this with the labels your model was trained on!
class_labels = ['cataract', 'Diabetic Retenopathy', 'Glucouma', 'Normal']

# Prediction function
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
        return f"Error: {e}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>OCULUS AI</h1>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Fundus Image")
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            output = gr.Textbox(label="Predicted Condition")

    submit_btn.click(fn=predict, inputs=image_input, outputs=output)
    clear_btn.click(fn=lambda: (None, ""), inputs=[], outputs=[image_input, output])

demo.launch()
