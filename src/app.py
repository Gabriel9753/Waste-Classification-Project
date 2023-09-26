import gradio as gr
from utils import load_specific_model, inference
import markdown

current_model = None  # Initialize the current model as None

# Define a set of example images
example_images = [
    ("Beispielbild Glas", "src/examples/Glas.jpg"),
    ("Beispielbild Organic", "src/examples/Organic.jpg"),
    ("Beispielbild Papier", "src/examples/Papier.jpg"),
    ("Beispielbild Restmüll", "src/examples/Restmuell.jpg"),
    ("Beispielbild Wertstoff", "src/examples/Wertstoff.jpg")
]

def load_model(model_name):
    global current_model
    if model_name is None:
        raise gr.Error("No model selected!")
    if current_model is not None:
        current_model = None

    current_model = load_specific_model(model_name)
    current_model.eval()

def predict(inp):
    global current_model
    if current_model is None:
        raise gr.Error("No model loaded!")

    confidences = inference(current_model, inp)
    return confidences

with gr.Blocks() as demo:
    with open('src/app_template.md', 'r') as f:
        markdown_string = f.read()
    header = gr.Markdown(markdown_string)
    
    with gr.Row(variant="panel", equal_height=True):

        user_image = gr.Image(
            type="pil",
            label="Upload Your Own Image",
            info="You can also upload your own image for prediction.",
            scale=2,
            height=350,
        )
        
        with gr.Column():
            output = gr.Label(
                num_top_classes=3,
                label="Output",
                info="Top three predicted classes and their confidences.",
                scale=2,
            )

            model_dropdown = gr.Dropdown(
                ["EfficientNet-B3", "EfficientNet-B4", "vgg19", "resnet50", "dinov2_vits14"],
                label="Model",
                info="Select a model to use.",
                scale=1,
            )
            model_dropdown.change(load_model, model_dropdown, show_progress=True, queue=True)
            predict_button = gr.Button(label="Predict", info="Click to make a prediction.", scale=1)
            predict_button.click(fn=predict, inputs=user_image, outputs=output, queue=True)

    gr.Markdown("## Example Images")
    gr.Markdown("You can just drag and drop these images into the image uploader above!")
    
    with gr.Row():
        for name, image_path in example_images:
            example_image = gr.Image(
                value=image_path,
                label=name,
                type="pil",
                height=220,
                interactive=False,
            )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
