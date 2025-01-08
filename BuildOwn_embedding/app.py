from Odiya_Tokenizer import Tokenizer_BPE
import pickle
import gradio as gr

# Define the TextProcessor class
class TextProcessor:
    def __init__(self, tokenizer_path):
        self.loaded_tokenizer = Tokenizer_BPE.load(tokenizer_path)
        
    def encode(self, example_text):
        encoded_text = self.loaded_tokenizer.encode(example_text)
        return str(encoded_text)  # Convert to string for Gradio output

    def decode(self, encoded_text):
        decoded_text = self.loaded_tokenizer.decode(eval(encoded_text))  # Convert the input string back to a list
        return decoded_text

# Instantiate the TextProcessor with the tokenizer path
tokenizer = TextProcessor("odiya_tokenizer.pkl")

# Define the Gradio app layout
def Odiya_tokenizer_app():
    
    with gr.Blocks(css="""
        #encode-header, #decode-header {
            font-size: 22px;
            font-weight: bold;
            color: #2D87D6;
            text-align: center;
        }
        #input-textbox, #token-input {
            border-radius: 10px;
            border: 2px solid #2D87D6;
            background-color: #E9F2FB;
            padding: 12px;
            margin-bottom: 10px;
            font-size: 16px;
            width: 100%;
        }
        #encoded-output, #decoded-output {
            border-radius: 10px;
            border: 2px solid #2D87D6;
            background-color: #E9F2FB;
            padding: 12px;
            font-size: 16px;
            width: 100%;
        }
        #encode-btn, #decode-btn {
            background-color: #2D87D6;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            border: none;
            padding: 12px;
            width: 100%;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        #encode-btn:hover, #decode-btn:hover {
            background-color: #1C6BB2;
        }
        .gr-button {
            margin-top: 15px;
        }
    """) as app:
        gr.Markdown(
        """
        <h1 style="text-align: center; font-size: 2.5em;">ଏହା ଏକ ଓଡିଆ ଟୋକେନାଇଜର୍ ଆପ୍| {This is a Odiya tokenizer app} Copy text in Encoder to see the Tokens.</h1>
        <p>Odiya Tokenizer (BPE Encoding and Decoding)</p>
        """,
        elem_id="title"
        )
        
        with gr.Row():
            # Left Column: Encode Text
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### **Encode Text**", elem_id="encode-header")
                input_text = gr.Textbox(
                    label="Enter Odiya Text", 
                    lines=10, 
                    placeholder="ଆମେ ସମସ୍ତେ ଭାରତୀୟ। କିନ୍ତୁ ଆମେ ପ୍ରଥମ ମଣିଷ |",
                    elem_id="input-textbox"
                )
                encode_button = gr.Button("Encode", elem_id="encode-btn")
                encoded_output = gr.Textbox(
                    label="Encoded Tokens", 
                    lines=10, 
                    interactive=False, 
                    placeholder="Encoded tokens will appear here.",
                    elem_id="encoded-output"
                )

            # Right Column: Decode Tokens
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### **Decode Tokens**", elem_id="decode-header")
                token_input = gr.Textbox(
                    label="Enter Encoded Tokens (comma-separated)", 
                    lines=10, 
                    placeholder="Example: [256, 474, 4786, 1501, 763, 607, 3672, 474, 4707, 300, 1858, 1326]",
                    elem_id="token-input"
                )
                decode_button = gr.Button("Decode", elem_id="decode-btn")
                decoded_output = gr.Textbox(
                    label="Decoded Text", 
                    lines=10, 
                    interactive=False, 
                    placeholder="Decoded text will appear here.",
                    elem_id="decoded-output"
                )

        # Function calls when buttons are clicked
        encode_button.click(fn=tokenizer.encode, inputs=input_text, outputs=encoded_output)
        decode_button.click(fn=tokenizer.decode, inputs=token_input, outputs=decoded_output)

    return app

# Running the app
app = Odiya_tokenizer_app()
app.launch()