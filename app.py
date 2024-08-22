# Import neccessary libraries
import spaces
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig, AutoProcessor
import gradio as gr
from threading import Thread
from PIL import Image
import subprocess

# Install flash-attn if not already installed
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Model and tokenizer for the chatbot
MODEL_ID1 = "microsoft/Phi-3.5-mini-instruct"
MODEL_LIST1 = ["microsoft/Phi-3.5-mini-instruct"]
HF_TOKEN = os.environ.get("HF_TOKEN", None)

device = "cuda" if torch.cuda.is_available() else "cpu"  # for GPU usage or "cpu" for CPU usage / But you need GPU :)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID1)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID1,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config)

# Chatbot tab function
@spaces.GPU()
def stream_chat(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    top_k: int = 20,
    penalty: float = 1.2,
):
    print(f'message: {message}')
    print(f'history: {history}')

    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])

    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens = max_new_tokens,
        do_sample = False if temperature == 0 else True,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        eos_token_id=[128001,128008,128009],
        streamer=streamer,
    )

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

# Vision model setup
models = {
    "microsoft/Phi-3.5-vision-instruct": AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True, torch_dtype="auto", _attn_implementation="flash_attention_2").cuda().eval()
}

processors = {
    "microsoft/Phi-3.5-vision-instruct": AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
}

user_prompt = '\n'
assistant_prompt = '\n'
prompt_suffix = "\n"

# Vision model tab function
@spaces.GPU()
def stream_vision(image, text_input=None, model_id="microsoft/Phi-3.5-vision-instruct"):
    model = models[model_id]
    processor = processors[model_id]

    # Prepare the image list and corresponding tags
    images = [Image.fromarray(image).convert("RGB")]
    placeholder = "<|image_1|>\n"  # Using the image tag as per the example

    # Construct the prompt with the image tag and the user's text input
    if text_input:
        prompt_content = placeholder + text_input
    else:
        prompt_content = placeholder

    messages = [
        {"role": "user", "content": prompt_content},
    ]

    # Apply the chat template to the messages
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the inputs with the processor
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    # Generation parameters
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate the response
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove input tokens from the generated response
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

    # Decode the generated output
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response

# CSS for the interface
CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""

PLACEHOLDER = """
<center>
<p>Hi! I'm your assistant. Feel free to ask your questions</p>
</center>
"""

TITLE = "<h1><center>Phi-3.5 Chatbot & Phi-3.5 Vision</center></h1>"

EXPLANATION = """
<div style="text-align: center; margin-top: 20px;">
    <p>This app supports both the microsoft/Phi-3.5-mini-instruct model for chat bot and the microsoft/Phi-3.5-vision-instruct model for multimodal model.</p>
    <p>Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.</p>
    <p>Phi-3.5-mini is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data. The model belongs to the Phi-3 model family and supports 128K token context length. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.</p>
</div>
"""

footer = """
<div style="text-align: center; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a> |
    <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct" target="_blank">microsoft/Phi-3.5-mini-instruct</a> |
    <a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct" target="_blank">microsoft/Phi-3.5-vision-instruct</a>
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

# Gradio app with two tabs
with gr.Blocks(css=CSS, theme="small_and_pretty") as demo:
    gr.HTML(TITLE)
    gr.HTML(EXPLANATION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot(height=600, placeholder=PLACEHOLDER)
        gr.ChatInterface(
            fn=stream_chat,
            chatbot=chatbot,
            fill_height=True,
            additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
            additional_inputs=[
                gr.Textbox(
                    value="You are a helpful assistant",
                    label="System Prompt",
                    render=False,
                ),
                gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.8,
                    label="Temperature",
                    render=False,
                ),
                gr.Slider(
                    minimum=128,
                    maximum=8192,
                    step=1,
                    value=1024,
                    label="Max new tokens",
                    render=False,
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                    label="top_p",
                    render=False,
                ),
                gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=20,
                    label="top_k",
                    render=False,
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.2,
                    label="Repetition penalty",
                    render=False,
                ),
            ],
            examples=[
                ["How to make a self-driving car?"],
                ["Give me a creative idea to establish a startup"],
                ["How can I improve my programming skills?"],
                ["Show me a code snippet of a website's sticky header in CSS and JavaScript."],
            ],
            cache_examples=False,
        )
    with gr.Tab("Vision"):
        with gr.Row():
            input_img = gr.Image(label="Input Picture")
        with gr.Row():
            model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value="microsoft/Phi-3.5-vision-instruct")
        with gr.Row():
            text_input = gr.Textbox(label="Question")
        with gr.Row():
            submit_btn = gr.Button(value="Submit")
        with gr.Row():
            output_text = gr.Textbox(label="Output Text")

        submit_btn.click(stream_vision, [input_img, text_input, model_selector], [output_text])

    gr.HTML(footer)

# Launch the combined app
demo.launch(debug=True)