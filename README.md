# gradioMed


### MedGemma 27B Chat UI on Google Colab

A quick-start Colab notebook and Gradio-based chat interface to interact with Google’s MedGemma 27B medical assistant model (text-only instruction-tuned).

> **Note**: This is intended for research and prototyping purposes only. **Do not** use for clinical diagnosis without appropriate regulatory approval and human oversight.

---

## 📋 Features

- **4-bit quantized** MedGemma 27B (text-only) running on a single GPU (≥ 16 GB VRAM).
- Optional **full-precision** load (requires ≥ 40 GB VRAM, e.g. A100/V100).
- Lightweight, browser-based chat UI via [Gradio](https://gradio.app/).
- Simple Colab integration — no local setup required.
- History tracking and auto-scrolling chatbox.

---

## 🚀 Quick Start

1. **Open the Colab notebook**  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/…/MedGemma-27B-Chat-UI.ipynb)

2. **Install dependencies**  
   ```bash
   !pip install -q transformers accelerate bitsandbytes gradio
    ````

3. **Load & Quantize the Model**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   import torch

   model_id = "google/medgemma-27b-text-it"
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
   )

   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       quantization_config=bnb_config,
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   ```

4. **Define Chat Function**

   ```python
   import gradio as gr

   def chat_with_medgemma(user_message, history):
       messages = [{"role":"system", "content":"You are a helpful medical assistant."}]
       for u, a in history:
           messages += [{"role":"user","content":u}, {"role":"assistant","content":a}]
       messages.append({"role":"user", "content": user_message})

       inputs = tokenizer.apply_chat_template(
           messages, add_generation_prompt=True, tokenize=True,
           return_tensors="pt"
       ).to(model.device)
       input_len = inputs["input_ids"].shape[-1]

       with torch.inference_mode():
           gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)
       reply = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)

       history.append((user_message, reply))
       return history, history
   ```

5. **Launch Gradio UI**

   ```python
   with gr.Blocks() as demo:
       chatbot = gr.Chatbot()
       user_input = gr.Textbox(placeholder="Type your message…")
       state = gr.State([])

       user_input.submit(
           chat_with_medgemma,
           inputs=[user_input, state],
           outputs=[chatbot, state]
       )

   demo.launch(server_name="0.0.0.0", share=True)
   ```

6. **Chat!**

   * Click the generated link or “Open in new tab” to start your MedGemma conversation.

---

## 🛠️ Requirements

* **Python** ≥ 3.8
* **GPU**:

  * **Quantized (4-bit)**: ≥ 16 GB VRAM (e.g. T4, RTX 3090, A100)
  * **Full-Precision**: ≥ 40 GB VRAM (e.g. A100, V100)
* **Colab**: Free/Pro for 4-bit; Enterprise for full precision
* **Internet**: to download model weights from Hugging Face

---

## ⚙️ Configuration

* **Model ID**

  * Default: `google/medgemma-27b-text-it` (instruction-tuned text-only)
  * For multimodal (image+text), use the appropriate MedGemma multimodal checkpoint when available.

* **Quantization**

  * Tweak `BitsAndBytesConfig` parameters for alternative quant types (`"fp4"`, `"fp8"`, etc.).

* **Max Tokens**

  * Adjust `max_new_tokens` in `model.generate()` per your use case.

---

## 📝 File Structure

```
.
├── MedGemma-27B-Chat-UI.ipynb    # Colab notebook
└── README.md                    # This file
```

---

## 🤝 Contributing

Feel free to open issues or pull requests for:

* Supporting multimodal MedGemma checkpoints
* Integrating image-to-text pipeline (TorchVision + transformers)
* UI enhancements (themes, auto-scroll, streaming)
* Adding rate-limiting, API endpoints, or containerization

---

## 📜 License

This project is provided under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
MedGemma model weights and licenses are governed by Google’s model release terms.
