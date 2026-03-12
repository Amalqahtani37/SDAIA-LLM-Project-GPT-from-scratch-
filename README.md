# Arabic Instruction GPT
## Project Overview

This project implements a simplified Arabic Instruction-Following GPT model using PyTorch.
The goal is to demonstrate the full pipeline of building and deploying a language model that can follow Arabic instructions.

The system takes an instruction and optional input text, then generates a response using a fine-tuned GPT-2 model.

Example:
<img width="1920" height="944" alt="image" src="https://github.com/user-attachments/assets/6d68b616-1ed3-4af0-b092-136f0d8c0140" />
<img width="1920" height="954" alt="image" src="https://github.com/user-attachments/assets/02182be6-0d2a-4cc2-bbb6-d84ee763485e" />

## Project Pipeline

<img width="488" height="687" alt="image" src="https://github.com/user-attachments/assets/39993cb2-b20e-44cd-880a-635c7f71beec" />

## Dataset

Two datasets were used in this project:

Instruction Dataset

Used for instruction fine-tuning.
File: 
arabic_instruction_dataset_clean_500.json

Example entry:
  {
    "instruction": "اكتب جملة قصيرة عن اللغة العربية باسلوب سهل",
    "input": "",
    "output": "اللغة العربية لغة جميلة وغنية بالتعبير."
  }
Pretraining Text Dataset

Used to expose the model to general Arabic language patterns.

File:
arabic_pretrain_data_clean_2000.txt
This dataset contains cleaned Arabic text.

Model
The system uses GPT-2 Small (124M parameters).

## Configuration:

Layers: 12
Embedding Size: 768
Attention Heads: 12
Context Length: 1024
Vocabulary: 50257

The model was loaded using pretrained GPT-2 weights and then fine-tuned on the Arabic instruction dataset.

## Implementation Based on M07

This project builds directly on Module M07: Instruction Fine-Tuning.

The following components were reused:

GPTModel architecture from previous_chapters.py
GPT-2 pretrained weights loader
instruction prompt template
autoregressive text generation
Instruction template used during training:

Below is an instruction that describes a task.

### Instruction:
{instruction}

### Input:
{input}

### Response:
Evaluation

Model performance was evaluated by comparing:
expected dataset responses
generated model outputs
Training and validation loss were also monitored during training.

<img width="1186" height="613" alt="image" src="https://github.com/user-attachments/assets/a095f32a-2bc5-46d9-ac82-c95e6c3d28b0" />
<img width="824" height="498" alt="image" src="https://github.com/user-attachments/assets/cd39a402-def2-4f39-b5b2-ca7d1597cefd" />

## Deployment
A Streamlit interface was created to interact with the model.
<img width="1891" height="449" alt="image" src="https://github.com/user-attachments/assets/33c3344c-d410-4909-b3e7-db4b0ace1181" />

Users can:
enter an instruction
provide input text
generate a response
The application was exposed using ngrok for external access.

## Technical Limitations

This project has several limitations:
small dataset size compared to real LLMs
GPT-2 tokenizer optimized mainly for English
limited training time
small model size (124M parameters)
These constraints affect the model’s ability to generalize.

## Conclusion

This project demonstrates how instruction-following language models are built using the GPT architecture.
<img width="644" height="386" alt="image" src="https://github.com/user-attachments/assets/34a2c4b0-0185-421e-a43d-8bd71f17728c" />


Although simplified, it provides practical insight into how modern LLM systems work.
