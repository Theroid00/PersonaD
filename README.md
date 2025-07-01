# GPT-2 Fine-Tuning for Instagram DM Chat History

This project fine-tunes a Hugging Face Transformers GPT-2 model using Instagram DM chat history to create a personalized chatbot that mimics a specific person's conversation style.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- tqdm

Install the required packages using:

```
pip install -r requirements.txt
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Theroid00/loml1.git
cd loml1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
This project uses Instagram chat export data in JSON format. You need to:

1. Request your Instagram data: 
   - Go to your Instagram profile
   - Settings > Privacy and Security > Data Download
   - Request Download > JSON format

2. Place your JSON files in the project directory (rename them to message_1.json, message_2.json, etc.)

### 4. Train Your Model
```bash
python clean_and_finetune.py --data_files message_1.json message_2.json --epochs 5 --augment
```

The trained model will be saved in the `fine_tuned_model` directory.

### 5. Chat With Your Model
```bash
python chat_with_model.py
```

## Data Format

The script expects Instagram export JSON files with messages in the following format (message_*.json):

```json
{
  "participants": [
    {"name": "User Name"},
    {"name": "Friend Name"}
  ],
  "messages": [
    {
      "sender_name": "User Name",
      "timestamp_ms": 1625097600000,
      "content": "Message text"
    },
    {
      "sender_name": "Friend Name",
      "timestamp_ms": 1625097660000,
      "content": "Reply text"
    }
  ]
}
```

## Command Line Options

### Fine-Tuning Script (clean_and_finetune.py)

- `--data_files`: Paths to Instagram exported JSON files
- `--model_name`: Model name (default: "gpt2", options: "gpt2", "gpt2-medium")
- `--output_dir`: Directory to save the fine-tuned model (default: "fine_tuned_model")
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 1)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 256)
- `--chat`: Start chat after training (flag)
- `--chunk_size`: Number of messages per conversation chunk (default: 16)
- `--augment`: Augment small datasets with repetition (flag, recommended for Instagram DMs)
- `--force-cpu`: Force using CPU even if CUDA is available
- `--max_chunks`: Maximum number of chunks to create (for testing)

### Chat Script (chat_with_model.py)

```bash
python chat_with_model.py --model_dir fine_tuned_model
```

## Recommended Settings for Instagram DMs

Since Instagram DM datasets are typically small, use these settings for best results:

```bash
python clean_and_finetune.py --model_name gpt2 --epochs 10 --batch_size 1 --learning_rate 3e-5 --augment
```

## Model Too Large for GitHub

The trained model files (*.safetensors) are not included in this repository due to GitHub's 100MB file size limitations. You'll need to train your own model using the instructions above, or download a pre-trained model from an external source.

## Using the Model in Other Applications

To use your fine-tuned model in other applications:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate a response
def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(input_ids[0]),
        temperature=1.0,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()

# Example usage
conversation = "User: How are you today?\nFriend:"
response = generate_response(conversation)
print(response)
```

## Notes

- The model will learn the conversation style of your friend based on the messages
- For better results with small datasets, the `--augment` flag is recommended
- Consider running for more epochs (10-15) with small datasets
- The quality of the generated responses depends on the amount and quality of training data
