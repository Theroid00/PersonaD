# GPT-2 Fine-Tuning for Instagram DM Chat History

This script fine-tunes a Hugging Face Transformers GPT-2 model using Instagram DM chat history stored in a JSON file.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- tqdm

Install the required packages using:

```
pip install -r requirements.txt
```

## Data Format

The script expects a JSON file with messages in the following format:

```json
[
  {
    "sender": "user",
    "text": "Message text",
    "timestamp": 1625097600
  },
  {
    "sender": "friend",
    "text": "Reply text",
    "timestamp": 1625097660
  },
  ...
]
```

Where:
- `sender`: Can be "user" (account owner) or any other value (treated as "friend")
- `text`: The message content
- `timestamp`: Unix timestamp of the message

## Usage

### Basic Usage

```
python fine_tune_gpt2.py --data_file dms.json
```

### For Small Datasets

If you have a small dataset (like most Instagram DM conversations), use the augment flag:

```
python fine_tune_gpt2.py --data_file dms.json --augment --epochs 10
```

This repeats your data several times to improve training and increases epochs.

### Interactive Chat

To start an interactive chat session after training:

```
python fine_tune_gpt2.py --data_file dms.json --augment --chat
```

### All Options

- `--data_file`: Path to DM chat JSON file (default: "dms.json")
- `--model_name`: Model name (default: "gpt2", options: "gpt2", "gpt2-medium", "gpt2-large")
- `--output_dir`: Directory to save the fine-tuned model (default: "fine_tuned_model")
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 2)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--chat`: Start chat after training (flag)
- `--augment`: Augment small datasets with repetition (flag, recommended for Instagram DMs)

## Recommended Settings for Instagram DMs

Since Instagram DM datasets are typically small, use these settings for best results:

```
python fine_tune_gpt2.py --model_name gpt2-medium --epochs 10 --batch_size 2 --learning_rate 3e-5 --augment --chat
```

## Loading a Saved Model

To chat with a previously saved model:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_path = "fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start a conversation
prompt = "User: How are you?\nFriend:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

# Generate response
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=100 + len(input_ids[0]),
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

## Notes

- For Instagram DMs, the script assumes messages from "user" are from the account owner
- The script works best with more data - if possible, export more of your chat history
- For better results with small datasets, the `--augment` flag is recommended
- Consider running for more epochs (10-15) with small datasets
- The model will learn the conversation style of your friend based on the messages
