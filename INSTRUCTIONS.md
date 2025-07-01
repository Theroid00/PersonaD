# Instagram Chat GPT-2 Fine-Tuning Instructions

## Error Resolution

You encountered an error with the `AdamW` optimizer in the original script:

```
ImportError: cannot import name 'AdamW' from 'transformers'
```

This happened because in newer versions of the Transformers library, `AdamW` has been moved to `torch.optim`. I've created two updated scripts:

1. `fine_tune_gpt2_updated.py` - Fixed version of the original script
2. `instagram_chat_fine_tune.py` - New script specifically for Instagram's exported chat format

## How to Use the Instagram Export Script

The `instagram_chat_fine_tune.py` script is designed to work with the Instagram DM export format you have in `message_1.json`, `message_2.json`, and `message_3.json`.

### Basic Usage

```powershell
python instagram_chat_fine_tune.py --data_file message_1.json
```

### Recommended Settings

For optimal results with Instagram export data:

```powershell
python instagram_chat_fine_tune.py --data_file message_1.json --augment --epochs 10 --model_name gpt2-medium --batch_size 2
```

### Training on Multiple Files

If you want to train on multiple message files, you'll need to combine them first (the script can currently only process one file at a time).

### Interactive Chat After Training

```powershell
python instagram_chat_fine_tune.py --data_file message_1.json --augment --chat
```

### All Available Options

- `--data_file`: Path to Instagram exported JSON file (default: "message_1.json")
- `--model_name`: Model name (default: "gpt2", options: "gpt2", "gpt2-medium", "gpt2-large")
- `--output_dir`: Directory to save the fine-tuned model (default: "fine_tuned_model")
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 2)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--chat`: Start chat after training (flag)
- `--chunk_size`: Number of messages per conversation chunk (default: 20)
- `--augment`: Augment training data with repetition (flag, recommended)

## How It Works

The script:

1. Loads the Instagram JSON export format, which contains participants and messages
2. Splits the chat history into chunks of 20 messages (customizable)
3. Formats each chunk as a dialogue with "User:" and "Friend:" prefixes
4. Tokenizes and trains a GPT-2 model on this data
5. Saves the fine-tuned model for future use

## Tips for Best Results

1. Use the `--augment` flag, especially if you have less than 1000 messages
2. Try the medium-sized GPT-2 model (`--model_name gpt2-medium`) for better quality
3. Train for more epochs (10-15) with small datasets
4. Use a smaller chunk size (10-15) if messages are very long
5. For large datasets, increase batch size to 4 or 8 if you have enough memory

## Chatting with a Saved Model

After training, you can chat with your model anytime using:

```powershell
python chat_with_model.py --model_dir fine_tuned_model
```
