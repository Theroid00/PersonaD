import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class DMChatDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        for text in data:
            encodings_dict = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            self.input_ids.append(encodings_dict['input_ids'])
            self.attn_masks.append(encodings_dict['attention_mask'])
            
        self.input_ids = torch.cat(self.input_ids)
        self.attn_masks = torch.cat(self.attn_masks)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'labels': self.input_ids[idx]
        }

def load_dm_data(file_path):
    """Load and parse the DM chat JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        dm_data = json.load(f)
    
    return dm_data

def format_dm_conversations(dm_data):
    """Format DM chat data into dialogue format"""
    conversations = []
    current_conversation = []
    
    # Sort messages by timestamp
    for message in sorted(dm_data, key=lambda x: x.get('timestamp', 0)):
        sender = message.get('sender', 'Unknown')
        text = message.get('text', '')
        
        if not text:  # Skip empty messages
            continue
        
        # Format as User: or Friend: based on sender
        # For Instagram DMs, we assume messages from "user" are from the account owner
        # and any other sender is treated as "Friend"
        sender_tag = "User: " if sender.lower() == "user" else "Friend: "
        formatted_message = f"{sender_tag}{text}"
        current_conversation.append(formatted_message)
    
    # Join all messages into a single string with newlines
    if current_conversation:
        conversations.append("\n".join(current_conversation))
    
    return conversations

def train(model, train_dataloader, optimizer, scheduler, device, epochs=5):
    """Train the model"""
    model.train()
    total_loss = 0
    best_loss = float('inf')
    no_improvement_count = 0
    patience = 3  # early stopping patience
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss
        
        # Check for early stopping (especially useful for small datasets to prevent overfitting)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            model.save_pretrained(os.path.join("checkpoints", "best_model"))
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            # Load the best model before returning
            if os.path.exists(os.path.join("checkpoints", "best_model")):
                model = GPT2LMHeadModel.from_pretrained(os.path.join("checkpoints", "best_model")).to(device)
            break
    
    # If we completed all epochs, check if we should load the best model
    if no_improvement_count > 0 and os.path.exists(os.path.join("checkpoints", "best_model")):
        logger.info("Loading the best model checkpoint")
        model = GPT2LMHeadModel.from_pretrained(os.path.join("checkpoints", "best_model")).to(device)
    
    logger.info(f"Training completed. Average Loss across epochs: {total_loss/(epoch+1):.4f}")
    return model

def generate_response(model, tokenizer, prompt, device, max_length=100):
    """Generate a response using the fine-tuned model"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the newly generated text (after the prompt)
    generated_text = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    
    # If the generated text contains a "User:" marker, truncate at that point
    user_marker_index = generated_text.find("\nUser:")
    if user_marker_index != -1:
        generated_text = generated_text[:user_marker_index]
    
    return generated_text.strip()

def chat_with_model(model, tokenizer, device, exit_phrases=("quit", "exit", "bye")):
    """Interactive chat with the fine-tuned model"""
    logger.info("Starting chat with the model. Type 'quit', 'exit', or 'bye' to end the conversation.")
    logger.info("This model is trained on your Instagram DM history and will attempt to respond like your friend.")
    
    conversation_history = ""
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in exit_phrases:
            logger.info("Ending conversation.")
            break
        
        # Add user input to conversation history
        if not conversation_history:
            conversation_history = f"User: {user_input}\nFriend:"
        else:
            conversation_history += f"\nUser: {user_input}\nFriend:"
        
        # Generate model response
        response = generate_response(model, tokenizer, conversation_history, device)
        
        # Add response to conversation history and display
        conversation_history += f" {response}"
        print(f"Friend: {response}")
        
        # Keep conversation history from growing too large by keeping only the last 10 exchanges
        if conversation_history.count("User:") > 10:
            # Find the position of the second occurrence of "User:"
            start_pos = conversation_history.find("User:")
            start_pos = conversation_history.find("User:", start_pos + 1)
            conversation_history = conversation_history[start_pos:]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Instagram DM chat data")
    parser.add_argument("--data_file", type=str, default="dms.json", help="Path to DM chat JSON file")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--chat", action="store_true", help="Start chat after training")
    parser.add_argument("--augment", action="store_true", help="Augment small datasets with repetition")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading {args.model_name} model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    
    # GPT-2 tokenizer doesn't have a padding token, so we'll set it to the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Move model to device
    model.to(device)
    
    # Load and format the data
    logger.info(f"Loading data from {args.data_file}...")
    dm_data = load_dm_data(args.data_file)
    conversations = format_dm_conversations(dm_data)
    
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # For small datasets (like Instagram DMs), we might want to repeat the data
    # to get better training results
    if args.augment and len(conversations) < 10:
        logger.info("Small dataset detected. Augmenting data...")
        augmented_conversations = []
        # Repeat each conversation 5 times for better training
        for _ in range(5):
            augmented_conversations.extend(conversations)
        conversations = augmented_conversations
        logger.info(f"Augmented to {len(conversations)} training examples")
    
    # Create dataset and dataloader
    logger.info("Creating dataset and dataloader...")
    dataset = DMChatDataset(tokenizer, conversations, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train the model
    logger.info(f"Starting training for {args.epochs} epochs...")
    model = train(model, dataloader, optimizer, scheduler, device, epochs=args.epochs)
    
    # Save the model and tokenizer
    logger.info(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Model training and saving completed")
    
    # Start chat if requested
    if args.chat:
        chat_with_model(model, tokenizer, device)

if __name__ == "__main__":
    main()
