import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
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

def load_instagram_export_data(file_paths):
    """Load and parse multiple Instagram exported chat JSON files"""
    all_messages = []
    participants_info = None
    
    for file_path in file_paths:
        logger.info(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Save participants info from the first file
            if participants_info is None:
                participants_info = data.get("participants", [])
            
            # Extract messages
            messages = data.get("messages", [])
            logger.info(f"Found {len(messages)} messages in {file_path}")
            all_messages.extend(messages)
    
    logger.info(f"Total messages loaded: {len(all_messages)}")
    
    # Return a dict with the same structure as a single file
    return {"participants": participants_info, "messages": all_messages}

def filter_text_messages(messages):
    """Filter out attachments and keep only text messages"""
    text_messages = []
    
    for message in messages:
        # Check if the message has content and doesn't have attachment indicators
        if "content" in message and not (
            message.get("share") or 
            message.get("photos") or 
            message.get("videos") or 
            message.get("content") == "You sent an attachment." or
            message.get("content") == "User2 sent an attachment."
        ):
            text_messages.append(message)
    
    logger.info(f"Filtered to {len(text_messages)} text-only messages")
    return text_messages

def format_instagram_conversations(ig_data, chunk_size=20, max_chunks=None):
    """Format Instagram chat data into dialogue format
    
    Args:
        ig_data: The loaded Instagram export JSON data
        chunk_size: Number of messages to include in each conversation chunk
        max_chunks: Maximum number of chunks to create (for testing purposes)
    
    Returns:
        List of conversation strings formatted for training
    """
    conversations = []
    
    # Extract participants
    participants = [p["name"] for p in ig_data.get("participants", [])]
    if len(participants) != 2:
        logger.warning(f"Expected 2 participants, but found {len(participants)}. Using generic names.")
        participants = ["User", "Friend"]
    
    # In this case, we're explicitly identifying the participants
    user_name = "User1"  # This is who we want the model to talk TO
    friend_name = "User2"  # This is who we want the model to talk AS
    
    logger.info(f"User: {user_name}, Friend (model persona): {friend_name}")
    
    # Get messages and filter out attachments
    messages = filter_text_messages(ig_data.get("messages", []))
    
    # Sort messages by timestamp (oldest first)
    messages.sort(key=lambda x: x.get("timestamp_ms", 0))
    
    # Process messages in chunks to create multiple training examples
    chunk_count = 0
    
    # First, let's extract all of User2's messages to create a personality profile
    User2_messages = [m.get("content", "") for m in messages if m.get("sender_name", "") == friend_name]
    
    # Add a special conversation at the beginning that describes who User2 is
    if User2_messages:
        User2_profile = "\n".join([
            "User: Tell me about yourself.",
            f"Friend: I'm User2. I like to talk about interesting topics, share memes, and connect with friends. "
            f"I sometimes use Hindi and English mixed together. I'm generally friendly and enjoy conversations.",
            "User: How would you describe your texting style?",
            f"Friend: I'm casual but thoughtful in my texts. I use emojis sometimes. I can be funny and sarcastic, "
            f"but also caring. I respond to what people say and ask questions to keep the conversation going."
        ])
        conversations.append(User2_profile)
        chunk_count += 1
    
    # Create sliding window conversations
    for i in range(0, len(messages), chunk_size//2):  # Overlap windows by 50%
        if max_chunks and chunk_count >= max_chunks:
            break
            
        chunk = messages[i:i+chunk_size]
        conversation = []
        
        # Keep track of consecutive messages by the same person
        last_sender = None
        current_content = []
        
        for message in chunk:
            sender = message.get("sender_name", "")
            content = message.get("content", "")
            
            # Skip messages without content
            if not content:
                continue
            
            # If we have a new sender, add the previous content
            if last_sender and sender != last_sender and current_content:
                sender_tag = "User: " if last_sender == user_name else "Friend: "
                formatted_message = f"{sender_tag}{' '.join(current_content)}"
                conversation.append(formatted_message)
                current_content = []
            
            # Add this content to the current batch
            current_content.append(content)
            last_sender = sender
        
        # Add the last batch if there is one
        if last_sender and current_content:
            sender_tag = "User: " if last_sender == user_name else "Friend: "
            formatted_message = f"{sender_tag}{' '.join(current_content)}"
            conversation.append(formatted_message)
        
        # Only add conversations with at least 2 messages and ensure they end with Friend
        if len(conversation) >= 2:
            # If the conversation doesn't end with Friend's message, remove the last User message
            if conversation[-1].startswith("User:") and len(conversation) > 2:
                conversation = conversation[:-1]
                
            # Only use if it ends with Friend's message
            if conversation[-1].startswith("Friend:"):
                conversations.append("\n".join(conversation))
                chunk_count += 1
    
    logger.info(f"Created {len(conversations)} conversation chunks for training")
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
        
        # Check for early stopping
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
                model = AutoModelForCausalLM.from_pretrained(os.path.join("checkpoints", "best_model")).to(device)
            break
    
    # If we completed all epochs, check if we should load the best model
    if no_improvement_count > 0 and os.path.exists(os.path.join("checkpoints", "best_model")):
        logger.info("Loading the best model checkpoint")
        model = AutoModelForCausalLM.from_pretrained(os.path.join("checkpoints", "best_model")).to(device)
    
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
        temperature=1.0,  # Higher temperature for more creativity
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.5,  # Further increased to reduce repetition
        no_repeat_ngram_size=3,  # Avoid repeating 3-grams
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        typical_p=0.95  # Use typical p sampling for more diverse outputs
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
    logger.info("This model is trained to respond like User2 based on Instagram DM history.")
    
    # Start with a personality prompt to guide the model
    conversation_history = (
        "User: Tell me about yourself.\n"
        "Friend: I'm User2. I like to talk about interesting topics, share memes, and connect with friends. "
        "I sometimes use Hindi and English mixed together. I'm generally friendly and enjoy conversations.\n"
        "User: How would you describe your texting style?\n"
        "Friend: I'm casual but thoughtful in my texts. I use emojis sometimes. I can be funny and sarcastic, "
        "but also caring. I respond to what people say and ask questions to keep the conversation going."
    )
    
    # Add a system prompt for creativity
    conversation_history += "\nUser: Let's have a real conversation. Please be yourself and don't repeat my words or previous conversations."
    
    while True:
        user_input = input("You (as User1): ")
        if user_input.lower() in exit_phrases:
            logger.info("Ending conversation.")
            break
        
        # Add user input to conversation history
        conversation_history += f"\nUser: {user_input}\nFriend:"
        
        # Generate model response
        response = generate_response(model, tokenizer, conversation_history, device)
        
        # Add response to conversation history and display
        conversation_history += f" {response}"
        print(f"Friend (as User2): {response}")
        
        # Keep conversation history from growing too large by keeping only the last 10 exchanges
        # But always keep the initial personality prompt
        if conversation_history.count("User:") > 12:  # 2 for personality prompt + 10 for conversation
            # Find the position after the personality prompt (4th occurrence of "User:")
            user_occurrences = 0
            start_pos = 0
            for _ in range(4):
                start_pos = conversation_history.find("User:", start_pos + 1)
                if start_pos == -1:
                    break
                user_occurrences += 1
                
            if start_pos != -1:
                conversation_history = conversation_history[:start_pos] + conversation_history[start_pos:]

def main():
    parser = argparse.ArgumentParser(description="Clean, combine, and fine-tune on Instagram DM chat data")
    parser.add_argument("--data_files", nargs='+', default=["message_1.json", "message_2.json", "message_3.json"], 
                        help="Paths to Instagram exported JSON files")
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Model name (default: gpt2)")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--chat", action="store_true", help="Start chat after training")
    parser.add_argument("--chunk_size", type=int, default=16, 
                        help="Number of messages per conversation chunk")
    parser.add_argument("--augment", action="store_true", help="Augment training data with repetition")
    parser.add_argument("--force-cpu", action="store_true", help="Force using CPU even if CUDA is available")
    parser.add_argument("--max_chunks", type=int, default=None, 
                        help="Maximum number of chunks to create (for testing)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading {args.model_name} model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    except Exception as e:
        logger.error(f"Error loading model {args.model_name}: {str(e)}")
        logger.info("Falling back to gpt2 model")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Move model to device
    model.to(device)
    
    # Load and format the data from all files
    logger.info(f"Loading data from {len(args.data_files)} files...")
    ig_data = load_instagram_export_data(args.data_files)
    conversations = format_instagram_conversations(
        ig_data, 
        chunk_size=args.chunk_size, 
        max_chunks=args.max_chunks
    )
    
    logger.info(f"Created {len(conversations)} conversation chunks for training")
    
    # For small datasets, we might want to augment by repeating
    if args.augment and len(conversations) < 200:
        logger.info("Augmenting data by repetition...")
        # Calculate how many times to repeat to get roughly 200-400 examples
        repeat_factor = max(1, min(10, 400 // len(conversations)))
        augmented_conversations = []
        for _ in range(repeat_factor):
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
    
    # Use the scheduler
    scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer,
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
