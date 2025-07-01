import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
        repetition_penalty=1.5,  # Increased to reduce repetition
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
    logger.info("This model is trained to respond like Vishwa Joshi based on Instagram DM history.")
    
    # Add a personality prompt to help guide the model's responses
    personality_prompt = "The following is a conversation between User (Pranay Kapoor) and Friend (Vishwa Joshi). " \
                         "Vishwa is creative, friendly, and has a unique way of expressing herself. " \
                         "She responds naturally in a conversational manner.\n\n"
    
    conversation_history = personality_prompt
    
    while True:
        user_input = input("You (as Pranay): ")
        if user_input.lower() in exit_phrases:
            logger.info("Ending conversation.")
            break
        
        # Add user input to conversation history
        conversation_history += f"User: {user_input}\nFriend:"
        
        # Generate model response
        response = generate_response(model, tokenizer, conversation_history, device)
        
        # Add response to conversation history and display
        conversation_history += f" {response}\n"
        print(f"Vishwa: {response}")
        
        # Keep conversation history from growing too large by keeping only the last 10 exchanges
        if conversation_history.count("User:") > 10:
            # Find the position of the second occurrence of "User:"
            start_pos = conversation_history.find("User:")
            start_pos = conversation_history.find("User:", start_pos + 1)
            # Preserve the personality prompt
            conversation_history = personality_prompt + conversation_history[start_pos:]

def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned GPT-2 model")
    parser.add_argument("--model_dir", type=str, default="fine_tuned_model", help="Directory of the fine-tuned model")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading model from {args.model_dir}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    
    # GPT-2 tokenizer doesn't have a padding token, so we'll set it to the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Move model to device
    model.to(device)
    
    # Start chat
    chat_with_model(model, tokenizer, device)

if __name__ == "__main__":
    main()
