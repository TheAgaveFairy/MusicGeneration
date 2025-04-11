import numpy as np
import pandas as pd
import os
import transformers
from tqdm import tqdm

from helpers import list_files

MIDI_TEXT_FILENAME = 'midi_text_data.txt'

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import transformers

transformers.utils.logging.set_verbosity(transformers.logging.INFO)
transformers.utils.logging.enable_progress_bar

def fine_tune_gpt2(training_file, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add special tokens if they're not already there
    special_tokens = {
        'additional_special_tokens': ['<|startofpiece|>', '<|endofpiece|>']
    }
    #tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    print("Model loaded")
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=training_file,
        block_size=128  # Adjust based on your data
    )

    print("Dataset loaded")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        logging_dir="./logs",
        logging_steps=10,  # Log every 10 steps
        logging_first_step=True,
        logging_strategy="steps",
        
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10000,
        fp16=True,
        save_total_limit=2,
    )

    print("Device is", training_args.device)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Start Training")
    
    # Train
    trainer.train()    

    print("Training done, saving to", output_dir)
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
def generate_music(model_path, prompt="Composer: Bach", max_length=512, num_return_sequences=1, temperature=1.0):
    """
    Generate music using your fine-tuned GPT-2 model
    
    Args:
        model_path: Path to the fine-tuned model
        prompt: Text prompt to start generation (e.g., "Composer: Bach")
        max_length: Maximum length of generated sequence
        num_return_sequences: Number of different sequences to generate
        temperature: Controls randomness (lower = more deterministic)
        
    Returns:
        List of generated MIDI objects
    """
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Format the prompt with the right structure
    formatted_prompt = f"<|startofpiece|>\n{prompt}\n"
    
    # Encode the prompt
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generate continuation
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    
    # Decode and parse the output
    generated_sequences = []
    for sequence in output_sequences:
        text = tokenizer.decode(sequence, skip_special_tokens=False)
        
        # Make sure we stop at the end of piece token if present
        if "<|endofpiece|>" in text:
            text = text.split("<|endofpiece|>")[0] + "<|endofpiece|>"
            
        generated_sequences.append(text)
        
    # Convert text to MIDI
    midi_objects = [text_to_midi(seq) for seq in generated_sequences]
    
    return midi_objects, generated_sequences

  
if __name__ == '__main__':
    print("hello")
    fine_tune_gpt2(MIDI_TEXT_FILENAME, './modeloutput/')
    #midi_objects, generated_sequences = generate_music(model_path, prompt="Composer: Bach", max_length=512, num_return_sequences=1, temperature=1.0)
