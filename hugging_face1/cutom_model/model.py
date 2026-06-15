from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

# --- STEP 1: PREPARE MODEL AND DATASET ---
model_id = "gpt2" # Using GPT-2 as a small example model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load a custom dataset (e.g., a small sample of text)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# --- STEP 2: FINE-TUNE WITH TRAINER ---
training_args = TrainingArguments(
    output_dir="./my_fine_tuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train() # The model "brain" learns the new dataset here [cite: 51, 704]
trainer.save_model("./my_fine_tuned_model") # Save the learned weights [cite: 85]

# --- STEP 3: CUSTOM GENERATION ---
# Load your newly trained model for inference
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./my_fine_tuned_model")

# Set up your "Output Behavior" (Hyperparameters)
gen_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,      # Enable creative mode [cite: 212, 344]
    temperature=0.9,     # High creativity [cite: 213, 373]
    repetition_penalty=1.2 # Prevent loops [cite: 221, 387]
)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate using the new brain + custom settings
outputs = fine_tuned_model.generate(**inputs, generation_config=gen_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))