import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Constants
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
print("Loading model and tokenizer...")
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds.")

# Define a sample query and context
query = "What is the capital of France?"
context_chunks = [
    "France is a country in Europe.",
    "The capital of France is Paris.",
    "Paris is known for its art, fashion, and culture."
]

# Format the prompt with the query and context
context = "\n".join(context_chunks)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
]

# Apply the chat template
print("Applying chat template...")
start_time = time.time()
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Chat template applied in {time.time() - start_time:.2f} seconds.")

# Tokenize the input
print("Tokenizing input...")
start_time = time.time()
model_inputs = tokenizer(
    [text], 
    return_tensors="pt", 
    padding=True, 
    truncation=True
).to(DEVICE)
print(f"Input tokenized in {time.time() - start_time:.2f} seconds.")

# Generate the response
print("Generating response...")
start_time = time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id
)
print(f"Response generated in {time.time() - start_time:.2f} seconds.")

# Decode the generated response
print("Decoding response...")
start_time = time.time()
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Response decoded in {time.time() - start_time:.2f} seconds.")

# Print the response
print("\nGenerated Response:")
print(response)