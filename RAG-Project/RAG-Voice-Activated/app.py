import os
import whisper
import chromadb
from sentence_transformers import SentenceTransformer
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For text splitting
import torch

# Constants
AUDIO_FILE = "user_input.wav"
RESPONSE_AUDIO_FILE = "response.wav"  # File to save the generated audio response
PDF_FILE = "Insurance_Handbook_20103.pdf"  # Replace with your PDF file path
SAMPLE_RATE = 16000  # Sample rate for audio recording
WAKE_WORD = "Hi"  # Wake word to activate the system
SIMILARITY_THRESHOLD = 0.4  # Threshold for wake word detection
MAX_ATTEMPTS = 5  # Maximum number of attempts to detect the wake word

# Step 1: Speech-to-Text (STT) using Whisper
def record_audio(filename, duration=5, samplerate=SAMPLE_RATE):
    """Record audio from the microphone."""
    print("Listening... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    # Save as WAV file
    write(filename, samplerate, (audio * 32767).astype(np.int16))

def transcribe_audio(filename):
    """Transcribe audio using Whisper."""
    print("Transcribing audio...")
    model = whisper.load_model("base.en")
    result = model.transcribe(filename)
    return result["text"].strip().lower()

# Step 2: Load and Process PDF
def load_and_chunk_pdf(pdf_file):
    """Load a PDF and split it into chunks."""
    from PyPDF2 import PdfReader
    print("Loading and chunking PDF...")
    reader = PdfReader(pdf_file)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,  # Size of each chunk
        chunk_overlap=50,  # Overlap between chunks to maintain context
        separators=["\n\n", "\n", " ", ""]  # Splitting hierarchy
    )
    chunks = text_splitter.split_text(all_text)
    return chunks

# Step 3: Initialize ChromaDB and SentenceTransformers
def setup_chromadb(chunks):
    """Set up ChromaDB and store text chunks with embeddings."""
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Delete existing collection (if needed)
    try:
        client.delete_collection(name="knowledge_base")
        print("Deleted existing collection: knowledge_base")
    except Exception as e:
        print(f"Collection does not exist or could not be deleted: {e}")

    # Create a new collection
    collection = client.create_collection(name="knowledge_base")

    # Add text chunks to the collection
    for i, chunk in enumerate(chunks):
        embedding = text_embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            metadatas=[{"source": "pdf", "chunk_id": i}],
            documents=[chunk]
        )
    print("Text chunks and embeddings stored in ChromaDB.")
    return collection

# Step 4: Query ChromaDB for Relevant Chunks
def query_chromadb(collection, query, top_k=3):
    """Query ChromaDB for relevant chunks."""
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = text_embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # Flatten the nested lists in results["documents"]
    relevant_chunks = [chunk for sublist in results["documents"] for chunk in sublist]
    return relevant_chunks

# Step 5: Generate Response using a Hugging Face Model
def generate_response(query, context_chunks):
    """Generate a response using a Hugging Face model."""
    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format the prompt with the query and context
    context = "\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input and move to the correct device
    model_inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)

    # Generate the response
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Step 6: Text-to-Speech (TTS) using Bark
def text_to_speech(text, output_file):
    """Convert text to speech using Bark and save to a file."""
    from transformers import AutoProcessor, BarkModel
    print("Generating speech...")
    
    # Load the processor and model
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")

    # Tokenize the input text (no padding or truncation needed for Bark)
    inputs = processor(text, return_tensors="pt")

    # Generate the audio
    audio_array = model.generate(**inputs)

    # Convert the audio to a numpy array
    audio = audio_array.cpu().numpy().squeeze()

    # Save the audio to a file
    write(output_file, 22050, (audio * 32767).astype(np.int16))
    print(f"Audio response saved to {output_file}")
    return audio

def play_audio(audio, samplerate=22050):
    """Play audio using sounddevice."""
    print("Playing response...")
    sd.play(audio, samplerate=samplerate)
    sd.wait()

# Step 7: Wake Word Detection using Embedding Similarity
def detect_wake_word(max_attempts=MAX_ATTEMPTS):
    """Detect wake word using embedding similarity."""
    print("Waiting for wake word...")
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    wake_word_embedding = text_embedding_model.encode(WAKE_WORD).reshape(1, -1)

    attempts = 0
    while attempts < max_attempts:
        record_audio(AUDIO_FILE, duration=3)  # Record 3 seconds of audio
        transcription = transcribe_audio(AUDIO_FILE)
        print(f"Transcription: {transcription}")

        # Compute embedding for the transcription
        transcription_embedding = text_embedding_model.encode(transcription).reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(wake_word_embedding, transcription_embedding)[0][0]
        print(f"Similarity with wake word: {similarity:.2f}")

        if similarity >= SIMILARITY_THRESHOLD:
            print(f"Wake word detected: {WAKE_WORD}")
            return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}. Please try again.")
    print("Wake word not detected. Exiting.")
    return False

# Step 8: Main Function
def main():
    # Step 1: Load and chunk the PDF
    chunks = load_and_chunk_pdf(PDF_FILE)

    # Step 2: Set up ChromaDB
    collection = setup_chromadb(chunks)

    # Step 3: Detect wake word with embedding similarity
    if not detect_wake_word():
        return  # Exit if wake word is not detected

    # Step 4: Record and transcribe user input
    record_audio(AUDIO_FILE, duration=5)  # Record 5 seconds of audio
    user_input = transcribe_audio(AUDIO_FILE)
    print(f"User Input: {user_input}")

    # Step 5: Query ChromaDB for relevant chunks
    relevant_chunks = query_chromadb(collection, user_input)
    print(f"Relevant Chunks: {relevant_chunks}")

    # Step 6: Generate response using a Hugging Face model
    response = generate_response(user_input, relevant_chunks)
    print(f"Generated Response: {response}")

    # Step 7: Convert response to speech, save it, and play it
    audio = text_to_speech(response, RESPONSE_AUDIO_FILE)
    play_audio(audio)

    # Clean up
    os.remove(AUDIO_FILE)  # Delete the temporary audio file

if __name__ == "__main__":
    main()