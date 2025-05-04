import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch
import torch.nn.functional as F

# Function for adding user question to prompt
def addQuestionToPrompt(prompt, question):
    prompt += f"\n<|user|>{question}\n"
    return prompt

# Function for adding retrieved information to prompt
def addInfoToPrompt(prompt, question):
    retrievedInfo = RAG(question, knowledge_base= entries)
    prompt += f"\n<|additional_information|>{retrievedInfo}\n"
    return prompt

# Get embedding of sentence
def get_embedding(sent):

    # Load embedding model and tokenizer
    embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model.eval()

    # Calculate embedding
    with torch.no_grad():
        encoded_input = embedding_tokenizer(sent, return_tensors='pt', truncation=True, padding=True)
        model_output = embedding_model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings

# RAG function
def RAG(user_prompt, knowledge_base):
    embed = get_embedding(user_prompt)  # Tensor
    mostRelevant = None
    highestRelevancy = -1 # For getting the highest score

    for _, sentence in knowledge_base:
        db_embed = get_embedding(sentence)  # Tensor
        relevancy = F.cosine_similarity(embed, db_embed).item() # Finding the most relevant sentence by cosine similarity

        if relevancy > highestRelevancy:
            highestRelevancy = relevancy
            mostRelevant = sentence

    return mostRelevant

# Model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device= 0)

# Load XML file
tree = ET.parse('knowledge_base.xml') # Get the tree
root = tree.getroot() # Get root of tree

# Save all entries
entries = []
for entry in root.findall('entry'):
    ID = entry.find('id').text
    text = entry.find('text').text
    entries.append((ID, text))
# Now, 'entries' is a list in [(id, text), (id, text), ...] format.

question = input("Enter your sentence: ") # Get user question

# Prompt scheme
prompt = (
"<|system|>\nYou are a chatbot that answers questions.\n"

        # User question will be added here

        # LLM answer will be added here
)

# RAG prompt scheme
prompt_RAG = (
    "<|system|>\nYou are a chatbot that answers questions. Pay attention to additional information provided to you before answering.\n"

        # Additional information will be added here

        # User question will be added here

        # LLM answer will be added here
)

prompt_RAG = addInfoToPrompt(prompt_RAG, question) # Retrieveal is made inside this function

prompt_RAG = addQuestionToPrompt(prompt_RAG, question) # Added user question to prompt
prompt = addQuestionToPrompt(prompt, question) # Added user question to prompt

prompt_RAG += "\n<|assistant|>\n" # Add this to get answer
prompt += "\n<|assistant|>\n" # Add this to get answer

response_RAG = pipe(prompt_RAG, max_new_tokens=30, temperature=0.7, do_sample=True) # Get response with RAG
response = pipe(prompt, max_new_tokens=30, temperature=0.7, do_sample=True) # Get response without RAG

# Responses
print(f"Response:\n-------{response[0]["generated_text"]}\n")
print(f"Response with RAG:\n-------{response_RAG[0]["generated_text"]}")