from sentence_transformers import SentenceTransformer
from utils import *
import requests

class Sentence_Transformer:
    def __init__(self, model_name, device):
        
        if model_name == "bge-large-en-v1.5":
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
            self.model.eval()
        elif model_name == "all-MiniLM-L6-v2":
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
            self.model.eval()
        elif model_name == "all-MiniLM-L12-v2":
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=device)
            self.model.eval()
        elif model_name == "UAE-Large-V1":
            self.model = SentenceTransformer('WhereIsAI/UAE-Large-V1', device=device)
            self.model.eval()
        elif model_name == "mxbai-embed-large-v1":
            self.model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device=device)
            self.model.eval()
        else:
            """
            You could modify the code here to make it compatible with other models
            """
            raise ValueError(f"Model {model_name} is not currently supported!")
            
    def encode(self, prompt):
        embedding = self.model.encode(prompt, normalize_embeddings=True)
        return embedding