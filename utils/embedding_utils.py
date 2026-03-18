import numpy as np
from huggingface_hub import InferenceClient


class EmbeddingClient:
    def __init__(self, hf_token):
        self.client = InferenceClient(
            model="BAAI/bge-large-en-v1.5",
            token=hf_token
        )

    def embed(self, text):
        emb = self.client.feature_extraction(text)
        emb = np.array(emb)

        # L2 normalization (VERY important for cosine similarity)
        emb = emb / np.linalg.norm(emb)

        return emb.tolist()