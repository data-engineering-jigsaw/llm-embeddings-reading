import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine

api_key = ""
client = OpenAI(
    api_key=api_key
)  # get API key from platform.openai.com

MODEL = "text-embedding-3-small"
inputs = [
        "Tree", "Bagel", "Software Developer", "King", "Queen", "Prince"
    ]
res = client.embeddings.create(
    input=inputs, model=MODEL
)
df = pd.DataFrame(inputs, columns = ['text'])
vectors = res.data

np_embeddings = [np.array(vector.embedding) for vector in vectors]
df = df.assign(embedding = np_embeddings)
embeddings = df['embedding']


question = "King"
q_embeddings = client.embeddings.create(input=question, model=MODEL)
q_embedding_array = np.array(q_embeddings.data[0].embedding)


distances = [cosine(q_embedding_array, embedding) for embedding in embeddings]
df['distances'] = distances

df.sort_values('distances')[:3]
context = '\n\n'.join(df.sort_values('distances')[:3].text)
