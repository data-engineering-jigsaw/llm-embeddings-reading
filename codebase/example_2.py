import numpy as np
import pandas as pd
from openai import OpenAI
# from openai.embeddings_utils import distances_from_embeddings
from scipy.spatial.distance import cosine

from utils import *

api_key = ""
client = OpenAI(
    api_key=api_key
)  # get API key from platform.openai.com

MODEL = "text-embedding-3-small"
inputs = [
        "Susan is CEO and makes $160,000 per year.",
        "Fiona is a software developer and earns $100,000 per year",
        "Sam is a marketer and earns $80,000 per year",
        "Sally is a vice-president and earns $90,000 per year",
        "Mark is a COO and earns 150,000 per year"
    ]
res = client.embeddings.create(
    input=inputs, model=MODEL
)

df = pd.DataFrame(inputs, columns = ['text'])
vectors = res.data

np_embeddings = [np.array(vector.embedding) for vector in vectors]
df = df.assign(embedding = np_embeddings)


question = "What is the name of the CEO?"
q_embeddings = client.embeddings.create(input=question, model=MODEL)
q_embedding_array = np.array(q_embeddings.data[0].embedding)

embeddings = df['embedding']


distances = [cosine(q_embedding_array, embedding) for embedding in embeddings]
df['distances'] = distances
df.sort_values('distances')[:3]
context = '\n\n'.join(df.sort_values('distances')[:3].text)

prompt = generate_prompt(question, context)

# get_answer(prompt, api_key)

