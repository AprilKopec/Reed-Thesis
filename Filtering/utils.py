from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import numpy as np
import pickle

model_name = "roneneldan/TinyStories-33M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

def embedding_vectors(doc):
    tokens = tokenizer.encode(doc, add_special_tokens=False)
    return [model.transformer.wte.weight[token].detach().numpy() for token in tokens]

def cos_sim(u, v):
    u = model.transformer.wte.weight[u].detach().numpy()
    v = model.transformer.wte.weight[v].detach().numpy()
    if np.linalg.norm(u)*np.linalg.norm(v) == 0.0:
        return 0
    else:
        return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))

with open('Filtering/token_counts.pkl', 'rb') as f:
        token_counts = pickle.load(f)

def idf(t):
    C = 2119718
    num = C - token_counts[t] + 0.5
    denom = token_counts[t] + 0.5
    return np.log(num/denom + 1)

# This is used to build token_counts.pkl
# It requires the TinyStories training dataset
def IDF_Calc_Constants():
    document_count = 0
    token_counts = {x: 0 for x in range(50257)}
    i=0
    doc = []
    with open('TinyStories-train.txt', 'r', encoding='utf-8') as file:
        for line in file:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            if tokens[0] == 50256:
                document_count += 1
                for t in set(doc):
                    token_counts[t] += 1
                doc = []
            else:
                doc += tokens

            i += 1
            if (i % 100000 == 0):
                print(document_count)
                print(i/14815488)

            if (i % 1000000 == 0):
                print(token_counts)

    print(document_count)
    with open('token_counts.pkl', 'wb') as file:
        pickle.dump(token_counts, file)

def split_string_by_tokens(s):
    tokens = tokenizer.encode(s, add_special_tokens=False)
    print([tokenizer.decode(t) for t in tokens])

def embedding_vector(token):
    return tokenizer.encode(token, add_special_tokens=False)[0]