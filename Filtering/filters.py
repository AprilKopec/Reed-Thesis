from utils import cos_sim, idf
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset
import random

model_name = "roneneldan/TinyStories-33M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

dataset = load_dataset("roneneldan/TinyStories")


query = '''Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground. She picked it up and took a bite. It was so juicy and delicious!
Suddenly, she heard a loud noise. It was a big, scary dog! Lily was scared and didn't know what to do. But then, she remembered the apple she had picked earlier. She took a bite and it was even more delicious than before!
Lily learned that sometimes things that look scary can be delicious. She also learned that it's important to be brave and not give up. From that day on, Lily always carried an apple with her, just in case she needed to face her fears.'''

query = tokenizer.encode(query, add_special_tokens=False)
documents = [tokenizer.encode(dataset['train'][random.randint(0, len(dataset['train']))]['text'], add_special_tokens=False) for _ in range(100)]
print(documents[0])

def tf_idf_score(Q, D):
    total = 0
    for t in Q:
        total += idf(t) * (t in D)
    return total

def test_score(Q, D):
    total = 0
    for t in Q:
        for s in D:
            total += abs(cos_sim(t, s)) * idf(t)
    total /= len(Q)
    total /= len(D)
    return 1000*total



#with open("Filtering/random_sample.text", "w") as f:
    x = [(tf_idf_score(query, d), tokenizer.decode(d)) for d in documents]
    x.sort()
    print(x)
    f.write(str(x))

    y = [(test_score(query, d), tokenizer.decode(d)) for d in documents]
    y.sort()
    print(y)
    f.write(str(y))

print(tf_idf_score(query, query))