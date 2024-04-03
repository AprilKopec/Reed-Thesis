from utils import cos_sim, idf
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "roneneldan/TinyStories-33M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)


queries = ['''One day, a little girl named Lily found a needle in her room. She knew it was sharp and could hurt her. Lily wanted to show her mom the needle, so she went to find her. Lily found her mom in the kitchen. "Mom, look what I found!" she said. Her mom looked at the needle and said, "Oh no, Lily! That's a terrible needle. It can hurt you. Let's put it away." Lily and her mom put the needle in a safe place. They found a soft cloth and put it away. Lily's mom said, "Now, let's not play with the needle. Let's play with your toys instead." Lily smiled and they played together happily.''',
           '''Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn. Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.''',
           '''One day, a little fish named Fin was swimming near the shore. He saw a big crab and wanted to be friends. "Hi, I am Fin. Do you want to play?" asked the little fish. The crab looked at Fin and said, "No, I don't want to play. I am cold and I don't feel fine." Fin felt sad but wanted to help the crab feel better. He swam away and thought of a plan. He remembered that the sun could make things warm. So, Fin swam to the top of the water and called to the sun, "Please, sun, help my new friend feel fine and not freeze!" The sun heard Fin's call and shone its warm light on the shore. The crab started to feel better and not so cold. He saw Fin and said, "Thank you, little fish, for making me feel fine. I don't feel like I will freeze now. Let's play together!" And so, Fin and the crab played and became good friends.''']


documents = ['''One day, a little girl named Lily found a needle in her room. She knew it was sharp and could hurt her. Lily wanted to show her mom the needle, so she went to find her. Lily found her mom in the kitchen. "Mom, look what I found!" she said. Her mom looked at the needle and said, "Oh no, Lily! That's a terrible needle. It can hurt you. Let's put it away." Lily and her mom put the needle in a safe place. They found a soft cloth and put it away. Lily's mom said, "Now, let's not play with the needle. Let's play with your toys instead." Lily smiled and they played together happily.''',
             '''One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.''',
             '''Once upon a time, there was a little car named Beep. Beep loved to drive around and explore the world. One day, Beep drove to a big park. In the park, there was a big tree with a swing. Beep loved to swing back and forth, feeling the wind in his face. As Beep was swinging, he saw a little girl named Lily. Lily was sad because she lost her toy. Beep wanted to help, so he drove around the park looking for the toy. He looked under the slide, behind the swings, and even in the sandbox. Finally, Beep found the toy near the big tree. Lily was so happy and thanked Beep for his help. Beep felt proud that he could help Lily. They became best friends and played together every day. And from that day on, Beep always remembered to help others when they needed it.''',
             '''One day, a little fish named Fin swam up to the top of the water. He saw a big, shiny rock. Fin wanted to show his friends the shiny rock. He swam fast to find them. Fin found his friends near the rock. They all looked at the shiny rock and said, "Wow, that's a pretty rock!" They all wanted to see the rock too. So, they swam around the rock and looked at it. Fin and his friends decided to share the shiny rock. They took turns showing it to each other. They were all very happy. They played together and swam around the rock all day.''']

queries = [tokenizer.encode(query, add_special_tokens=False) for query in queries]
documents = [tokenizer.encode(document, add_special_tokens=False) for document in documents]

def tf_idf_score(Q, D):
    total = 0
    for t in Q:
        total += idf(t) * (t in D)
    return 5/3 * total

def test_score(Q, D):
    total = 0
    for t in Q:
        for s in D:
            total += abs(cos_sim(t, s)) * idf(t)
    total /= len(Q)
    total /= len(D)
    return 1000*total

for i in range(len(queries)):
    for j in range(len(documents)):
        q = queries[i]
        d = documents[j]
        print(f'influence of doc {j} on query {i}')
        tf = tf_idf_score(q, d)
        test = test_score(q, d)
        print(f'tf-idf: {tf}')
        print(f'test: {test}')
