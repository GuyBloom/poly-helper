import pandas as pd
import tiktoken
import openai
import numpy as np
from embeddings_utils import distances_from_embeddings
from ast import literal_eval
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv, dotenv_values 

load_dotenv()
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('disctionary-of-cultural-and-criticaltheory.csv', index_col=0)

df.columns = ['text']

df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

df.n_tokens.hist()

max_tokens = 500

def split_into_many(text, max_tokens = max_tokens):

    sentences = text.split('. ')

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):


        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0


        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


shortened = []

for row in df.iterrows():

    if row[1]['text'] is None:
        continue

    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    else:
        shortened.append( row[1]['text'] )



df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()




df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-3-small').data[0].embedding)



df.to_csv('embeddings.csv')
# df.head()



# df=pd.read_csv('embeddings.csv', index_col=0)
# df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# df.head()


# def create_context(
#     question, df, max_len=1800, size="ada"
# ):
#     """
#     Create a context for a question by finding the most similar context from the dataframe
#     """

#     q_embeddings = client.embeddings.create(input=question, model='text-embedding-3-small').data[0].embedding

#     df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


#     returns = []
#     cur_len = 0

#     for _, row in df.sort_values('distances', ascending=True).iterrows():

#         cur_len += row['n_tokens'] + 4

#         if cur_len > max_len:
#             break

#         returns.append(row["text"])

#     return "\n\n###\n\n".join(returns)


# def answer_question(
#     df,
#     model="gpt-3.5-turbo",
#     question="Am I allowed to publish model outputs to Twitter, without a human review?",
#     max_len=1800,
#     size="ada",
#     debug=False,
#     max_tokens=150,
#     stop_sequence=None
# ):
#     """
#     Answer a question based on the most similar context from the dataframe texts
#     """
#     context = create_context(
#         question,
#         df,
#         max_len=max_len,
#         size=size,
#     )
#     if debug:
#         print("Context:\n" + context)
#         print("\n\n")

#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": f"You are a confident Capital One virtual assistant with decades of training. Answer the questions with this context: {context}" },
#                 {"role": "user", "content": question},
                
#             ],
#             temperature=0.7,
#             max_tokens=max_tokens,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=stop_sequence,
#         )
#         return str(response.choices[0].message).strip()
#     except Exception as e:
#         print(e)
#         return ""





# print(answer_question(df, question="What are you?", debug=False))
# print(answer_question(df, question="How do I order checks online for my business?"))
# print(answer_question(df, question="What happens when you file a dispute through Capital One?"))
# print(answer_question(df, question="What makes Capital One unique?"))