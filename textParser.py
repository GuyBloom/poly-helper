import pandas as pd
import pickle
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
print(os.environ.get("OPENAI_API_KEY"))
client = OpenAI(
    api_key = "sk-proj-MlVpZ6ucgi9woUZCAbRvT3BlbkFJ3ZKhFsbS7Lecgah5ns6U"
)

def parse_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    terms_and_definitions = []
    current_term = ""
    current_definition = ""
    for line in lines:
        # If line starts with a tilde character, it's a new term
        
        if line.strip() and line.startswith("`"):
            if current_term and current_definition:
                terms_and_definitions.append((current_term, current_definition.strip()))
            current_term = line.strip()[1:]
            current_definition = ""
        else:
            current_definition += line

    # Add the last term and definition
    if current_term and current_definition:
        terms_and_definitions.append((current_term, current_definition.strip()))

    return terms_and_definitions

def create_dataframe(terms_and_definitions):
    df = pd.DataFrame(terms_and_definitions, columns=['Term', 'Definition'])
    return df
def isNotCut(data):
    for index, row in data.iterrows():
        if(row['n_tokens'] > 500):
            return True
    return False
def maxTokens(data):
    max = 0
    for index, row in data.iterrows():
        if(row['n_tokens'] > max):
            max = row['n_tokens']
    return max
    

# Example usage
file_path = 'dictionary-of-cultural-and-criticaltheory.txt'
terms_and_definitions = parse_text_file(file_path)
df = create_dataframe(terms_and_definitions)
totalDictionary = dict(skibidi = 'biden')
for _, row in df.iterrows():
    totalDictionary[row['Term']] = [row['Definition']]
with open('person_data.pkl', 'wb') as fp:
    pickle.dump(totalDictionary, fp)
    print('dictionary saved successfully to file')
# tokenizer = tiktoken.get_encoding("cl100k_base")
# df['n_tokens'] = df.Definition.apply(lambda x: len(tokenizer.encode(x)))
# while(isNotCut(df)):
#     df['n_tokens'] = df.Definition.apply(lambda x: len(tokenizer.encode(x)))
#     for index, row in df.iterrows():
#         if(row['n_tokens'] > 500):
#             tempText = row['Definition'][1500:]
#             row['Definition'] = row['Definition'][:1500]
#             df.loc[index] = row
#             newLine = pd.Series({"Term" : row['Term'], 'Definition': tempText, 'n_tokens' : tokenizer.encode(tempText)})
#             df = pd.concat([df, newLine.to_frame().T], ignore_index= True)
#     df['n_tokens'] = df.Definition.apply(lambda x: len(tokenizer.encode(x)))

# df['embeddings'] = df.Definition.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-3-small').data[0].embedding)
# df.to_csv('embeddings.csv')

