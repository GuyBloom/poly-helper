import pandas as pd
from pandas import DataFrame
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
instructions_file = open('instructions.txt', 'r', encoding='utf-8')
instructions = instructions_file.read()

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

df=pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
with open('person_data.pkl', 'rb') as fp:
    uncut_dict = pickle.load(fp)

df.head()
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    q_embeddings = client.embeddings.create(input=question, model='text-embedding-3-small').data[0].embedding

    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    interm_df = df.sort_values('distances', ascending=True, ignore_index = True)
    top_names = []
    top_dist = []
    counter = 0
    while(len(top_names)<10):
        if(top_names.count(interm_df.loc[counter].at["Term"])==0):
            top_names.append(interm_df.loc[counter].at["Term"])
            top_dist.append(interm_df.loc[counter].at['distances'])
        counter = counter + 1
    info_compiled = {'Term': top_names, 'distances': top_dist}
    topdf = pd.DataFrame(data = info_compiled)
    topdf['Definition'] = topdf.Term.apply(lambda x: uncut_dict[x])
    returns = []
    cur_len = 0

    list_context = []
    
    for _, row in topdf.iterrows():
        new_line = (row['Term'], row['Definition'], row['distances'])
        list_context.append(new_line)
    return (list_context)

colorList = ["green", "red", "blue", "purple"]
tester = "Type the first 8 numbers of the fibonacci statement, creating a newline between each number"
def answer_question(
    df,
    model="gpt-4o",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=5000,
    size="ada",
    debug=False,
    max_tokens=3000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    
    if debug:
        print("Context:\n")
        print(context)
        print("\n\n")
    try:
        response = client.chat.completions.create(
            model=model,        
            messages=[
                {"role": "system", "content": (instructions + f"{context}") },
                {"role": "user", "content": question},
                
            ],
            temperature=0.7,
            seed=123,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return (str(response.choices[0].message.content).strip(), context)
    except Exception as e:
        print(e)
        return ""
def split_answer(answer, separator, context_list, debug = False):

    init_list = answer.split('*')
    paragraph = init_list.pop(0)
    exp_list = []
    init_list[0] = init_list[0].strip()
    inter_list=init_list[0].splitlines()
    for x in inter_list:
        if(x!=''):
            exp_list.append(x)
    name_list = []
    distance_list = []
    for x in context_list:
        name_list.append(x[0])
        distance_list.append(x[2])
    if(debug):
        dict = {'name': name_list, 'distance': distance_list, 'expl': exp_list}
        print('name_list length: ' + str(len(name_list)) + '\n')
        print('distance_list length: ' + str(len(distance_list)) + '\n')
        print('exp_list length: ' + str(len(exp_list)) + '\n')
        for x in name_list:
            print(x + '\n')
        for x in distance_list:
            print(str(x) + '\n')
        for x in exp_list:
            print(x + '\n')
    final_df = pd.DataFrame(data = dict)
    final_df['percent_match'] = final_df.distance.apply(lambda x: 100*round(1-float(x), 4))
    if(debug):
        print(paragraph)
        print(final_df)
        final_df.to_csv('final_test.csv')
    return (paragraph, final_df)
    


    
file = open('question.txt', 'r', encoding='utf-8')
answer_raw = answer_question(df, question=file.read()                  
, debug=True)
answer = answer_raw[0]
final_raw =split_answer(answer, '*', answer_raw[1], debug= True)
print(final_raw[0])
print(final_raw[1])

# answerlist = answer[1]
# answerlist.to_csv('responses.csv')
