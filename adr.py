import os
from icecream import ic
import numpy as np
import time
import torch
import pandas as pd

import evaluate
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
from evaluate import load
bertscore = load("bertscore")

############################################################################################################

def getdata():
    path = '/home2/rudra.dhar/Data/SE/adr/data_folder/'

    directory = path + 'architecture-master'
    context, decision = [], []
    for filename in os.scandir(directory):
        #if filename.path.endswith(".md") and filename.is_file():
        with open(filename.path, 'r') as f:
            content = f.read()
            context.append(content[content.find('## Context')+12:content.find('## Decision')-2])
            decision.append(content[content.find('## Decision')+13:content.find('## Status')])
        

    directory = path + 'adr'
    #context, decision = [], []
    for filename in os.scandir(directory):
        #if filename.path.endswith(".md") and filename.is_file():
        with open(filename.path, 'r') as f:
            content = f.read()
            context.append(content[2:content.find('## Decision')-2])
            if ('## Pros and Cons' in content):
                decision.append(content[content.find('## Decision')+21:content.find('## Pros and Cons')])
            elif ('## License' in content):
                decision.append(content[content.find('## Decision')+21:content.find('## License')])
            
        

    directory = path + 'examples_extracted'
    #context, decision = [], []
    for filename in os.scandir(directory):
        with open(filename.path, 'r') as f:
            content = f.read()
            context.append(content[12:content.find('## Decision')-2])
            decision.append(content[content.find('## Decision')+13:])
        

    directory = path + 'cardano'
    #context, decision = [], []
    for filename in os.scandir(directory):
        with open(filename.path, 'r', encoding='utf-8') as f:
            content = f.read()
            context.append(content[12:content.find('## Decision')-2])
            decision.append(content[content.find('## Decision')+17:-2])
        

    directory = path + 'island'
    #context, decision = [], []
    for filename in os.scandir(directory):
        with open(filename, "r", encoding='utf-8') as f:
            content = f.read()
            if('\nConsidered Options' in content):
                context.append(content[content.find('Context and Problem Statement')+30:content.find('\nConsidered Options')])
            elif('\nDecision Drivers' in content):
                context.append(content[content.find('Context and Problem Statement')+30:content.find('\nDecision Drivers')])
            else:
                context.append(content[content.find('Context and Problem Statement')+30:content.find('\nDecision Outcome')])

            if('\nPros and Cons of ' in content):
                decision.append(content[content.find('Decision Outcome')+17:content.find('\nPros and Cons of ')])
            else:
                decision.append(content[content.find('Decision Outcome')+17:])

        
    context_input_gpt3 = ["Architectural Decision Record\n\n## Context\n" +c+ "\n\n## Decision\n" for c in context]
    context_input_chatGPT = ["## Context\n\n" +c for c in context]
    context_input_T5 = ["This is a Architectural Decision Record. Provide a Decision for the Context given below.\n\n## Context\n" +c for c in context]
    
    return context, decision, context_input_gpt3, context_input_chatGPT, context_input_T5
        
    
############################################################################################################

def gpt_2(context, decision, model):

    # Total length of tokens should be less than 1024
    # approx less than 512 tokens for context and 512 tokens for decision
    # approx 512 tokens = 2000 characters
    context_truncated = [c[:2000]+"\n\n## Decision" for c in context]
    decision_reference_truncated = [c[:2000] for c in decision]

    generator = pipeline('text-generation', model=model, device = 0)

    prediction = generator(context_truncated, max_length=1000, num_return_sequences=1)
    prediction = [p[0]['generated_text'] for p in prediction]

    predicted_decision = []
    for p in range(len(prediction)):
        predicted_decision.append(prediction[p][len(context_truncated[p]):])

    results = rouge.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bleu.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = meteor.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bertscore.compute(predictions=predicted_decision, references=decision_reference_truncated, lang="en")
    results = {
        'precision': np.average(results['precision']), 'recall': np.average(results['recall']),
        'f1': np.average(results['f1']), 'hashcode': results['hashcode']
        }
    ic(results)


for model in ['gpt2-large']:
    ic('#############################################\n', model)
    gpt_2(context, decision, model)


############################################################################################################

import openai
from dotenv import load_dotenv
path = '/home2/rudra.dhar/codes/SE/adr/'
load_dotenv(path + '.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

def askGPT3(text, model, max_tokens):    
    response = openai.Completion.create(
        model=model,
        prompt=text,
        max_tokens=max_tokens,
    )
    return response.choices[0].text

def gpt_3(context, decision, model, max_tokens):

    # Total length of tokens should be less than 2048 for ada, 4096 for text-davinci-003
    # approx half for context and half for decision
    # approx 1000 tokens = 4000 characters

    max_char = max_tokens*4
    context_truncated = [c[:max_char] for c in context]
    decision_reference_truncated = [c[:max_char] for c in decision]

    predicted_decision = []
    for c in context_truncated:
        predicted_decision.append(askGPT3(c, model, max_tokens))
    
    for i in range(len(predicted_decision)):
        if('## Decision' in predicted_decision[i]):
            predicted_decision[i] = predicted_decision[i][predicted_decision[i].find('## Decision')+12:]
            predicted_decision[i] = predicted_decision[i][:predicted_decision[i].find('\n## ')]

    results = rouge.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bleu.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = meteor.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bertscore.compute(predictions=predicted_decision, references=decision_reference_truncated, lang="en")
    results = {
        'precision': np.average(results['precision']), 'recall': np.average(results['recall']),
        'f1': np.average(results['f1']), 'hashcode': results['hashcode']
        }
    ic(results)


context, decision, context_input, _ = getdata()
ic(len(context_input))
for m in [('ada', 1000), ('text-davinci-003', 2000)]:
    model, max_tokens = m
    ic('#############################################\n',model)
    gpt_3(context_input, decision, model, max_tokens)


############################################################################################################

def ask_chatGPT(text, model, max_tokens):    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": "This is an Architectural Decision Record. Give a ## Decision corresponding to the ## Context provided by the user"
            },
            {
            "role": "user",
            "content": text
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def chatGPT(context, decision, model, max_tokens):
    max_char = max_tokens*4
    context_truncated = [c[:max_char] for c in context]
    decision_reference_truncated = [c[:max_char] for c in decision]

    predicted_decision = []
    for c in context_truncated:
        if(model == 'gpt-4'):
            time.sleep(30)
        predicted_decision.append(ask_chatGPT(c, model, max_tokens))

    for i in range(len(predicted_decision)):
        if('## Decision' in predicted_decision[i]):
            predicted_decision[i] = predicted_decision[i][predicted_decision[i].find('## Decision')+12:]
            predicted_decision[i] = predicted_decision[i][:predicted_decision[i].find('\n## ')]

    
    results = rouge.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bleu.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = meteor.compute(predictions=predicted_decision, references=decision_reference_truncated)
    ic(results)
    results = bertscore.compute(predictions=predicted_decision, references=decision_reference_truncated, lang="en")
    results = {
        'precision': np.average(results['precision']), 'recall': np.average(results['recall']),
        'f1': np.average(results['f1']), 'hashcode': results['hashcode']
        }
    ic(results)


context, decision, _, context_input = getdata()
ic(len(context_input))
#for m in [('gpt-3.5-turbo', 2000), ('gpt-4', 4000)]:
for m in [('gpt-4', 4000)]:
    model, max_tokens = m
    ic('#############################################\n',model)
    chatGPT(context_input, decision, model, max_tokens)

############################################################################################################

from transformers import T5Tokenizer, T5ForConditionalGeneration

def ask_t5(context, decision, model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='/scratch/rudra.dhar/cache', model_max_length=4096)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='/scratch/rudra.dhar/cache', device_map="balanced")

    predicted_decision = []
    
    for c in context:
        input_ids = tokenizer(c, return_tensors="pt").input_ids.cuda()
        #print(len(input_ids[0]))
        outputs = model.generate(input_ids, max_length=len(input_ids[0])*4, min_length= int(len(input_ids[0])/8))
        predicted_decision.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    results = rouge.compute(predictions=predicted_decision, references=decision)
    ic(results)
    results = bleu.compute(predictions=predicted_decision, references=decision)
    ic(results)
    results = meteor.compute(predictions=predicted_decision, references=decision)
    ic(results)
    results = bertscore.compute(predictions=predicted_decision, references=decision, lang="en")
    results = {
        'precision': np.average(results['precision']), 'recall': np.average(results['recall']),
        'f1': np.average(results['f1']), 'hashcode': results['hashcode']
        }
    ic(results)

    model_name = model_name.replace('/', '_')
    df = pd.DataFrame(list(zip(context, decision, predicted_decision)), columns =['context', 'decision', 'predicted_decision'])
    df.to_csv('/home2/rudra.dhar/Data/SE/adr/output/'+model_name+'.csv', index=False)
    del tokenizer, model, input_ids, outputs, predicted_decision, df
    torch.cuda.empty_cache()


_, decision, _, _, context = getdata()
ic(len(context))
t5_models = ['t5-small', 't5-base', 't5-large', 't5-3b', 'bigscience/T0_3B',
             'google/flan-t5-small','google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
t5_models = ['google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
for model in t5_models:
    ic('#############################################\n',model)
    ask_t5(context, decision, model)

