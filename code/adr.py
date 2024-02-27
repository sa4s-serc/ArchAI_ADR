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

def getdata(experiment='0_shot', path='../data/'):
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
            else:
                context.append(content[content.find('Context and Problem Statement')+30:content.find('Considered Options')])

            if('\nPros and Cons of ' in content):
                decision.append(content[content.find('Decision Outcome')+17:content.find('\nPros and Cons of ')])
            else:
                decision.append(content[content.find('Decision Outcome')+17:])

    if(experiment=='0_shot'):    
        context_input_gpt3 = ["Architectural Decision Record\n\n## Context\n" +c+ "\n\n## Decision\n" for c in context]
        context_input_chatGPT = ["## Context\n\n" +c for c in context]
        context_input_T5 = ["This is a Architectural Decision Record. Provide a Decision for the Context given below.\n\n## Context\n" +c for c in context]
        
        #return context, decision, context_input_gpt3, context_input_chatGPT, context_input_T5
        df = pd.DataFrame(list(zip(context, decision, context_input_gpt3, context_input_chatGPT, context_input_T5)),
                          columns =['context', 'decision', 'context_input_gpt3', 'context_input_chatGPT', 'context_input_T5'])
        df.to_csv(os.path.join(path, '0_shot.csv'), index=False)
    else:
        with open(os.path.join(path, 'prompt.txt'), 'r') as f:
            prefix = f.read()
        context_input_gpt3 = [prefix+ "\n\n## Context\n" +c+ "\n\n## Decision\n" for c in context]
        #return context, decision, context_input_gpt3
        df = pd.DataFrame(list(zip(context, decision, context_input_gpt3)),
                          columns =['context', 'decision', 'context_input_gpt3'])
        df.to_csv(os.path.join(path, 'few_shot.csv'), index=False)

    
def load_data(model, experiment='0_shot', path='../data/'):
    if(experiment=='0_shot'):
        df = pd.read_csv(os.path.join(path, '0_shot.csv'))
        if(model=='gpt3'):
            return df['context_input_gpt3'].tolist(), df['decision'].tolist()
        elif(model=='chatGPT'):
            return df['context_input_chatGPT'].tolist(), df['decision'].tolist()
        elif(model=='T5'):
            return df['context_input_T5'].tolist(), df['decision'].tolist()
        else:
            return df['context'].tolist(), df['decision'].tolist()
    else:
        df = pd.read_csv(os.path.join(path, 'few_shot.csv'))
        if(model=='gpt3'):
            return df['context_input_gpt3'].tolist(), df['decision'].tolist()
        else:
            return df['context'].tolist(), df['decision'].tolist()



############################################################################################################

def select_data(context, decision, length):
    context_new, decision_new = [], []
    for c,d in zip(context, decision):
        if(len(c) < length and len(d) < length):
            context_new.append(c)
            decision_new.append(d)
        
    return context_new, decision_new

def print_results(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    ic(results)
    results = bleu.compute(predictions=predictions, references=references)
    ic(results)
    results = meteor.compute(predictions=predictions, references=references)
    ic(results)
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    results = {
        'precision': np.average(results['precision']), 'recall': np.average(results['recall']),
        'f1': np.average(results['f1']), 'hashcode': results['hashcode']
        }
    ic(results)

def store_output(model_name, context, decision, predicted_decision, experiment='0_shot'):
    df = pd.DataFrame(list(zip(context, decision, predicted_decision)), columns =['context', 'decision', 'predicted_decision'])
    model_name = model_name.replace('/', '_')
    if(experiment=='0_shot'):
        df.to_csv('../output/0_shot/'+model_name+'.csv', index=False)
    else:
        df.to_csv('../output/few_shot/'+model_name+'.csv', index=False)


############################################################################################################

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def gpt_2(context, decision, model_name, experiment='0_shot'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, device_map="auto")
    prediction = []
    for c in context:
        encoded_input = tokenizer(c, return_tensors='pt').input_ids.to('cuda')
        if(experiment=='0_shot'):
            output = model.generate(encoded_input, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        else:
            output = model.generate(encoded_input, max_new_tokens=250, pad_token_id=tokenizer.eos_token_id)
        prediction.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    predicted_decision = []
    for p in range(len(prediction)):
        predicted_decision.append(prediction[p][len(context[p]):])

    store_output(model_name, context, decision, predicted_decision, experiment)

    del tokenizer, model
    torch.cuda.empty_cache()
    print_results(predicted_decision, decision)



experiment = 'few_shot'

if(experiment=='0_shot'):
    context, decision = load_data('gpt3')
    context, decision = select_data(context, decision, 2000)
else:
    context, decision = load_data('gpt3', experiment='few_shot')
    context, decision = select_data(context, decision, 3000)

ic(len(context))
for model in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    ic('#############################################\n',model)
    gpt_2(context, decision, model, experiment=experiment)


############################################################################################################

import openai
from dotenv import load_dotenv
path = ''
load_dotenv(path + '.env')
openai.api_key = os.getenv("YOUR_KEY")

def askGPT3(text, model, max_tokens):    
    response = openai.Completion.create(
        model=model,
        prompt=text,
        max_tokens=max_tokens,
    )
    return response.choices[0].text

def gpt_3(context, decision, model, max_tokens, experiment='0_shot'):
    if(experiment=='0_shot'):
        context, decision = select_data(context, decision, 2*max_tokens)
    else:
        context, decision = select_data(context, decision, 3*max_tokens)
        
    predicted_decision = []
    for c in context:
        if(experiment=='0_shot'):
            predicted_decision.append(askGPT3(c, model, int(max_tokens/2)))
        else:
            predicted_decision.append(askGPT3(c, model, int(max_tokens/4)))

    store_output(model, context, decision, predicted_decision, experiment)
    print_results(predicted_decision, decision)


experiment= 'few_shot'
context, decision = load_data('gpt3', experiment=experiment)
ic(len(context))
# Total length of tokens should be less than 2048 for ada, 4096 for text-davinci-003
# approx less than 1000 tokens for context and 1000 tokens for decision for ada
# 500 tokens approx 4000 characters

for m in [('ada', 2000), ('text-davinci-003', 4000)]:
    model, max_tokens = m
    ic('#############################################\n',model)
    gpt_3(context, decision, model, max_tokens, experiment)


# ############################################################################################################

import openai
from dotenv import load_dotenv
path = ''
load_dotenv(path + '.env')
openai.api_key = os.getenv("YOUR_KEY")

def ask_chatGPT(text, model, max_tokens, count):
    # time.sleep(30)
    try:
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
            # request_timeout=600
        )
        return response.choices[0].message.content
    except Exception as e:
        print(count)
        print(e)
    


def ask_chatGPT_fewshot(text, model, max_tokens, count):
    # time.sleep(30)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                "role": "system",
                "content": "These are architecture decision records. Follow the examples to get return Decision based on Context provided by the USER"
                },
                {
                "role": "user",
                "content": "## Context\n\nIOG is undertaking a company-wide effort to restructure and standardize its repositories, favoring mono-repos and enforcing shared GitOps and DevOps processes. Parallel to this, a new CI infrastructure is being developed.\n\nExamples of this are:\n\n    input-output-hk/cardano-world\n\n    input-output-hk/ci-world\n\n    input-output-hk/atala-world\n\nThis initiative appears to be championed by the SRE team who are the creators of divnix/std. Indeed std is at the heart of the standardization dream."
                },
                {
                "role": "assistant",
                "content": "## Decision\n\n    Standardization of the repositories has been deemed a worthwhile endeavour, though of very low priority.\n\n    Phase 1 of the standardization process will be carried out in parallel with Move Marconi to a separate repository. A separate repository will be created for Marconi, and from the very beginning it will use std. This way the benefits, limitations and integration costs of std can be experienced and measured, and an informed, definitive decision on standardizing plutus-core and plutus-apps themselves can be made."
                },
                {
                "role": "user",
                "content": "## Context\n\nWe need to decide on which database management system (DBMS) to use for Project X. The database will be used to store and manage large amounts of data from multiple sources. We need a DBMS that can handle transactions, offer scalability, and provide high reliability and security. Among various options available, we are considering MySQL as a possible choice.\n\n### Decision Considerations\n\n- Ease of use and maintenance\n\n- Community support and resources\n\n- Performance and scalability\n\n- Security and reliability\n\n- Cost and licensing\n\n- Compatibility with our technology stack\n\n### Considered Options\n\n- MySQL\n\n- PostgreSQL\n\n- Oracle\n\n- Microsoft SQL Server\n\n- MongoDB"
                },
                {
                "role": "assistant",
                "content": "## Decision\n\nAfter evaluating the above options based on our decision considerations, we have decided to choose MySQL as our DBMS for Project X.\n\nMySQL is a popular open-source system with a strong development community and a large pool of resources for problem-solving and knowledge sharing. It is well-known for its excellent performance and scalability capabilities, making it ideal for handling vast amounts of data with high levels of efficiency. The platform is secure, reliable, and has a wide range of features that are essential for our project, including ACID compliance for transactions, flexible data model, and support for various programming languages and frameworks.\n\nMySQL is also compatible with the majority of our technology stack, including our web development framework, hosting solutions, and other essential tools. Plus, its cost and licensing terms are competitive compared to other proprietary systems like Oracle and Microsoft SQL Server."
                },
                {
                "role": "user",
                "content": text
                },
            ],
            max_tokens=max_tokens,
            # request_timeout=60
        )
        return response.choices[0].message.content
    except Exception as e:
        print(count)
        print(e)
    


def gpt_3_4(context, decision, model, max_tokens, experiment='0_shot'):
    if(experiment=='0_shot'):
        context, decision = select_data(context, decision, 2*max_tokens)
    else:
        context, decision = select_data(context, decision, 3*max_tokens)
        
    ic(len(context), len(decision))
    predicted_decision = []
    count = 0
    for c in context:
        count += 1
        c = "## Context\n\n" + c
        if(experiment=='0_shot'):
            predicted_decision.append(ask_chatGPT(c, model, int(max_tokens/2), count))
        else:
            predicted_decision.append(ask_chatGPT_fewshot(c, model, int(max_tokens/4), count))
        if(count%10==0):
            ic(count)

    store_output(model, context, decision, predicted_decision, experiment)
    print_results(predicted_decision, decision)


experiment= '0_shot'
context, decision = load_data('gpt3_4', experiment=experiment)
ic(len(context))

for m in [('gpt-3.5-turbo', 4000)]:
    model, max_tokens = m
    ic('#############################################\n',model)
    gpt_3_4(context, decision, model, max_tokens, experiment)



experiment= 'few_shot'
context, decision = load_data('gpt3_4', experiment=experiment)
ic(len(context))

for m in [('gpt-3.5-turbo', 4000)]:
    model, max_tokens = m
    ic('#############################################\n',model)
    gpt_3_4(context, decision, model, max_tokens, experiment)

# ############################################################################################################

from transformers import T5Tokenizer, T5ForConditionalGeneration

def ask_t5(context, decision, model_name, experiment='0_shot'):
    if(experiment=='0_shot'):
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='../cache', model_max_length=2000)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='../cache', model_max_length=2500)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='../cache', device_map="balanced")

    predicted_decision = []
    
    for c in context:
        input_ids = tokenizer(c, return_tensors="pt").input_ids.cuda()
        #print(len(input_ids[0]))
        outputs = model.generate(input_ids, max_length=len(input_ids[0])*4, min_length= int(len(input_ids[0])/8))
        predicted_decision.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    store_output(model_name, context, decision, predicted_decision, experiment)
    print_results(predicted_decision, decision)
    del tokenizer, model
    torch.cuda.empty_cache()


experiment= 'few_shot'

if(experiment=='0_shot'):
    context, decision = load_data('gpt3', experiment=experiment)
    context, decision = select_data(context, decision, 8000)
else:
    context, decision = load_data('gpt3', experiment=experiment)
    context, decision = select_data(context, decision, 10000)

ic(len(context))

t5_models = ['t5-small', 't5-base', 't5-large', 't5-3b', 'bigscience/T0_3B',
             'google/flan-t5-small','google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
for model in t5_models:
    ic('#############################################\n',model)
    ask_t5(context, decision, model, experiment)

