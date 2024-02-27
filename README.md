# Can LLMs Generate Architectural Design Decisions?- An Exploratory Empirical study

This repository contains both the code and data associated with the paper: 
"Can LLMs Generate Architectural Design Decisions?- An Exploratory Empirical study"

## Abstract

Architectural Knowledge Management (AKM) involves the organized handling of information related to architectural decisions and design within a project or organization. An essential artefact of AKM is the Architecture Decision Records (ADR), which documents key design decisions. ADRs are documents that capture decision context, decision made and various aspects related to a design decision, thereby promoting transparency, collaboration, and understanding. Despite their benefits, ADR adoption in software development has been slow due to challenges like time constraints and inconsistent uptake. Recent advancements in Large Language Models (LLMs) may help bridge this adoption gap by facilitating ADR generation. However, the effectiveness of LLM for ADR generation or understanding is something that has not been explored. To this end, in this work, we perform an exploratory study which aims to investigate the feasibility of using LLM for the generation of ADRs given the decision context. In our exploratory study, we utilize GPT and T5-based models with 0-shot, few-shot, and fine-tuning approaches to generate the Decision of an ADR given its Context. Our results indicate that in a 0-shot setting, state-of-the-art models such as GPT-4 generate relevant and accurate Design Decisions, although they fall short of human-level performance. Additionally, we observe that more cost-effective models like GPT-3 can achieve similar outcomes in a few-shot setting, and smaller models such as Flan-T5 can yield comparable results after fine-tuning. To conclude, this exploratory study suggests that LLM can generate Design Decisions, but further research is required to attain human-level generation and establish standardized widespread adoption.

## Repository structure

1. code
    - test_env.yml contains the required packages. One can create a conda env from this file with the command: <br> conda env create -f test_env.yml
    - adr.py contains code for all 0-shot and few-shot experiments, for all models.
    - training.ipynb contains code for finetuning and inference.
2. data
    - The collected data after preprocessing is in 0_shot.csv and few_shot.csv. The models ingest data from these two files.
    - To view the original collected data unzip the zip files provided.
3. Simplefied_Example
       Running the whole code with all the data will take time, compute power and OpenAI api credits. Hence in the [Simplified_Example.md](https://github.com/sa4s-serc/ArchAI_ADR/blob/main/Simplified_Example.md) file we record the way to execute a simgle sample.

## Steps to run

1. Create a conda environment with code/test_env.yml file with the command: <br> conda env create -f test_env.yml
2. Creat a .env file and save you OpenAI api key as: <br> YOUR_KEY=******
3. Run the adr.py file to execute all the 0-shot and few-shot experiments.
    - This might take many hours and OpemAI credits ($10 - $20)
    - To run only experiments based on GPT models, comment T5 specific codes (after line 354)
    - To run only experiments based on T5 models, comment GPT specific codes (from like 150 to 354)
4. Execute training.ipynb notebook to train and test LLMs for generating Design Decisions

## System Specification

1. The experiments were run on Ubuntu version 18.04 with python installed.
2. The experiments were run on a high-performance compute node with 40 cores, 80GB ram and 4 GPUs ranging from 'NVIDIA GeForce RTX 2080' to 'NVIDIA GeForce RTX 3080' each with 12GB GPU RAM.
