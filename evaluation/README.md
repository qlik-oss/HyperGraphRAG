# HyperGraphRAG Evaluation

### Preparation
First, to evaluate HyperGraphRAG, we should use ```evaluation``` as the working directory. 
```bash
cd evaluation
```
Then, we need to set openai api key in ```openai_api_key.txt``` file. (We use [www.apiyi.com](https://www.apiyi.com/) for LLM server.)

Last, we need download the contexts and datasets in data folder and put them in the ```contexts``` and ```datasets``` folders. 

```
HyperGraphRAG/
└── evaluation/
    ├── contexts/   
        ├── fiction_contexts.json   
        ├── cs_contexts.json                  
        ├── maud_contexts.json                    
    ├── datasets/           
        ├── fiction/                             
            └── questions.json     
        ├── cs/                            
            └── questions.json 
        ├── maud/                            
            └── questions.json 
    └── openai_api_key.txt                               
```

### Step1. Knowledge HyperGraph Construction
```bash
nohup python script_insert.py --cls fiction > result_fiction_insert.log 2>&1 &
# nohup python script_insert.py --cls agriculture > result_agriculture_insert.log 2>&1 &
# nohup python script_insert.py --cls cs > result_cs_insert.log 2>&1 &
# nohup python script_insert.py --cls legal > result_legal_insert.log 2>&1 &
# nohup python script_insert.py --cls mix > result_mix_insert.log 2>&1 &
```

### Step2. Retrieve Knowledge of HyperGraphRAG

```bash
python script_hypergraphrag.py --data_source fiction 
# python script_standardrag.py --data_source hypertension
# python script_naivegeneration.py --data_source hypertension
```
With PPR optimization:  

```bash
python script_hypergraphrag.py --data_source fiction --ppr 
```

### Step3. Generate Based on Retrieved Knowledge
```bash
python get_generation.py --data_sources fiction --methods HyperGraphRAG
# python get_generation.py --data_sources hypertension --methods StandardRAG
# python get_generation.py --data_sources hypertension --methods NaiveGeneration
```

### Step4. Evaluate the Generation
```bash
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source fiction --method HyperGraphRAG
# CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source hypertension --method StandardRAG
# CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source hypertension --method NaiveGeneration
```

### Step5. See Evaluation Results
```bash
python see_score.py --data_source fiction --method HyperGraphRAG
# python see_score.py --data_source hypertension --method StandardRAG
# python see_score.py --data_source hypertension --method NaiveGeneration
```