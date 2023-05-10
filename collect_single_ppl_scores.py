_author_ = "lorenzoscottb"

import os, nltk

cuda_device = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset

corpus = "DB"
model_name  = "gpt2"

if corpus == "DB":
    dataset    = load_dataset("DReAMy-lib/DreamBank-dreams-en")
    dataset_df = pd.DataFrame(dataset["train"])
    dataset_df["No.Words"] = [
        len(nltk.word_tokenize(report)) for report in tqdm(dataset_df["dreams"])
    ]

    data_as_list = dataset_df["dreams"].tolist()

elif corpus == "WK":
    input_texts = datasets.load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )["text"]
    
    # remove empty lists, and titles
    data_as_list = [s for s in tqdm(input_texts) if (s!='') and ("= \n" not in s)]
    No_Words = [
        len(nltk.word_tokenize(report)) for report in tqdm(input_texts)
    ]
    
    dataset_df = pd.DataFrame(
        {
            "Text":data_as_list,
            "No.Words":No_Words
        }
    )
    
else: 
    print("No code for such dataset: {}".format(corpus))
    quit()
    
# Check the space (https://huggingface.co/spaces/evaluate-metric/perplexity) for more info
perplexity = load("perplexity", module_type="metric")

p = perplexity.compute(predictions=data_as_list, model_id=model_name, add_start_token=False)

# lets now update (and save) our test DataFrame
dataset_df["GPT2_perplexities"] = list(p["perplexities"])

dataset_df.to_csv("{}_en_pptx_GPT2.csv".format(corpus))
