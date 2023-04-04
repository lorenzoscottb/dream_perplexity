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

dataset    = load_dataset("DReAMy-lib/DreamBank-dreams-en")
dataset_df = pd.DataFrame(dataset["train"])
dataset_df["No.Words"] = [
    len(nltk.word_tokenize(report)) for report in tqdm(dataset_df["dreams"])
]

# Since the perplexity metric from hugging face is still not 100% hacakble 
# (i.e., is not easy to set truncation), we'll have to select those reports with a 
# "limited" ammount of words
dataset_df_test = dataset_df[dataset_df["No.Words"] <= 700]
dream_list = dataset_df_test["dreams"].tolist()


# Check the space (https://huggingface.co/spaces/evaluate-metric/perplexity) for more info
perplexity = load("perplexity", module_type="metric")

p = perplexity.compute(predictions=dream_list, model_id='gpt2', add_start_token=False)


# lets now update (and save) our test DataFrame
dataset_df_test["GPT2_perplexities"] = list(p["perplexities"])

dataset_df_test.to_csv("DreamBank_en_pptx_GPT2.csv")
