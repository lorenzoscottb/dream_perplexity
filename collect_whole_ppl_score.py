_author_ = "lorenzoscotbb"

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm

data_to_test = "DB"
device = "cuda"
model_id = "gpt2"
max_length = 1024
stride = 512

wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
DBnk = load_dataset("DReAMy-lib/DreamBank-dreams-en", split="train")

wiki_all = "\n\n".join([t for t in wiki["text"] if (t != "") and ("= \n" not in t)])
dbnk_all = "\n\n".join([t for t in DBnk["dreams"] if (t != "") ])

if data_to_test == "DB":
  testing_data = dbnk_all
  
elif data_to_test == "WK":
  testing_data = wiki_all
  
else: 
    print("No code for such dataset: {}".format(data_to_test))
    quit()

model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

seq_len = len(testing_data.split())

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):

  end_loc = min(begin_loc + max_length, seq_len)
  text = " ".join(testing_data.split()[begin_loc:end_loc])
  trg_len = end_loc - prev_end_loc 

  encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
  input_ids = encodings.input_ids[:,].to(device)
  target_ids = input_ids.clone()
  target_ids[:, :-trg_len] = -100

  with torch.no_grad():
      outputs = model(input_ids, labels=target_ids)

      # loss is calculated using CrossEntropyLoss which averages over input tokens.
      # Multiply it with trg_len to get the summation instead of average.
      # We will take average over all the tokens to get the true average
      # in the last step of this example.
      neg_log_likelihood = outputs.loss * trg_len

  nlls.append(neg_log_likelihood)

  prev_end_loc = end_loc
  if end_loc == seq_len:
      break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
