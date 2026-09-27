import os
import re
from tqdm import tqdm
import tiktoken
import numpy as np
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds

def clean_text(text: str) -> str:
    """Normalize whitespace: collapse multiple blank lines, strip per-line."""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)

num_proc = 8
enc = tiktoken.get_encoding("gpt2")
def prepare_dataset():
    #cfg = "cfg"
    hf_dataset="HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
    local_dir = snapshot_download(
        repo_id=hf_dataset,
        repo_type="dataset",
    )
    print(local_dir)
    dataset = ds.dataset(
        local_dir, format="parquet", exclude_invalid_files=True
    )

    tokens_budget = 1e9 # this will later be read from the cfg itself
    arr_len = int(tokens_budget)
    filename = "data.bin" # to be replaced

    dtype = np.uint16 if enc.max_token_value < 65536 else np.uint32
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    idx = 0 
    
    for batch in dataset.to_batches(columns=["text"], batch_size=10_000):
        texts = batch.column("text").to_pylist()
        token_lists = enc.encode_ordinary_batch(
            texts, num_threads=num_proc
        ) # tokenize the text in batches
        for ids in token_lists: # I go over each tokenized sequence
            ids.append(enc.eot_token) # and append the eos token
            # then I append it to the mmaped file
            # first I check len of ids and if it exceeded the budget
            n = len(ids)
            if idx + n > arr_len: 
                n = arr_len - idx # if it did, I only take n tokens that get me exactly at the budget 
                ids = ids[:n]
            arr[idx:idx+n] = ids # I add them to the mmaped file
            idx += n # and I updated the idx, which is just a pointer to  what token position ive last reached 
        if idx >= arr_len:
            break
    arr.flush()
            
                
        
if __name__=="__main__":
    prepare_dataset()