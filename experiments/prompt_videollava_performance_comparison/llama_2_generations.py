import sys
from constants import LLAMA_DIR, MP, LLAMA_2_7B,TOKENIZER_PATH, LLAMA_2_7B_CHAT
sys.path.append(LLAMA_DIR)
from llama import Llama
from typing import List
import json
from tqdm import tqdm
import json 

def llama2_chat_completion (videollava_gen:str, ckpt_dir: str=LLAMA_2_7B_CHAT,
    tokenizer_path: str= TOKENIZER_PATH,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int =2024,
    max_gen_len: int = 1024,  
    max_batch_size: int = 4,):
    
    max_prompt_len = len( videollava_gen.split(" "))
    if max_prompt_len > max_seq_len:
        return "Prompt length is greater than max_seq_len 2024"

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        [{
            "role": "user","content":f"""Considering the following sentences describing activities happening in the video, does the video contain any unusual activities? Reply Yes or No first, then explain why. Description:
        {videollava_gen}"""
        }],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # print(f"results: {results[0]['generation']['content']}")
    
    return results[0]['generation']['content']

def llama2_text_completion (prompts:str, ckpt_dir: str=LLAMA_2_7B,
    tokenizer_path: str= TOKENIZER_PATH,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print(f"prompts: {prompts}")
    result = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"result: {result}")
    return result

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent = 4)


if __name__ == "__main__":
    result = "./_results/p3_videollava_x_oops_baseline_res.json"
    with open(result, "r") as f:
        p_3_videollava_x_baseline_res = json.load(f)

        llm_prompts =[
    """Considering the following sentences describing activities happening in a video, does the video contain any unusual activities? Reply Yes or No first, then explain why. Description:
        """
    ]


    for k in tqdm(list(p_3_videollava_x_baseline_res.keys())):

        v = p_3_videollava_x_baseline_res[k]
        # print(k,v)
        videollava_gen =  v[f'videollava_generation_{3}'].split("<")[0]
        p = llm_prompts[0]+videollava_gen

        # print(p)
        try: 
            result= llama2_chat_completion(p)
        except Exception as e:
            print(e)
            result = "Error"
            print(k)
        v[f'videollava_gen{3}_llama2_generation']= result
        # print(results)
        # v[f'llama2_videollava_generation_{3}']= llama2_generation(p)
        
        # print("done")
        save_json(f"_results/p3_videollava_x_oops_baseline_res.json", p_3_videollava_x_baseline_res)
  
# torchrun --nproc_per_node 1