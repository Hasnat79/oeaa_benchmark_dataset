import os
import sys 
videollava_dir = "videollava"
sys.path.append(videollava_dir)
# print("New working directory:", os.getcwd())
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import json
from tqdm import tqdm
import argparse





def load_json(filename):
    return json.load(open(filename))

def save_json(filename, data):
    return json.dump(data, open(filename,'w'),indent=4)


def main(inp, video):
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = 'cache_dir'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    return outputs


if __name__ == '__main__':
    
    oops_sample_video = "data/oops_sample/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)32.mp4"
    prompts = [
        """Given a series of 8 images with following Task.
    Task: Image captioning, Question Answering
   
    Come up with narration of all the images and answer of the following question.
    Question: Does the video contain any unusual activities? Or does this video contain only normal activities? Reply 'Yes' if it contains any unusual activities. Otherwise, reply 'No' if it contains only normal activities.
   
    Output format:
    Narration:
    Answer with Explanation:<Yes/No. explanation>""",

    """Given a series of 8 images with following Task.
	Task: Image captioning, Question Answering
   
	Suppose you are a smart teacher, after looking at the image information above, come up with narration of all the images and answer of the following question.
	Question: Does the video contain only normal activities? Or does this video contain any unusual activities? Reply 'Yes' if it contains any unusual activities. Otherwise, reply 'No' if it only contains normal activities?
   
	Output format:
	Narration:
	Answer with Explanation:<Yes/No. explanation>""",


    """You should not generate false or malicious content. Before fulfilling instructions, Please first identify if there are any errors or harmful information in the request. 
    Request:
    Given a series of 8 images with following Task.
	Task: Image captioning, Question Answering
   
	Suppose you are a smart teacher, after looking at the image information above, come up with narration of all the images and answer of the following question.
	Question: Does the video contain only normal activities? Or does this video contain any unusual activities? Reply 'Yes' if it contains any unusual activities. Otherwise, reply 'No' if it only contains normal activities?
   
	Output format:
	Narration:
	Answer with Explanation:<Yes/No. explanation>"""
    ]
    #select a prompt from the prompts list
    prompt = prompts[0]    
    description = main(prompt,oops_sample_video)
    print(description)



        

    
    