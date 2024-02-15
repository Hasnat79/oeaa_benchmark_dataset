# this file evaluates all the generated outputs and ground truths with 
# the help of chatgpt and prompting

import os
import openai
from openai import OpenAI
from tqdm import tqdm
import time

#api key for openai
# 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
# api_key = "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"
# export OPENAI_API_KEY="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"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def timeout(t=30):
    for _ in tqdm(range(t), desc='waiting'):
        time.sleep(1)
wentwrong_1= "When the guy jumps, the son flies up and hits the wall."
goal_1 = "A guy jumps onto a bed where his son is."
gen= "Narration: The video shows a man and a little boy playing on a bed. The man lifts the boy up and throws him on the bed. The boy then jumps up and runs out of the room.\n\nAnswer with Explanation: Yes. The video contains unusual activities as it shows a man lifting a child and throwing him on the bed. This is not a typical activity that one would expect to see in a video.</s>"

score_based_prompt = f"""Given the source document: {wentwrong_1+' '+goal_1}
Given the model-generated text: {gen}
Please score the quality of the generated text from 1 (worst) to 5 (best)
"""
likert_prompt = f"""Given the source document: {wentwrong_1+' '+goal_1}
Given the model-generated text: {gen}
Is the generated text consistent with the source document? (Answer Yes or No)
"""
description = ''
isException = False
try:
    timeout(60)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": "howdy!"},
        ]
    )
    print('response', response)
    # description = response['choices'][0]['message']['content']
except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)

# print(description)