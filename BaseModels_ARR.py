from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import pandas as pd
import openai
import os
import json
import time

SYSTEM_PROMPT = "You are an expert detective."
USER_PROMPT_PREFIX = \
"You are provided with a situation of deliberate misrepresentation. There are three contestants, Number One, Number Two, and Number Three, all claiming to be the same person. However, there is only of them who is actually the person they are all claiming to be. There is an affidavit that includes truthful information about the claimed person. The affidavit is publicly available. However, the affidavit does not contain all truth about the person, and it is possible that new information is obtained in the conversation which is actually true.\n\n" \
"There is a judge who cross-questions all three contestants to uncover the truth. All the contestants will get monetary rewards if they can deceive the judge. However, the contestant, who is the real person, is sworn to speak truthfully. But they will most often utter half-truths to deceive the judges.\n\n" \
"The name of the claimed person is given as Name:\n" \
"The affidavit of the claimed person is given as Affidavit:\n" \
"The conversation (in a question-answer form) between the judge and the contestants is given as Conversations:\n" \
"If a question starts with addressing a particular contestant (e.g., Number One), then that question and all the following questions are addressed to that contestant unless a different contestant (e.g., Number Two) is explicitly addressed."


# load examples
data_path = ""
tttt_gold_inputs = []
for file in os.listdir(data_path)[:1]:
    with open(os.path.join(data_path, file), "r") as f:
        tttt_input = f.readlines()
    tttt_gold_inputs.append(''.join(tttt_input))

MODEL = 'gpt-4'
# MODEL = 'gpt-3.5-turbo-16k'
# MODEL = 'gpt-3.5-turbo'

# load gold answers
gold_answers = pd.read_csv("")
gold_answers_map = {}
answer_map = defaultdict(str, {'1': "number one", '2': "number two", '3': "number three"})
for _, row in gold_answers.iterrows():
    gold_answers_map["e{}s{}".format(str(row['episode']), str(row['session']))] = answer_map[str(row['cc'])]


def predict(tttt_input, user_prompt_prefix, user_prompt_suffix, system_prompt, model, temp=.7, tokens=750):

    full_input = user_prompt_prefix + "\n\nBased on the affidavit and the conversation, predict the contestant who is not an imposter. First, generate your rationale behind your prediction. Then, write ### followed by the single option from {Number One, Number Two, Number Three} as the answer." + "\n\n" + tttt_input + "\n\nAnswer:"

    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": full_input}],
                    temperature=temp, 
                    max_tokens=tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        except Exception as e:
            time.sleep(2)

    output_summary = response["choices"][0]["message"]["content"]
    return output_summary

def calculate_accuracy(predictions, save=False): 
    correct = 0

    for file, prediction in predictions:
        if prediction['prediction'].lower() == gold_answers_map[file]:
            correct += 1.0
    
    accuracy = correct/len(predictions)
    print("Accuracy: {}".format(accuracy))

    if save:
        # saving predictions
        pred_json = json.dumps(dict(predictions))
        time = datetime.now().strftime("%b_%d_%Y_%H:%M:%S")

        save_dir = ""
        with open(os.path.join(save_dir, "Acc_{}_Run_{}_M_{}.json".format(round(accuracy*100, 2), time, MODEL)), "w") as outfile:
            outfile.write(pred_json)

def top_2_acc(predictions, save=False): 
    correct = 0

    for file, prediction in predictions:
        # if 
        if gold_answers_map[file] in prediction['rank'][:2]:
            correct += 1.0
    
    accuracy = correct/len(predictions)
    print("Accuracy@2: {}".format(accuracy))

    if save:
        # saving predictions
        pred_json = json.dumps(dict(predictions))
        time = datetime.now().strftime("%b_%d_%Y_%H:%M:%S")

        save_dir = ""
        with open(os.path.join(save_dir, "Acc_{}_Run_{}_M_{}.json".format(round(accuracy*100, 2), time, MODEL)), "w") as outfile:
            outfile.write(pred_json)

# read TTTT
data_path = ""
tttt_inputs = []
for idx, file in enumerate(os.listdir(data_path)):
    if file.endswith(".txt"):
        with open(os.path.join(data_path, file), "r") as f:
            tttt_input = f.readlines()
        tttt_inputs.append((file[:-4], ''.join(tttt_input)))

predictions = []
reasons = []
for file, tttt_input in tqdm(tttt_inputs[2:]):
    if MODEL == 'gpt-4':
        time.sleep(2)
        
    prediction = predict(
        tttt_input=tttt_input,
        user_prompt_prefix=USER_PROMPT_PREFIX,
        user_prompt_suffix="",
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temp=0,
        tokens=2048
        )
    
    rationale = prediction.split('###')[0]
    ranked_list = [x.lstrip().rstrip().lower() for x in prediction.split("###")[1].split(",")]
    print(file, ranked_list, gold_answers_map[file])
    
    predictions.append((file, {
        "rank": ranked_list,
        "prediction": ranked_list[0],
        "gold": gold_answers_map[file],
        "rationale": rationale,
        "verdict": ranked_list[0] == gold_answers_map[file].lower()
        }))
    
    top_2_acc(predictions, save=False)

top_2_acc(predictions, save=True)
