import os
import json
import logging
import random
import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


def get_target_data(dataset, mode=None):
    image_list = []
    if dataset == "advbench":
        csv_file = os.path.join("dataset", "advbench", "harmful_behaviors.csv")
        data = pd.read_csv(csv_file)["goal"].to_list()

    elif dataset == "safebench_tiny":
        csv_file = os.path.join("dataset", "safebench", "question", "SafeBench-Tiny.csv")
        images = os.path.join("dataset", "safebench", "question", "")
        data = pd.read_csv(csv_file)["question"].to_list()

    elif dataset == "mm-safebench":
        csv_file = os.path.join("dataset", "mm-safebench", "samples.csv")
        data = pd.read_csv(csv_file)["content"].to_list()

    elif dataset == "xstest":
        csv_file = os.path.join("dataset", "xstest", "xstest_v2_prompts.csv")
        data = pd.read_csv(csv_file)["prompt"].to_list()
        # data = []
        # for idx, row in pd.read_csv(csv_file).iterrows():
        #     if str(row["type"]) == "safe_contexts":
        #         data.append(str(row["prompt"]))

    elif "safebench" in dataset:
        csv_file = os.path.join("dataset", "safebench", "question", dataset+".csv")
        data = pd.read_csv(csv_file)["question"].to_list()



    if mode == "mm":
        return data, image_list
    else:
        return data

def get_logger(log_file_path):
    logger = logging.getLogger("my logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    log_file_path = log_file_path + ".log"  # Specify the path to your log file.
    file_handler = logging.FileHandler(log_file_path)

    logger.addHandler(file_handler)

    logger.addHandler(handler)

    return logger


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
# print(get_target_data("advbench"))
# load_image("view.jpg")


def prepare_prompt(inp, model, model_name, image):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("the total prompt is:", prompt)
    return prompt

def get_conv(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    return conv

def exp_img_parse(img_ll):
    exp_imgs = dict()
    if img_ll is None:
        return None
    else:
        indices = set([img.split("_")[1] for img in img_ll])
        for idx in indices:
            exp_imgs[idx] = []

        for img in img_ll:
            idx = img.split("_")[1]
            exp_imgs[idx].append(img)
        for idx in indices:
            exp_imgs[idx] = sorted(exp_imgs[idx], key=lambda x: int(x.split("_")[-2]))
        return exp_imgs

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + '\n')

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def create_mm_safebench():
    total_content = []
    total_source = []
    df_new = pd.DataFrame()
    files = os.listdir(os.path.join("dataset", "mm-safebench"))
    text_files = [file for file in files if file.endswith(".txt")]
    for file in text_files:
        with open(os.path.join(os.path.join("dataset", "mm-safebench", file))) as infile:
            content = infile.readlines()
            content = [item.strip() for item in content]
            total_content += random.sample(content, 10)
            total_source += [file.split(".")[0] for _ in range(10)]

    # print(total_source)
    # print(total_content)
    df_new["source"] = total_source
    df_new["content"] = total_content
    df_new.to_csv(os.path.join(os.path.join("dataset", "mm-safebench", "samples.csv")))

# create_mm_safebench()

def safebench_split():
    csv_file = os.path.join("dataset", "safebench", "question", "safebench.csv")
    df = pd.read_csv(csv_file)
    split_size = len(df) // 5
    splits = [df.iloc[i:i + split_size] if i + split_size < len(df) else df.iloc[i:]
              for i in range(0, len(df), split_size)]
    for idx, split in enumerate(splits):
        file_name = csv_file.split(".")[0] + str(idx+1) + ".csv"
        split.to_csv(file_name, index=True)
        print(file_name, "saved!")
        # data = pd.read_csv(csv_file)["question"].to_list()

# safebench_split()

def test_vlguard():
    json_file = os.path.join("results", "test_vlguard.json")
    content = json.load(open(json_file))
    df_safe = pd.DataFrame()
    df_unsafe = pd.DataFrame()

    safe_ll, unsafe_ll = [], []
    cnt = 0
    for item in content:
        if item['safe']:
            for subitem in item["instr-resp"]:
                if "safe_instruction" in subitem:
                    safe_ll.append(subitem["safe_instruction"])
                elif "unsafe_instruction" in subitem:
                    unsafe_ll.append(subitem["unsafe_instruction"])
    df_safe["question"] = safe_ll
    df_unsafe["question"] = unsafe_ll
    df_safe.to_csv(os.path.join("results", "vlguard_safe.csv"))
    df_unsafe.to_csv(os.path.join("results", "vlguard_unsafe.csv"))

        # print(item)
        # cnt += 1

    # print(cnt)

# print(get_target_data("xstest"))
# test_vlguard()

