import json
import os

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
import yadisk

import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer, XCLIPTextModel


# всякие переменные и пути
DEVICE = 'cuda'
video_folder_path = '../dataset/videos/'
no_subs_path = '../dataset/no_subs.json'
download = False
sub_folder_path = '../dataset/subs/'
sub_mapping_file_path = '../dataset/id2name.json'
embedding_path = '../dataset/processed/embeddings/'
info_path = '../dataset/processed/info/'



video_model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")

text_model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")


text_model.to(DEVICE)
video_model.to(DEVICE)


def encode_frame(frame):
    return torch.tensor([0.1,0.2,0.3])

def encode_video(frames):
    # внутри 1280 x 720 -> 224 x 224
    with torch.no_grad():
        while len(frames) < 8:
            frames.append(frames[-1])
        inputs = processor(videos=[frames], return_tensors="pt").to(DEVICE)
        video_features = video_model.get_video_features(**inputs)
        video_features = video_features.to('cpu')
    return video_features[0]

def encode_text(text):
    with torch.no_grad():
        inputs = tokenizer([text], 
                        truncation=True,
                        padding=True, 
                        return_tensors="pt").to(DEVICE)
        outputs = text_model(**inputs).pooler_output.to('cpu')
        return outputs
    
def find_sub_vid_overlap(sub_s, sub_e, vid_s, vid_e, min_overlap=3):
    sub = [i for i in range(int(sub_s), int(sub_e)+1, 1)]
    vid = [i for i in range(int(vid_s), int(vid_e)+1, 1)]
    overlap = len(set(sub) & set(vid))
    if overlap == len(sub):
        return True
    elif overlap > min_overlap:
        return True
    return False


def extract_frames(cap, sub_df, reuse_frames=True,
                   window_size=11, overlap=5, new_fps=3):
    '''
    sub_df = dataframe with columns "text", "start", "end"
    overlap = non-overlap between video pieces in seconds (sliding window step)
    window_size = size of video pieces in seconds
    new_fps = fps we read the video with

    if no overlap needed, just set overlap = window_size
    '''

    video_embeddings = [] # эмбеддинг видео кусочка
    text_embeddings = [] # эмбеддинг соответствующего текста
    seconds = []
    texts = []

    # определим кадры, которые пойдут в наши кусочки
    good_frames_ids = []
    fps = cap.get(cv2.CAP_PROP_FPS) # собственное fps видео
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = fps // new_fps # например, 25//3 = 8

    for i in range(0, int(frame_count), int(fps)):
        good_frames_ids.extend([i+k*frame_step for k in range(new_fps)])

    good_frames_ids = set(good_frames_ids) # для ускорения поиска
    # прочитаем эти кусочки
    good_frames = [] # это возможно не считается на маломощной машине
    frame_embeddings = []
    m = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if m in good_frames_ids:
            # .to_ndarray(format="rgb24")
            if reuse_frames:
                frame_embeddings.append(encode_frame(frame))
            else:
                good_frames.append(frame)
        m += 1

    # теперь идём по ним с нужным шагом и вынимаем субтитры
    for step in range(0, max(len(good_frames), len(frame_embeddings)), new_fps*overlap):
        seconds_start = step / new_fps
        seconds_end = seconds_start + window_size

        frame_sub_df = sub_df[(((sub_df['start'] > seconds_start) & # начало текста внутри видео
                                (sub_df['start'] < seconds_end)) |
                                ((sub_df['end'] > seconds_start) & # конец текста внутри видео
                                (sub_df['end'] < seconds_end)) |
                                ((sub_df['end'] > seconds_end) & # видео внутри текста
                                (sub_df['start'] < seconds_start)) 
                                )]
        subs = ''
        for _, row in frame_sub_df.iterrows():
            if find_sub_vid_overlap(row['start'], row['end'], 
                                        seconds_start, seconds_end, 
                                        min_overlap=2):
                subs += row['text'] + ' '
            
        frames_start = step
        frames_end = frames_start + window_size*new_fps
        if reuse_frames:
            initial_frames_emb = frame_embeddings[frames_start:frames_end]
            frames_emb = torch.concatenate(initial_frames_emb)
            frames_emb = frames_emb.reshape(len(initial_frames_emb), len(initial_frames_emb[0]))
            frames_emb = torch.mean(frames_emb, axis=0)
        else:
            frames_emb = encode_video(good_frames[frames_start:frames_end])

        text_embeddings.append(encode_text(subs.strip(' ')))
        video_embeddings.append(frames_emb)
        seconds.append([seconds_start, seconds_end])
        texts.append(subs)
        torch.cuda.empty_cache()
        del frames_emb

    return seconds, video_embeddings, text_embeddings, texts


#  было бы неплохо привести это в норм вид потом для публичного датасета 
def read_subtitles(sub_folder_path, sub_mapping_file_path):
    id2name = json.load(open(sub_mapping_file_path, 'r', encoding='utf-8'))
    bad_symbols = '?\":/|'

    caption_dict = dict() # Название видео : DataFrame(columns=["text", "start", "end"])
    #new_caption_dict = dict()
    for caption_file in tqdm([sf for sf in os.listdir(sub_folder_path)]):
        fn = sub_folder_path+ caption_file
        this_cap_list = []
        with open(fn, 'r', encoding='utf-8') as f:
            for caption in json.load(f):
                cap_dict = caption.copy()
                this_cap_list.append({'text': cap_dict['text'], 
                                    'start': cap_dict['start'],
                                    'end': cap_dict['start'] + cap_dict['duration']})
        
        name = caption_file.split('/')[-1].split('.')[0]
        video_name = id2name[name]+'.mp4'
        for bs in bad_symbols:
            video_name = video_name.replace(bs, '')
        if video_name == 'Интервью с двукратным сурдлимпийским чемпионом Владиславом Винником. 2 часть.mp4':
            video_name = "Интервью с двукратным сурдлимпийским чемпионом Владиславом Винником. 2 часть. С субтитрами.mp4"
        caption_dict[video_name] = pd.DataFrame(data=this_cap_list)
    return caption_dict


# Все видео (316 по итогу)
no_subs_set = set() # видео без субтитров нас не интересуют
with open(no_subs_path, 'r', encoding='utf-8') as f:
    no_subs = json.load(f)
    for no_sub in no_subs:
        no_subs_set.add(no_sub['vid_path'].replace('\"', ''))


caption_dict = read_subtitles(sub_folder_path, sub_mapping_file_path)


files = os.listdir(video_folder_path) # или list(client.listdir("disk:/SLR Project"))
for file in tqdm(files):
    if download:
        video_name = file['path'].split('/')[-1]
        path_on_disk = file['path']
        path_to_store = './' + video_name
        # и прописать сюда скачивание
        pass
    else:
        video_name = file.split('/')[-1]
    if video_name not in no_subs_set and video_name in caption_dict:
        sub_df = caption_dict[video_name]
        
        cap = cv2.VideoCapture(video_folder_path + file)
        seconds, video_embeddings, text_embeddings, texts = extract_frames(cap, sub_df,
                                                                           window_size=4,
                                                                           new_fps=2,
                                                                           overlap=4,
                                                                           reuse_frames=False)
        torch.save([video_embeddings, text_embeddings], f'{embedding_path}{video_name}.pt')
        with open(f'{info_path}{video_name}.json', 'w', encoding='utf-8') as f:
            json.dump([seconds, texts], f, ensure_ascii=False, indent=4)
        
    if download:
        # удалить скачанное видео
        pass