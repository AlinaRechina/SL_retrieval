import logging
from typing import List, Callable

import os
import json

from http import HTTPStatus
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import yadisk

from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
import numpy as np

import config
from pymorphy2 import MorphAnalyzer

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./logs/myapp.log', level=logging.INFO)

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"tryItOutEnabled": True}
)

class SearchRequest(BaseModel):
    text: List[str]
    topn: int
    token: str #yadisk.Client
    disk_emb_path: str

class SearchResponse(BaseModel):
    idxs: List[List[int]]

class EdaResponse(BaseModel):
    result: List[List]

# При запуске
def _startup_model(app: FastAPI) -> None:
    logger.info('Starting the app')

    #app.state.model = CLIPTextModelWithProjection.from_pretrained(
    #config.model, device=config.device)
    #app.state.tokenizer = AutoTokenizer.from_pretrained(
    #config.model, device=config.device)
    logger.info('Loaded the model')
    app.state.vid_embs = None

def start_app_handler(app: FastAPI) -> Callable:
    '''При запуске приложения'''
    def startup() -> None:
        _startup_model(app)
    return startup

app.add_event_handler("startup", start_app_handler(app))

@app.post("/get_vids", response_model=SearchResponse, status_code=HTTPStatus.OK)
async def search(request:SearchRequest):
    '''Главная ручка'''
    if 'embeddings.pt' not in os.listdir(config.corpus['folder_for_embedings']):
        url = request.disk_emb_path+'embeddings.pt'
        output = config.corpus['folder_for_embedings']+'embeddings.pt'
        client = yadisk.Client(token=request.token)
        client.download(url, output) 
        logger.info('Downloaded embeddings')
    if not app.state.vid_embs:
        app.state.vid_embs = KeyedVectors(config.dim)
        vectors = torch.load(config.corpus['folder_for_embedings']+'embeddings.pt')
        app.state.vid_embs.add_vectors([i for i in range(vectors.shape[0])], 
                                   torch.mean(vectors, 1).detach().numpy()) 
    # среднее по 50, т.к. 1 видео = 1 эмбеддинг
    logger.info('Loaded the video corpus')

    #inputs = app.state.tokenizer(request.text, padding=True, return_tensors="pt")
    #outputs = app.state.model(**inputs)
    #text_embeds = outputs.text_embeds
    text_embeds = np.random.random((len(request.text), 768)).astype('float32')
    logger.info('Embedded the queries')
    results = []
    for text_emb in text_embeds:
        search_res = app.state.vid_embs.most_similar(text_emb, topn=request.topn)
        topn_res = []
        for i, _ in search_res: # index from test.json
            topn_res.append(i)
        results.append(topn_res)
    return SearchResponse(idxs=results) # список индексов



def preprocess_text(text) -> List[str]:
    '''Лемматизация'''
    morph = MorphAnalyzer()
    lemmas = []
    for word in text.split(' '):
        word = word.strip('!@#$%^&*()_+-=?><.,\'\":;][{}]`~\n\t—»« ').lower()
        ana = morph.parse(word)
        lemmas.append(ana[0].normal_form)
    return lemmas

@app.post("/eda", response_model=EdaResponse, status_code=HTTPStatus.OK)
async def eda(request:SearchRequest):
    '''Эту ручку мы сделали случайно, использовать не планируем пока что'''
    # Это должно идти при запуске приложения, но раз мы не используем пока - будет тут
    corpus, tokenized_corpus, meta_info_corpus = json.load(
        open(config.corpus['sub_path'], 'r', encoding='utf-8'))
    bm25 = BM25Okapi(tokenized_corpus)
    app.state.bm25 = bm25
    app.state.corpus = corpus
    app.state.meta_info_corpus = meta_info_corpus
    logger.info('Created bm25 corpus')

    # Основная часть ручки
    result = []
    for q in request.text:
        tokenized_query = preprocess_text(q)
        doc_scores = app.state.bm25.get_scores(tokenized_query)
        doc_ind = np.argpartition(doc_scores, -config.topn_eda)[-config.topn_eda:]

        topx = []
        for i in doc_ind:
            topx.append({'text': app.state.corpus[i],
                        'video_name': app.state.meta_info_corpus[i][0],
                        'start': app.state.meta_info_corpus[i][1],
                        'end': app.state.meta_info_corpus[i][2]})
        result.append(topx)

    return EdaResponse(result=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
