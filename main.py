import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from http import HTTPStatus
from typing import Dict, List, Union, Callable
import logging

import os 
#import sys
#import gdown
import json

from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
import numpy as np

import config
from src.preprocess_text import preprocess_text
# Наружу должен торчать порт

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"tryItOutEnabled": True}
)

class SearchRequest(BaseModel):
    text: List[str] 

class SearchResponse(BaseModel):
    video_name: str
    start: float
    end: float
    #subtitles: Union[str, None] # и для размеченных, и для не размеченных данных

class EdaResponse(BaseModel):
    result: List[List]

# При запуске
def _startup_model(app: FastAPI) -> None:
    logger.info('Starting the app')

    # Для eda
    corpus, tokenized_corpus, meta_info_corpus = json.load(open(config.corpus['sub_path'], 
                                                                'r', encoding='utf-8'))
    bm25 = BM25Okapi(tokenized_corpus)
    app.state.bm25 = bm25
    app.state.corpus = corpus
    app.state.meta_info_corpus = meta_info_corpus
    logger.info('Created bm25 corpus')

    #app.state.model = CLIPTextModelWithProjection.from_pretrained(config.model, device=config.device)
    #app.state.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32", device=config.device)
    logger.info('Loaded the model')

    # if 'embeddings.pt' not in os.listdir('./data/'): # тоже конфиг # это не работает
    #     url = 'https://drive.google.com/file/d/1d1x2Z15eTV1r2CEaKJZx8TU6Vj_MsZpu/view?usp=sharing' # тоже должно быть в конфиге
    #     output = './data/embeddings.pt'
    #     gdown.download(url, output, quiet=False)

    app.state.vid_embs = KeyedVectors(config.dim)
    vectors = torch.load(config.corpus['embed_path'])
    app.state.vid_embs.add_vectors([i for i in range(vectors.shape[0])], 
                                   torch.mean(vectors, 1).detach().numpy()) 
    # среднее по 50, т.к. 1 видео = 1 эмбеддинг
    logger.info('Loaded the video corpus')

def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _startup_model(app)
    return startup

app.add_event_handler("startup", start_app_handler(app))

@app.post("/get_vids", response_model=List[SearchResponse], status_code=HTTPStatus.OK)
async def search(request:SearchRequest):
    #inputs = app.state.tokenizer(request.text, padding=True, return_tensors="pt")
    #outputs = app.state.model(**inputs)
    #text_embeds = outputs.text_embeds
    text_embeds = np.random.random((len(request.text), 768)).astype('float32')
    logger.info('Embedded the queries')
    results = []
    for text_emb in text_embeds:
        search_res = app.state.vid_embs.most_similar(text_emb, topn=config.topn_search)
        with open(config.corpus['main_path'], 'r', encoding='utf-8') as f:
            test_vids = json.load(f)
        results = []
        for i, dist in search_res: # index from test.json
            idx_info = test_vids[i]
            results.append(SearchResponse(
                video_name=idx_info['video'],
                start=idx_info['start'],
                end=idx_info['end'],
                #subtitles=idx_info['subtitles']
                ))
    return results # список индексов

@app.post("/eda", response_model=EdaResponse, status_code=HTTPStatus.OK)
async def eda(request:SearchRequest):
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