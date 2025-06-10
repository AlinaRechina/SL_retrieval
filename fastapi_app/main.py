import logging
from typing import List, Callable

import os
from http import HTTPStatus
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import yadisk
import cv2

from transformers import AutoTokenizer, XCLIPTextModel, AutoProcessor, AutoModel
import torch

from config import device, custom_model

if not os.path.exists('./logs'):
    os.mkdir('./logs')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='./logs/myapp.log', level=logging.INFO)

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"tryItOutEnabled": True}
)


class InitRequest(BaseModel):
    token: str  # yadisk.Client
    disk_path: str


class Vid(BaseModel):
    start: float
    end: float
    title: str


class InitResponse(BaseModel):
    vids: List[Vid]


class SearchRequest(BaseModel):
    text: str
    topn: int


class SearchResponse(BaseModel):
    idxs: List[int]


def embed_text(text, model, tokenizer):
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.pooler_output


def embed_vid(vid, title, model, processor):
    out_meta = []
    embeddings = None

    fps = vid.get(cv2.CAP_PROP_FPS)  # собственное fps видео
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_rate = round(fps / 2)

    for i in range(0, frame_count - frame_rate * 8, frame_rate * 8):
        frames = []
        for j in range(i, i + frame_rate * 8, frame_rate):
            vid.set(cv2.CAP_PROP_POS_FRAMES, j)
            frames.append(torch.tensor(vid.read()[1][:, :, ::-1].copy()))

        inputs = processor(videos=frames, return_tensors="pt")
        video_features = model.get_video_features(**inputs)

        if embeddings is None:
            embeddings = video_features
        else:
            embeddings = torch.concat((embeddings, video_features))

        out_meta.append(Vid(
            start=i / fps,
            end=(i + frame_rate * 8) / fps,
            title=title
        ))
    return out_meta, embeddings


# При запуске
def _startup_model(app: FastAPI) -> None:
    logger.info('Starting the app')

    model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
    app.state.embed_text = lambda text: embed_text(text, model, tokenizer)

    video_model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    app.state.embed_vid = lambda vid, title: embed_vid(vid, title, video_model, processor)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')


def start_app_handler(app: FastAPI) -> Callable:
    '''При запуске приложения'''

    def startup() -> None:
        _startup_model(app)

    return startup


app.add_event_handler("startup", start_app_handler(app))


@app.post("/load_vids", response_model=InitResponse, status_code=HTTPStatus.OK)
async def search(request: InitRequest):
    client = yadisk.Client(token=request.token)
    meta = []
    app.state.embs = None
    for vid in client.listdir(request.disk_path):
        title = vid.field('name')
        vid.download('tmp/' + title)
        info, embeddings = app.state.embed_vid(cv2.VideoCapture('tmp/' + title), title)
        meta.extend(info)
        if app.state.embs is not None:
            app.state.embs = torch.concat(
                (app.state.embs, embeddings)
            )
        else:
            app.state.embs = embeddings

        os.unlink('tmp/' + title)
    logger.info('Loaded the video corpus')
    return InitResponse(vids=meta)  # список индексов


@app.post("/get_vids", response_model=SearchResponse, status_code=HTTPStatus.OK)
async def search(request: SearchRequest):
    '''Главная ручка'''
    text_embed = app.state.embed_text(request.text)
    logger.info('Embedded the queries')
    res = (text_embed @ app.state.embs.T)[0].argsort()[-request.topn:]
    return SearchResponse(idxs=res.tolist())  # список индексов


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
