import requests
import aiohttp
import datetime
import asyncio
import yadisk
from translate import Translator
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
import pandas as pd
from plotly.express import violin
from collections import Counter
import json
import streamlit as st
import spacy

t = Translator(to_lang='en', from_lang='ru')
nlp = spacy.load("ru_core_news_sm")
tmp_path = './tmp/'


async def get_vids(query: str, top: int, token: str, disk_emb_path: str) -> List[int]:
    link = "http://127.0.0.1:8000" # вот это я хз, правильно ли
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{link}/get_vids',
                                json={'text':[query], 
                                'topn':2,
                                'token': token,
                                'disk_emb_path': disk_emb_path}, 
                                timeout=300) as resp:
            resp_dict = await resp.json()
            idxs = resp_dict["idxs"][0]
    return idxs


async def clear_path(path: Path):
    if path.exists():
        for node in path.iterdir():
            node.unlink()
    else:
        path.mkdir(parents=True)


def get_data() -> Tuple[str, str, Any]:
    token = st.text_input(
        "Enter your YaDisk Key:",
        type="password"
    )

    path = st.text_input("Enter your YaDisk path:")

    meta_file = st.file_uploader(
        "Drop your JSON metadata:",
        type='json'
    )
    return token, path, meta_file


def count_stats(meta: List[Dict]) -> Tuple[Counter, Counter, Counter, List[float], List[int]]:
    lemmas = Counter()
    bigrams = Counter()
    trigrams = Counter()
    durations = []
    word_counts = []

    for phrase in meta:
        word_counts.append(0)
        durations.append(phrase['end'] - phrase['start'])
        doc = nlp(phrase['text'])
        for i in range(len(doc)):
            if not (doc[i].is_punct or doc[i].is_space):
                word_counts[-1] += 1

                if not doc[i].is_stop:
                    lemmas.update([doc[i].lemma_])

                if i > 0 and not (doc[i - 1].is_punct or doc[i - 1].is_space):
                    bigrams.update([doc[i - 1].lemma_ + ' ' + doc[i].lemma_])

                    if i > 1 and not (doc[i - 2].is_punct or doc[i - 2].is_space):
                        trigrams.update(
                            [doc[i - 2].lemma_ + ' ' + doc[i - 1].lemma_ + ' ' + doc[i].lemma_]
                        )
    return lemmas, bigrams, trigrams, durations, word_counts


async def draw_graphs(meta: List[Dict]):
    lemmas, bigrams, trigrams, durations, word_counts = count_stats(meta)
    left, right = st.columns(2)

    with left:
        option = st.selectbox(
            "A phrase feature to display distribution for:",
            ("Word count", "Duration")
        )

        if option == 'Word count':
            st.plotly_chart(
                violin(
                    {"num. of words": word_counts},
                    box=True
                ),
                use_container_width=True
            )
        elif option == 'Duration':
            st.plotly_chart(
                violin(
                    {"duration, s": durations},
                    box=True
                ),
                use_container_width=True
            )

    with right:
        num = st.selectbox(
            "Most popular text features to show:",
            ("Lemmas", "Bigrams", "Trigrams")
        )

        if num == "Lemmas":
            st.bar_chart(
                pd.DataFrame([
                    {
                        'lemmas': pair[0],
                        'values': pair[1]
                    }
                    for pair in lemmas.most_common(50)
                ]),
                x="lemmas",
                y="values"
            )
        elif num == "Bigrams":
            st.bar_chart(
                pd.DataFrame([
                    {
                        'bigrams': pair[0],
                        'values': pair[1]
                    }
                    for pair in bigrams.most_common(50)
                ]),
                x="bigrams",
                y="values"
            )
        elif num == "Trigrams":
            st.bar_chart(
                pd.DataFrame([
                    {
                        'trigrams': pair[0],
                        'values': pair[1]
                    }
                    for pair in trigrams.most_common(50)
                ]),
                x="trigrams",
                y="values"
            )


async def process_query(query: str, top_n: int, meta: List[Dict], 
                        path: str, client: yadisk.Client, token: str):
    query = t.translate(query)
    vid_idxs = await get_vids(query, top_n, token, path)
    for idx in vid_idxs:
        if not Path(tmp_path + meta[idx]['video']).exists():
            client.download(path + meta[idx]['video'], tmp_path + meta[idx]['video'])
        st.video(
            tmp_path + meta[idx]['video'],
            start_time=meta[idx]['start'],
            end_time=meta[idx]['end'] + 1,
            muted=True,
            loop=True
        )


async def main():
    token, path, meta_file = get_data()

    if token and path and meta_file:
        meta = json.loads(meta_file.getvalue())
        await draw_graphs(meta)

        client = yadisk.Client(token=token)
        path += '/' if not path.endswith('/') else ''

        left, right = st.columns([5, 1])
        with left:
            query = st.text_area(
                "Enter your query:",
                placeholder="Интервью с сурдолимпийским чемпионом..."
            )

        with right:
            top_n = st.number_input(
                "Enter maximum vids:",
                min_value=1,
                max_value=10,
                value="min",
                step=1
            )

        if query:
            await process_query(query, top_n, meta, path, client, token)


if __name__ == '__main__':
    asyncio.run(main())
