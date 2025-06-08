from typing import List
from pathlib import Path
from collections import Counter
import json

import asyncio
import aiohttp
import yadisk
from translate import Translator
import pandas as pd
from plotly.express import violin
import streamlit as st
import spacy

t = Translator(to_lang='en', from_lang='ru')
nlp = spacy.load("ru_core_news_sm")
TMP_PATH = './tmp/'
Path(TMP_PATH).mkdir(parents=True, exist_ok=True)


async def get_vids(query: str, top: int, token: str, disk_emb_path: str) -> List[int]:
    '''Обращение к бэку'''
    link = "http://fastapi:8000"
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(f'{link}/get_vids',
                                json={'text': [query],
                                      'topn': top,
                                      'token': token,
                                      'disk_emb_path': disk_emb_path},
                                timeout=300) as resp:
            resp_dict = await resp.json()
            idxs = resp_dict["idxs"][0]
    return idxs


def get_client():
    if st.session_state['token']:
        st.session_state['yadisk_client'] = yadisk.Client(token=st.session_state['token'])


def set_path():
    if st.session_state['yadisk_path']:
        st.session_state['yadisk_path'] += '/' \
            if not st.session_state['yadisk_path'].endswith('/') \
            else ''


def get_data():
    '''Скачивание данных с Яндекс Диска'''
    st.text_input(
        "Enter your YaDisk Key:",
        type="password",
        key='token',
        on_change=get_client,
    )

    st.text_input(
        "Enter your YaDisk path:",
        key='yadisk_path',
        on_change=set_path
    )

    st.file_uploader(
        "Drop your JSON metadata:",
        type='json',
        on_change=count_stats,
        key='meta_file'
    )


def count_stats():
    '''EDA'''
    if st.session_state['meta_file'] is None:
        Path(TMP_PATH + 'stats.json').unlink()
        st.session_state.pop('meta_list')
        return None

    meta = json.loads(st.session_state['meta_file'].getvalue())
    st.session_state['meta_list'] = meta

    lemmas = Counter()
    bigrams = Counter()
    trigrams = Counter()
    durations = []
    word_counts = []

    for phrase in meta:
        word_counts.append(0)
        durations.append(phrase['end'] - phrase['start'])
        doc = nlp(phrase['text'])
        for i, _ in enumerate(doc):
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
    json.dump(
        {
            'lemmas': [[val, lemmas[val]] for val in sorted(lemmas, key=lemmas.get, reverse=True)[:50]],
            'bigrams': [[val, bigrams[val]] for val in sorted(bigrams, key=bigrams.get, reverse=True)[:50]],
            'trigrams': [[val, trigrams[val]] for val in sorted(trigrams, key=trigrams.get, reverse=True)[:50]],
            'duration': durations,
            'word count': word_counts
        },
        Path(TMP_PATH + 'stats.json').open('w', encoding='utf-8'),
        indent=2,
        ensure_ascii=False
    )


async def draw_graphs():
    stats = json.load(Path(TMP_PATH + 'stats.json').open('r', encoding='utf-8'))

    left, right = st.columns(2)

    with left:
        option = st.selectbox(
            "A phrase feature to display distribution for:",
            ("word count", "duration"),
            format_func=lambda x: x.title(),
            index=0
        )

        st.plotly_chart(
            violin(
                {option: stats[option]},
                box=True
            ),
            use_container_width=True
        )

    with right:
        num = st.selectbox(
            "Most popular text features to show:",
            ("lemmas", "bigrams", "trigrams"),
            format_func=lambda x: x.title(),
            index=0
        )

        st.bar_chart(
            pd.DataFrame([
                {
                    num: pair[0],
                    'values': pair[1]
                }
                for pair in stats[num]
            ]),
            x=num,
            y="values"
        )


async def process_query(query: str, top_n: int):
    query = t.translate(query)
    meta = st.session_state['meta_list']
    vid_idxs = await get_vids(query, top_n, st.session_state['token'], st.session_state['yadisk_path'])
    for idx in vid_idxs:
        if not Path(TMP_PATH + meta[idx]['video']).exists():
            st.session_state['yadisk_client'].download(
                st.session_state['yadisk_path'] + meta[idx]['video'], TMP_PATH + meta[idx]['video']
            )
        st.video(
            TMP_PATH + meta[idx]['video'],
            start_time=meta[idx]['start'],
            end_time=meta[idx]['end'] + 1,
            muted=True,
            loop=True
        )


async def main():
    get_data()

    if st.session_state['meta_file']:
        await draw_graphs()

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

    if (query and
            'meta_list' in st.session_state and
            st.session_state['yadisk_path'] and
            'yadisk_client' in st.session_state):
        await process_query(query, top_n)


if __name__ == '__main__':
    asyncio.run(main())
