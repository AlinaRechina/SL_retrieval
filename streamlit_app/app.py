from typing import List
from pathlib import Path
import asyncio
import aiohttp
import yadisk
import streamlit as st


TMP_PATH = './tmp/'
FASTAPI = "http://fastapi:8000"
Path(TMP_PATH).mkdir(parents=True, exist_ok=True)


async def get_vids(query: str, top: int) -> List[int]:
    '''Обращение к бэку'''
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(
                f'{FASTAPI}/get_vids',
                json={
                    'text': query,
                    'topn': top
                },
                timeout=300
        ) as resp:
            resp_dict = await resp.json()
            idxs = resp_dict["idxs"]
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


async def send_load_request(token: str, disk_emb_path: str):
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(f'{FASTAPI}/load_vids',
                                json={'token': token,
                                      'disk_path': disk_emb_path}) as resp:
            st.session_state['meta_list'] = resp.json()


async def process_query(query: str, top_n: int):
    meta = st.session_state['meta_list']
    vid_idxs = await get_vids(query, top_n)
    for idx in vid_idxs:
        if not Path(TMP_PATH + meta[idx]['video']).exists():
            st.session_state['yadisk_client'].download(
                st.session_state['yadisk_path'] + meta[idx]['title'], TMP_PATH + meta[idx]['title']
            )
        st.video(
            TMP_PATH + meta[idx]['title'],
            start_time=meta[idx]['start'],
            end_time=meta[idx]['end'] + 1,
            muted=True,
            loop=True
        )


async def main():
    get_data()
    get_client()
    await send_load_request(st.session_state['token'], st.session_state['yadisk_path'])
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

    if query and 'meta_list' in st.session_state:
        await process_query(query, top_n)


if __name__ == '__main__':
    asyncio.run(main())
