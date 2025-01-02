device = 'cpu'
model = 'microsoft/xclip-base-patch32'
dim = 768

corpus = dict(
    embed_path = "./data/embeddings.pt",
    sub_path = "./data/test_captions.json",
    folder_for_embedings = "./data/",
    yadisk_emb_path = 'disk:/embeddings.pt'
)

topn_search = 1
topn_eda = 5
