device = 'cpu'
model = 'microsoft/xclip-base-patch32'
dim = 768

corpus = dict(
    embed_path = "./data/embeddings.pt",
    main_path = "./data/test.json",
    sub_path = "./data/test_captions.json"
)

topn_search = 1
topn_eda = 5
