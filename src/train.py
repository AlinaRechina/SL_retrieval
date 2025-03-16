from datasets import Dataset
import os 
import torch
import json
from tqdm import tqdm
import numpy as np

from torch import nn
from torch.nn.functional import log_softmax
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_cosine_similarity
import wandb

device='cuda'
wandb.login()
emb_dir = '../dataset/processed/embeddings/'
info_dir = '../dataset/processed/info/'


class RSL_Dataset(): # Dataset
    def __init__(self):
        self.video_embs = []
        self.text_embs = []
        
    def __len__(self):
        return len(self.video_embs)

    def __getitem__(self, idx):
        return self.video_embs[idx], self.text_embs[idx]
    
    def add_item(self, video_emb, text_emb):
        self.video_embs.append(video_emb.squeeze())
        self.text_embs.append(text_emb.squeeze())

# тестовые видео оставляем теми же
test_videos = set()
with open('../dataset/test.json', 'r') as f:
    tvds = json.load(f)
    for tvd in tvds:
        test_videos.add(tvd['video'])


train_dataset = RSL_Dataset()
eval_dataset = RSL_Dataset()
test_dataset = RSL_Dataset()

for emb_file in tqdm(os.listdir(emb_dir)):
    if emb_file[:-3] in test_videos:
        with open(info_dir + '/' + emb_file[:-3]+'.json', 'r') as f:
            info = json.load(f)
        texts = info[1]
        for t, temb, vemb in zip(texts, tembs, vembs):
            if t != '':
                test_dataset.add_item(vemb.to(device), temb.to(device))

    else:
        with open(info_dir + '/' + emb_file[:-3]+'.json', 'r') as f:
            info = json.load(f)
        texts = info[1]
        vembs, tembs = torch.load(emb_dir + '/' + emb_file)

        good_vembs = []
        good_tembs = []
        good_texts = []
        for t, temb, vemb in zip(texts, tembs, vembs):
            if t != '':
                good_vembs.append(vemb)
                good_tembs.append(temb)
                good_texts.append(t)

        ten_last_i = int(len(good_tembs)*0.9)
        for i, e in enumerate(zip(good_tembs, good_vembs, good_texts)):
            if i > 0 and (i%9 == 0 or i > ten_last_i):
                eval_dataset.add_item(e[0].to(device), e[1].to(device))
            else:
                train_dataset.add_item(e[0].to(device), e[1].to(device))


class MLPClassifier(nn.Module):
    def __init__(
          self,
          vids_emb_dim,
          text_emb_dim,
          hidden_layer_dim=2048,
          dropout_p=0.2
        ):
        super().__init__()

        self.vids_emb_layer = nn.Sequential(
            nn.Linear(vids_emb_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim)
        )

        self.text_emb_layer = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_layer_dim, hidden_layer_dim)
        )

    def forward(self, vids_emb, text_emb):
        return pairwise_cosine_similarity(
            self.vids_emb_layer(vids_emb),
            self.text_emb_layer(text_emb)
        )

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(log_softmax(logits, dim=dim))
    return -neg_ce

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    cap2img_match_idx = similarity.argmax(dim=0)
    cap_acc = (cap2img_match_idx == y).float()

    return cap_acc


epoch_num = 30
lrs = [0.1, 0.01, 0.001]
dropout_ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
hidden_layer_dims= [512, 1024, 2048, 4096]
batch_size = 32
emb_dim = 512

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

for lr in lrs:
    for dropout_p in dropout_ps:
        for hidden_layer_dim in hidden_layer_dims:
            
            model = MLPClassifier(
                vids_emb_dim=emb_dim,
                text_emb_dim=emb_dim,
                hidden_layer_dim=hidden_layer_dim,
                dropout_p=dropout_p
            )
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            exp_name = f"mlp_5layers_bs32_lr{str(lr)}_drp{str(dropout_p)}_dim{str(hidden_layer_dim)}".replace('.', '@')
            logging = wandb.init(
                        project="RSL",
                        name=exp_name,
                        config={
                            "learning_rate": lr,
                            "epochs": epoch_num,
                            "hidden_layers_dim": hidden_layer_dim,
                            "dropout_rate": dropout_p,
                            "batch_size": 32,
                        }
                    )
            os.mkdir(f'./checkpoints/{exp_name}')


            for epoch in range(epoch_num):
                for vids_emb, text_emb in tqdm(train_dataloader):
                    optimizer.zero_grad()
                    pred = model(vids_emb, text_emb)
                    loss = clip_loss(pred).mean()
                    logging.log({"train_loss": loss.item()})
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
                    optimizer.step()

                with torch.no_grad():
                    model.eval()
                    loss = torch.tensor([])
                    accuracy = torch.tensor([])
                    for vids_emb, text_emb in tqdm(eval_dataloader):
                        pred = model(vids_emb, text_emb)
                        pred_norm = (pred + 1) / 2
                        target = torch.diag(torch.ones(vids_emb.shape[0]))
                        loss = torch.concat((loss, clip_loss(pred).to('cpu')), dim=0)
                        accuracy = torch.concat((accuracy, metrics(pred).to('cpu')), dim=0)

                    logging.log({
                        "val_loss": loss.mean().item(),
                        "val_acc": accuracy.mean().item()
                    })
                    
                    torch.save(model.state_dict(), f'./checkpoints/{exp_name}/{epoch}.pt')
                    model.train()
            

            logging.finish()

