# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import nncore
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.dataset import DATASETS
from nncore.ops import temporal_iou
from nncore.parallel import DataContainer
from torch.utils.data import Dataset
from torchtext import vocab

from transformers import AutoTokenizer, AutoModel
from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder
import numpy as np
import os
import cv2
from tqdm import tqdm
import csv

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def video_clip_preprocessing(video_path, start_sec, end_sec, fnum=8):
    """Extract frames from the video within the time interval [start_sec, end_sec], then perform VideoCLIP-XL preprocessing."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f'Fail to open video: {video_path}')

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frames = []
    idx = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        t = idx / fps
        idx += 1

        if t < start_sec:
            continue
        if t > end_sec:
            break

        frames.append(frame)

    video.release()

    if len(frames) == 0:
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frames.append(frame)
        video.release()

    if len(frames) == 0:
        raise RuntimeError(f'No frame found in video: {video_path}')

    # Downsample to fnum frames.
    step = max(len(frames) // fnum, 1)
    frames = frames[::step][:fnum]

    vid_tube = []
    for fr in frames:
        fr = fr[:, :, ::-1]  # BGR -> RGB
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), axis=(0, 1))  # [1,1,H,W,3]
        vid_tube.append(fr)
    vid_tube = np.concatenate(vid_tube, axis=1)  # [1,T,H,W,3]
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # [1,T,3,H,W]
    vid_tube = torch.from_numpy(vid_tube)

    return vid_tube


@DATASETS.register()
class CharadesSTA(Dataset):

    def __init__(self,
                 modality,
                 label_path,
                 video_path,
                 optic_path=None,
                 audio_path=None,
                 use_bert=False,
                 bert_name='bert-base-uncased',
                 bert_max_len=32,
                 bert_freeze=True,

                 raw_video_root=None,
                 raw_video_ext='.mp4',

                 use_videoclip=False,
                 videoclip_ckpt='./VideoCLIP-XL/VideoCLIP-XL.bin',
                 videoclip_device='cuda',

                 save_pred_csv=False,
                 pred_csv_path='charades_preds.csv'
                 ):
        assert modality in ('va', 'vo')
        self.label = nncore.load(label_path)

        self.modality = modality
        self.label_path = label_path
        self.video_path = video_path
        self.optic_path = optic_path
        self.audio_path = audio_path

        self.use_bert = use_bert
        self.bert_max_len = bert_max_len
        self.bert_freeze=bert_freeze

        if self.use_bert:
            # Initialize BERT
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
            self.bert = AutoModel.from_pretrained(bert_name)
            self.bert.eval()

            if self.bert_freeze:
                for p in self.bert.parameters():
                    p.requires_grad = False  # freeze the rest of the model and train only UMT.
        else:
            # Keep the original GloVe logic
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

        self.raw_video_root = raw_video_root
        self.raw_video_ext = raw_video_ext
        self.use_videoclip = use_videoclip
        self.videoclip_device = videoclip_device

        self.save_pred_csv = save_pred_csv
        self.pred_csv_path = pred_csv_path

        if self.use_videoclip:
            assert self.raw_video_root is not None, \
                'When use_videoclip=True, you must provide raw_video_root, which is used to read the raw video files.'

            self.videoclip_xl = VideoCLIP_XL()
            state_dict = torch.load(videoclip_ckpt, map_location='cpu')
            self.videoclip_xl.load_state_dict(state_dict)
            self.videoclip_xl.to(self.videoclip_device).eval()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        query = self.get_query(idx)

        num_clips = min(c.size(0) for c in (video, audio))

        boundary = self.get_boundary(idx)
        saliency = torch.ones(num_clips)

        data = dict(
            video=DataContainer(video[:num_clips]),
            audio=DataContainer(audio[:num_clips]),
            query=DataContainer(query, pad_value=float('inf')),
            saliency=DataContainer(saliency, pad_value=-1),
            meta=DataContainer(self.label[idx], cpu_only=True))

        if boundary is not None:
            data['boundary'] = DataContainer(boundary, pad_value=-1)

        return data

    def parse_boundary(self, label):
        boundary = label.split('##')[0].split()[1:]
        if float(boundary[1]) < float(boundary[0]):
            boundary = [boundary[1], boundary[0]]
        return torch.Tensor([[float(s) for s in boundary]])

    def get_video(self, idx):
        vid = self.label[idx].split()[0]
        video = nncore.load(nncore.join(self.video_path, f'{vid}.npy'))
        return F.normalize(torch.from_numpy(video).float())

    def get_audio(self, idx):
        vid = self.label[idx].split()[0]
        path = self.audio_path if self.modality == 'va' else self.optic_path
        audio = nncore.load(nncore.join(path, f'{vid}.npy'))
        return F.normalize(torch.from_numpy(audio).float())

    def get_query(self, idx):
        query = self.label[idx].split('##')[-1][:-1]

        if self.use_bert:
            # Use BERT to encode the entire sentence.
            encoded = self.tokenizer(
                query,
                truncation=True,
                max_length=self.bert_max_len,
                return_tensors='pt'
            )
            with torch.no_grad():
                out = self.bert(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )

            sent_emb = out.last_hidden_state
            return sent_emb.squeeze(0)

        # The original GloVe logic.
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def get_boundary(self, idx):
        return self.parse_boundary(self.label[idx])

    def evaluate(self,
                 blob,
                 method='gaussian',
                 nms_thr=0.3,
                 sigma=0.5,
                 rank=[1, 5],
                 iou_thr=[0.5, 0.7],
                 **kwargs):
        assert method in ('fast', 'normal', 'linear', 'gaussian')

        blob = nncore.to_dict_of_list(blob)
        results = dict()

        print('Performing temporal NMS...')
        boundary = []

        for bnd in blob['boundary']:
            bnd = bnd[0]

            if method == 'fast':
                iou = temporal_iou(bnd[:, :-1], bnd[:, :-1]).triu(diagonal=1)
                keep = iou.amax(dim=0) <= nms_thr
                bnd = bnd[keep]
            else:
                for i in range(bnd.size(0)):
                    max_idx = bnd[i:, -1].argmax(dim=0)
                    bnd = nncore.swap_element(bnd, i, max_idx + i)
                    iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

                    if method == 'normal':
                        bnd[i + 1:, -1][iou >= nms_thr] = 0
                    elif method == 'linear':
                        bnd[i + 1:, -1] *= 1 - iou
                    else:
                        bnd[i + 1:, -1] *= (-iou.pow(2) / sigma).exp()

            boundary.append(bnd)

        if self.save_pred_csv and len(boundary) > 0:
            rows = []
            for idx, bnd in enumerate(boundary):
                meta_str = blob['meta'][idx][0]
                parts = meta_str.split('##')
                head = parts[0].split()
                vid = head[0]
                text = parts[-1][:-1]

                # GT boundary
                gt = self.parse_boundary(meta_str)[0]
                gt_start = float(gt[0].item())
                gt_end = float(gt[1].item())

                for rank_idx in range(bnd.size(0)):
                    pred_start = float(bnd[rank_idx, 0].item())
                    pred_end = float(bnd[rank_idx, 1].item())
                    pred_score = float(bnd[rank_idx, 2].item())

                    rows.append(dict(
                        index=idx,
                        vid=vid,
                        text=text,
                        gt_start=gt_start,
                        gt_end=gt_end,
                        pred_rank=rank_idx,
                        pred_start=pred_start,
                        pred_end=pred_end,
                        pred_score=pred_score
                    ))

            fieldnames = [
                'index', 'vid', 'text',
                'gt_start', 'gt_end',
                'pred_rank', 'pred_start', 'pred_end', 'pred_score'
            ]
            with open(self.pred_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f'Prediction CSV saved to {self.pred_csv_path}')

        for k in rank:
            for thr in iou_thr:
                print(f'Evaluating Rank{k}@{thr}...')
                hits = 0

                for idx, bnd in enumerate(boundary):
                    inds = torch.argsort(bnd[:, -1], descending=True)
                    keep = inds[:k]
                    bnd = bnd[:, :-1][keep]

                    gt = self.parse_boundary(blob['meta'][idx][0])
                    iou = temporal_iou(gt, bnd)

                    if iou.max() >= thr:
                        hits += 1

                results[f'Rank{k}@{thr}'] = hits / len(self.label)

        # VideoCLIP-XL text-video alignment score
        if self.use_videoclip:
            print('Evaluating VideoCLIP-XL ClipScore (text-video alignment)...')
            Tmp = 100.0
            total_score = 0.0
            valid_count = 0

            for idx, bnd in tqdm(enumerate(boundary)):
                # bnd: [num_preds, 3] -> [start, end, score]
                # Select the prediction with the highest score.
                best_idx = torch.argmax(bnd[:, -1])
                pred = bnd[best_idx]  # [start, end, score]
                start_sec = float(pred[0].item())
                end_sec = float(pred[1].item())

                # Parse the video ID and text description from the meta data.
                meta_str = blob['meta'][idx][0]
                parts = meta_str.split('##')
                head = parts[0].split()
                vid = head[0]
                text = parts[-1][:-1]

                video_path = os.path.join(
                    self.raw_video_root, f'{vid}{self.raw_video_ext}')

                # Preprocess the video segment.
                try:
                    vid_tensor = video_clip_preprocessing(
                        video_path, start_sec, end_sec, fnum=8).float()
                except Exception as e:
                    print(f'[VideoCLIP-XL] Skip sample {vid} due to error: {e}')
                    continue

                vid_tensor = vid_tensor.to(self.videoclip_device)

                with torch.no_grad():
                    # Video features
                    video_features = self.videoclip_xl.vision_model \
                        .get_vid_features(vid_tensor).float()
                    video_features = F.normalize(
                        video_features, dim=-1, p=2)  # [1, D]

                    # Text features
                    text_inputs = text_encoder.tokenize(
                        [text], truncate=True).to(self.videoclip_device)
                    text_features = self.videoclip_xl.text_model \
                        .encode_text(text_inputs).float()
                    text_features = F.normalize(
                        text_features, dim=-1, p=2)  # [1, D]

                    # sim_matrix: [1,1]
                    sim_matrix = (text_features @ video_features.T) * Tmp
                    score = sim_matrix.item()

                total_score += score
                valid_count += 1

            if valid_count > 0:
                results['VideoCLIP-XL-ClipScore'] = total_score / valid_count
            else:
                results['VideoCLIP-XL-ClipScore'] = 0.0

        return results

