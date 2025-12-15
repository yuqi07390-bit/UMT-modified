_base_ = [
    '../_base_/models/umt_base.py', '../_base_/plugins/mr.py',
    '../_base_/datasets/charades.py', '../_base_/schedules/100e.py',
    '../_base_/runtime.py'
]
# model settings
model = dict(audio_enc=dict(dims=[4096, 256]), query_gen=dict(dims=[768, 256]))
# dataset settings
data = dict(train=dict(modality='vo', use_bert=True, bert_freeze=False), val=dict(modality='vo', use_bert=True,bert_freeze=False))
