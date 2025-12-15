_base_ = [
    '../_base_/models/umt_base.py', '../_base_/plugins/mr.py',
    '../_base_/datasets/charades.py', '../_base_/schedules/100e.py',
    '../_base_/runtime.py'
]
# model settings
model = dict(audio_enc=dict(dims=[4096, 256]))
# dataset settings
data = dict(train=dict(modality='vo', raw_video_root="data/Charades_v1_480", raw_video_ext='.mp4',use_videoclip=True,save_pred_csv=True, pred_csv_path='original_preds.csv'),
            val=dict(modality='vo', raw_video_root="data/Charades_v1_480", raw_video_ext='.mp4',use_videoclip=True,save_pred_csv=True, pred_csv_path='original_preds.csv'))
