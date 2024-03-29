import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from data.ecc import ECC
from data.common import Collate
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--load_model', default=None)
parser.add_argument('--test_path', default='segmented-test1')
parser.add_argument('--model', default='proposed', choices=['baseline', 'baseline_gru', 'proposed'])
args = parser.parse_args()

if args.gpu < 0:
    device = "cpu"
else:
    device = "cuda:%d" % args.gpu

if args.model == 'baseline':
    from hparams import baseline as hparams
    from model.baseline import Baseline, FakeMST
    model = Baseline(hparams)
    fake = FakeMST(hparams.fake_mst)
    load_model = Path(args.load_model)
    fake.load_state_dict(torch.load(load_model.parent / f'fake_{load_model.name}', map_location='cpu'))
    fake.to(device)
elif args.model == 'baseline_gru':
    from hparams import baseline_gru as hparams
    from model.baseline_gru import BaselineGRU
    from model.baseline import FakeMST
    model = BaselineGRU(hparams)
    fake = FakeMST(hparams.fake_mst)
    load_model = Path(args.load_model)
    fake.load_state_dict(torch.load(load_model.parent / f'fake_{load_model.name}', map_location='cpu'))
    fake.to(device)
elif args.model == 'proposed':
    from hparams import proposed as hparams
    from model.proposed import Proposed
    model = Proposed(hparams)

test_dataset = ECC(args.test_path, chunk_size=hparams.input.chunk_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False, collate_fn=Collate(device))

model.load_state_dict(torch.load(args.load_model, map_location='cpu'))
model.to(device)

with torch.no_grad():
    predicted_gst = []
    predicted_wst = []
    predicted_gst_only = []
    for data in tqdm(test_dataloader):
        length, speaker, bert, gst, wst, gst_only, sbert = data
        current_length = [i[-1] for i in length]

        if args.model == 'baseline':
            history_gst_only = [i[:-1] for i in gst_only]
            current_bert = [i[-1] for i in bert]

            predicted_gst_only.append(model(length, speaker, bert, history_gst_only))
            _predicted_gst, _predicted_wst = fake(current_length, current_bert, predicted_gst_only[-1].detach())
        if args.model == 'baseline_gru':
            history_gst_only = [i[:-1] for i in gst_only]
            current_bert = [i[-1] for i in bert]

            predicted_gst_only.append(model(length, speaker, bert, history_gst_only, sbert))
            _predicted_gst, _predicted_wst = fake(current_length, current_bert, predicted_gst_only[-1].detach())
        if args.model == 'proposed':
            history_gst = [i[:-1] for i in gst]
            history_wst = [i[:-1] for i in wst]

            _predicted_gst, _predicted_wst = model(length, speaker, bert, history_gst, history_wst)

        predicted_gst.append(_predicted_gst)
        predicted_wst += _predicted_wst

    if args.model in ['baseline', 'baseline_gru']:
        predicted_gst_only = torch.cat(predicted_gst_only, dim=0)

    predicted_gst = torch.cat(predicted_gst, dim=0)

    current_utterances = [chunk[-1] for chunk in test_dataset.chunks]
    for i in range(len(current_utterances)):
        key = current_utterances[i].wav.stem
        if args.model in ['baseline', 'baseline_gru']:
            np.save(test_dataset.path / f'{key}.p_gst_only.npy', predicted_gst_only[i].cpu().numpy())
        np.save(test_dataset.path / f'{key}.p_gst.npy', predicted_gst[i].cpu().numpy())
        np.save(test_dataset.path / f'{key}.p_wst.npy', predicted_wst[i].cpu().numpy())
