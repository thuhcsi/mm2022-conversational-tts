import os
import sys
import torch
import argparse
from tqdm import tqdm
from data.ecc import ECC
from data.common import Collate
from save import Save

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--load_model', default=None)
parser.add_argument('--train_path', default='segmented-train')
parser.add_argument('--test_path', default='segmented-test')
parser.add_argument('--model', default='proposed', choices=['baseline', 'proposed'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu

if args.model == 'baseline':
    from hparams import baseline as hparams
    from model.baseline import Baseline, FakeMST
    model = Baseline(hparams)
    fake = FakeMST(hparams.fake_mst)
    fake.to(device)
elif args.model == 'proposed':
    from hparams import proposed as hparams
    from model.proposed import Proposed
    model = Proposed(hparams)

train_dataset = ECC(args.train_path)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)
test_dataset = ECC(args.test_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

if args.load_model:
    #model_dict = model.state_dict()
    #state_dict = torch.load(args.load_model)
    #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aligner.')}
    #model_dict.update(state_dict)
    #model.load_state_dict(model_dict)
    model.load_state_dict(torch.load(args.load_model, map_location='cpu'))

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if args.name is None:
    args.name = args.model
#else:
args.name = args.model + '_' + args.name

save = Save(args.name)
save.save_parameters(hparams)

step = 1
for epoch in range(hparams.max_epochs):
    save.logger.info('Epoch %d', epoch)

    batch = 1
    for data in train_dataloader:
        length, speaker, bert, gst, wst, gst_only = data
        current_gst = [i[-1] for i in gst]
        current_length = [i[-1] for i in length]
        current_wst = [i[-1, :l] for i, l in zip(wst, current_length)]
        current_gst = torch.stack(current_gst)

        if args.model == 'baseline':
            current_gst_only = [i[-1] for i in gst_only]
            current_gst_only = torch.stack(current_gst_only)
            current_bert = [i[-1] for i in bert]
            history_gst = [i[:-1] for i in gst_only]

            predicted_gst = model(length, speaker, bert, history_gst)
            gst_only_loss = model.gst_loss(predicted_gst, current_gst_only)
            save.writer.add_scalar(f'training/gst_only_loss', gst_only_loss, step)
            predicted_gst, predicted_wst = fake(current_length, current_bert, predicted_gst.detach())
            wst_loss = fake.wst_loss(predicted_wst, current_wst)
        if args.model == 'proposed':
            history_gst = [i[:-1] for i in gst]
            history_wst = [i[:-1] for i in wst]

            predicted_gst, predicted_wst = model(length, speaker, bert, history_gst, history_wst)
            wst_loss = model.wst_loss(predicted_wst, current_wst)

        gst_loss = model.gst_loss(predicted_gst, current_gst)
        save.writer.add_scalar(f'training/gst_loss', gst_loss, step)
        save.writer.add_scalar(f'training/wst_loss', wst_loss, step)
        loss = gst_loss + wst_loss
        save.writer.add_scalar(f'training/loss', loss, step)

        save.save_log('training', epoch, batch, step, loss)

        optimizer.zero_grad()
        if args.model == 'baseline':
            gst_only_loss.backward()
        loss.backward()
        optimizer.step()

        if step % 20000 == 0:
            save.save_model(model, f'{step // 1000}k')

        step += 1
        batch += 1

    with torch.no_grad():
        predicted_gst = []
        predicted_wst = []
        predicted_gst_only = []
        current_gst = []
        current_wst = []
        current_gst_only = []
        for data in tqdm(test_dataloader):
            length, speaker, bert, gst, wst, gst_only = data
            current_gst += [i[-1] for i in gst]
            current_length = [i[-1] for i in length]
            current_wst += [i[-1, :l] for i, l in zip(wst, current_length)]

            if args.model == 'baseline':
                current_gst_only += [i[-1] for i in gst_only]
                history_gst_only = [i[:-1] for i in gst_only]
                current_bert = [i[-1] for i in bert]

                predicted_gst_only.append(model(length, speaker, bert, history_gst_only))
                _predicted_gst, _predicted_wst = fake(current_length, current_bert, predicted_gst_only[-1].detach())
            if args.model == 'proposed':
                history_wst = [i[:-1] for i in wst]

                _predicted_gst, _predicted_wst = model(length, speaker, bert, history_gst, history_wst)

            predicted_gst.append(_predicted_gst)
            predicted_wst += _predicted_wst

        if args.model == 'baseline':
            current_gst_only = torch.stack(current_gst_only)
            predicted_gst_only = torch.cat(predicted_gst_only, dim=0)
            gst_only_loss = model.gst_loss(predicted_gst_only, current_gst_only)
            save.writer.add_scalar(f'test/gst_only_loss', gst_only_loss, epoch)
            wst_loss = fake.wst_loss(predicted_wst, current_wst)
        if args.model == 'proposed':
            wst_loss = model.wst_loss(predicted_wst, current_wst)

        current_gst = torch.stack(current_gst)
        predicted_gst = torch.cat(predicted_gst, dim=0)
        gst_loss = model.gst_loss(predicted_gst, current_gst)
        save.writer.add_scalar(f'test/gst_loss', gst_loss, epoch)
        save.writer.add_scalar(f'test/wst_loss', wst_loss, epoch)

        loss = gst_loss + wst_loss
        save.save_log('test', epoch, batch, epoch, loss)
