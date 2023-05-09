import numpy as np
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

def sort_key(i):
    return [int(k) for k in i.stem.split('.')[0].split('-')[:-1]]

class Utterance:

    def __init__(self, text):
        with open(text) as f:
            self.text = f.readline()
        self.speaker = text.stem[-1]
        self.wav = text.parent / f'{text.stem}.wav'
        self.bert = text.parent / f'{text.stem}.bert.npy'
        self.sbert = text.parent / f'{text.stem}.sbert.npy'
        self.gst = text.parent / f'{text.stem}.gst.npy'
        self.wst = text.parent / f'{text.stem}.wst.npy'
        self.gst_only = text.parent / f'{text.stem}.gst_only.npy'

class ECC(torch.utils.data.Dataset):

    def __init__(self, segmented, chunk_size=6):
        super().__init__()
        self.path = Path(segmented)
        self.chunk_size = chunk_size

        texts = sorted([i for i in self.path.rglob('*.txt')], key=sort_key)

        self.conversations = []
        previous = None
        for i in texts:
            current = sort_key(i)[-1]

            if current - 1 != previous:
                self.conversations.append([])
            self.conversations[-1].append(Utterance(i))
            previous = current

        self.conversations = [[j for j in i if j.gst.exists() ] for i in self.conversations]
        self.conversations = [i for i in self.conversations if len(i) >= chunk_size]
        self.chunks = [i[j:j+chunk_size] for i in self.conversations for j in range(len(i)-chunk_size)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        speaker = []
        length = []
        bert = []
        sbert = []
        gst = []
        wst = []
        gst_only = []
        speaker_cache = ''
        for i in self.chunks[index]:
            if not i.speaker in speaker_cache:
                speaker_cache += i.speaker
            speaker.append(speaker_cache.find(i.speaker))
            bert.append(torch.as_tensor(np.load(i.bert)))
            sbert.append(torch.as_tensor(np.load(i.sbert)))
            length.append(bert[-1].shape[0])
            gst.append(torch.as_tensor(np.load(i.gst)))
            wst.append(torch.as_tensor(np.load(i.wst)))
            gst_only.append(torch.as_tensor(np.load(i.gst_only)))
        speaker = np.array(speaker)
        length = np.array(length)
        bert = pad_sequence(bert, batch_first=True)
        sbert = torch.stack(sbert)
        gst = torch.stack(gst)
        wst = pad_sequence(wst, batch_first=True)
        gst_only = torch.stack(gst_only)
        return length, speaker, bert, gst, wst, gst_only, sbert

def process_bert(model, tokenizer, utterance: Utterance):
    text = ''.join([i for i in utterance.text.lower() if i in "abcedfghijklmnopqrstuvwxyz' "])
    words = text.split(' ')
    words = [i for i in words if i != '']
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    sbert = outputs.pooler_output[0].detach().numpy()
    bert = outputs.last_hidden_state[0][1:-1].detach().numpy()
    result = []
    start = 0
    for word in words:
        subwords = tokenizer.tokenize(word)
        if len(subwords) > 1:
            result.append(np.mean(bert[start:start+len(subwords)], axis=0, keepdims=False))
        elif len(subwords) == 1:
            result.append(bert[start])
        start += len(subwords)
    try:
        np.save(utterance.bert, np.stack(result))
        np.save(utterance.sbert, sbert)
    except:
        print(utterance.text, utterance.bert)

if __name__ == '__main__':
    import sys
    from functools import partial
    from tqdm.contrib.concurrent import process_map, thread_map

    dataset = ECC(sys.argv[1], chunk_size=6)
    print(len(dataset.conversations), len(dataset.chunks))

    if not dataset.chunks[0][0].bert.exists():
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #process_bert(model, tokenizer, [i for chunk in dataset.chunks for i in chunk][0])
        thread_map(partial(process_bert, model, tokenizer), [i for chunk in dataset.chunks for i in chunk])

    from .common import Collate
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True)
    for batch in data_loader:
        for _list in batch:
            print([i.shape for i in _list])
            print([i.dtype for i in _list])
        break
