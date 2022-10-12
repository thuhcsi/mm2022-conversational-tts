import numpy as np
import torch
from pathlib import Path

def process_bert(model, tokenizer, line):
    inputs = tokenizer(line[1], return_tensors="pt")
    outputs = model(**inputs)
    np.save(line[2], outputs.last_hidden_state[0].detach().numpy())
    np.save(line[3], outputs.pooler_output[0].detach().numpy())

class ECC(torch.utils.data.Dataset):

    def __init__(self, path, segmented=None, chunk_size=5):
        super().__init__()
        self.path = Path(path)
        self.segmented = Path(segmented) if segmented else self.path / 'segmented'
        self.chunk_size = chunk_size

        conversations = self.path / 'conversations'
        self.conversations = []
        for filename in conversations.rglob('*.txt'):
            i = filename.name[21:23]

            file = open(filename)
            lines = file.readlines()
            file.close()

            breaks = [j for j, line in enumerate(lines) if line == '\n']
            breaks.insert(0, -1)
            if breaks[-1] != len(lines) - 1:
                breaks.append(len(lines)-1)

            lines = [i.strip('\r\n').split('\t')[::3] for i in lines]

            self.conversations.append([lines[breaks[j]+1:breaks[j+1]] for j in range(len(breaks)-1)])
            for j, conversation in enumerate(self.conversations[-1]):
                for k, line in enumerate(conversation):
                    self.conversations[-1][j][k].append(Path(self.segmented / f'{i}-{j}-{k}-{line[0]}.bert.npy'))
                    self.conversations[-1][j][k].append(Path(self.segmented / f'{i}-{j}-{k}-{line[0]}.sbert.npy'))
                    self.conversations[-1][j][k].append(Path(self.segmented / f'{i}-{j}-{k}-{line[0]}.gst.npy'))
                    self.conversations[-1][j][k].append(Path(self.segmented / f'{i}-{j}-{k}-{line[0]}.wst.npy'))

        self.conversations = [j for i in self.conversations for j in i]
        self.conversations = [i for i in self.conversations if len(i) >= chunk_size]
        self.chunks = [i[j:j+chunk_size] for i in self.conversations for j in range(len(i)-chunk_size)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        result = []
        for i in self.chunks[index]:
            bert = np.load(i[2])
            sbert = np.load(i[3])
            #gst = np.load(self.chunks[index][4])
            #wst = np.load(self.chunks[index][5])
            result.append([bert, sbert])

        return bert, sbert#, gst, wst

if __name__ == '__main__':
    import sys
    from functools import partial
    from tqdm.contrib.concurrent import process_map, thread_map

    dataset = ECC(sys.argv[1], sys.argv[2])
    print(len(dataset.conversations), len(dataset.chunks))

    if not dataset.conversations[0][0][2].exists():
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        thread_map(partial(process_bert, model, tokenizer), [line for conversation in dataset.conversations for line in conversation])

    #from .common import Collate
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True)
    #for batch in data_loader:
    #    for _list in batch:
    #        print([i.shape for i in _list])
    #        print([i.dtype for i in _list])
    #    break
