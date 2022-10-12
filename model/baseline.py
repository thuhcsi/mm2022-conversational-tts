import torch
from torch import nn
from torch_geometric.nn.conv import GraphConv

from .tacotron import Prenet, CBHG
from .graph import RGCNConv
from .attention import BahdanauAttention

def pad_sequence(sequences, **kwargs):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, **kwargs)

class GlobalEncoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.prenet = Prenet(hparams.input_dim, sizes=hparams.prenet.sizes)
        self.cbhg = CBHG(hparams.cbhg.dim, K=hparams.cbhg.K, projections=hparams.cbhg.projections)

    def forward(self, inputs, input_lengths=None):
        x = self.prenet(inputs)
        x = self.cbhg(x, input_lengths)
        return x

class DialogueGCN(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.global_attention = BahdanauAttention(hparams.global_attention.input_dim, hparams.global_attention.input_dim, hparams.global_attention.input_dim, hparams.global_attention.dim)
        self.rgcn = RGCNConv(hparams.global_feature_dim, hparams.rgcn.dim, 2 * hparams.length ** 2)
        self.gcn = GraphConv(hparams.rgcn.dim, hparams.gcn.dim)

        self.edges = [(i, j) for i in range(hparams.length) for j in range(hparams.length)]
        edge_types = [[f'{i}{j}0', f'{i}{j}1'] for i in range(hparams.length) for j in range(hparams.length)]
        edge_types = [j for i in edge_types for j in i]
        self.edge_type_to_id = {}
        for i, edge_type in enumerate(edge_types):
            self.edge_type_to_id[edge_type] = i

    def forward(self, global_features, speaker):
        edges = torch.tensor(self.edges).T.to(global_features.device)
        edge_type = []
        for i in range(len(speaker)):
            for j in range(len(speaker)):
                direction = 0 if i < j else 1
                edge_type.append(self.edge_type_to_id[f'{speaker[i]}{speaker[j]}{direction}'])
        edge_type = torch.tensor(edge_type).to(global_features.device)

        global_attention_keys = torch.stack([global_features for i in range(len(speaker))])
        _, global_attention_weights = self.global_attention(global_features, global_attention_keys, global_attention_keys)
        edge_weight = torch.flatten(global_attention_weights)

        x = self.rgcn(global_features, edges, edge_type, edge_weight=edge_weight)
        x = self.gcn(x, edges)
        return x

class Baseline(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.global_encoder = GlobalEncoder(hparams.global_encoder)
        self.gcn = DialogueGCN(hparams.dialogue_gcn)
        self.global_attention = BahdanauAttention(hparams.global_attention.query_dim, hparams.global_attention.key_dim, hparams.global_attention.key_dim, hparams.global_attention.dim)
        self.global_linear = nn.Linear(hparams.global_linear.input_dim, hparams.global_linear.output_dim)
        self.mse = nn.MSELoss()

    def forward(self, length, speaker, bert, history_gst):
        batch_size = len(bert)

        global_features = []
        for i in range(batch_size):
            length[i] = length[i].cpu()
            global_features.append(self.global_encoder(bert[i], length[i]))
            global_features[-1] = global_features[-1][range(global_features[-1].shape[0]), (length[i] - 1).long(), :]
        current_global_features = [i[-1] for i in global_features]
        history_global_features = [torch.cat([i[:-1], j], dim=-1) for i, j in zip(global_features[:], history_gst)]

        for i in range(batch_size):
            history_global_features[i] = self.gcn(history_global_features[i], speaker[i][:-1])

        current_speaker = torch.stack([i[-1] for i in speaker])
        current_speaker = nn.functional.one_hot(current_speaker, num_classes=len(speaker[0]))
        history_global_features = torch.stack(history_global_features)
        current_global_features = torch.stack(current_global_features)
        current_global_features = torch.cat([current_global_features, current_speaker], dim=-1)
        context_vector, _ = self.global_attention(current_global_features, history_global_features, history_global_features)
        context_vector = torch.cat([current_global_features, context_vector], dim=-1)
        current_gst = self.global_linear(context_vector)
        return current_gst

    def gst_loss(self, p_gst, gst):
        return self.mse(p_gst, gst)

class FakeMST(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.global_linear = nn.Linear(hparams.global_linear.input_dim, hparams.global_linear.output_dim)
        self.local_linear = nn.Linear(hparams.local_linear.input_dim, hparams.local_linear.output_dim)
        self.mse = nn.MSELoss()

    def forward(self, length, bert, gst):
        predicted_gst = self.global_linear(gst)
        bert = pad_sequence(bert)
        gst = torch.tile(torch.unsqueeze(gst, dim=1), (1, bert.shape[1], 1))
        predicted_wst = self.local_linear(torch.cat([bert, gst], dim=-1))
        predicted_wst = [i[:l] for i, l in zip(predicted_wst, length)]
        return predicted_gst, predicted_wst

    def gst_loss(self, p_gst, gst):
        return self.mse(p_gst, gst)

    def wst_loss(self, p_wst, wst):
        p_wst = torch.cat(p_wst, dim=0)
        wst = torch.cat(wst, dim=0)
        return self.mse(p_wst, wst)

if __name__ == '__main__':
    from data.ecc import ECC
    from data.common import Collate
    from hparams import baseline

    device = 'cpu'
    data_loader = torch.utils.data.DataLoader(ECC('segmented-train'), batch_size=2, shuffle=True, collate_fn=Collate(device))

    model = Baseline(baseline)
    fake = FakeMST(baseline.fake_mst)
    model.to(device)

    for batch in data_loader:
        length, speaker, bert, gst, wst, gst_only = batch
        history_gst = [i[:-1] for i in gst_only]
        predicted_gst = model(length, speaker, bert, history_gst)
        print(predicted_gst.shape)

        current_length = [i[-1] for i in length]
        current_wst = [i[-1, :l] for i, l in zip(wst, current_length)]
        current_bert = [i[-1] for i in bert]
        predicted_gst, predicted_wst = fake(current_length, current_bert, predicted_gst.detach())
        print([i.shape for i in predicted_wst])
        break
