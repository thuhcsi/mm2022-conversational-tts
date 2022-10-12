import torch
from torch import nn

from .tacotron import Prenet, CBHG
from .graph import RGCNConv_FG
from .attention import BahdanauAttention, BidirectionalAttention
from .baseline import pad_sequence, GlobalEncoder

def pad_attention_weights(attention_weights, length):
    result = [[torch.cat([j, torch.zeros((length - j.shape[0], j.shape[1]), device=j.device)], dim=0) for j in i] for i in attention_weights]
    result = [[torch.cat([j, torch.zeros((j.shape[0], length - j.shape[1]), device=j.device)], dim=1) for j in i] for i in result]
    result = torch.cat([torch.stack(i) for i in result])
    return result

class DialogueGCN_FG(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.global_attention = BahdanauAttention(hparams.global_attention.input_dim, hparams.global_attention.input_dim, hparams.global_attention.input_dim, hparams.global_attention.dim)
        self.local_attention = BidirectionalAttention(hparams.local_attention.k1_dim, hparams.local_attention.k2_dim, hparams.local_attention.v1_dim, hparams.local_attention.v2_dim, hparams.local_attention.dim)
        self.rgcn = RGCNConv_FG(hparams.local_feature_dim, hparams.rgcn.dim, 2 * hparams.length ** 2)
        self.gcn = RGCNConv_FG(hparams.rgcn.dim, hparams.gcn.dim, 1)

        self.edges = [(i, j) for i in range(hparams.length) for j in range(hparams.length)]
        edge_types = [[f'{i}{j}0', f'{i}{j}1'] for i in range(hparams.length) for j in range(hparams.length)]
        edge_types = [j for i in edge_types for j in i]
        self.edge_type_to_id = {}
        for i, edge_type in enumerate(edge_types):
            self.edge_type_to_id[edge_type] = i

    def forward(self, global_features, local_features, speaker, length):
        edges = torch.tensor(self.edges).T.to(global_features.device)
        edge_type = []
        for i in range(len(speaker)):
            for j in range(len(speaker)):
                direction = 0 if i < j else 1
                edge_type.append(self.edge_type_to_id[f'{speaker[i]}{speaker[j]}{direction}'])
        edge_type = torch.tensor(edge_type).to(global_features.device)

        global_attention_keys = torch.stack([global_features for i in range(len(speaker))])
        _, global_attention_weights = self.global_attention(global_features, global_attention_keys, global_attention_keys)
        global_attention_weights = torch.flatten(global_attention_weights)

        local_attention_weights = []
        for i in range(len(speaker)):
            local_attention_k1 = torch.stack([local_features[i] for j in range(len(speaker))])
            local_attention_k2 = torch.stack([local_features[j] for j in range(len(speaker))])
            local_attention_k1_length = torch.stack([length[i] for j in range(len(speaker))])
            local_attention_k2_length = torch.stack([length[j] for j in range(len(speaker))])
            _, _, w1, w2, _ = self.local_attention(local_attention_k1, local_attention_k2, local_attention_k1, local_attention_k2, k1_lengths=local_attention_k1_length, k2_lengths=local_attention_k2_length)
            local_attention_weights.append(w1)
        #local_attention_weights = torch.cat(local_attention_weights)
        local_attention_weights = pad_attention_weights(local_attention_weights, local_features.shape[1])

        edge_weight = torch.stack([global_attention_weights[i] * local_attention_weights[i] for i in range(len(self.edges))])

        x = self.rgcn(local_features, edges, edge_type, edge_weight=edge_weight)

        edge_type = torch.zeros(edge_type.shape, device=edge_type.device)
        edge_weight = local_attention_weights
        x = self.gcn(x, edges, edge_type, edge_weight=edge_weight)

        return x

class Proposed(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.global_encoder = GlobalEncoder(hparams.global_encoder)
        self.local_encoder = GlobalEncoder(hparams.local_encoder)
        self.gcn = DialogueGCN_FG(hparams.dialogue_gcn)
        self.post_global_encoder = GlobalEncoder(hparams.post_global_encoder)
        self.global_attention = BahdanauAttention(hparams.global_attention.query_dim, hparams.global_attention.key_dim, hparams.global_attention.key_dim, hparams.global_attention.dim)
        self.local_attention = BidirectionalAttention(hparams.local_attention.k1_dim, hparams.local_attention.k2_dim, hparams.local_attention.v1_dim, hparams.local_attention.v2_dim, hparams.local_attention.dim)
        self.global_linear = nn.Linear(hparams.global_linear.input_dim, hparams.global_linear.output_dim)
        self.local_linear = nn.Linear(hparams.local_linear.input_dim, hparams.local_linear.output_dim)
        self.mse = nn.MSELoss()

    def forward(self, length, speaker, bert, history_gst, history_wst):
        local_features = []
        global_features = []
        batch_size = len(bert)
        for i in range(batch_size):
            length[i] = length[i].cpu()
            features = torch.cat([history_wst[i], torch.zeros((1, ) +  history_wst[i].shape[1:], device=history_wst[i].device)])
            features = torch.cat([features, bert[i]], dim=-1)
            local_features.append(self.local_encoder(features, length[i]))
            global_features.append(self.global_encoder(features, length[i]))
            global_features[-1] = global_features[-1][range(global_features[-1].shape[0]), (length[i] - 1).long(), :]

        current_global_features = [i[-1] for i in global_features]
        history_global_features = [torch.cat([i[:-1], j], dim=-1) for i, j in zip(global_features[:], history_gst)]
        current_local_features = [i[-1] for i in local_features]
        history_local_features = [i[:-1] for i in local_features]

        for i in range(batch_size):
            history_local_features[i] = self.gcn(history_global_features[i], history_local_features[i], speaker[i][:-1], length[i])

        for i in range(batch_size):
            history_global_features[i] = self.post_global_encoder(history_local_features[i], length[i][:-1])
            history_global_features[i] = history_global_features[i][range(history_global_features[i].shape[0]), (length[i][:-1] - 1).long(), :]

        current_speaker = torch.stack([i[-1] for i in speaker])
        current_speaker = nn.functional.one_hot(current_speaker, num_classes=len(speaker[0]))
        history_global_features = torch.stack(history_global_features)
        current_global_features = torch.stack(current_global_features)
        current_global_features = torch.cat([current_global_features, current_speaker], dim=-1)
        global_context_vector, global_attention_weights = self.global_attention(current_global_features, history_global_features, history_global_features)
        global_context_vector = torch.cat([current_global_features, global_context_vector], dim=-1)
        current_gst = self.global_linear(global_context_vector)
        #print(global_context_vector.shape, global_attention_weights.shape, current_gst.shape)

        local_attention_weights = []
        for i in range(batch_size):
            local_attention_k1 = torch.stack([current_local_features[i] for j in range(len(history_local_features[i]))])
            local_attention_k2 = history_local_features[i]
            local_attention_k1_length = torch.stack([length[i][-1] for j in range(len(history_local_features[i]))])
            local_attention_k2_length = length[i][:-1]
            _, _, w1, w2, _ = self.local_attention(local_attention_k1, local_attention_k2, local_attention_k1, local_attention_k2, k1_lengths=local_attention_k1_length, k2_lengths=local_attention_k2_length)
            local_attention_weights.append(pad_attention_weights([w1], history_local_features[i].shape[1]))
        #print([i.shape for i in local_attention_weights])
        #local_attention_weights = [pad_attention_weights(i, )

        attention_weights = []
        for i in range(batch_size):
            attention_weights.append(torch.stack([global_attention_weights[i][j] * local_attention_weights[i][j] for j in range(len(global_attention_weights[i]))]))
        #print([i.shape for i in attention_weights])

        local_context_vector = [torch.sum(torch.bmm(attention_weights[i], history_local_features[i]), dim=0) for i in range(batch_size)]
        #print([i.shape for i in local_context_vector])

        local_context_vector = [torch.cat([local_context_vector[i], current_local_features[i]], dim=-1) for i in range(batch_size)]
        #print([i.shape for i in local_context_vector])

        local_context_vector = pad_sequence(local_context_vector)
        current_wst = self.local_linear(local_context_vector)

        current_wst = [current_wst[i, :length[i][-1]] for i in range(batch_size)]
        #print([i.shape for i in current_wst])

        return current_gst, current_wst

    def gst_loss(self, p_gst, gst):
        return self.mse(p_gst, gst)

    def wst_loss(self, p_wst, wst):
        p_wst = torch.cat(p_wst, dim=0)
        wst = torch.cat(wst, dim=0)
        return self.mse(p_wst, wst)

if __name__ == '__main__':
    from data.ecc import ECC
    from data.common import Collate
    from hparams import proposed

    device = 'cpu'
    data_loader = torch.utils.data.DataLoader(ECC('segmented-train'), batch_size=2, shuffle=True, collate_fn=Collate(device))

    model = Proposed(proposed)
    model.to(device)

    for batch in data_loader:
        length, speaker, bert, gst, wst, _ = batch
        history_gst = [i[:-1] for i in gst]
        history_wst = [i[:-1] for i in wst]
        current_gst = [i[-1] for i in gst]
        current_gst = torch.stack(current_gst)
        current_length = [i[-1] for i in length]
        current_wst = [i[-1, :l] for i, l in zip(wst, current_length)]

        predicted_gst, predicted_wst = model(length, speaker, bert, history_gst, history_wst)
        print(model.gst_loss(predicted_gst, current_gst))
        print(model.wst_loss(predicted_wst, current_wst))
        break
