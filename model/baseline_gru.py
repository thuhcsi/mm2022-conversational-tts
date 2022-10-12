import torch
from torch import nn

from .baseline import GlobalEncoder, FakeMST, pad_sequence

class BaselineGRU(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.gru = nn.GRU(hparams.gru.input_dim, hparams.gru.dim, 1, batch_first=True)
        self.global_encoder = GlobalEncoder(hparams.global_encoder)
        self.global_linear = nn.Linear(hparams.global_linear.input_dim, hparams.global_linear.output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.mse = nn.MSELoss()

    def forward(self, length, speaker, bert, history_gst, sbert):
        batch_size = len(length)

        history_sbert = torch.stack([i[:-1] for i in sbert])
        history_global_features, _ = self.gru(history_sbert)
        history_global_features = history_global_features[:,-1]

        current_length = [i[-1] for i in length]
        current_length = torch.stack(current_length)
        current_bert = [i[-1] for i in bert]
        current_bert = pad_sequence(current_bert)
        current_sbert = [i[-1] for i in sbert]
        current_sbert = torch.stack(current_sbert)
        current_global_feature = self.global_encoder(current_bert, current_length.cpu())
        current_global_feature = current_global_feature[:,-1]

        current_speaker = torch.stack([i[-1] for i in speaker])
        current_speaker = nn.functional.one_hot(current_speaker, num_classes=len(speaker[0]))
        current_global_feature = torch.cat([history_global_features, current_sbert, current_global_feature, current_speaker], dim=-1)

        current_gst = self.global_linear(current_global_feature)
        current_gst = current_gst.contiguous().view(batch_size, 4, 10)
        current_gst = self.softmax(current_gst)
        current_gst = current_gst.contiguous().view(batch_size, 40)
        return current_gst

    def gst_loss(self, p_gst, gst):
        return self.mse(p_gst, gst)

if __name__ == '__main__':
    from data.ecc import ECC
    from data.common import Collate
    from hparams import baseline_gru

    device = 'cpu'
    data_loader = torch.utils.data.DataLoader(ECC('segmented-train'), batch_size=2, shuffle=True, collate_fn=Collate(device))

    model = BaselineGRU(baseline_gru)
    fake = FakeMST(baseline_gru.fake_mst)
    model.to(device)

    for batch in data_loader:
        length, speaker, bert, gst, wst, gst_only, sbert = batch
        history_gst = [i[:-1] for i in gst_only]
        predicted_gst = model(length, speaker, bert, history_gst, sbert)
        print(predicted_gst.shape)

        current_length = [i[-1] for i in length]
        current_wst = [i[-1, :l] for i, l in zip(wst, current_length)]
        current_bert = [i[-1] for i in bert]
        predicted_gst, predicted_wst = fake(current_length, current_bert, predicted_gst.detach())
        print([i.shape for i in predicted_wst])
        break
