from copy import deepcopy

class Options(dict):

    def __getitem__(self, key):
        if not key in self.keys():
            self.__setitem__(key, Options())
        return super().__getitem__(key)

    def __getattr__(self, attr):
        if not attr in self.keys():
            self[attr] = Options()
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

    def __deepcopy__(self, memo=None):
        new = Options()
        for key in self.keys():
            new[key] = deepcopy(self[key])
        return new

baseline = Options()
baseline.max_epochs = 50000
baseline.batch_size = 16
baseline.learning_rate = 1e-4
baseline.chunk_size = 6

baseline.input.bert_dim = 768
baseline.input.gst_dim = 768

baseline.global_encoder.input_dim = baseline.input.bert_dim
baseline.global_encoder.prenet.sizes = [256, 128]
baseline.global_encoder.cbhg.dim = 128
baseline.global_encoder.cbhg.K = 16
baseline.global_encoder.cbhg.projections = [128, 128]
baseline.global_encoder.output_dim = baseline.global_encoder.cbhg.dim * 2

baseline.dialogue_gcn.length = baseline.chunk_size - 1
baseline.dialogue_gcn.global_feature_dim = baseline.global_encoder.output_dim + baseline.input.gst_dim
baseline.dialogue_gcn.attention.input_dim = baseline.dialogue_gcn.global_feature_dim
baseline.dialogue_gcn.attention.dim = 128
baseline.dialogue_gcn.rgcn.dim = 128
baseline.dialogue_gcn.gcn.dim = 128
baseline.dialogue_gcn.output_dim = baseline.dialogue_gcn.gcn.dim

baseline.attention.dim = 128
baseline.attention.query_dim = baseline.global_encoder.output_dim + baseline.chunk_size
baseline.attention.key_dim = baseline.dialogue_gcn.output_dim
baseline.attention.value_dim = baseline.dialogue_gcn.output_dim

baseline.linear.input_dim = baseline.attention.value_dim + baseline.attention.query_dim
baseline.linear.output_dim = baseline.input.gst_dim

proposed = deepcopy(baseline)
proposed.input.wst_dim = 768
