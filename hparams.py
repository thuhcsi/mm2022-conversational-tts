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
baseline.max_epochs = 100
baseline.batch_size = 32
baseline.learning_rate = 1e-4

baseline.input.chunk_size = 6
baseline.input.bert_dim = 768
baseline.input.gst_dim = 40
baseline.input.wst_dim = 40

baseline.global_encoder.input_dim = baseline.input.bert_dim
baseline.global_encoder.prenet.sizes = [256, 128]
baseline.global_encoder.cbhg.dim = 128
baseline.global_encoder.cbhg.K = 16
baseline.global_encoder.cbhg.projections = [128, 128]
baseline.global_encoder.output_dim = baseline.global_encoder.cbhg.dim * 2

baseline.dialogue_gcn.length = baseline.input.chunk_size - 1
baseline.dialogue_gcn.global_feature_dim = baseline.global_encoder.output_dim + baseline.input.gst_dim
baseline.dialogue_gcn.global_attention.input_dim = baseline.dialogue_gcn.global_feature_dim
baseline.dialogue_gcn.global_attention.dim = 128
baseline.dialogue_gcn.rgcn.dim = 128
baseline.dialogue_gcn.gcn.dim = 128
baseline.dialogue_gcn.output_dim = baseline.dialogue_gcn.gcn.dim

baseline.global_attention.dim = 128
baseline.global_attention.query_dim = baseline.global_encoder.output_dim + baseline.input.chunk_size
baseline.global_attention.key_dim = baseline.dialogue_gcn.output_dim
baseline.global_attention.value_dim = baseline.dialogue_gcn.output_dim

baseline.global_linear.input_dim = baseline.global_attention.value_dim + baseline.global_attention.query_dim
baseline.global_linear.output_dim = baseline.input.gst_dim

baseline.fake_mst.global_linear.input_dim = baseline.input.gst_dim
baseline.fake_mst.global_linear.output_dim = baseline.input.gst_dim
baseline.fake_mst.local_linear.input_dim = baseline.input.gst_dim + baseline.input.bert_dim
baseline.fake_mst.local_linear.output_dim = baseline.input.wst_dim

proposed = deepcopy(baseline)

proposed.local_encoder = deepcopy(proposed.global_encoder)

proposed.dialogue_gcn.local_feature_dim = proposed.local_encoder.output_dim
proposed.dialogue_gcn.local_attention.dim = 128
proposed.dialogue_gcn.local_attention.k1_dim = proposed.dialogue_gcn.local_feature_dim
proposed.dialogue_gcn.local_attention.k2_dim = proposed.dialogue_gcn.local_feature_dim
proposed.dialogue_gcn.local_attention.v1_dim = proposed.dialogue_gcn.local_attention.k1_dim
proposed.dialogue_gcn.local_attention.v2_dim = proposed.dialogue_gcn.local_attention.k2_dim

proposed.post_global_encoder = deepcopy(proposed.global_encoder)
proposed.post_global_encoder.input_dim = proposed.dialogue_gcn.output_dim

proposed.global_attention.key_dim = proposed.post_global_encoder.output_dim
proposed.global_attention.value_dim = proposed.post_global_encoder.output_dim

proposed.local_attention.dim = 128
proposed.local_attention.k1_dim = proposed.local_encoder.output_dim
proposed.local_attention.k2_dim = proposed.dialogue_gcn.output_dim
proposed.local_attention.v1_dim = proposed.local_attention.k1_dim
proposed.local_attention.v2_dim = proposed.local_attention.k2_dim

proposed.global_linear.input_dim = proposed.global_attention.value_dim + proposed.global_attention.query_dim

proposed.local_linear.input_dim = proposed.local_attention.v2_dim + proposed.local_encoder.output_dim
proposed.local_linear.output_dim = proposed.input.wst_dim
