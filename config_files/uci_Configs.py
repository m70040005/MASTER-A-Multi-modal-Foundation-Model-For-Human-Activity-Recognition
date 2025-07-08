class Config(object):
    def __init__(self):
        # modality configs
        self.modalities = ['acc', 'gyro', 'total_acc']
        self.missing_modality_selection_range = ['acc', 'gyro', 'total_acc']
        self.modality_num = 3
        self.chunk_length = [20, 20, 20]
        self.data_time_max_len = [120, 120, 120]
        self.features_len = [1, 1, 1]
        self.features_max_len = 19

        # models configs
        self.empty_fill = 0
        self.embedding_dim = 128
        self.mlm_probability = 0.15
        self.transformer_depth = 4
        self.transformer_heads = 3
        self.mlp_ratio = 1
        self.temperature = 0.1
        self.align_ratio = 2
        self.align_loss_lamda = 0.1
        self.pred_dropout = 0.1
        self.num_classes = 6

        # training configs
        self.batch_size = 128
        self.num_epoch = 100
        self.num_exchange_freeze_epoch = 3
        self.num_exchange_unfreeze_epoch = 3
        self.early_stop_step = 30

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4