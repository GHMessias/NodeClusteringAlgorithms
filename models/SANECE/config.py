class Config:
    def __init__(self):
        # cora
        self.reg = 5
        self.epsilon = 1e-3
        self.dimension = 64
        self.gcnlayers = [None, 128, self.dimension]
        self.layers = [None, 512, self.dimension]
        self.lr = 1e-3
        self.lr_sg = 1e-3
        self.batch_size = 512
        self.batch_size_sg = 512
        self.keep_prob = 1.0
        self.p = 1.0
        self.q = 1.0
        self.num_walks = 20
        self.walk_length = 10
        self.window_size = 10
        self.num_sampled = 5
        self.margin = 0.8
