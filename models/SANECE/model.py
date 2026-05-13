import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from layer import *
class Model:
    def __init__(self, config, Adj):
        self.config = config
        self.W = {}
        self.b = {}
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_idx = tf.placeholder(tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int32)
        self.Adj = tf.constant(Adj, dtype=tf.float32)
        self.A = tf.gather(tf.gather(self.Adj, self.batch_idx, axis=0), self.batch_idx, axis=1)
        self.A_normalize = self.normalize(self.A)
        self.X = tf.placeholder(tf.float32, [None, config.m])
        self.X_neg = tf.placeholder(tf.float32, [None, config.m])
        self.X_hidden1 = GraphConvolution(input_dim=config.gcnlayers[0],
                                         output_dim=config.gcnlayers[1],
                                         adj=self.A_normalize,
                                         act=tf.nn.elu,
                                         dropout=(1-config.keep_prob))(self.X)
        self.Y = GraphConvolution(input_dim=config.gcnlayers[1],
                                 output_dim=config.gcnlayers[2],
                                 adj=self.A_normalize,
                                 act=lambda x: x,
                                 dropout=(1-config.keep_prob))(self.X_hidden1)

        for i in range(len(config.layers) - 1):
            W_name = 'encoder_W' + str(i)
            b_name = 'encoder_b' + str(i)
            self.W[W_name] = tf.get_variable(W_name, shape=[config.layers[i], config.layers[i + 1]],
                                             initializer=tf.glorot_uniform_initializer())
            self.b[b_name] = tf.get_variable(b_name, shape=[config.layers[i + 1]], initializer=tf.zeros_initializer())

        self.Q = self.makeGraph(config, self.X)
        self.Q_neg = self.makeGraph(config, self.X_neg)
        self.Y_final = tf.concat([self.Y, self.Q], 1)
        self.loss_network = self.get_network_loss(config, self.Y, self.Q, self.Q_neg)
        self.train_net = tf.train.AdamOptimizer(config.lr).minimize(self.loss_network)

        ############ define variables for skipgram  ####################
        # construct variables for nce loss
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.nce_weights = tf.get_variable("nce_weights", [
            config.n, config.dimension * 2], initializer=tf.glorot_uniform_initializer())
        self.nce_biases = tf.get_variable(
            "nce_biases", [config.n], initializer=tf.zeros_initializer())
        self.loss_sg = self.make_skipgram_loss(config)
        self.train_sg = tf.train.AdamOptimizer(config.lr_sg).minimize(self.loss_sg)

    def normalize(self, A):
        A = A + tf.eye(self.batch_size)
        rowsum = tf.reduce_sum(A, axis=1)+1e-10
        D = tf.pow(rowsum, -0.5)
        d_inv_sqrt = tf.diag(D)
        return tf.matmul(tf.transpose(tf.matmul(A, d_inv_sqrt)), d_inv_sqrt)

    def makeGraph(self, config, X):
        def encoder(X):
            for i in range(len(config.layers) - 1):
                W_name = 'encoder_W' + str(i)
                b_name = 'encoder_b' + str(i)
                Wx_plus_b = tf.matmul(X, self.W[W_name]) + self.b[b_name]
                Wx_plus_b = tf.nn.dropout(Wx_plus_b, self.keep_prob)
                mean, var = tf.nn.moments(Wx_plus_b, [0, 1])
                scale = tf.Variable(tf.ones([config.layers[i + 1]]))
                offset = tf.Variable(tf.zeros([config.layers[i + 1]]))
                epsilon = config.epsilon
                Wx_plus_b_normalized = tf.nn.batch_normalization(Wx_plus_b, mean, var, offset, scale, epsilon)
                X = tf.nn.elu(Wx_plus_b_normalized)
            return X

        Q = encoder(X)
        return Q

    def get_siamese_loss(self, X1, X2, X3):
        eucudian_dist_positive = tf.reduce_sum(tf.pow((X1 - X2), 2))
        eucudian_dist_negative = tf.reduce_sum(tf.pow((X1 - X3), 2))
        triplet_loss = tf.maximum((eucudian_dist_positive - eucudian_dist_negative + self.config.batch_size * self.config.margin),0)
        return triplet_loss

    def get_reg_loss(self, Ws, bs):
        loss_reg = tf.add_n([tf.nn.l2_loss(W) for W in Ws.values()])
        loss_reg += tf.add_n([tf.nn.l2_loss(b) for b in bs.values()])
        return loss_reg

    def get_network_loss(self, config, Y, Q_pos, Q_neg):
        loss_siamese = self.get_siamese_loss(Y, Q_pos, Q_neg)
        loss_reg = self.get_reg_loss(self.W, self.b)
        loss = loss_siamese + config.reg * loss_reg
        return loss

    def make_skipgram_loss(self,config):
        loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=self.labels,
            inputs=self.Y_final,
            num_sampled=config.num_sampled,
            num_classes=config.n))
        return loss
