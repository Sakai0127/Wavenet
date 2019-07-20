import tensorflow as tf

class ResidualNet:
    def __init__(self, n_loop=2, layers_per_loop=10, k_size=2, res_channel=64, skip_channel=256, conditioning=True, name='ResNet'):
        self.dilations = [2**i for i in range(layers_per_loop)]*n_loop
        self.k_size = k_size
        self.res_channel = res_channel
        self.skip_channel = skip_channel
        self.conditioning = conditioning
        self.name = name
    
    def resblock(self, input, dilation, cond=None, name='ResBlock'):
        with tf.variable_scope(name):
            length = input.shape[1]
            x = tf.pad(input, [[0, 0],[dilation * (self.k_size-1), 0], [0, 0]])
            x = tf.layers.conv1d(x, self.res_channel*2, self.k_size, dilation_rate=dilation)
            x = x[:,:length,:]
            if self.conditioning and cond is not None:
                x += cond
            tanh_z, sig_z = tf.split(x, 2, 2)
            z = tf.tanh(tanh_z)*tf.sigmoid(sig_z)
            res = tf.layers.conv1d(z, self.res_channel, 1) + input
            skip_connection = tf.layers.conv1d(z, self.skip_channel, 1)
        return res, skip_connection
    
    def __call__(self, input, condition=None, activation=tf.nn.relu):
        with tf.variable_scope(self.name):
            x = input
            for idx, (r, c) in enumerate(zip(self.dilations, condition)):
                x, skip = self.resblock(x, r, c, name='ResBlock_%d'%(idx+1))
                if idx == 0:
                    skip_connection = skip
                else:
                    skip_connection += skip
            if activation:
                skip_connection = activation(skip_connection)
        return skip_connection

class UpsampleNet:
    def __init__(self, layers, out_channels, channels=[128, 128], scales=[16, 16], name='Upsample'):
        self.layers = layers
        self.out_channels = out_channels
        self.channels = channels
        self.scales = scales
        self.name = name
        assert len(self.channels) == len(self.scales)
    
    def upsampling(self, input):
        with tf.variable_scope(self.name):
            conditions = []
            with tf.variable_scope('Deconvolution'):
                x = tf.expand_dims(input, 1)
                for c, s in zip(self.channels, self.scales):
                    x = tf.layers.conv2d_transpose(x, c, (1, s), (1, s))
                    x = tf.nn.relu(x)
                x = tf.squeeze(x, 1)
            with tf.variable_scope('Encode_feature'):
                for _ in range(self.layers):
                    conditions.append(tf.layers.conv1d(x, self.out_channels, 1))
            return conditions
    
    def __call__(self, input):
        return self.upsampling(input)

class Wavenet:
    def __init__(self, n_loop=2, layers_per_loop=10, k_size=2, res_channel=64, skip_channel=256, out_channel=256, conditioning=True, name='Wavenet'):
        self.n_loop = n_loop
        self.layers_per_loop = layers_per_loop
        self.k_size = k_size
        self.res_channel = res_channel
        self.skip_channel = skip_channel
        self.out_channel = out_channel
        self.conditioning = conditioning
        self.name = name

        self.Resnet = ResidualNet(self.n_loop, self.layers_per_loop, self.k_size, self.res_channel, self.skip_channel, self.conditioning)
    
    def decode(self, input, cond=None):
        with tf.variable_scope(self.name):
            with tf.variable_scope('Embed_convolution'):
                x = tf.layers.conv1d(input, self.res_channel, 2, padding='same', use_bias=False)
                x = tf.nn.tanh(x)
            skips = self.Resnet(x, cond)
            with tf.variable_scope('Projection'):
                z = tf.layers.conv1d(skips, self.skip_channel, 1, use_bias=False)
                z = tf.nn.relu(z)
                z = tf.layers.conv1d(z, self.out_channel, 1)
        return z
    
    def __call__(self, input, cond=None):
        return self.decode(input, cond)