import tensorflow as tf
from tensorflow.contrib import signal

class MuLaw:
    def __init__(self, mu=256, int_type=tf.int32, float_type=tf.float32):
        self.mu = float(mu-1)
        self.int_type = int_type
        self.float_type = float_type
    
    def transform(self, x, name='MuLaw_Encode'):
        assert isinstance(x, tf.Tensor), 'input must be tf.Tensor'
        with tf.name_scope(name):
            if x.dtype != self.float_type:
                x = tf.cast(x, self.float_type)
            signal = tf.sign(x)*(tf.log1p(self.mu*tf.abs(x)) / tf.log1p(self.mu))
            signal = (signal + 1)/2*self.mu + 0.5
            signal = tf.cast(signal, self.int_type)
        return signal
    
    def reverse(self, y, name='MuLaw_Decode'):
        assert isinstance(y, tf.Tensor), 'input must be tf.Tensor'
        with tf.name_scope(name):
            if y.dtype != self.float_type:
                y = tf.cast(y, self.float_type)
            y = 2 * (y / self.mu) - 1
            y = tf.sign(y) * (1.0 / self.mu) * (tf.pow((1.0 + self.mu), tf.abs(y)) - 1.0)
        return y

class Spectorogram:
    def __init__(self, sampling_rate, nfft, hop_length):
        self.sr = sampling_rate
        self.nfft = nfft
        self.hop_length = hop_length
    
    def make_pow_spectorogram(self, waveform, name='make_spectorogram'):
        assert isinstance(waveform, tf.Tensor), 'input must be tf.Tensor'
        with tf.name_scope(name):
            frames = signal.stft(waveform, frame_length=self.nfft, frame_step=self.hop_length, fft_length=self.nfft, pad_end=True)
            pow_spectorogram = tf.pow(tf.abs(frames), 2.0)
        return pow_spectorogram
    
    def make_mel_spectorogram(self, waveform, mel_bins=128, name='make_mel_spectorogram'):
        assert isinstance(waveform, tf.Tensor), 'input must be tf.Tensor'
        with tf.name_scope(name):
            power_spec = self.make_pow_spectorogram(waveform)
            mel_matrix = signal.linear_to_mel_weight_matrix(num_mel_bins=mel_bins, num_spectrogram_bins=int(self.nfft/2+1), sample_rate=self.sr, lower_edge_hertz=0.0, upper_edge_hertz=int(self.sr/2))
            #ax = len(power_spec.shape)-1
            #mel_spec = tf.tensordot(power_spec, mel_matrix, ax)
            mel_spec = tf.einsum('ijk,kl->ijl', power_spec, mel_matrix)
        return mel_spec
    
    def power2db(self, spec, top_db=80.0, normalize=True, name='power_to_db'):
        assert isinstance(spec, tf.Tensor), 'input must be tf.Tensor'
        with tf.name_scope(name):
            eps = 1e-10
            ref = tf.reduce_max(spec)
            log_spec = 10.0 * self.log_base(tf.clip_by_value(spec, clip_value_min=eps, clip_value_max=ref), base=10.0)
            log_spec -= 10.0 * self.log_base(tf.maximum(eps, ref), base=10.0)
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec)-top_db)
            if normalize:
                log_spec = (log_spec + (top_db / 2.0)) / (top_db / 2.0)
        return log_spec
    
    def log_base(self, x, base=10.0):
        assert isinstance(x, tf.Tensor), 'x must be tf.Tensor'
        assert base > 0 and (type(base) in [type(tf.Tensor), float]), 'base must be floating number and larger than 0'
        with tf.name_scope('log_with_base'):
            y = tf.log(x)
            d = tf.log(base)
            return y/d