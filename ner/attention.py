import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    全连接层
    :param x: 输入
    :param scope: 变量域名
    :param nf: 输出维度
    :param w_init_stdev:
    :return:
    """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        with tf.device('/cpu:0'):
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def attn(x, scope, n_state, hparams):
    """
    注意力层
    :param x: 原始输入 [batch,seq_len,embedding_size]
    :param scope: 变量域名
    :param n_state: 词向量维度
    :param past: 上文
    :param hparams: 超参数
    :return:
    """
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0

    def split_heads(x):
        """
        将输入文本的对应词向量分割，并转化为[batch, heads,sequence,features]的形式
        :param x:
        :return:
        """
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x_reshape = split_states(x, hparams.n_head)
        return tf.transpose(x_reshape, [0, 2, 1, 3])

    def multihead_attn(q, k, v):
        """
        计算多头注意力, q,k,v都是相同值经过不同矩阵线性变化而来
        :param q:
        :param k:
        :param v:
        :return:
        """
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        rsqrt = tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        w = w * rsqrt
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        c_split = tf.split(c, 3, axis=2)
        q, k, v = map(split_heads, c_split)

        a = multihead_attn(q, k, v)
        a = conv1d(a, 'c_proj', 1)
        return a