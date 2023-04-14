import tensorflow as tf
print(tf.__version__)

from random import seed
from keras.optimizers import Optimizer
from keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

# Proposed BSG opitmizer code
# Using Binary search technique with Adam
class BSG(Optimizer):
    def __init__(self,
                 l_r=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 alpha=10000,
                 amsgrad=False,
                 type=1,
                 **kwargs):
        super(BSG, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.l_r = K.variable(l_r, name='l_r')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.alpha = alpha
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.type = type

    def _create_all_weights(self, params):
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats
        return ms, vs, vhats

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        seed(1)
        N = [K.variable(np.full(fill_value=0.00, shape=shape)) for shape in shapes]
        P = [K.variable(np.full(fill_value=0.00, shape=shape)) for shape in shapes]
        X = [K.zeros(shape) for shape in shapes]
        Y = [K.zeros(shape) for shape in shapes]
        c = K.variable(2, dtype='float32', name='c')
        s = K.variable(1.0, dtype='float32', name='d')

        self.updates = []

        lr = self.l_r
        alpha = self.alpha
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                    1. /
                    (1. +
                     self.decay * math_ops.cast(self.iterations, K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        lr_t = lr * (
                K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                (1. - math_ops.pow(self.beta_1, t)))

        ms, vs, vhats = self._create_all_weights(params)
        for p, g, m, v, vhat, x, y, n, po in zip(params, grads, ms, vs, vhats, X, Y, N, P):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            r = po - n
            x = K.identity(g)
            y = K.identity(g)
            x = tf.math.truediv(tf.cast(x > 0, x.dtype), s)  # positive gradient
            y = tf.math.truediv(tf.cast(y <= 0, y.dtype), s)  # negative gradient

            temp_p_t = (p - p_t)

            if self.iterations == 1:
                n_new = (p - temp_p_t) * y + (
                            p - alpha * lr * temp_p_t) * x  # for negative gtadient x is zero. so for positive case value will not change
                p_new = (p - temp_p_t) * x + (
                            p - alpha * lr * temp_p_t) * y  # for positive gtadient y is zero and for negative gradient y is one.  so for negative case value will not change

            else:

                r = tf.math.truediv(tf.cast(r < 0, r.dtype), s)  # positive is less than negative
                n_new = (p - temp_p_t) * y + n * x * (1 - r) + (
                            p - alpha * lr * temp_p_t) * x * r  # for negative gtadient x is zero. so for positive case value will not change
                p_new = (p - temp_p_t) * x + po * y * (1 - r) + (
                            p - alpha * lr * temp_p_t) * y * r  # for positive gtadient y is zero and for negative y is one.  so for negative case value will not change

            w_new = tf.math.truediv((n_new + p_new), c)
            if (self.type == 1):
                new_p = w_new  # BDO
            else:
                new_p = p_t  # this is for adam

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            self.updates.append(state_ops.assign(p, new_p))
            self.updates.append(state_ops.assign(n, n_new))
            self.updates.append(state_ops.assign(po, p_new))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(BSG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
