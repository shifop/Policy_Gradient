import gym
import tensorflow as tf
import numpy as np
import math

"""
流程：
1. 采集游戏过程
2. 优化参数
3. 再次采集
"""

class Actor(object):
    def __init__(self, config):
        self.env = gym.make('Pong-v0')
        self.config = config
        self.__create_graph()

    def __get_gradients(self, value):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.config.lr)
        gradients = opt.compute_gradients(value, params)
        # train_op = opt.apply_gradients(zip(gradients, params))
        return gradients

    def __create_graph(self):
        """
        创建actor的模型结构，简单的卷积
        :return:
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            observation = tf.placeholder(dtype=tf.float32, shape=[None, 210, 160, 3], name='observation')
            action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='action')
            reward = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='reward')
            # 计算概率
            ft = tf.layers.conv2d(observation, 16, [5, 5], padding='SAME', name='conv1')
            ft = tf.layers.max_pooling2d(ft, [2,2], 2, padding='SAME')
            ft = tf.layers.conv2d(ft, 16, [5, 5], padding='SAME', name='conv2')
            ft = tf.layers.max_pooling2d(ft, [2, 2], 2, padding='SAME')
            ft = tf.layers.conv2d(ft, 1, [5, 5], padding='SAME', name='conv3')
            ft = tf.layers.max_pooling2d(ft, [2, 2], 2, padding='SAME')
            ft = tf.reshape(ft, [-1, 27*20])

            p = tf.layers.dense(ft, self.config.action_size, name='dense')

            # 计算logp
            p = tf.nn.softmax(p, axis=-1)
            logp = tf.log(p)
            # 计算用于优化参数的reward
            def r(reward, logp, action):
                """

                :param reward: [-1]
                :param logp: [-1]
                :param action: [-1]
                :return:
                """
                rt = 0.0
                rate = np.array([math.pow(0.9, x) for x in range(len(reward)+1)], dtype=np.float32)
                for index, x in enumerate(reward):
                    rt+=np.sum(rate[:-(index+1)]*reward[index:])*logp[index][action[index]]
                return np.array(rt, np.float32)

            r_hat,_,_ = tf.py_func(r, [tf.reshape(reward, [-1]), tf.reshape(logp, [-1, self.config.action_size]), tf.reshape(action, [-1])], [tf.float32, tf.float32, tf.float32])
            gradients = self.__get_gradients(r_hat)
            # 计算梯度
    def train(self):
        """
        训练，包含采集以及更新参数的过程
        # TODO 合并多个游戏过程的梯度，更新参数
        :return:
        """




class Config(object):
    def __init__(self):
        self.action_size = 6
        self.lr = 1e-3

if __name__=="__main__":
    config = Config()
    oj=Actor(config)
