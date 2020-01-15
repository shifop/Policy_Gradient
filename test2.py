import gym
import tensorflow as tf
import numpy as np
import math
import logging
import os
from datetime import timedelta
import time
logging.basicConfig(level=logging.INFO)

def get_loss(action, action_size, p):
    logp = tf.math.log(p+1e-8)
    tag = tf.one_hot(tf.reshape(tf.cast(action, dtype=tf.int32), [-1]), depth=action_size, axis=-1)
    logp = tf.reduce_sum(logp*tag, axis=-1)
    return logp

class Actor(object):
    def __init__(self, config):
        self.config = config
        self.__create_graph()

    def __get_gradients(self, value):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.config.lr)
        gradients = opt.compute_gradients(value, params)
        train_op = opt.apply_gradients(zip(gradients, params))
        return train_op

    def __create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 210, 160, 1], name='observation')
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='action')
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')

            ft = tf.reshape(self.observation, [-1, 210 * 160])
            ft = tf.layers.dense(ft, 256, activation=tf.tanh, name='dense')
            ft = tf.layers.dense(ft, 256, activation=tf.tanh, name='dense2')
            p = tf.layers.dense(ft, self.config.action_size, name='dense3')

            self.p_st = tf.nn.softmax(p)
            dist = tf.distributions.Categorical(p)
            self.selected_actions = dist.sample()
            # self.log_scores = dist.log_prob(tf.reshape(self.action, [-1]))
            self.log_scores = get_loss(self.action, self.config.action_size, tf.nn.softmax(p))

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_sum(-self.log_scores * tf.reshape(self.reward,[-1]))
                self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)


    def __get_tau(self, count, sess):
        # env = gym.make('Pong-v0')
        env = gym.make('Pong-v0')
        env.seed(11037)
        tau = []
        rate = np.reshape(np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), [1, 1, 3])
        for index in range(count):
            observation = env.reset()
            observation = (rate * observation).sum(axis=-1, keepdims=True)
            next_observation, _, _, _ = env.step(0)
            next_observation = (rate * next_observation).sum(axis=-1, keepdims=True)
            flag=True
            step=0
            while(flag):
                # time.sleep(0.05)
                cache = [next_observation - observation]
                # env.render()
                p_st = sess.run(self.p_st, feed_dict={self.observation: np.array(cache, dtype=np.float32)})
                if p_st[0][0] > np.random.uniform():
                    next_action = [0]
                else:
                    next_action = [1]
                observation = next_observation
                next_observation, reward, done, info = env.step(next_action[0]+2)
                next_observation = (rate * next_observation).sum(axis=-1, keepdims=True)
                cache = cache+[next_action, reward]
                tau.append(cache)
                step += 1
                flag = not done
        env.close()
        return tau

    def train(self):
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            logging.info('episode:%d, batch_size:%d, lr:%f'%(self.config.episode, self.config.batch_size, self.config.lr))
            start_time = time.time()
            for step in range(self.config.episode):
                # logging.info("episode:%d"%(step))
                tau = self.__get_tau(10, sess)
                all_rewoards = sum([_[-1] for _ in tau if _[-1]>0])
                time_dif = get_time_dif(start_time)
                cache = tau
                obervation = np.concatenate([np.expand_dims(x[0], axis=0) for x in cache], axis=0)
                action = np.concatenate([np.expand_dims(x[1], axis=0) for x in cache], axis=0)
                reward = np.array([x[-1] for x in cache])
                reward = discount_reward(reward, 0.99)
                reward = np.concatenate([np.expand_dims([x], axis=0) for x in reward], axis=0)
                loss,_ = sess.run([self.loss, self.train_op], feed_dict={self.observation:obervation, self.action:action, self.reward:reward})
                msg = 'step: {0:>6}, loss: {1:>6} total recward: {2:>6}, Time: {3}'
                logging.info(msg.format(step, loss, all_rewoards, time_dif))

class Config(object):
    def __init__(self):
        self.action_size = 2
        self.lr = 1e-4
        self.episode = 10000
        self.batch_size = 2048

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def discount_reward(r_dic, gamma):
    r = 0
    for i in range(len(r_dic) - 1, -1, -1):
        if r_dic[i] != 0:
            r = r_dic[i]
        else:
            r = r * gamma
            r_dic[i] = r
    r_dic = (r_dic - r_dic.mean()) / (r_dic.std() + 1e-8)
    return r_dic

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = Config()
    oj=Actor(config)
    oj.train()
    # oj.get_tau(2)
