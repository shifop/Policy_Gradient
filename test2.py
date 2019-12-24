import gym
import tensorflow as tf
import numpy as np
import math
import logging
import os
import time
logging.basicConfig(level=logging.INFO)

"""
流程：
1. 采集游戏过程
2. 优化参数
3. 再次采集
"""
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
        """
        创建actor的模型结构，简单的卷积
        :return:
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 210, 160, 1], name='observation')
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='action')
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
            # 计算概率
            ft = tf.reshape(self.observation, [-1, 210 * 160])
            ft = tf.layers.dense(ft, 256, activation=tf.tanh, name='dense')
            ft = tf.layers.dense(ft, 256, activation=tf.tanh, name='dense2')
            p = tf.layers.dense(ft, self.config.action_size, name='dense3')

            dist = tf.distributions.Categorical(p)
            self.selected_actions = dist.sample()
            # self.log_scores = dist.log_prob(tf.reshape(self.action, [-1]))
            self.log_scores = get_loss(self.action, self.config.action_size, tf.nn.softmax(p))

            with tf.variable_scope('loss'):
                loss = tf.reduce_sum(-self.log_scores * tf.reshape(self.reward,[-1]))
                self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)


    def __get_tau(self, count, sess):
        """
        采集指定此时的游戏过程
        0-不动, 1-不动， 2-快向上，3-慢向下，4-慢向上，5-快向下
        一些特殊情况的处理：
        1. 每5帧采集一次（做一次动作）
        2. 结束条件是任意方得分超过21或者死亡
        :param count:
        :return: 采集的游戏过程，[[],[]]
        """
        # env = gym.make('Pong-v0')
        env = gym.make('Pong-v0').unwrapped
        env.seed(123456)
        tau = []
        for index in range(count):
            tau.append([])
            observation = env.reset()
            next_observation = observation
            flag=True
            score = 0
            step=0
            while(flag):
                # time.sleep(0.05)
                rate = np.reshape(np.array([0.2126,0.7152,0.0722], dtype=np.float32),[1,1,3])
                cache = [(rate*next_observation).sum(axis=-1,keepdims=True)-(rate*observation).sum(axis=-1,keepdims=True)]
                env.render()
                next_action = sess.run(self.selected_actions, feed_dict={self.observation: np.array([cache[0]], dtype=np.float32)})
                # print(next_action)
                # 并非选择概率最高的，按分布随机选择
                # next_action = np.random.choice([0, 1, 2, 3, 4, 5], 1, replace=True, p=next_action[0])[0]

                # next_action = next_action.argmax()
                observation = next_observation
                next_observation, reward, done, info = env.step(next_action[0])
                if reward>0:
                    score += reward
                cache = cache+[next_action, reward]
                tau[-1].append(cache)
                step += 1
                if done or score==-21 or score==21:
                    flag=False
        env.close()
        return tau

    def train(self):
        """
        训练，包含采集以及更新参数的过程
        :return:
        """
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            logging.info('开始训练，部分参数如下：')
            logging.info('episode:%d，batch_size:%d，lr:%f'%(self.config.episode, self.config.batch_size, self.config.lr))
            for step in range(self.config.episode):
                logging.info("episode:%d"%(step))
                # 采集游戏过程
                tau = self.__get_tau(1, sess)[0]
                logging.info("采集结果，时长：%d，总得分:%d，得分:%d" % (len(tau), sum([x[2] for x in tau]), sum([x[2] for x in tau if x[2]>0])))
                # 选择若干步进行更新参数
                for n in range(len(tau)//self.config.batch_size+1):
                    cache = tau[n*self.config.batch_size:(n+1)*self.config.batch_size]
                    if len(cache)==0:
                        continue
                    obervation = np.concatenate([np.expand_dims(x[0], axis=0) for x in cache], axis=0)
                    action = np.concatenate([np.expand_dims(x[1], axis=0) for x in cache], axis=0)
                    reward = np.concatenate([np.expand_dims([x[2]], axis=0) for x in cache], axis=0)
                    sess.run(self.train_op, feed_dict={self.observation:obervation, self.action:action, self.reward:reward})

class Config(object):
    def __init__(self):
        self.action_size = 6
        self.lr = 1e-5
        self.episode = 10000
        self.batch_size = 2048


if __name__=="__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config()
    oj=Actor(config)
    oj.train()
    # oj.get_tau(2)
