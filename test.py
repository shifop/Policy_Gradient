import gym
import tensorflow as tf
import numpy as np
import math
import logging
import os
logging.basicConfig(level=logging.INFO)

"""
流程：
1. 采集游戏过程
2. 优化参数
3. 再次采集
"""

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
            self.reward = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='reward')
            # 计算概率
            ft = tf.layers.conv2d(self.observation, 4, [5, 5], activation=tf.nn.relu, padding='SAME', name='conv1')
            ft = tf.layers.max_pooling2d(ft, [2,2], 2, padding='SAME')
            ft = tf.layers.conv2d(ft, 1, [5, 5], activation=tf.nn.relu, padding='SAME', name='conv2')
            ft = tf.layers.max_pooling2d(ft, [2, 2], 2, padding='SAME')
            ft = tf.reshape(ft, [-1, 53*40])
            ft = tf.layers.dense(ft, 300, activation=tf.tanh, name='dense')

            p = tf.layers.dense(ft, self.config.action_size, name='dense1')

            # 计算logp
            p = tf.nn.softmax(p, axis=-1)
            logp = tf.math.log(p+1e-8)
            #logp = p
            tag = tf.one_hot(tf.reshape(self.action, [-1]), depth=self.config.action_size, axis=-1)
            self.logp = logp
            logp = tf.reduce_sum(logp*tag, axis=-1)

            self.logp_1 = logp
            # 计算用于优化参数的reward
            def r(reward):
                """

                :param reward: [-1]
                :param logp: [-1]
                :param action: [-1]
                :return:
                """
                rt = []
                rate = np.array([math.pow(0.9, x) for x in range(len(reward)+1)], dtype=np.float32)
                length = len(reward)
                for index, x in enumerate(reward):
                    rt.append(np.sum(rate[:length-index]*(reward[index:]-0.1)))
                return np.array(rt, dtype=np.float32)

            r = tf.py_func(r, [tf.reshape(self.reward, [-1])], [tf.float32])
            r = tf.reshape(r, [-1])
            r_hat = r*logp
            self.r_hat = tf.reshape(tf.reduce_sum(r_hat), [-1,1])
            # self.opt = self.__get_gradients(-r_hat)
            self.opt = tf.train.GradientDescentOptimizer(self.config.lr).minimize(-self.r_hat)
            self.p = p
            self.r = r
            # 计算梯度

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
        env = gym.make('Pong-v0')
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
                rate.sum()
                cache = [(rate*next_observation).sum(axis=-1,keepdims=True)-(rate*observation).sum(axis=-1,keepdims=True)]
                env.render()
                next_action = sess.run(self.p, feed_dict={self.observation: np.array([cache[0]], dtype=np.float32)})
                # print(next_action)
                # 并非选择概率最高的，按分布随机选择
                next_action = np.random.choice([0, 1, 2, 3, 4, 5], 1, replace=True, p=next_action[0])[0]
                observation = next_observation
                next_observation, reward, done, info = env.step(next_action)
                score += reward
                cache = cache+[next_action, reward]
                tau[-1].append(cache)
                step += 1
                if done or score==2 or score==-2:
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
                logging.info("采集结果，时长：%d，总得分:%d" % (len(tau), sum([x[2] for x in tau])))
                # 选择若干步进行更新参数
                for n in range(len(tau)//self.config.batch_size+1):
                    cache = tau[n*self.config.batch_size:(n+1)*self.config.batch_size]
                    if len(cache)==0:
                        continue
                    obervation = np.concatenate([np.expand_dims(x[0], axis=0) for x in cache], axis=0)
                    action = np.concatenate([np.expand_dims([x[1]], axis=0) for x in cache], axis=0)
                    reward = np.concatenate([np.expand_dims([x[2]], axis=0) for x in cache], axis=0)
                    r_hat, _, r, logp = sess.run([self.r_hat, self.opt, self.p, self.logp], feed_dict={self.observation:obervation, self.action:action, self.reward:reward})
                    #print(r)

class Config(object):
    def __init__(self):
        self.action_size = 6
        self.lr = 1e-3
        self.episode = 10000
        self.batch_size = 150


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config()
    oj=Actor(config)
    oj.train()
    # oj.get_tau(2)
