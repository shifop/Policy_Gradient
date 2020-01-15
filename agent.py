import scipy.misc
import numpy as np
import gym
import time
from datetime import timedelta
from tools import get_logger
import tensorflow as tf

logger = get_logger("main")



def get_logp2(action, action_size, p):
    logp = tf.math.log(p+1e-8)
    tag = tf.one_hot(tf.reshape(action, [-1]), depth=action_size, axis=-1)
    logp = tf.reduce_sum(logp*tag, axis=-1)
    return logp

def get_logp(action, p):
    exp_p = tf.exp(p)
    logp = tf.math.log(tf.reduce_sum(exp_p, axis=-1))
    x_p = tf.reshape(tf.batch_gather(p, action), [-1])
    logp = -x_p+logp
    return logp


def prepro(o, image_size=[80, 80]):
    """
    turn rgb -> gray
    Input:
        np.array [210, 160, 3]
    Output:
        np.array [1, 1, 80, 80]
    https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    """
    o = o[35:, :, :]
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    new_img = resized.reshape(1, 1, 80, 80)
    return new_img


def discount_reward(r_dic, gamma):
    """
    change reward like below
    [0, 0, 0, 0, 1] -> [0.99^4, 0.99^3, 0.99^2, 0.99, 1]
    and then normalize
    """
    r = 0
    for i in range(len(r_dic) - 1, -1, -1):
        if r_dic[i] != 0:
            r = r_dic[i]
        else:
            r = r * gamma
            r_dic[i] = r
    r_dic = (r_dic - r_dic.mean()) / (r_dic.std() + 1e-8)
    return r_dic


class Agent():
    def __init__(self, args):
        print("Init Agent")
        self.config = args
        self.__create_graph()

    def _set_env(self, env):
        _ = env.reset()

        state, reward, done, info = env.step(0)
        state = prepro(state)
        pre_state = state

        state, reward, done, info = env.step(0)
        state = prepro(state)
        diff_state = state - pre_state

        return done, pre_state, diff_state

    def __create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80], name='observation')
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='action')
            self.action_p = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='action_p')
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')

            ft = tf.reshape(self.observation, [-1, 80*80])
            ft = tf.layers.dense(ft, 200, activation=tf.nn.relu, name='dense')
            self.p = tf.layers.dense(ft, 2, name='dense1')

            self.p_st = tf.nn.softmax(self.p)
            dist = tf.distributions.Categorical(tf.nn.softmax(self.p))
            self.selected_actions = dist.sample()
            # self.log_scores = dist.log_prob(tf.reshape(self.action, [-1]))
            tag = tf.one_hot(tf.reshape(self.action, [-1]), depth=2, axis=-1)
            self.log_scores = tf.nn.softmax_cross_entropy_with_logits(labels=tag, logits=self.p)
            # self.log_scores = get_logp2(self.action, 2, tf.nn.softmax(self.p))
            # self.log_scores = get_logp(self.action, self.p)


            with tf.variable_scope('loss'):
                # self.loss = tf.reduce_sum(self.log_scores * tf.reshape(self.reward,[-1]))
                self.loss = tf.reduce_sum(tf.multiply(self.log_scores, tf.reshape(self.reward,[-1])))
                self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

            self.saver = tf.train.Saver()

            for index, x in enumerate(tf.trainable_variables()):
                logger.info('%d:%s' % (index, x))

    def get_action(self, sess, state):
        action_p = sess.run(self.p_st, feed_dict={self.observation: state})

        if action_p[0][0]>np.random.uniform():
            action_int = 0
        else:
            action_int = 1
        return action_int, action_p[0]

    def __update(self, sess, diff_states_dic, act_dic, r_dic):
        loss, log_scores, _ = sess.run([self.loss, self.log_scores, self.train_op], feed_dict={
            self.observation: diff_states_dic,
            self.action: act_dic,
            self.reward: r_dic
        })
        return loss

    def train(self, args):
        print("start training")
        # for visualize
        step = 0
        reward_per_game = 0

        env = gym.make('Pong-v0')
        env.seed(11037)

        diff_state = None
        pre_state = None

        start_time = time.time()
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())

            # start train
            for e in range(args.epochs):
                # init training set for an epoch
                act_dic = []
                act_p_dic = []
                r_dic = []
                diff_states = []
                num_game = 0
                done, pre_state, diff_state = self._set_env(env)
                reward_r = 0
                while True:
                    if done:
                        num_game = num_game + 1
                        step = step + 1
                        reward_per_game = 0
                        # end of the epoch
                        if num_game == args.batch_size:
                            break
                        done, pre_state, diff_state = self._set_env(env)
                    else:
                        # play the game
                        act, act_p = self.get_action(sess, diff_state[0])  # 第一个是选择的操作，第二个是预测出的分布概率
                        state, reward, done, _ = env.step(act+2)
                        # env.render()
                        if reward > 0:
                            reward_r += reward

                        diff_states.append(diff_state[0][0])
                        act_dic.append(act)
                        act_p_dic.append(act_p)
                        r_dic.append(reward)

                        state = prepro(state)
                        diff_state = state - pre_state
                        pre_state = state

                        reward_per_game = reward_per_game + reward

                # dic -> numpy -> tensor
                diff_states_dic = np.stack(diff_states)
                act_p_dic = np.stack(act_p_dic)
                act_dic = np.array(act_dic)
                r_dic = np.array(r_dic, dtype=np.float32)

                # discount_reward
                r_dic = discount_reward(r_dic, args.gamma)

                """
                act_p_dic: 两种动作的得分
                act_dic ：采取的动作（sample）
                r_dic ：处理后的reward
                """
                # update model
                act_dic = np.reshape(act_dic, [-1, 1])
                r_dic = np.reshape(r_dic, [-1, 1])
                loss = self.__update(sess, diff_states_dic, act_dic, r_dic)
                time_dif = get_time_dif(start_time)
                msg = 'step: {0:>6}, loss: {1:>6.5}, total recward: {2:>6}, Time: {3}'
                logger.info(msg.format(step, loss, reward_r, time_dif))
                reward_r = 0
                # save model
                if (e % 100 == 0):
                    self.saver.save(sess=sess, save_path='./model/model.ckpt')
                    logger.info("save:%d" % e)

    def load(self):
        pass

    def test(self):
        pass


def save_img(path, img):
    img = img.reshape(80, 80)
    scipy.misc.imsave(path, img)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    print("development mode")