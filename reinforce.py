import argparse

parser = argparse.ArgumentParser(description='gym reinforce')

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--hidden_dim', type=int, default=4)
parser.add_argument('--render', default=False, action='store_true')

args = parser.parse_args()


import gym
env = gym.make('CartPole-v0').unwrapped
env.seed(args.seed)

print(env.observation_space.shape)
IN_DIM = env.observation_space.shape[0]
OUT_DIM = env.action_space.n


import tensorflow as tf
import numpy as np

tf.set_random_seed(args.seed)


def get_reward(reward):
    import math
    def r(reward):
        rt = []
        rate = np.array([math.pow(0.5, x) for x in range(len(reward) + 1)], dtype=np.float32)
        length = len(reward)
        for index, x in enumerate(reward):
            rt.append(np.sum(rate[:length - index] * (reward[index:])))
        return np.array(rt, dtype=np.float32)

    r = tf.py_func(r, [tf.reshape(reward, [-1])], [tf.float32])
    r = tf.reshape(r,[-1])
    return r

def get_loss(action, action_size, p):
    logp = tf.math.log(p+1e-8)
    tag = tf.one_hot(tf.reshape(tf.cast(action, dtype=tf.int32), [-1]), depth=action_size, axis=-1)
    logp = tf.reduce_sum(logp*tag, axis=-1)
    return logp

class Policy(object):
    def __init__(self, in_dim, out_dim, h_dim):
        with tf.variable_scope('init_variables'):
            self.state = tf.placeholder(
                tf.float32, [None, in_dim], name="state")
            self.rewards = tf.placeholder(
                tf.float32, [None], name="rewards")
            self.selected_actions = tf.placeholder(
                tf.float32, [None], name="actions")

        with tf.variable_scope('init_layers'):
            lr1 = tf.keras.layers.Dense(h_dim)
            lr2 = tf.keras.layers.Dense(out_dim)

        with tf.variable_scope('init_graph'):
            hidden = lr1(self.state)
            prop = lr2(hidden)
            props = tf.nn.softmax(prop)

            dist = tf.distributions.Categorical(props)
            self.action = dist.sample()

            # self.log_scores = dist.log_prob(self.selected_actions)
            # self.log_scores = -tf.nn.softmax_cross_entropy_with_logits_v2(
            #     labels=tf.one_hot(tf.cast(self.selected_actions, dtype=tf.int32), depth=2, dtype=tf.float32), logits=prop)
            self.log_scores = get_loss(self.selected_actions,out_dim,props)

        with tf.variable_scope('loss'):
            loss = tf.reduce_sum(-self.log_scores * self.rewards)
            self.train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)


from itertools import count

policy = Policy(IN_DIM, OUT_DIM, args.hidden_dim)


def get_state(data):
    # rate = np.reshape(np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), [1, 1, 3])
    # data = (rate * data).sum(axis=-1)
    # data = np.reshape(data,[-1])
    return data

def train():
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    more = 0

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        for epoch in count(1):
            state = env.reset()
            state = get_state(state)
            if args.render:
                env.render()
            states, policy_rewards, actions = [state], [], []

            for step in range(10000):
                # env.render()
                action = sess.run(
                    [policy.action], feed_dict={policy.state: [state]})[0][0]
                state, reward, done, _ = env.step(action)
                state = get_state(state)
                policy_rewards.append(reward if not done else -10)
                actions.append(action)
                if done:
                    # print('done')
                    break
                states.append(state)

            R, rewards = 0, []
            for r in policy_rewards[::-1]:
                R = r + args.gamma * R
                rewards.insert(0, R)

            rewards = np.asarray(rewards)
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)

            feed_dict = {
                policy.state: np.asarray(states),
                policy.rewards: rewards,
                policy.selected_actions: np.asarray(actions),
            }
            sess.run([policy.train_op], feed_dict)

            if more < step:
                print('Epoch {}\tlength: {:5d}\t'.format(epoch, step))
                more = step


if __name__ == '__main__':
    train()