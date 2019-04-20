# coding:utf-8
import os
import tensorflow as tf
import gym, time, threading
import numpy as np
from mpdrl.envs import MpdrlEnv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ENV = 'mpdrl-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]   
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NONE_STATE = np.zeros(NUM_STATES)

MIN_BATCH = 2048
BUFFER_SIZE = MIN_BATCH * 10
MAX_STEPS = 500
EPOCH = 3
EPSILON = 0.2
LOSS_V = 0.2
LOSS_ENTROPY = 1e-3
LEARNING_RATE = 1e-3

GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** (N_STEP_RETURN)
LAMBDA = 0.95
NUM_HIDDENS = [512, 512, 512]

N_WORKERS = 8

TARGET_SCORE = 7.0
GLOBAL_EP = 0 
MODEL_SAVE_INTERVAL = 1000
MODEL_DIR = './models/'
SUMMARY_DIR = './logs/'
NN_MODEL = os.path.join(os.path.dirname(__file__),'models/ppo_model.ckpt')

def build_summaries():
    with tf.variable_scope("summaries"):
        reward = tf.Variable(0.,name="reward")
        tf.summary.scalar("Reward",reward)
        entropy = tf.Variable(0.,name="entropy")
        tf.summary.scalar("Entropy",entropy)
        learning_rate = tf.Variable(0.,name="learning_rate")
        tf.summary.scalar("Learning_Rate",learning_rate)
        policy_loss = tf.Variable(0.,name="policy_loss")
        tf.summary.scalar("Policy_Loss",policy_loss)
        value_loss = tf.Variable(0.,name="value_loss")
        tf.summary.scalar("Value_Loss",value_loss)
        value_estimate = tf.Variable(0.,name="value_estimate")
        tf.summary.scalar("Value_Estimate",value_estimate)

    summary_vars = [reward,entropy,learning_rate,policy_loss,value_loss,value_estimate]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class PPONet(object):
    def __init__(self,sess):
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False,name="gloabal_step")
        self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 1000, 0.98, staircase=True,name="learning_rate")

        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES),name="s_t")
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS),name="a_t")
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1),name="r_t") 

        self.alpha, self.beta, self.v, self.params = self._build_net('pi',trainable=True)
        self.old_alpha, self.old_beta, _, old_params = self._build_net('old_pi',trainable =False)
        self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        with tf.variable_scope("assign"):
            self.assign_op = [old_params[i].assign(self.params[i]) for i in range(len(self.params))]
        self.graph = self.build_graph()

    def _build_net(self,name,trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s_t, NUM_HIDDENS[0],tf.nn.relu,trainable=trainable,name="l1")
            l2 = tf.layers.dense(l1, NUM_HIDDENS[1],tf.nn.relu,trainable=trainable,name="l2")
            l3 = tf.layers.dense(l2, NUM_HIDDENS[2],tf.nn.relu,trainable=trainable,name="l3")
            alpha = tf.layers.dense(l3,NUM_ACTIONS,tf.nn.softplus,trainable=trainable,name="alpha")
            beta = tf.layers.dense(l3,NUM_ACTIONS,tf.nn.softplus,trainable=trainable,name="beta")
            value = tf.layers.dense(l3,1,trainable=trainable,name="value")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)

        return alpha, beta, value, params

    def build_graph(self):
        with tf.variable_scope("beta_dist"):
            x = (self.a_t - A_BOUNDS[0]) / (-A_BOUNDS[0] + A_BOUNDS[1])
            beta_dist = tf.contrib.distributions.Beta(self.alpha + 1, self.beta + 1)
            self.prob = beta_dist.prob( x ) + 1e-8

            beta_dist_old = tf.contrib.distributions.Beta(self.old_alpha + 1, self.old_beta + 1)
            self.prob_old = tf.stop_gradient(beta_dist_old.prob( x ) + 1e-5)
            self.A = beta_dist.sample(1) * (-A_BOUNDS[0] + A_BOUNDS[1]) + A_BOUNDS[0]
        with tf.variable_scope("advantage"):
            self.advantage = self.r_t - self.v
        with tf.variable_scope("clip_loss"):
            r_theta = self.prob / self.prob_old
            loss_CPI = r_theta * tf.stop_gradient(self.advantage)
            self.r_clip = tf.clip_by_value(r_theta, 1.0-EPSILON, 1.0+EPSILON)
            clipped_loss_CPI = self.r_clip * tf.stop_gradient(self.advantage)
            self.loss_CLIP = -tf.reduce_mean(tf.minimum(loss_CPI, clipped_loss_CPI))
        with tf.variable_scope("loss_value"):
            self.loss_value = tf.reduce_mean(tf.square(self.advantage))

        with tf.variable_scope("loss_entropy"):
            self.entropy = tf.reduce_mean(beta_dist.entropy())
        
        with tf.variable_scope("loss_total"):
            self.loss_total = self.loss_CLIP + LOSS_V * self.loss_value - LOSS_ENTROPY * self.entropy

        minimize = self.opt.minimize(self.loss_total, global_step=self.global_step)
        return minimize

    def update_parameter_server(self):
        if len(self.train_queue[0]) < BUFFER_SIZE:
            return
        queue = self.train_queue
        self.train_queue = [[], [], [], [], []]
        Buffer = np.array(queue).T
        [self.sess.run(self.assign_op[i]) for i in range(len(self.assign_op))]
        for i in range(EPOCH):
            print("EPOCH:" + str(i+1))
            n_batches = int(BUFFER_SIZE / MIN_BATCH)
            batch = np.random.permutation(Buffer)
            for n in range(n_batches):
                s, a, r, s_, s_mask = np.array(batch[n * MIN_BATCH: (n + 1) * MIN_BATCH]).T
                s = np.vstack(s)
                a = np.vstack(a)
                r = np.vstack(r)
                s_ = np.vstack(s_)
                s_mask = np.vstack(s_mask)

                v = self.sess.run(self.v, feed_dict={self.s_t:s_})

                r = r + LAMBDA * GAMMA_N * v * s_mask

                feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}
                minimize = self.graph
                self.sess.run(minimize, feed_dict)

        summary_str = self.sess.run(summary_ops, feed_dict = {
            summary_vars[0]: r.mean(),
            summary_vars[1]: self.sess.run(self.entropy, feed_dict={self.s_t: s}),
            summary_vars[2]: self.sess.run(self.learning_rate),
            summary_vars[3]: self.sess.run(self.loss_CLIP, feed_dict),
            summary_vars[4]: self.sess.run(self.loss_value, feed_dict={self.s_t: s, self.r_t: r}),
            summary_vars[5]: self.sess.run(self.v, feed_dict={self.s_t: s}).mean()
        })
        writer.add_summary(summary_str,GLOBAL_EP)
        writer.flush()

    def predict_a(self, s):
        a = self.sess.run(self.A, feed_dict={self.s_t: s})
        return a

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

class Agent:
    def __init__(self, brain):
        self.brain = brain
        self.memory = []
        self.R = 0

    def act(self, s):
        s = np.array([s])
        a = self.brain.predict_a(s).reshape(-1)
        return a

    def advantage_push_brain(self, s, a, r, s_):
        def get_sample(memory):
            s, a, _, _ = memory[0]
            return s, a, self.R

        self.memory.append((s, a, r, s_))

        self.R = (self.R + LAMBDA * r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                self.R = self.R - LAMBDA * self.memory[0][2] + self.memory[0][2]
                s, a, r = get_sample(self.memory)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            self.R = self.R - LAMBDA * self.memory[0][2] + self.memory[0][2]
            s, a, r = get_sample(self.memory)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

class Environment:
    total_reward_vec = np.array([0 for i in range(20)])

    def __init__(self, name, brain):
        self.name = name
        self.env = gym.make(ENV)
        self.agent = Agent(brain)

    def run(self):
        global frames
        global isLearned
        global GLOBAL_EP

        s = self.env.reset().reshape(-1)
        R = 0
        step = 0
        while True:
            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)
            s_ = s_.reshape(-1)
            a = a.reshape(-1)

            step += 1
            frames += 1

            if step > MAX_STEPS or done:
                s_ = None

            self.agent.advantage_push_brain(s, a, r, s_)
            s = s_
            R += r
            if len(self.agent.brain.train_queue[0]) >= BUFFER_SIZE:
                if not isLearned:
                    self.agent.brain.update_parameter_server()
            if step > MAX_STEPS or done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], R))
                break
 
        print(
            self.name,
            "| EP: %d" % GLOBAL_EP,
            "| reword: %f" % R,
            "| step: %d" % step,
            "| running_reward: %f" % self.total_reward_vec.mean(),
            )
        GLOBAL_EP += 1
        if GLOBAL_EP % MODEL_SAVE_INTERVAL == 0:
            saver.save(SESS,MODEL_DIR + "/ppo_model.ckpt") 
        if self.total_reward_vec.mean() > TARGET_SCORE:
        #if GLOBAL_EP > 60000:
            isLearned = True
            time.sleep(2.0)

class Worker_thread:
    def __init__(self, thread_name, brain):
        self.environment = Environment(thread_name, brain)

    def run(self):
        while True:
            self.environment.run()
            if isLearned:
                break

if __name__ == "__main__":
    frames = GLOBAL_EP * MAX_STEPS
    isLearned = False

    with tf.Session() as SESS:
        brain = PPONet(SESS)
        workers = []
        for i in range(N_WORKERS):
            worker_name = 'W_%i' % i 
            workers.append(Worker_thread(thread_name=worker_name, brain=brain))


        summary_ops, summary_vars = build_summaries()
        COORD = tf.train.Coordinator()
        SESS.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR,SESS.graph)
        saver = tf.train.Saver()

        nn_model = NN_MODEL
        if os.path.exists(nn_model):
            saver.restore(SESS,nn_model)
            print("Model restored!!")

        worker_threads = []
        for worker in workers:
            job = lambda: worker.run()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
