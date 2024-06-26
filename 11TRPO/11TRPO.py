'''
策略梯度算法
'''
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 参数区
learning_rate = 1e-3
num_episodes = 1000
gamma = 0.98
lam = 0.95
kl_constraint = 0.00005
env_name = "CartPole-v0"
play_mode = "rgb_array"


# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# 价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Trajectory:
    '''
    轨迹
    '''

    def __init__(self):
        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.done = []

    def push(self, state, action, reward, done, next_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.next_state.append(next_state)


class TRPO:
    '''
    TRPO 算法实现
    '''

    def __init__(self, state_dim, action_dim, learning_rate, gamma, lam, kl_constraint):
        '''
        TRPO算法初始化
        :param state_dim:
        :param action_dim:
        :param learning_rate:价值网络学习率
        :param gamma: reward的衰减率
        :param kl_constraint: KL散度限制
        '''
        # 策略网络
        self.policy_net = PolicyNet(state_dim, action_dim)
        # 策略网络优化不需要优化器
        self.value_net = ValueNet(state_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        # reward折扣因子
        self.gamma = gamma
        self.lam = lam
        self.kl_constraint = kl_constraint  # KL距离最大限制

    def take_action(self, state):  # 根据动作概率分布随机采样
        '''
        根据动作概率分布随机采样
        :param state:状态
        :return:
        '''
        state = torch.tensor([state], dtype=torch.float)
        # 获得当前状态下的softmax概率分布
        probs = self.policy_net(state)
        # 归一化处理，得到动作的概率分布
        action_dist = torch.distributions.Categorical(probs)
        # 抽样
        action = action_dist.sample()
        return action.item()

    def surrogate_loss(self, old_log_probs, log_probs, advantage):
        '''
        代理损失函数：一般是指当目标函数非凸、不连续时，数学性质不好，优化起来比较复杂，这时候需要使用其他的性能较好的函数进行替换。
        :param old_log_probs: 原概率分布的对数
        :param log_probs: 当前网络的概率分布的对数
        :param advantage: 优势函数
        :return:
        '''
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def gae(self, gamma, lam, td_delta):
        '''
        广义优势估计
        :return:
        '''
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in reversed(td_delta):
            advantage = gamma * lam * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def cg(self, Ax, b, cg_iter: 10):
        '''
        conjugate graient algorithm 共轭梯度法求解方程
        :param Ax:
        :param b:
        :return:
        '''
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rdotr = np.dot(r, r)
        for _ in range(cg_iter):
            z = Ax(p)
            alpha = rdotr / np.dot(p, z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = np.dot(r, r)
            if new_rdotr < 1e-8:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def update(self, trajectory):
        '''
        根据trajectory，优化策略和价值网络
        :param transition_dict:
        :return:
        '''
        state_list = trajectory.state
        action_list = trajectory.action
        reward_list = trajectory.reward
        next_state_list = trajectory.next_state
        done_list = trajectory.done

        td_target = reward_list + self.gamma * self.value_net(next_state_list) * (1 - done_list)
        td_delta = td_target - self.value_net(state_list)
        advantage = self.gae(self.gamma, self.lam, td_delta)
        action_dists = self.policy_net(state_list)

        old_log_prob = action_dists.log_prob(action_list)
        value_loss =torch.mean(F.mse_loss(action_dists, td_target.detach()))
        # 清空value_net梯度
        self.value_optimizer.zero_grad()
        # 反向传播
        value_loss.backward()
        # 梯度下降
        self.value_optimizer.step()


        # 策略优化
        surrogate = self.surrogate_loss(old_log_prob,
                                        self.policy_net(state_list).log_prob(action_list),
                                        advantage)
        print(surrogate)


def train(env, agent):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # trajectory的总回报
                episode_return = 0
                # trajectory的字典
                trajectory = Trajectory()
                state = env.reset()
                # 初始状态
                state = state[0]
                done = False
                # 玩一局游戏，得到一个trajectory，直到游戏结束
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    # 记录动作
                    trajectory.push(state, action, reward, done, next_state)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(trajectory)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)


def init_env(env_name, mode):
    '''
    初始化游戏环境
    :param env_name: 游戏名称
    :return:
    '''
    env = gym.make(env_name, render_mode=mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim


def main():
    env, state_dim, action_dim = init_env(env_name, play_mode)
    agent = TRPO(state_dim, action_dim, learning_rate, gamma, lam, kl_constraint)
    train(env, agent)


if __name__ == "__main__":
    main()
