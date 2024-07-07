'''
PPO
'''
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt



# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=0)


# 价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
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


class PPO:
    def __init__(self, state_dim, action_dim, plr, vlr, gamma, lam, eps):
        '''
        PPO算法初始化
        :param state_dim:
        :param action_dim:
        :param plr: 策略网络学习率
        :param vlr: 价值网络学习率
        :param gamma: reward的衰减率
        :param lam: 广义优势估计的加权因子
        :param eps: clip范围，[1-eps, 1+eps]
        '''
        # 策略网络
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=plr)
        # 价值网络
        self.value_net = ValueNet(state_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=vlr)
        # reward折扣因子
        self.gamma = gamma
        # 广义优势估计-加权因子
        self.lam = lam
        # 截断范围
        self.eps = eps

    def take_action(self, state):
        '''
        根据动作概率分布随机采样
        :param state:状态
        :return:
        '''
        # state转换为tensor
        state = torch.tensor(state, dtype=torch.float)
        # 获得当前状态下的概率分布
        probs = self.policy_net(state)
        # 归一化处理，得到动作的类别分布
        action_dist = torch.distributions.Categorical(probs)
        # 抽样
        action = action_dist.sample()
        return action.item()

    def update(self, trajectory):
        '''
        根据trajectory，优化策略网络
        :param transition_dict:
        :return:
        '''
        # {s_1, a_1, r_1, ... , s_{t-1}, a_{t-1}, r_{t-1}}
        state_list = torch.tensor(np.array(trajectory.state), dtype=torch.float)
        action_list = torch.tensor(trajectory.action, dtype=torch.int64).view(-1, 1)
        reward_list = torch.tensor(trajectory.reward, dtype=torch.float).view(-1, 1)
        next_state_list = torch.tensor(np.array(trajectory.next_state), dtype=torch.float)
        done_list = torch.tensor(trajectory.done, dtype=torch.float).view(-1, 1)

        td_target = reward_list + self.gamma * self.value_net(next_state_list) * (1 - done_list)
        td_delta = td_target - self.value_net(state_list)
        advantage_list = gae(self.gamma, self.lam, td_delta)
        old_log_prob = torch.log(self.policy_net(state_list).gather(1, action_list)).detach()

        for _ in range(10):
            # 计算新的log概率分布
            log_prob = torch.log(self.policy_net(state_list).gather(1, action_list))
            ratio = torch.exp(log_prob - old_log_prob)
            # 估计1
            surr1 = ratio * advantage_list
            # 估计2
            surr2 = torch.clamp(ratio, 1-eps, 1+eps)*advantage_list
            policy_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(self.value_net(state_list), td_target.detach()))
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.requires_grad_(True)
            policy_loss.backward()
            value_loss.requires_grad_(True)
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()




def train(env, agent):
    return_list = []
    for i in range(epochs):
        episode = int(num_episodes / epochs)
        with tqdm(total=episode, desc='Iteration %d' % int(i+1)) as pbar:
            for i_episode in range(episode):
                # trajectory的字典
                trajectory = Trajectory()
                state = env.reset()
                done = False
                # 玩一局游戏，得到一个trajectory，直到游戏结束
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # 记录动作
                    trajectory.push(state, action, reward, done, next_state)
                    state = next_state
                # 将本局游戏的总reward放入return_list
                return_list.append(np.sum(trajectory.reward))
                agent.update(trajectory)
                if (i_episode + 1) % 10 == 0:
                    # 每玩10局打印
                    pbar.set_postfix({
                        'episode':
                            '%d' % (episode * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    pbar.update(10)
    return return_list

def draw(return_list):
    smooth_list=[]
    for i in range(0, len(return_list), 10):
        smooth_list.append(np.mean(return_list[i:i+10]))

    episodes_list = list(range(len(smooth_list)))
    plt.plot(episodes_list, smooth_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()


def gae(gamma, lam, td_delta):
    '''
    广义优势估计 advantage
    :param gamma: 折扣因子
    :param lam: 加权因子
    :param td_delta: 时序差分误差
    :return:
    '''
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in reversed(td_delta):
        advantage = gamma * lam * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def init_env(env_name, mode):
    '''
    初始化游戏环境
    :param env_name: 游戏名称
    :return:
    '''
    env = gym.make(env_name, render_mode=mode)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim


# 参数区
plr = 5e-5
vlr = 5e-4
gamma = 0.96
lam = 0.95
eps = 0.2

num_episodes = 10000
epochs = 200

env_name = "CartPole-v0"
play_mode = "rgb_array"


def main():
    env, state_dim, action_dim = init_env(env_name, play_mode)
    agent = PPO(state_dim, action_dim, plr, vlr, gamma, lam, eps)
    return_list = train(env, agent)
    draw(return_list)


if __name__ == "__main__":
    main()
