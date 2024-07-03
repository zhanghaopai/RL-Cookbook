'''
策略梯度算法
'''
import copy

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

    def __init__(self, state_dim, action_dim, learning_rate, gamma, lam, kl_constraint, alpha):
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
        self.alpha = alpha  # 线性搜索超参数

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

    def gae(self, gamma, lam, td_delta):
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
        return torch.tensor(advantage_list, dtype=torch.float)

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

    def conjugate_gradients(self, Ax, b, cg_iter: 10):
        '''
        conjugate graient algorithm 共轭梯度法求解方程,计算x=h^{-1}g的值
        :param Ax:
        :param b:
        :return:
        '''
        x = np.zeros_like(b)  # 初始化x
        r = b.copy()  # 初始化梯度值
        p = r.copy()  # 初始化梯度更新方向
        rdotr = np.dot(r, r)  # r的转置乘r的结果
        for _ in range(cg_iter):
            z = Ax(p)
            alpha = rdotr / np.dot(p, z)  # 计算步长
            x += alpha * p  # 更新迭代值
            r -= alpha * z  # 更新梯度
            new_rdotr = np.dot(r, r)  # r'
            if new_rdotr < 1e-8:
                break
            beta = new_rdotr / rdotr  # 计算组合系数
            p = r + beta * p  # 计算共轭方向
            rdotr = new_rdotr  # 更新
        return x

    def line_search(self, states, actions, advantage, old_log_prob, old_action_dists, max_vec):
        '''
        线性搜索法的实现
        :param states: 状态
        :param actions: 动作
        :param advantage: 优势函数
        :param old_log_prob:
        :param old_action_dists:
        :param max_vec: 最大步长
        :return: 合适的策略网络参数
        '''
        old_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(self.policy_net.parameters())
        old_obj = self.surrogate_loss(old_log_prob, log_probs(self.policy_net, states, actions), advantage)

        for i in range(15):
            coef = self.alpha ** i
            new_parameters = old_parameters + coef * max_vec
            # 初始化一个新策略
            new_policy_net = copy.deepcopy(self.policy_net)
            # 将新参数更新到新策略中
            torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, new_policy_net.parameters())
            # 新策略的action输出
            new_actions_dist = torch.distributions.Categorical(new_policy_net())
            # 计算新旧kl散度
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_actions_dist))
            # 计算新obj
            new_obj = self.surrogate_loss(old_log_prob, log_probs(new_policy_net, states, actions), advantage)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                # 如何新的surrogate_loss比原来好，且在新旧分布差异在信任区间内，则返回新参数
                return new_parameters
        return old_parameters

    def update(self, trajectory):
        '''
        根据trajectory，优化策略和价值网络
        :param transition_dict:
        :return:
        '''
        state_list = torch.tensor(trajectory.state, dtype=torch.float)
        next_state_list = torch.tensor(trajectory.next_state, dtype=torch.float)
        action_list = torch.tensor(trajectory.action, dtype=torch.int64).view(1, -1)
        reward_list = torch.tensor(trajectory.reward, dtype=torch.float).view(1, -1)
        done_list = torch.tensor(trajectory.done, dtype=torch.float).view(1, -1)
        # 计算当前网络的Q值
        td_target = reward_list + self.gamma * self.value_net(next_state_list) * (1 - done_list)
        # 计算当前时序差分误差
        td_delta = td_target - self.value_net(state_list)
        # 计算优势函数
        advantage = self.gae(self.gamma, self.lam, td_delta)
        # 计算旧的策略网络的log
        old_log_probs = torch.log(self.policy_net(state_list).gather(1, action_list)).detach()
        # 计算旧的action分布
        old_action_dists = torch.distributions.Categorical(self.policy_net(state_list).detach())
        # 计算出旧的 value_net的loss函数
        value_loss = torch.mean(F.mse_loss(self.value_net(state_list), td_target.detach()))
        # 清空value_net梯度
        self.value_optimizer.zero_grad()
        # 反向传播
        value_loss.backward()
        # 梯度下降
        self.value_optimizer.step()

        # 策略优化
        # 计算当前策略目标
        surrogate_obj = self.surrogate_loss(old_log_probs, log_probs(self.policy_net, state_list, action_list), advantage)
        # 计算当前policy_net的梯度值
        grads = torch.autograd.grad(surrogate_obj, self.policy_net.parameters())
        # 把梯度值改为向量
        obj_grads = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度计算x
        descent_direction = self.conjugate_gradients(obj_grads, )
        # 计算出最大步长(平方根(2*δ/(xT*H*x)))
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        # 利用线性搜索法计算出新的策略网络的参数
        new_para = self.line_search(state_list, action_list, advantage, old_log_probs, old_action_dists,
                                    descent_direction * max_coef)
        # 把计算出的网络参数更新至actor网络中
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.policy_net.parameters())  # 用线性搜索后的参数更新策略



        print(surrogate_obj)


def log_probs(policy_net, states, actions):
    return torch.log(policy_net(states).gather(1, actions))


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
