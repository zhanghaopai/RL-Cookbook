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
kl_constraint = 0.00005
env_name = "Pendulum-v0"
play_mode = "human"


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


class PPO:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, kl_constraint):
        '''
        PPO算法初始化
        :param state_dim:
        :param action_dim:
        :param learning_rate:价值网络学习率
        :param gamma: reward的衰减率
        :param kl_constraint: KL散度限制
        '''
        # 策略网络
        self.policy_net = PolicyNet(state_dim, action_dim)
        # 策略网络优化不需要优化器，
        # todo：为什么不需要优化器
        self.value_net = ValueNet(state_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        # reward折扣因子
        self.gamma = gamma
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

    def hessian_matrix_vector_product(self,
                                      states,
                                      old_action_dists,
                                      vector,
                                      damping=0.1):
        mu, std = self.policy_net(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))
        kl_grad = torch.autograd.grad(kl,
                                      self.policy_net.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x


    def update(self, transition_dict):
        '''
        根据trajectory，优化策略网络
        :param transition_dict:
        :return:
        '''
        # {s_1, a_1, r_1, ... , s_{t-1}, a_{t-1}, r_{t-1}}
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.policy_optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            # 获取reward
            reward = reward_list[i]
            # 获取state
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float)
            # 获取action
            action = torch.tensor([action_list[i]]).view(-1, 1)
            # 当前网络参数情况下，state状态算出来的action概率，即P(a|s),并将结果求对数
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            # 用累计奖励值G作为Q(s,a)的无偏估计
            G = self.gamma * G + reward
            # 每一步的损失函数
            loss = -log_prob * G
            # 反向传播计算梯度
            loss.backward()
        self.policy_optimizer.step()  # 梯度下降


def train(env, agent):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # trajectory的总回报
                episode_return = 0
                # trajectory的字典
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                # 初始状态
                state = state[0]
                done = False
                # 玩一局游戏，得到一个trajectory，直到游戏结束
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
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
    agent = PPO(state_dim, action_dim, learning_rate, gamma, kl_constraint)
    train(env, agent)


if __name__ == "__main__":
    main()
