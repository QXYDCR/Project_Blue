from Utils import *



def update_critic(func, a, b):
    return func(a, b)


def update_actor(actions_pair, pred_actions, true_actions, H):
    L_L_ = (true_actions == pred_actions.squeeze()).sum().item() / 30
    R = H - 0.5 * L_L_
    selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
    loss_actor = selected_logprobs.mean()
    return loss_actor


def update_fc(func, a, b):
    return func(a, b)

import torch
x = torch.randn((1,4),dtype=torch.float32,requires_grad=True)
y = x ** 2
z = y * 4
output1 = z.mean()
output2 = z.sum()
output1.backward(retain_graph = True)    # 这个代码执行正常，但是执行完中间变量都free了，所以下一个出现了问题
output2.backward()    # 这时会引发错误



















