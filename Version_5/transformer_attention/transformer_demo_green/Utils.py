import numpy as np
import torch
import torch.nn.functional as F
import data
import math

def mask_x(x, max_len = 11):
    # b句话,每句话11个词,这里是还没embed的
    # x = [b, 11]
    # 判断每个词是不是<PAD>
    # 在数字化文本进行掩码表示，在字典中为<PAD>的进行掩饰, result:[true,false, ....]
    mask = x == data.zidian['<PAD>']

    # [b, 11] -> [b, 1, 1, 11]
    mask = mask.reshape(-1, 1, 1, 11)

    # [b, 1, 1, 11] -> [b, 1, 11, 11]
    # 重复11个(1, 11)
    mask = mask.expand(-1, 1, 11, 11)

    return mask

# mask_y_fast 和 mask_y是一样的结果，只是写法不同
def mask_y_fast(y):
    """ torch.triu()
    [[0, 1, 1],
    [0, 0, 1 ],
    [0, 0, 0 ]]
    """
    # 上三角矩阵,[11, 11]
    triangle = torch.triu(torch.ones((11, 11), dtype=torch.long), diagonal=1)

    # 判断每个词是不是<PAD>, [b, 11]
    y_eq_pad = y == data.zidian['<PAD>']

    # torch.where()函数的作用是按照一定的规则合并两个tensor类型
    # 每个y每个词是否等于pad,组合全1的矩阵和triangle矩阵
    mask = torch.where(y_eq_pad.reshape(-1, 1, 11), torch.ones(1, 11, 11, dtype=torch.long),
                       triangle.reshape(1, 11, 11))

    return mask.bool().reshape(-1, 1, 11, 11)


def mask_y(y):
    # return mask_y_fast(y)
    # b句话,每句话11个词,这里是还没embed的
    # y = [b, 11]

    b = y.shape[0]

    # b句话,11*11的矩阵表示每个词对其他词是否可见
    mask = torch.zeros(b, 11, 11)

    # 遍历b句话
    for bi in range(b):
        # 遍历11个词
        for i in range(11):
            # 如果词是pad,则这个词完全不可见
            if y[bi, i] == data.zidian['<PAD>']:
                mask[bi, :, i] = 1
                continue

            # 这个词之前的词都可见,之后的词不可见
            col = [1] * i + [0] * 11
            col = col[:11]
            mask[bi, :, i] = torch.LongTensor(col)

    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)

    return mask


def attention(Q, K, V, mask):
    # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 11, 8]

    # [b, 4, 11, 8] * [b, 4, 8, 11] -> [b, 4, 11, 11]
    # Q,K矩阵相乘,结果除以根号头数,这里完全是公式
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(8)

    # mask遮盖,mask是true的地方都被替换成-inf
    # mask = [b, 1, 11, 11]
    score = score.masked_fill_(mask, -np.inf)
    score = F.softmax(score, dim=-1)

    # 这一步也是公式
    # [b, 4, 11, 11] * [b, 4, 11, 8] -> [b, 4, 11, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 11, 8] -> [b, 11, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 11, 32)

    return score



if __name__ == '__main__':
    print(mask_x(data.get_sample()[0][:1]).shape)
    y1 = mask_y(data.get_sample()[1][:, :-1])
    y2 = mask_y_fast(data.get_sample()[1][:, :-1])
    print(torch.all(y1 == y2))


