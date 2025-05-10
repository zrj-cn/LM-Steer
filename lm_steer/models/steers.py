import torch
import torch.nn as nn


class Projected_Adaptor(nn.Module):
    def __init__(self, lm_head, adaptor_class, num_steers, embed_dim,
                 vocab_size, rank, epsilon, init_var, position="output"):
        super().__init__()
        assert rank > 0
        if adaptor_class == "multiply":
            self.projector1 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
            self.projector2 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
        elif adaptor_class == "add":
            self.add_vec = nn.Parameter(torch.randn(
                num_steers, embed_dim
            ))
        elif adaptor_class == "offset":
            self.offset_vec = nn.Parameter(torch.randn(
                num_steers, vocab_size
            ))
        else:
            raise NotImplementedError()

        self.adaptor_class = adaptor_class
        self.rank = rank
        self.lm_head = lm_head
        self.epsilon = epsilon
        self.position = position
        self.num_steers = num_steers
        self.init_var = init_var
        self.steer_values = torch.zeros(num_steers)

    def set_value(self, steer_values):
        self.steer_values = steer_values

    def forward(self, state):
        if self.steer_values.abs().sum() == 0:
            return state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        if self.adaptor_class == "multiply":
            # 这是为了确保后续的矩阵乘法操作能够正确进行
            if self.position == "input":
                state = self.lm_head(state)
            # 获取 state 的批次大小
            batch_size = state.shape[0]

            # 将 self.steer_values 赋值给局部变量 steer_values
            steer_values = self.steer_values 
            print("steer_values.shape:", steer_values.shape)  # 打印 steer_values 的形状

            
            # 检查 steer_values 是否是一维的
            if steer_values.dim() == 1:
                # 使其形状变为 [batch_size, num_steers]
                steer_values = steer_values.unsqueeze(0).expand(batch_size, -1)

            print("steer_values.shape:", steer_values.shape)  # 打印 steer_values 的形状
            # 检查 steer_values 的形状是否与预期相符
            if steer_values.shape != (batch_size, self.num_steers):
                raise ValueError(f"steer_values should have shape [batch_size, num_steers], "
                                 f"but got {steer_values.shape}")
            # 计算 delta 的第一部分：
            print("state[:, None].shape:", state[:, None].shape)  # 打印 state[:, None] 的形状
            print("self.projector1[None].shape:", self.projector1[None].shape)  # 打印 self.projector1[None] 的形状
            # print("delta.shape:", delta.shape)  # 打印 delta 的形状
            print("self.steer_values[:, :, None, None].shape:", self.steer_values[:, :, None, None].shape)  # 打印 self.steer_values[:, :, None, None] 的形状
            delta = state[:, None].matmul(self.projector1[None]) *\
                self.steer_values[:, :, None, None] 
            
            delta = delta.matmul(
                self.projector2.transpose(1, 2)[None]).sum(1)

            # projected_state 的计算：将原始 state 加上经过 epsilon 缩放的 delta
            projected_state = state + self.epsilon * delta
            
            # 计算最终的 logits：将调整后的 projected_state 与语言模型的头部权重进行矩阵乘法
            logits = projected_state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        elif self.adaptor_class == "add":
            add_values = self.steer_values.matmul(self.add_vec)
            projected_state = state + self.epsilon * add_values[:, None]
            logits = projected_state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        elif self.adaptor_class == "offset":
            offset_values = self.steer_values.matmul(self.offset_vec)
            logits = state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
            logits = logits + self.epsilon * offset_values[:, None]
        return logits

    def regularization_term(self):
        if self.adaptor_class == "multiply":
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()
        elif self.adaptor_class == "add":
            return self.add_vec.pow(2).sum()
        elif self.adaptor_class == "offset":
            return self.offset_vec.pow(2).sum()

    def parameters(self):
        if self.adaptor_class == "multiply":
            return [self.projector1, self.projector2]
        elif self.adaptor_class == "add":
            return [self.add_vec]
        elif self.adaptor_class == "offset":
            return [self.offset_vec]

    def state_dict(self):
        if self.adaptor_class == "multiply":
            return {"projector1": self.projector1,
                    "projector2": self.projector2}
        elif self.adaptor_class == "add":
            return {"add_vec": self.add_vec}
        elif self.adaptor_class == "offset":
            return {"offset_vec": self.offset_vec}

    def load_state_dict(self, state_dict):
        if self.adaptor_class == "multiply":
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
        elif self.adaptor_class == "add":
            self.add_vec.data = state_dict["add_vec"]
        elif self.adaptor_class == "offset":
            self.offset_vec.data = state_dict["offset_vec"]
