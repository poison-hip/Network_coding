import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(
            self,
            time_module : int
    ):
        super().__init__()
        self.time_module = time_module
        self.linear_1 = nn.Linear(time_module // 4, time_module)
        self.linear_2 = nn.Linear(time_module, time_module)
        self.act_1 = nn.ReLU(time_module)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        d = self.time_module // 8
        freq = math.exp(-math.log(10000) / (d-1) * torch.arange(d, device=x.device))
        emd = d[:,None] * freq[None , :]
        torch.cat(math.sin(emd), math.cos(emd), dim=1)
        return self.linear_2(self.act_1(self.linear_1(emd)))
    

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        time_channel,
        norm_groups,
        dropout
    ):
        super().__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.time_channel = time_channel
        self.norm_groups = norm_groups
        self.dropout = dropout

        self.norm_1 = nn.GroupNorm(self.norm_groups, self.input_channel)
        self.conv_1 = nn.Conv2d(self.input_channel, self.out_channel,kernel_size=3,padding=1)
        self.act_1 = nn.SiLU(True)
        self.dropout_1 = nn.Dropout(self.dropout)

        self.time_act = nn.SiLU()
        self.time_emb_proj = nn.Linear(self.time_channel, self.out_channel)

        self.norm2 = nn.GroupNorm(self.norm_groups, self.out_channel)
        self.act_2 = nn.SiLU(True)
        self.dropout = nn.Dropout(True)
        self.conv_2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1)

        if self.input_channel != self.out_channel:
            self.shortcut = nn.Conv1d(self.input_channel, self.out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv(self.act_1(self.norm1(x)))
        h += self.time_emb_proj(self.time_act(t))[:,:,None,None]
        h += self.conv_2(self.act_2(self.norm2(h)))
        return self.shortcut(h)
    

class AttentionBlock(nn.Module):
    def __init__(
        self, 
        out_channel,
        num_head,
        d_k,
        num_group,
    ):
        super().__init__()
        self.out_channel = out_channel
        self.num_head = num_head
        self.num_group = num_group
        if d_k == None:
            assert out_channel % num_head == 0
            self.d_k = out_channel // num_head
        self.d_k = d_k

        self.norm_1 = nn.GroupNorm(self.num_group, self.out_channel)
        self.input_proj = nn.Linear(self.out_channel, self.d_k * self.num_head *3)
        self.output = nn.Linear(self.d_k * self.num_head, self.out_channel)

    def forward(self, x : torch.Tensor, t):
        b, c, h, w = x.shape()
        x_in = x
        seq_line = h * w
        x = self.norm_1(x)
        x = x.reshape(b, c, seq_line)

        x = x.permute(0, 2, 1)

        x = self.input_proj(x)

        qkv = x.reshape(b ,seq_line, self.num_head, self.d_k*3)
        q, k, v = qkv.chunk(qkv, 3, dim=-1)

        scale = self.d_k ** -0.5
        attention : torch.Tensor = torch.einsum("bind,bjhd->bijh", q, k) * scale

        attention_weight = attention.softmax(dim=2)
        attention_aggregation = torch.einsum("bind,bjhd->bijh", attention_weight, v)
        
        attention_aggregation = attention_aggregation.reshape(b, seq_line, -1)
        res : torch.Tensor = self.output(attention_aggregation)

        res = res.permute(0, 2, 1).reshape(b, c, h, w)

        return res+x_in


class DownBlock(nn.Module):
    def __init__(
            self, 
            input_channel,
            out_channel,
            time_channel,
            is_atten : bool=False
    ):
        super().__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.time_channle = time_channel

        self.resblock = ResidualBlock(self.input_channel, self.out_channel, self.time_channle, norm_groups=16)
        if is_atten:
            self.attention = AttentionBlock(self.out_channel,num_head=6,num_group=16)
        else:
            self.attention = nn.Identity()

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        x = self.resblock(x, t)
        x = self.attention(x)
        return x
    

class UPBlock(nn.Module):
    def __init__(
            self, 
            input_channel,
            out_channel,
            time_channel,
            is_atten : bool=False
    ):
        super().__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.time_channle = time_channel

        self.resblock = ResidualBlock(self.input_channel, self.out_channel, self.time_channle, norm_groups=16)
        if is_atten:
            self.attention = AttentionBlock(self.out_channel,num_head=6,num_group=16)
        else:
            self.attention = nn.Identity()

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        x = self.resblock(x, t)
        x = self.attention(x)
        return x
    

class DownSample(nn.Module):
    def __init__(
            self,
            out_channel
    ):
        super().__init__()
        self.out_channel = out_channel
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)
    

class UpSample(nn.Module):
    def __init__(
            self,
            out_channel
    ):
        super().__init__()
        self.out_channel = out_channel
        self.transpose_conv = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.transpose_conv(x)
    
class MiddleBlock(nn.Module):
    def __init__(
            self, 
            out_channel,
            time_channel
    ):
        super().__init__()
        self.out_channel = out_channel
        self.time_channel = time_channel

        self.res_1 = ResidualBlock(out_channel, out_channel, time_channel, norm_groups=16)
        self.res_2 = ResidualBlock(out_channel, out_channel, time_channel, norm_groups=16)
        self.atten = AttentionBlock(out_channel, num_head=6, num_group=16)

    def forward(self,x, t):
        x = self.res_1(x ,t)
        x = self.atten(x)
        x = self.res_2(x, t)
        return x
    

class DDPNModule(nn.Module):
    def __init__(
            self,
            img_channel,
            model_channel,
            channel_mutis : list[int] = [1,2,2,2],
            is_atten : list[bool] = [False, False, False, False],
            num_res_block : int = 2,
            num_groups : int = 32,
    ):
        super().__init__()
        self.image_channel = img_channel
        self.model_channel = model_channel
        self.channel_mutis = channel_mutis
        self.is_atten = is_atten
        self.num_res_block = num_res_block
        self.num_groups = num_groups
        self.times_channel = self.model_channel * 4

        self.img_proj = nn.Linear(self.image_channel, self.model_channel)
        self.time_embedding = TimeEmbedding(self.times_channel)

        self.downblock = nn.ModuleList()
        self.block_len = len(self.channel_mutis)
        self.cur_channel = model_channel
        self.down_channel = [self.cur_channel]
        
        for i in range (self.block_len):
            self.next_channel = self.cur_channel * self.channel_mutis[i]
            for _ in range (self.num_res_block):
                self.downblock.append(DownBlock(self.cur_channel, self.next_channel, self.times_channel, is_atten=self.is_atten[i]))
                self.cur_channel = self.next_channel

            self.down_channel.append(self.cur_channel)

            if i < self.block_len() - 1:
                self.downblock.append(DownSample(self.next_channel))


        self.upblock = nn.ModuleList()
        for i in range (self.block_len-1, -1 , -1):
            self.cur_channel = self.down_channel[i]
            if i < self.block_len - 1:
                self.upblock.append(UpSample(self.cur_channel))

            self.cur_channel = self.down_channel.pop()
            self.next_channel = self.down_channel.top()
            for _ in range(self.num_res_block):
                self.upblock.append(UPBlock(self.cur_channel, self.next_channel,self.times_channel, self.is_atten[i]))
                self.cur_channel = self.next_channel


        self.norm = nn.GroupNorm(self.num_groups)
        self.act = nn.SiLU(True)
        self.output = nn.Linear(model_channel, img_channel)


    def forward(self, x:torch.Tensor, t:torch.Tensor):
        x = self.img_proj(x)
        t = self.time_embedding(t)

        c = [x]
        for block in self.downblock:
            if isinstance(block, DownSample):
                x = block(x)
            else:
                x = block(x, t)
                c.append(x)

        for block in self.upblock:
            if isinstance(block, UpSample):
                x = block(x)
            else:
                x = torch.cat(c.pop(), x)
                x = block(x,t)

        x = self.norm(x)
        x = self.act(self.output(x))
        return x
                