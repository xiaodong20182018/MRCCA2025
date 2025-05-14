import torch
import numpy as np, torch.nn as nn, torch.nn.functional as F
from torch import softmax, relu, sigmoid
from torch_geometric.nn import GCNConv
from utils import cal_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

class Cross_attention(nn.Module):
    def __init__(self, n_segment, d_model, d_k, **kwargs):
        super().__init__()
        self.d_k=d_k
        self.n_segment = n_segment
        self.att_cross = CrissCrossAttention(1)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_segment * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(kwargs['dr_hi'])


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Linear(64, 64 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64 // 16, 64, bias=False),
            nn.Sigmoid()
        )
        # self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32, 2), stride=1)
    def forward(self, enc_input, X1, X2, X3):
        d_k, n_segment = self.d_k, self.n_segment
        sz_b, len, _ = enc_input.size()
        output = self.att_cross(X1, X2, X3)
        # output = self.att_cross(output, output, output)

        # X1_weights = sigmoid(self.conv(X1))
        # X2_weights = sigmoid(self.conv(X2))
        # X3_weights = sigmoid(self.conv(X3))
        # X1_W = X1_weights / (X1_weights + X2_weights + X3_weights)
        # X2_W = X2_weights / (X1_weights + X2_weights + X3_weights)
        # X3_W = X3_weights / (X1_weights + X2_weights + X3_weights)
        # X123 = X1 * X1_W + X2 * X2_W + X3 * X3_W
        #
        # output = output + X123

        output = output.reshape(sz_b, n_segment, len, d_k)

        # #增加软收缩去噪
        # y = self.global_avg_pool(output).view(sz_b, n_segment)  # 形状变为 (batch_size, channels)
        # # 通过全连接层生成通道注意力权重
        # y = self.avg_fc(y).view(sz_b, n_segment, 1, 1)   # 形状变为 (batch_size, channels)
        # output = torch.sign(output) * torch.maximum(torch.abs(output) - torch.abs(y), torch.zeros_like(output))


        output = output.transpose(1, 2).contiguous().view(sz_b, len, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + enc_input)
        return output
class MR_MLP(nn.Module):
    def __init__(self, n_segment, d_model, d_k, **kwargs):
        super().__init__()
        self.n_segment = n_segment
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_segment * d_k)
        self.w_ks = nn.Linear(d_model, n_segment * d_k)
        self.w_vs = nn.Linear(d_model, n_segment * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

    def forward(self, enc_input):
        d_k, n_segment = self.d_k, self.n_segment

        sz_b, len, _ = enc_input.size()

        X1 = self.w_qs(enc_input).view(sz_b, len, n_segment, d_k)
        X2 = self.w_ks(enc_input).view(sz_b, len, n_segment, d_k)
        X3 = self.w_vs(enc_input).view(sz_b, len, n_segment, d_k)

        X1 = X1.transpose(1, 2).contiguous().view(-1, 1, 32, 2)  # (n*b) x lq x dk
        X2 = X2.transpose(1, 2).contiguous().view(-1, 1, 32, 2)  # (n*b) x lk x dk
        X3 = X3.transpose(1, 2).contiguous().view(-1, 1, 32, 2)  # (n*b) x lv x dv

        return X1, X2, X3

class MR_CNN(nn.Module):
    def __init__(self, n_segment, d_model, d_k, **kwargs):
        super().__init__()
        self.n_segment = n_segment
        self.d_k = d_k
        self.conv_wq = nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=10)
        nn.init.normal_(self.conv_wq.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.conv_wk = nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=10)
        nn.init.normal_(self.conv_wk.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.conv_wv = nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=10)
        nn.init.normal_(self.conv_wv.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Linear(64, 64 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64 // 16, 64, bias=False),
            nn.Sigmoid()
        )
    def forward(self, enc_input):
        d_k, n_segment = self.d_k, self.n_segment
        sz_b, len, _ = enc_input.size()

        #wq替换conv
        X1h = enc_input[:, :1, :].view(sz_b, 1, 10, 10)
        X1h = self.conv_wq(X1h).view(sz_b, 1, -1)  #(sz_b, 32, 8, 8)
        X1r = enc_input[:, 1:, :].view(sz_b, 1, 10, 10)
        X1r = self.conv_wq(X1r).view(sz_b, 1, -1)
        conv_X1 = torch.cat((X1h,X1r),dim=1).view(sz_b, len, n_segment, d_k)
        X2h = enc_input[:, :1, :].view(sz_b, 1, 10, 10)
        X2h = self.conv_wk(X2h).view(sz_b, 1, -1)
        X2r = enc_input[:, 1:, :].view(sz_b, 1, 10, 10)
        X2r = self.conv_wk(X2r).view(sz_b, 1, -1)
        conv_X2 = torch.cat((X2h, X2r), dim=1).view(sz_b, len, n_segment, d_k)
        X3h = enc_input[:, :1, :].view(sz_b, 1, 10, 10)
        X3h = self.conv_wv(X3h).view(sz_b, 1, -1)
        X3r = enc_input[:, 1:, :].view(sz_b, 1, 10, 10)
        X3r = self.conv_wv(X3r).view(sz_b, 1, -1)
        conv_X3 = torch.cat((X3h, X3r), dim=1).view(sz_b, len, n_segment, d_k)

        conv_X1 = conv_X1.transpose(1, 2).contiguous()
        #增加软收缩去噪
        y1 = self.global_avg_pool(conv_X1).view(sz_b, n_segment)  # 形状变为 (batch_size, channels)
        # 通过全连接层生成通道注意力权重
        y1 = self.avg_fc(y1).view(sz_b, n_segment, 1, 1)   # 形状变为 (batch_size, channels)
        conv_X1 = torch.sign(conv_X1) * torch.maximum(torch.abs(conv_X1) - torch.abs(y1), torch.zeros_like(conv_X1))

        conv_X2 = conv_X2.transpose(1, 2).contiguous()
        #增加软收缩去噪
        y2 = self.global_avg_pool(conv_X2).view(sz_b, n_segment)  # 形状变为 (batch_size, channels)
        # 通过全连接层生成通道注意力权重
        y2 = self.avg_fc(y2).view(sz_b, n_segment, 1, 1)   # 形状变为 (batch_size, channels)
        conv_X2 = torch.sign(conv_X2) * torch.maximum(torch.abs(conv_X2) - torch.abs(y2), torch.zeros_like(conv_X2))

        conv_X3 = conv_X3.transpose(1, 2).contiguous()
        #增加软收缩去噪
        y3 = self.global_avg_pool(conv_X3).view(sz_b, n_segment)  # 形状变为 (batch_size, channels)
        # 通过全连接层生成通道注意力权重
        y3 = self.avg_fc(y3).view(sz_b, n_segment, 1, 1)   # 形状变为 (batch_size, channels)
        conv_X3 = torch.sign(conv_X3) * torch.maximum(torch.abs(conv_X3) - torch.abs(y3), torch.zeros_like(conv_X3))

        # X1 = conv_X1.transpose(1, 2).contiguous().view(-1, 1 , 8, 8)  # (n*b) x lq x dk
        # X2 = conv_X2.transpose(1, 2).contiguous().view(-1, 1, 8, 8)  # (n*b) x lk x dk
        # X3 = conv_X3.transpose(1, 2).contiguous().view(-1, 1, 8, 8)  # (n*b) x lv x dv

        return conv_X1.view(-1, 1 , 32, 2), conv_X2.view(-1, 1 , 32, 2), conv_X3.view(-1, 1 , 32, 2)

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, X1, X2, X3):
        m_batchsize, _, height, width = X1.size()
        proj_query = self.query_conv(X1)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key   = self.key_conv(X2)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(X3)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return out_H + out_W

class LinkPrediction(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3):
        super(LinkPrediction, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class mrcca(nn.Module):
    def __init__(self, d_input, n_segment, d_k,
                 d_model, d_inner, MR_method='mlp', **kwargs):
        super(mrcca, self).__init__()
        # parameters
        self.d_input = d_input

        self.n_segment = n_segment
        self.d_k = d_k
        self.d_model = d_model
        self.d_inner = d_inner
        self.MR_method = MR_method
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(kwargs['dr_in'])
        self.MR_mlp = MR_MLP(
            n_segment, d_model, d_k, **kwargs)
        self.MR_cnn = MR_CNN(
            n_segment, d_model, d_k, **kwargs)
        self.Ca = Cross_attention(
            n_segment, d_model, d_k, **kwargs)
        self.LP = LinkPrediction(
            d_model, d_inner, dropout=kwargs['dr_fm'])

    def forward(self, edges):
        enc_input = self.dropout(edges)
        if self.MR_method != 'cnn':
            X1, X2, X3 = self.MR_mlp(enc_input)
        else:
            X1, X2, X3 = self.MR_cnn(enc_input)
        enc_output = self.Ca(enc_input, X1, X2, X3)
        enc_output = self.LP(enc_output)
        return enc_output

#删dv
class MRCCA(nn.Module):

    def __init__(
            self, num_nodes,
            num_rels,
            d_embd, d_k,
            d_model, d_inner, num_segment,
            MR_method,
            label_smoothing=0.1,
            **kwargs
    ):
        super(MRCCA, self).__init__()

        self.mrcca = mrcca(d_embd,  num_segment, d_k, d_model, d_inner, MR_method, **kwargs)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_nodes)))
        self.num_nodes, self.num_rels = num_nodes, num_rels
        self.d_embd = d_embd
        self.init_parameters()
        self.ent_bn = nn.BatchNorm1d(d_embd)
        self.rel_bn = nn.BatchNorm1d(d_embd)
        self.label_smoothing = label_smoothing
        self.MR_method = MR_method
        self.GCN = GCNConv(d_model, d_model)
        self.dropout = nn.Dropout(0.3)
    def init_parameters(self):
        self.tr_ent_embedding = nn.Parameter(torch.Tensor(self.num_nodes, self.d_embd))
        self.rel_embedding = nn.Parameter(torch.Tensor(2 * self.num_rels, self.d_embd))
        nn.init.xavier_normal_(self.tr_ent_embedding.data)
        nn.init.xavier_normal_(self.rel_embedding.data)

    def cal_loss(self, scores, edges, label_smooth=True):
        labels = torch.zeros_like((scores), device='cuda')
        labels.scatter_(1, edges[:, 2][:, None].long(), 1.)
        if self.label_smoothing:
            labels = ((1.0 - self.label_smoothing) * labels) + (1.0 / labels.size(1))
        pred_loss = F.binary_cross_entropy_with_logits(scores, labels)

        return pred_loss

    def cal_score(self, edges, data, mode='train'):
        if self.MR_method == 'gcn':
            out_e = self.GCN(self.tr_ent_embedding, data.edge_index)
            out_e = self.tr_ent_embedding + self.dropout(out_e)
            h = out_e[edges[:, 0]]
        else:
            h = self.tr_ent_embedding[edges[:, 0]]
        r = self.rel_embedding[edges[:, 1]]
        h = self.ent_bn(h)[:, None]
        r = self.rel_bn(r)[:, None]
        feat_edges = torch.hstack((h, r))

        embd_edges = self.mrcca(feat_edges)
        src_edges = embd_edges[:, 1, :]
        scores = torch.mm(src_edges, out_e.transpose(0, 1))
        scores += self.b.expand_as(scores)
        return scores

    def forward(self, feat_edges, data):
        scores = self.cal_score(feat_edges, data, mode='train')
        return scores

    def predict(self, edges, data):
        labels = edges[:, 2]
        edges = edges[0][None, :]
        scores = self.cal_score(edges, data, mode='test')
        scores = scores[:, labels.view(-1)].view(-1)
        scores = torch.sigmoid(scores)
        acc = 0.0
        return scores, acc
