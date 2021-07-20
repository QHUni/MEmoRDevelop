import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
from model.TfEncoderWithMemory import BertEncoderWithMemory

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class AMER(BaseModel):
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device):

        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"]
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"]
        D_a = config["audio"]["feature_dim"]
        D_t = config["text"]["feature_dim"]
        D_p = config["personality"]["feature_dim"]
        
        self.attn = ScaledDotProductAttention((4 * D_e) ** 0.5, attn_dropout=0)
        
        #qhy add
        #注意这里的config格式不一样
        # self.config_from_tf = config_from_tf
        self.encoder = BertEncoderWithMemory(self.config_from_tf)
        self.encoder_no_mem = BertEncoderNoMemory(self.config_from_tf)

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),# default D_e = 128
            # qhy note
            # nn.Linear(in_features,out_features,bias=True)
            # D_v * 2 * D_e
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),# 这里的D_e需要改为768/2
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 14 * D_e

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c):
        # qhy note Encoders : 两层感知机，这一步要不要？
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p)
        # feature大小*2*128 
        U_all = []

        print('U_v shape...', U_v.shape) # [8, 40, 4302]
        print('U_t shape...', U_t.shape) # [8, 40, 1024]
        print('U_a shape...', U_a.shape) # [8, 40, 6373]
        print('U_p shape...', U_p.shape) # [8, 40, 118]
        
        print('m_v shape...',M_v.shape) #(8,40)
        print('m_a shape...', M_a.shape) #(8,40)
        print('m_t shape...', M_t.shape) #(8,40)
        print('target_loc shape...', target_loc.shape) #(8,40)

        print('V_e shape...', V_e.shape) # [8, 40, 256]
        print('T_e shape...', T_e.shape) # [8, 40, 256]
        print('A_e shape...', A_e.shape) # [8, 40, 256]
        print('P_e shape...', P_e.shape) # [8, 40, 256]

        #1.三个模态的tf处理
        # qhy add from MART:encoder
        # 参数1：prev_ms（这个初始为空，可以不用管，把config改一下就行）
        prev_ms_v = [None] * self.config_from_tf.num_hidden_layers
        # 2:embeddings（不用管前面的处理，只要输入是embedding且大小为(N, L, D)即可）
        # self.embeddings = BertEmbeddingsWithVideo(config, add_postion_embeddings=True)
        # embeddings = self.embeddings(input_ids, video_features, token_type_ids)  # (N, L, D)
        # 3:input masks
        # input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        # *或许可以直接input_mask = torch.randn(2, 5)
        # 调用路径：_load_indexed_video_feature得到mask（未完）
        # 和输入内容和格式直接相关
        # input_masks_list = [e["input_mask"] for e in batched_data]  # input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
        input_mask_v = torch.randn(config["data_loader"]["args"]["batch_size"], D_v)
        prev_ms_v, tf_output_v = self.encoder(
        prev_ms_v, V_e, input_mask_v, output_all_encoded_layers=False)  # both outputs are list

        input_mask_t = torch.randn(config["data_loader"]["args"]["batch_size"], D_a)
        prev_ms_t = [None] * self.config_from_tf.num_hidden_layers
        prev_ms_t, tf_output_t = self.encoder(
        prev_ms_t, T_e, input_mask_t, output_all_encoded_layers=False)  # both outputs are list

        input_mask_a = torch.randn(config["data_loader"]["args"]["batch_size"], D_a)
        prev_ms_a, tf_output_a = self.encoder(
        prev_ms_a, A_e, input_mask_a, output_all_encoded_layers=False)  # both outputs are list (N, L, D)

        '''
        问题7.17
        1.cross-attention分为main context和side_context,不是将两个融合为一个，而是将一个side融合进main里面，那么三个应该怎么融合呢？
        2.放在for循坏之外，那么还是对M片段中的N个人进行分析吗？
        3.cross-attention的mask？

        问题7.16
        1.U_v、V_e,inp_v的含义？
            原始的三种模态的feature和对应的mask
        2.双层感知机要不要？感知机的输出shape？
            要；[8,40,256]
        3.感知机输出为4*3*256，encoder要求的输入是(N, L, D)(15,length,768)，如何转变？
            可以直接对应：对应的含义是一致的
        
        问题7.14
        1.prev_ms是什么？
            prev_ms：previous memory state，初始为空
        2.N,L,D等参数的含义？
            N：batch_size,15
            L：序列的长度,也就是一句话有多少个单词，不固定
            D：embedding的维度，300
        3.encoder前的concat操作在哪里？
            在embedding类里面
        4.transformerXL?
            对比模型

        note：
        注意这是python2.7

        核心：
        1.三个输入和输出格式的适配
        2.config的适配
        '''

        #2.tf后和personality进行concat
        inp_V = torch.cat([tf_output_v, P_e], dim=2)
        inp_A = torch.cat([tf_output_a, P_e], dim=2)
        inp_T = torch.cat([tf_output_t, P_e], dim=2)


        # qhy add
        # 3.video和text的cross-attention
        # def cross_context_encoder(self, main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer, self_att_layer):
        # 仔细看图和函数，这个cross-attention分为main context和side_context,不是将两个融合为一个，而是将一个side融合进main里面，那么三个应该怎么融合呢？
        x_encoded_video_text = self.cross_context_encoder(
            encoded_video_feat, video_mask, encoded_text_feat, text_mask,
            self.video_cross_att, self.video_cross_layernorm, self.video_encoder2)  # (N, L, D)
        x_encoded_video_audio = self.cross_context_encoder(
            encoded_video_feat, video_mask, encoded_audio_feat, audio_mask,
            self.sub_cross_att, self.sub_cross_layernorm, self.video_encoder2)  # (N, L, D)

        # 4.cross-attention之后的concat
        middle_total = torch.cat([x_encoded_video_text, x_encoded_video_audio], dim=2)
        

        # 5.concat之后再放到decoder without memory 里
        input_mask_total = torch.randn(config["data_loader"]["args"]["batch_size"], D_a+D_v+D_t)
        encoded_layer_outputs = self.encoder_no_mem(
            middle_total, input_mask_total, output_all_encoded_layers=False)  # both outputs are list



        for i in range(M_v.shape[0]):
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1:
                    target_moment = j % int(seg_len[i].cpu().numpy())
                    target_character = int(j / seg_len[i].cpu().numpy())
                    break
            
            # 第1部分原始的特征变换
            # qhy note : 也就是说前一步得到的feature全是三维数组
            # qhy modified
            # inp_V = V_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_V = V_e[i, : seq_lengths[i], :].reshape((self.config_from_tf.num_hidden_layers, seg_len[i], -1)).transpose(0, 1)
            inp_T = T_e[i, : seq_lengths[i], :].reshape((self.config_from_tf.num_hidden_layers, seg_len[i], -1)).transpose(0, 1)
            inp_A = A_e[i, : seq_lengths[i], :].reshape((self.config_from_tf.num_hidden_layers, seg_len[i], -1)).transpose(0, 1)
            inp_P = P_e[i, : seq_lengths[i], :].reshape((self.config_from_tf.num_hidden_layers, seg_len[i], -1)).transpose(0, 1)

            # qhy note : 随机mask掉一部分特征做推理
            mask_V = M_v[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_T = M_t[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_A = M_a[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)

            print('inp_V shape...',inp_V.shape) #（5，3，256）
            print('inp_T shape...',inp_T.shape) #（5，3，256）
            print('inp_A shape...', inp_A.shape) #（5，3，256）
            print('inp_P shape...', inp_P.shape) #（5，3，256）

            print('mask_V shape...', mask_V.shape) #(5,3)
            print('mask_T shape...',mask_T.shape) #(5,3)
            print('mask_A shape...',mask_A.shape) #(5,3)

            # Concat with personality embedding
            # qhy modified
            inp_V = torch.cat([tf_output_v, inp_P], dim=2)
            inp_A = torch.cat([tf_output_a, inp_P], dim=2)
            inp_T = torch.cat([tf_output_t, inp_P], dim=2)

            U = []

            
            # 第2部分原始的cross-attention：两个循环
            for k in range(n_c[i]):# 对于每一个character
                new_inp_A, new_inp_T, new_inp_V = inp_A.clone(), inp_T.clone(), inp_V.clone(),
                
                # Modality-level inter-personal attention
                for j in range(seg_len[i]):#对于每一个moment
                    att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :])
                    att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :])
                    att_A, _ = self.attn(inp_A[j, :], inp_A[j, :], inp_A[j, :], mask_A[j, :])
                    new_inp_V[j, :] = att_V + inp_V[j, :]
                    new_inp_A[j, :] = att_A + inp_A[j, :]
                    new_inp_T[j, :] = att_T + inp_T[j, :]
                    # 对于每一个moment j ，得到的new_inp_V的每一行代表一个moment的inter-person attention

                # Modality-level intra-personal attention
                att_V, _ = self.attn(new_inp_V[:, k], new_inp_V[:, k], new_inp_V[:, k], mask_V[:, k])
                att_A, _ = self.attn(new_inp_A[:, k], new_inp_A[:, k], new_inp_A[:, k], mask_A[:, k])
                att_T, _ = self.attn(new_inp_T[:, k], new_inp_T[:, k], new_inp_T[:, k], mask_T[:, k])

                # Residual connection
                inner_V = (att_V[target_moment] + new_inp_V[target_moment][k]).squeeze()
                inner_A = (att_A[target_moment] + new_inp_A[target_moment][k]).squeeze()
                inner_T = (att_T[target_moment] + new_inp_T[target_moment][k]).squeeze()

                # 第3部分特征融合
                # Multimodal fusion
                inner_U = self.fusion_layer(torch.cat([inner_V, inner_A, inner_T, inp_P[0][k]]))

                U.append(inner_U)

            # 第3部分之后的又一次attention处理，需要删掉，注意输出格式
            if len(U) == 1:
                # Only one character in this sample
                U_all.append(U[0])
            else:
                # Person-level Inter-personal Attention
                U = torch.stack(U, dim=0)
                output, _ = self.attn(U, U, U)
                U = U + output
                U_all.append(U[target_character])

        U_all = torch.stack(U_all, dim=0)
        # Classification
        log_prob = self.out_layer(U_all)
        log_prob = F.log_softmax(log_prob)

        return log_prob

