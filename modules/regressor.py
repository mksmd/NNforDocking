import torch
from torch import nn
from torch.nn import functional as F


class regressor(nn.Module):
    def __init__(self):
        super(regressor, self).__init__()

        self.drop = nn.Dropout(p=0.0)

        ###
        ### peptide adjacency matrix ###
        ###
        # input Tensor - (N_batch, Channels = 1, Height = 300, Width = 300)
        self.conv_p_adj_r0 = nn.Conv2d(1, 8, kernel_size=(24, 24), stride=(4, 4))
        # output Tensor - (N_batch, Channels = 8, Height = 70, Width = 70)

        # input Tensor - (N_batch, Channels = 1, Height = 300, Width = 300)
        self.conv_p_adj_0 = nn.Conv2d(1, 4, kernel_size=(12, 12), stride=(2, 2))

        # input Tensor - (N_batch, Channels = 4, Height = 145, Width = 145)
        self.conv_p_adj_1 = nn.Conv2d(4, 8, kernel_size=(6, 6), stride=(2, 2))
        # output Tensor - (N_batch, Channels = 8, Height = 70, Width = 70)


        # input Tensor - (N_batch, Channels = 8, Height = 70, Width = 70)
        self.conv_p_adj_r1 = nn.Conv2d(8, 16, kernel_size=(14, 14), stride=(4, 4))
        # output Tensor - (N_batch, Channels = 16, Height = 15, Width = 15)

        # input Tensor - (N_batch, Channels = 8, Height = 70, Width = 70)
        self.conv_p_adj_2 = nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(2, 2))

        # input Tensor - (N_batch, Channels = 12, Height = 33, Width = 33)
        self.conv_p_adj_3 = nn.Conv2d(12, 16, kernel_size=(4, 4), stride=(2, 2))
        # output Tensor - (N_batch, Channels = 16, Height = 15, Width = 15)

        self.fc_p_adj_4 = nn.Linear(3600, 3600)

        ###
        ### peptide onehot ###
        ###
        # input Tensor - (N_batch, Channels = 1, Height = 300, Width = 10)
        self.conv_p_onehot_r0 = nn.Conv2d(1, 8, kernel_size=(24, 10), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 8, Height = 70, Width = 1)

        # input Tensor - (N_batch, Channels = 1, Height = 300, Width = 10)
        self.conv_p_onehot_0 = nn.Conv2d(1, 4, kernel_size=(12, 10), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 4, Height = 145, Width = 1)
        self.conv_p_onehot_1 = nn.Conv2d(4, 8, kernel_size=(6, 1), stride=(2, 1))


        # input Tensor - (N_batch, Channels = 8, Height = 70, Width = 1)
        self.conv_p_onehot_r1 = nn.Conv2d(8, 16, kernel_size=(14, 1), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 15, Width = 1)

        # input Tensor - (N_batch, Channels = 8, Height = 70, Width = 1)
        self.conv_p_onehot_2 = nn.Conv2d(8, 12, kernel_size=(6, 1), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 12, Height = 33, Width = 1)
        self.conv_p_onehot_3 = nn.Conv2d(12, 16, kernel_size=(4, 1), stride=(2, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 15, Width = 1)

        self.fc_p_onehot_4 = nn.Linear(240, 240)

        ###
        ### receptor coordinates ###
        ###
        # input Tensor - (N_batch, Channels = 1, Height = 4000, Width = 3)
        self.conv_r_coords_r0 = nn.Conv2d(1, 8, kernel_size=(24, 3), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)

        # input Tensor - (N_batch, Channels = 1, Height = 4000, Width = 3)
        self.conv_r_coords_0 = nn.Conv2d(1, 4, kernel_size=(12, 3), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 4, Height = 1995, Width = 1)
        self.conv_r_coords_1 = nn.Conv2d(4, 8, kernel_size=(6, 1), stride=(2, 1))


        # input Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)
        self.conv_r_coords_r1 = nn.Conv2d(8, 16, kernel_size=(14, 1), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 246, Width = 1)

        # input Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)
        self.conv_r_coords_2 = nn.Conv2d(8, 12, kernel_size=(6, 1), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 12, Height = 495, Width = 1)
        self.conv_r_coords_3 = nn.Conv2d(12, 16, kernel_size=(4, 1), stride=(2, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 246, Width = 1)

        self.fc_r_coords_4 = nn.Linear(3936, 3936)

        ###
        ### receptor onehot ###
        ###
        # input Tensor - (N_batch, Channels = 1, Height = 4000, Width = 10)
        self.conv_r_onehot_r0 = nn.Conv2d(1, 8, kernel_size=(24, 10), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)

        # input Tensor - (N_batch, Channels = 1, Height = 4000, Width = 10)
        self.conv_r_onehot_0 = nn.Conv2d(1, 4, kernel_size=(12, 10), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 4, Height = 1995, Width = 1)
        self.conv_r_onehot_1 = nn.Conv2d(4, 8, kernel_size=(6, 1), stride=(2, 1))


        # input Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)
        self.conv_r_onehot_r1 = nn.Conv2d(8, 16, kernel_size=(14, 1), stride=(4, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 246, Width = 1)

        # input Tensor - (N_batch, Channels = 8, Height = 995, Width = 1)
        self.conv_r_onehot_2 = nn.Conv2d(8, 12, kernel_size=(6, 1), stride=(2, 1))

        # input Tensor - (N_batch, Channels = 12, Height = 495, Width = 1)
        self.conv_r_onehot_3 = nn.Conv2d(12, 16, kernel_size=(4, 1), stride=(2, 1))
        # output Tensor - (N_batch, Channels = 16, Height = 246, Width = 1)

        self.fc_r_onehot_4 = nn.Linear(3936, 3936)

        ###
        ### all stacked ###
        ###
        self.fc_all_p_adj_0 = nn.Linear(3600, 3000)
        self.fc_all_p_onehot_0 = nn.Linear(240, 3000)
        self.fc_all_r_coords_0 = nn.Linear(3936, 3000)
        self.fc_all_r_onehot_0 = nn.Linear(3936, 3000)

        self.fc_all_p_adj_1 = nn.Linear(3600, 900)
        self.fc_all_p_onehot_1 = nn.Linear(240, 900)
        self.fc_all_r_coords_1 = nn.Linear(3936, 900)
        self.fc_all_r_onehot_1 = nn.Linear(3936, 900)

        self.fc_all_0 = nn.Linear(11712, 6000)
        self.fc_all_1 = nn.Linear(6000, 3000)
        self.fc_all_2 = nn.Linear(3000, 900)

        self.prelu = nn.PReLU()

    def conv_p_adj(self, x):
        x_r0 = F.relu(self.conv_p_adj_r0(x))
        x = F.relu(self.conv_p_adj_0(x))
        x = F.relu(self.conv_p_adj_1(x) + x_r0)
        del x_r0
        x_r1 = F.relu(self.conv_p_adj_r1(x))
        x = F.relu(self.conv_p_adj_2(x))
        x = F.relu(self.conv_p_adj_3(x) + x_r1)
        del x_r1
        x = F.relu(self.fc_p_adj_4(x.view(-1, 3600)))
        return x

    def conv_p_onehot(self, x):
        x_r0 = F.relu(self.conv_p_onehot_r0(x))
        x = F.relu(self.conv_p_onehot_0(x))
        x = F.relu(self.conv_p_onehot_1(x) + x_r0)
        del x_r0
        x_r1 = F.relu(self.conv_p_onehot_r1(x))
        x = F.relu(self.conv_p_onehot_2(x))
        x = F.relu(self.conv_p_onehot_3(x) + x_r1)
        del x_r1
        x = F.relu(self.fc_p_onehot_4(x.view(-1, 240)))
        return x

    def conv_r_coords(self, x):
        x_r0 = F.relu(self.conv_r_coords_r0(x))
        x = F.relu(self.conv_r_coords_0(x))
        x = F.relu(self.conv_r_coords_1(x) + x_r0)
        del x_r0
        x_r1 = F.relu(self.conv_r_coords_r1(x))
        x = F.relu(self.conv_r_coords_2(x))
        x = F.relu(self.conv_r_coords_3(x) + x_r1)
        del x_r1
        x = F.relu(self.fc_r_coords_4(x.view(-1, 3936)))
        return x

    def conv_r_onehot(self, x):
        x_r0 = F.relu(self.conv_r_onehot_r0(x))
        x = F.relu(self.conv_r_onehot_0(x))
        x = F.relu(self.conv_r_onehot_1(x) + x_r0)
        del x_r0
        x_r1 = F.relu(self.conv_r_onehot_r1(x))
        x = F.relu(self.conv_r_onehot_2(x))
        x = F.relu(self.conv_r_onehot_3(x) + x_r1)
        del x_r1
        x = F.relu(self.fc_r_onehot_4(x.view(-1, 3936)))
        return x

    def fc_all(self, x_p_a, x_p_o, x_r_c, x_r_o):
        x = torch.cat((x_p_a,
                       x_p_o,
                       x_r_c,
                       x_r_o), dim = 1)
        x = F.normalize(x)

        x_p_a = F.normalize(x_p_a)
        x_p_o = F.normalize(x_p_o)
        x_r_c = F.normalize(x_r_c)
        x_r_o = F.normalize(x_r_o)

        x_p_a_0 = F.relu(self.drop(self.fc_all_p_adj_0(x_p_a)))
        x_p_o_0 = F.relu(self.drop(self.fc_all_p_onehot_0(x_p_o)))
        x_r_c_0 = F.relu(self.drop(self.fc_all_r_coords_0(x_r_c)))
        x_r_o_0 = F.relu(self.drop(self.fc_all_r_onehot_0(x_r_o)))

        del x_p_a, x_p_o, x_r_c, x_r_o

        x = F.relu(self.drop(self.fc_all_0(x)))
        x = F.relu(self.drop(self.fc_all_1(x)) + x_p_a_0 + x_p_o_0 + x_r_c_0 + x_r_o_0)
        del x_p_a_0, x_p_o_0, x_r_c_0, x_r_o_0

        x = self.fc_all_2(x)

        return x

    def forward(self, x):
        h_p_a = self.conv_p_adj(x[0].view(-1, 1, 300, 300))
        h_p_o = self.conv_p_onehot(x[1].view(-1, 1, 300, 10))
        h_r_c = self.conv_r_coords(x[2].view(-1, 1, 4000, 3))
        h_r_o = self.conv_r_onehot(x[3].view(-1, 1, 4000, 10))
        h_all = self.fc_all(h_p_a, h_p_o, h_r_c, h_r_o)
        return h_all.view(-1, 300, 3)
