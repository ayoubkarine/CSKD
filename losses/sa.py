"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CriterionSA']

class CriterionSA(nn.Module):  # CKA between self-attention modules

    def __init__(self, temperature=1.0, temperaturesa=1.0):
    
        super(CriterionSA, self).__init__()
        self.temperature = temperature # Temperature of score map
        self.temperaturesa=temperaturesa # Temperature of feature map
        self.softmax = torch.nn.Softmax(dim=-1)
        self.gammacam = nn.Parameter(torch.zeros(1))
        self.gammapam = nn.Parameter(torch.zeros(1))
        self.kld = nn.KLDivLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n]).cuda()
        I = torch.eye(n).cuda()
        H = I - unit / n
        # H.cuda()
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA_loss(self, X, Y): # CKA between X and Y
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return -torch.log(torch.mean(torch.abs(torch.div(hsic, (var1 * var2)))) + 1e-8)
    
    def CAM(self, X): # Channel Attention Module
        m_batchsize, C, height, width = X.size()
        proj_query = X.contiguous().view(m_batchsize, C, -1) # reshape
        proj_key = X.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1) # reshape and transpose
        energy = torch.bmm(proj_query, proj_key) # multiplication
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new/self.temperaturesa)               
        proj_value = X.contiguous().view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.contiguous().view(m_batchsize, C, height, width)
        out = self.gammacam*out + X
        return out

    def PAM(self, X): # Positionnal Attention Module
        m_batchsize, C, height, width = X.size()
        in_dim = C
        self.query_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1).cuda() # to generate B
        self.key_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1).cuda() # to generate C
        self.value_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).cuda()
        proj_query = self.query_conv(X).contiguous().view(m_batchsize, -1, width*height).permute(0, 2, 1) # B
        proj_key = self.key_conv(X).contiguous().view(m_batchsize, -1, width*height) # C
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy/self.temperaturesa) # S
        proj_value = self.value_conv(X).contiguous().view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.contiguous().view(m_batchsize, C, height, width)
        out = self.gammapam*out + X
        return out  

    def forward(self,feat_S, feat_T, feature_transform="CAM"):  # CKA between CAM and between PAM
        # feat_S, feat_T = feat_S.cuda(), feat_T.cuda()
        m_batchsize, CS, height, width = feat_S.size()
        m_batchsize, CT, height, width = feat_T.size()
        loss_CAM, loss_PAM = 0, 0
        self.conv = nn.Conv2d(CS, CT, kernel_size=1, bias=False).cuda()

        if feature_transform=="CAM_MSE":
            feat_S = self.conv(feat_S) 
            CAM_S = self.CAM(feat_S)
            CAM_T = self.CAM(feat_T)
            loss_CAM = self.mse(CAM_S, CAM_T) 
            return loss_CAM
        elif feature_transform=="CAM_CKA":
            CAM_S = self.CAM(feat_S)
            CAM_T = self.CAM(feat_T)
            CAM_S = CAM_S.view(CAM_S.size(0), -1)
            CAM_T = CAM_T.view(CAM_T.size(0), -1)
            loss_CAM = self.linear_CKA_loss(CAM_S, CAM_T)
            return loss_CAM
        elif feature_transform=="PAM_CKA":
            PAM_S = self.PAM(feat_S)
            PAM_T = self.PAM(feat_T)
            PAM_S = PAM_S.view(PAM_S.size(0), -1)
            PAM_T = PAM_T.view(PAM_T.size(0), -1)
            loss_PAM = self.linear_CKA_loss(PAM_S, PAM_T)
            return loss_PAM
        elif feature_transform=="gridPAM_MSE":
            # Student feature maps to blocks
            feat_S = self.conv(feat_S)
            firstpart_S = torch.chunk(feat_S, 5, dim=2)
            partsPAM_S = []
            for i in firstpart_S:
                secondpart_S = torch.chunk(i, 5, dim=3)
                for j in secondpart_S:
                    partsPAM_S.append(self.PAM(j))
            # Teacher feature maps to blocks
            firstpart_T = torch.chunk(feat_T, 5, dim=2)
            partsPAM_T = []
            for i in firstpart_T:
                secondpart_T = torch.chunk(i, 5, dim=3)
                for j in secondpart_T:
                    partsPAM_T.append(self.PAM(j))
            # Loss computation 
            n = len(partsPAM_S)
            loss_PAM = 0
            for i in range(0, n):
                loss_PAM += self.mse(partsPAM_S[i], partsPAM_T[i])
            loss = loss_PAM/n
            return loss
        elif feature_transform == "gridPAM_CKA":
            # Student feature maps to blocks
            firstpart_S = torch.chunk(feat_S, 5, dim=2)
            partsPAM_S = []
            for i in firstpart_S:
                secondpart_S = torch.chunk(i, 5, dim=3)
                for j in secondpart_S:
                    partsPAM_S.append(self.PAM(j))
            # Teacher feature maps to blocks
            firstpart_T = torch.chunk(feat_T, 5, dim=2)
            partsPAM_T = []
            for i in firstpart_T:
                secondpart_T = torch.chunk(i, 5, dim=3)
                for j in secondpart_T:
                    partsPAM_T.append(self.PAM(j))
            # Loss computation 
            n = len(partsPAM_S)
            loss_PAM = 0
            for i in range(0, n):
                # -------- CKA gridCAM
                S_cam = partsPAM_S[i].view(partsPAM_S[i].size(0), -1)
                T_cam = partsPAM_T[i].view(partsPAM_T[i].size(0), -1)
                loss_PAM += self.linear_CKA_loss(S_cam, T_cam)
            loss = loss_PAM/n
            return loss
        elif feature_transform=="separately_CAMgridPAM_MSE":
            feat_S = self.conv(feat_S)
            # ---- gridPAM
            # Student feature maps to blocks
            firstpart_S = torch.chunk(feat_S, 5, dim=2)
            partsPAM_S = []
            for i in firstpart_S:
                secondpart_S = torch.chunk(i, 5, dim=3)
                for j in secondpart_S:
                    partsPAM_S.append(self.PAM(j))
            # Teacher feature maps to blocks
            firstpart_T = torch.chunk(feat_T, 5, dim=2)
            partsPAM_T = []
            for i in firstpart_T:
                secondpart_T = torch.chunk(i, 5, dim=3)
                for j in secondpart_T:
                    partsPAM_T.append(self.PAM(j))
            # Loss computation 
            n = len(partsPAM_S)
            loss_PAM = 0
            for i in range(0, n):             
                loss_PAM += self.mse( partsPAM_S[i], partsPAM_T[i])
            loss_PAM = loss_PAM/n
            # ---- CAM
            CAM_S = self.CAM(feat_S)
            CAM_T = self.CAM(feat_T)
            loss_CAM = self.mse(CAM_S, CAM_T)
            return loss_CAM, loss_PAM
        elif feature_transform=="separately_CAMgridPAM_CKA":
            # ---- gridPAM
            # Student feature maps to blocks
            firstpart_S = torch.chunk(feat_S, 5, dim=2)
            partsPAM_S = []
            for i in firstpart_S:
                secondpart_S = torch.chunk(i, 5, dim=3)
                for j in secondpart_S:
                    partsPAM_S.append(self.PAM(j))
            # Teacher feature maps to blocks
            firstpart_T = torch.chunk(feat_T, 5, dim=2)
            partsPAM_T = []
            for i in firstpart_T:
                secondpart_T = torch.chunk(i, 5, dim=3)
                for j in secondpart_T:
                    partsPAM_T.append(self.PAM(j))
            # Loss computation 
            n = len(partsPAM_S)
            loss_PAM = 0
            for i in range(0, n):             
                S_pam = partsPAM_S[i].view(partsPAM_S[i].size(0), -1)
                T_pam = partsPAM_T[i].view(partsPAM_T[i].size(0), -1)
                loss_PAM += self.linear_CKA_loss(S_pam, T_pam)
            loss_PAM = loss_PAM/n
            # ---- CAM
            CAM_S = self.CAM(feat_S)
            CAM_T = self.CAM(feat_T)
            CAM_S = CAM_S.view(CAM_S.size(0), -1)
            CAM_T = CAM_T.view(CAM_T.size(0), -1)
            loss_CAM = self.linear_CKA_loss(CAM_S, CAM_T)
            return loss_CAM, loss_PAM