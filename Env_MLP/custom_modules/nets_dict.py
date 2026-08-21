

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CarlaDictFeatureExtractor(nn.Module):

    def __init__(self, state_shape:dict, features_dim: int = 256):
        super().__init__()
        img_channels = state_shape["image"][0]
        kin_channels = state_shape["kinematics"][0]
        # Define the feature extractor for each component of the state dictionary
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), # stride 2 to learned downsampled
            nn.ReLU(), 
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(kin_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        with torch.no_grad():
            _c, _h, _w = state_shape["image"]
            cnn_out = self.cnn(torch.zeros(1, _c, _h, _w)).shape[1] 

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.kin_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(128 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, img_seq, kin_seq):

        B, T = img_seq.shape[:2]
        flat_img = img_seq.view(B*T, *img_seq.shape[2:])
        flat_kin = kin_seq.view(B*T, *kin_seq.shape[2:])

        f_img = self.cnn_proj(self.cnn(flat_img))
        f_kin = self.kin_proj(self.mlp(flat_kin))
        f_cat = torch.cat((f_img, f_kin), dim=1)
        features = self.linear(f_cat)

        return features.view(B, T, -1)  # Reshape back to (B, T, features_dim)

class LSTMDict_Actor(nn.Module):

    def __init__(self, action_dim, hidden_size, features_dim = 256):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=features_dim + action_dim,
                             hidden_size=hidden_size, batch_first=True)
        self.action_head = nn.Sequential(nn.Linear(features_dim + hidden_size, 128), 
                                         nn.LayerNorm(128),
                                         nn.ReLU(), 
                                         nn.Linear(128, action_dim))
        nn.init.uniform_(self.action_head[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.action_head[-1].bias)
        with torch.no_grad():
            self.action_head[-1].bias[1] = 0.5  # throttle, pre-tanh so approx 0.46 idle creep.

    def forward(self, features ,hidden_state, a_past):
        
        lstm_in = torch.cat((features, a_past), dim = -1)
        lstm_out, hidden_state = self.lstm(lstm_in, hidden_state)

        fusion = torch.cat((features, lstm_out), dim = -1)
        logits = self.action_head(fusion)
        action = torch.tanh(logits)

        return action, logits, hidden_state
    
class Double_LSTMDict_Critic(nn.Module):

    def __init__(self, action_dim, state_shape:dict, hidden_size, features_dim = 256):
        super().__init__()
        self.extractor = CarlaDictFeatureExtractor(state_shape, features_dim=features_dim)

        self.lstm1 = nn.LSTM(features_dim + action_dim, hidden_size, batch_first=True)
        self.q1_fusion_norm = nn.LayerNorm(features_dim + action_dim + hidden_size)
        self.q1_head = nn.Sequential(nn.Linear(features_dim + action_dim + hidden_size, 128), 
                                    nn.ReLU(), 
                                    nn.Linear(128, 1))
        #concatinate for better temporal awareness
        self.lstm2 = nn.LSTM(features_dim + action_dim, hidden_size=hidden_size, batch_first=True)
        self.q2_fusion_norm = nn.LayerNorm(features_dim + action_dim + hidden_size)
        self.q2_head = nn.Sequential(nn.Linear(features_dim + action_dim + hidden_size, 128), 
                                    nn.ReLU(), 
                                    nn.Linear(128, 1))

    def encode(self, s):
        """ Only cnn_kin pass, returns (B, T, features_dim)"""
        return self.extractor(s["image"], s["kinematics"])

    def forward(self, features, a_curr, hidden_state_1, hidden_state_2, a_past):
        # Q1
        lstm_in1 = torch.cat((features, a_past), dim = -1)
        lstm_out1, hidden_state_1 = self.lstm1(lstm_in1, hidden_state_1)

        fusion1 = self.q1_fusion_norm(torch.cat((features, a_curr, lstm_out1), dim = -1))
        q1 = self.q1_head(fusion1)

        # Q2
        lstm_in2 = torch.cat((features, a_past), dim = -1)
        lstm_out2, hidden_state_2 = self.lstm2(lstm_in2, hidden_state_2)
        
        fusion2 = self.q2_fusion_norm(torch.cat((features, a_curr, lstm_out2), dim = -1))
        q2 = self.q2_head(fusion2)

        return q1, q2, hidden_state_1, hidden_state_2
    
    def single_forward(self, features, a_curr, hidden_state_1, a_past):
       
        lstm_in1 = torch.cat((features, a_past), dim = -1)
        lstm_out1, hidden_state_1 = self.lstm1(lstm_in1, hidden_state_1)

        fusion1 = self.q1_fusion_norm(torch.cat((features, a_curr, lstm_out1), dim = -1))

        q1 = self.q1_head(fusion1)

        return q1, hidden_state_1
    