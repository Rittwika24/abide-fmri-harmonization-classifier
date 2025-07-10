# --- Step 5: CNN-LSTM Model (Modified for Vectorized Processing) ---
import torch.nn as nn
import torch.nn.functional as F

# -------- CNN for 2D slices --------
class SliceCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # Input to Conv2d is (N_slices, 1, X, Y)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_dim, 5, padding=1)
        # AdaptiveAvgPool2d(1) will pool the spatial dimensions (X, Y) to 1x1,
        # so the output feature size is `out_dim` regardless of original X, Y dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # x shape: (N_slices, 1, X, Y)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(self.conv3(x))
        return x.view(x.size(0), -1) # Output shape: (N_slices, out_dim)

# -------- CNN-LSTM + Classifier --------
class CNN_LSTM_Harmonizer(nn.Module):
    def __init__(self, cnn_out_dim=128, lstm_hidden=64, num_classes=2):
        super().__init__()
        self.cnn = SliceCNN(out_dim=cnn_out_dim)
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x_samples_list, return_feats=False):
        # x_samples_list is a list of (X, Y, Z, Time) tensors for the current batch
        device = next(self.parameters()).device
        
        batch_lstm_inputs = [] # Will store (Time, cnn_out_dim) for each sample
        seq_lengths = []       # Will store original Time_dim for each sample
        
        for sample_4d in x_samples_list:
            # sample_4d shape: (X, Y, Z, Time)
            X_dim, Y_dim, Z_dim, Time_dim = sample_4d.shape

            # Reshape the 4D sample into a batch of 2D slices for SliceCNN
            # Desired input for SliceCNN: (N_slices, 1, X, Y)
            # Permute to (Z, Time, X, Y), then reshape to (Z*Time, X, Y)
            all_2d_slices_from_sample = sample_4d.permute(2, 3, 0, 1).reshape(Z_dim * Time_dim, X_dim, Y_dim)
            
            # Add channel dimension (1) and move to device
            all_2d_slices_from_sample = all_2d_slices_from_sample.unsqueeze(1).to(device) # Shape: (Z*Time, 1, X, Y)
            
            # Process all 2D slices for this sample through SliceCNN in one go
            cnn_features_from_slices = self.cnn(all_2d_slices_from_sample) # Shape: (Z*Time, cnn_out_dim)
            
            # Reshape features back to (Z, Time, cnn_out_dim) for aggregation
            cnn_features_reshaped = cnn_features_from_slices.reshape(Z_dim, Time_dim, -1)
            
            # Aggregate features for each time point by averaging over Z slices
            # Resulting shape: (Time, cnn_out_dim)
            lstm_input_for_sample = cnn_features_reshaped.mean(dim=0) 
            
            batch_lstm_inputs.append(lstm_input_for_sample)
            seq_lengths.append(Time_dim) # Store the original time dimension as sequence length for LSTM padding

        # Pad sequences for LSTM and pack them
        padded_lstm_inputs = nn.utils.rnn.pad_sequence(batch_lstm_inputs, batch_first=True)
        packed_lstm_input = nn.utils.rnn.pack_padded_sequence(
            padded_lstm_inputs, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_out, (hn, cn) = self.lstm(packed_lstm_input)
        
        # Get the last hidden state for classification
        feats_out = hn[-1] # Shape: (batch_size, lstm_hidden)
        
        # Pass through classifier
        logits = self.classifier(feats_out)
        
        if return_feats:
            return logits, feats_out
        else:
            return logits
