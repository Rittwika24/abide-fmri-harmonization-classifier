# --- Step 4: DataLoaders and Collation Function ---
from torch.utils.data import DataLoader

def collate_fn(batch):
    # This collation function correctly handles variable-sized inputs (X,Y,Z,Time)
    # The padding for LSTM will happen later based on Time dimension.
    imgs = [item[0] for item in batch]  # list of (X, Y, Z, Time) tensors
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    sites = [item[2] for item in batch]
    return imgs, labels, sites

# --- Step 6: Helper to extract LSTM features (remains mostly same) ---
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from neuroHarmonize import harmonizationLearn, harmonizationApply

def extract_features(model, dataloader, device):
    model.eval()
    all_feats = []
    all_labels = []
    all_sites = []
    with torch.no_grad():
        for imgs, lbls, sites in dataloader:
            # imgs is a list of (X,Y,Z,Time) tensors
            _, feats = model(imgs, return_feats=True) # feats shape (batch_size, lstm_hidden)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_sites.extend(sites)
    
    # Concatenate all features
    feats = np.vstack(all_feats)
    return feats, np.array(all_labels), np.array(all_sites)
