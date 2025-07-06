# --- Step 3: Dataset Class (Modified for Lazy Loading) ---
from torch.utils.data import Dataset
import numpy as np

class Variable4DDataset(Dataset):
    def __init__(self, data_references, base_directory):
        self.data_references = data_references
        self.base_directory = base_directory
        # A simple cache to store recently loaded chunks. Adjust size as needed.
        self.chunk_cache = {}
        self.cache_max_size = 2 # Store up to 5 chunks in cache

    def __len__(self):
        return len(self.data_references)

    def __getitem__(self, idx):
        ref = self.data_references[idx]
        chunk_id = ref['chunk_id']
        index_in_chunk = ref['index_in_chunk']
        
        # Load chunk from cache or disk
        if chunk_id not in self.chunk_cache:
            # Simple cache eviction: remove oldest if cache is full
            if len(self.chunk_cache) >= self.cache_max_size:
                # Find and remove the least recently used (LRU) chunk if you want
                # For simplicity, we'll just pop an arbitrary item
                # A more robust LRU cache would require tracking access times
                self.chunk_cache.pop(next(iter(self.chunk_cache)))
            
            pkl_file = os.path.join(self.base_directory, f'{chunk_id}.pkl')
            with open(pkl_file, 'rb') as f:
                self.chunk_cache[chunk_id] = pickle.load(f)['images']
        
        # Access the image from the loaded chunk
        images_in_chunk = self.chunk_cache[chunk_id]
        img = images_in_chunk[index_in_chunk]
        
        # Convert to tensor and normalize
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected img np.ndarray or torch.Tensor, got {type(img)}")
        
        # Normalize voxel time series: (X, Y, Z, Time)
        original_shape = img.shape 
        flat_img = img.reshape(-1, original_shape[-1]) # Reshape to (Voxels, Time)
        
        mean_vox = flat_img.mean(dim=1, keepdim=True)
        std_vox = flat_img.std(dim=1, keepdim=True) + 1e-8
        norm_img = ((flat_img - mean_vox) / std_vox).reshape(original_shape)
        
        # Extract pre-processed label and site_id
        label = ref['label']
        site = ref['site_id']
        
        return norm_img, label, site
