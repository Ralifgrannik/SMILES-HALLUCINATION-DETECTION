"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""


import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.device != hidden_states.device:
        attention_mask = attention_mask.to(hidden_states.device)

    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # 1. Search for response tokens
    real_indices = attention_mask.nonzero(as_tuple=True)[0]
    num_real = len(real_indices)
    
    # We take the last 25 tokens
    response_len = min(num_real, 25)
    response_indices = real_indices[-response_len:]

    # 2. Layer Selection
    selected_layers = [14, 16, 18, 22]
    features = []
    
    for layer_idx in selected_layers:
        if layer_idx >= n_layers:
            continue
        layer = hidden_states[layer_idx]
        resp_tokens = layer[response_indices]
        
        if len(resp_tokens) > 0:
            # Mean pooling
            mean_vec = resp_tokens.mean(dim=0)
            # Max pooling
            max_vec = resp_tokens.max(dim=0)[0]
            
            features.append(mean_vec)
            features.append(max_vec)
    
    delta = hidden_states[-1, response_indices[-1]] - hidden_states[-2, response_indices[-1]]
    features.append(delta)

    return torch.cat(features, dim=0)

def extract_geometric_features(hidden_states, attention_mask):
    return torch.zeros(0, device=hidden_states.device)

def aggregation_and_feature_extraction(hidden_states, attention_mask, use_geometric=False):
    return aggregate(hidden_states, attention_mask)