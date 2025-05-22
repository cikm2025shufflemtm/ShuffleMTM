#%%

import torch
import torch.nn as nn

#%%

def create_patch(x, patch_len, stride=None):
    # x : (bs, seq_len, n_vars)
    stride = patch_len if stride is None else stride
    seq_len = x.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
    
    x = x[:, s_begin:, :]
    x = x.unfold(dimension=1, size=patch_len, step=stride) # x: [bs x num_patch x n_vars x patch_len]
    
    return x, num_patch
#%%
def random_masking(x_patch, mask_ratio=0.3, mask=None):
    if mask is not None:
        x_masked = x_patch * (1-mask.unsqueeze(-1))
        return x_masked
    else:
        bs, L, nvars, D = x_patch.shape
        x = x_patch.clone()
        
        len_keep = int(L * (1 - mask_ratio))
            
        noise = torch.rand(bs, L, nvars,device=x_patch.device)  # noise in [0, 1], bs x L x nvars
            
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
    
        # removed x
        x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=x_patch.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask

#%%
def permute_patch(x):
    b, n_patch, n_var, patch_len = x.shape
    # Generate random indices for each row
    # torch.randint generates random integers from 0 to n_cols for each element of the tensor
    random_indices = torch.randint(low=0, high=n_var, size=(n_patch, n_var))
    
    # Create a mask to ensure all elements in a row are unique
    # Using torch.argsort to sort the indices will ensure all values are unique per row
    sorted_indices = random_indices.argsort(dim=1)

    # Apply these indices to permute each row
    output = x[:, torch.arange(n_patch).unsqueeze(1), sorted_indices]

    return output

#%%
def flatten(x_patch):
    bs, n_patch, n_var, patch_len = x_patch.shape
    return x_patch.transpose(2, 3).reshape(bs, -1, n_var)
#%%

x = torch.randn((2, 24, 4))
x_patch, n = create_patch(x, 4)
x_patch_permute = permute_patch(x_patch)

x_masked, mask = random_masking(x_patch, 0.4)
x_shuffle_masked = random_masking(x_patch_permute, mask=mask)

xm_flatten = flatten(x_masked)
xm_shuffle_flatten = flatten(x_shuffle_masked)