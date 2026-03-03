import torch
from torch_scatter import scatter

# atomic energy calculated by qo2mol datatset. Reference: https://arxiv.org/abs/2410.19316
# used for stable training
atomref_list = torch.tensor([0, -3.73194827e+02, -2.76486389e-10,  1.27329258e-10,
  3.05590220e-10,  2.61934474e-10, -2.38999452e+04, -3.43228948e+04,
 -4.71770868e+04, -6.25983851e+04,  2.18278728e-11,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.14168133e+05,
 -2.49789503e+05, -2.88688114e+05,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.61511430e+06,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00, -1.86853102e+05,])

def atomref(energy, batch, atomic_numbers):
    """
    Energy Processor: get the predicted energy from Suiren model, reverse it to original value space.
    energy shape:      [b, 1] or [b]
    batch shape:       [natoms]
    atomic_numbers:    [natoms]
    """
    cur_atomref_list = atomref_list.to(device=atomic_numbers.device, dtype=energy.dtype)
    return energy - scatter(cur_atomref_list[atomic_numbers], index=batch, dim_size=energy.shape[0])
  
def reverse_atomref(pred_energy, batch, atomic_numbers):
    """
    Energy Processor: Subtract the specified energy of each atom from the original energy.
    Precision level: B3LYP/def2-SVP
    energy shape:      [b, 1] or [b]
    batch shape:       [natoms]
    atomic_numbers:    [natoms]
    """
    cur_atomref_list = atomref_list.to(device=atomic_numbers.device, dtype=pred_energy.dtype)
    return pred_energy + scatter(cur_atomref_list[atomic_numbers], index=batch, dim_size=pred_energy.shape[0])



