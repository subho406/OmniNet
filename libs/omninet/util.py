#
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""
Authors: Subhojeet Pramanik

Utilities 

"""


import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
import numpy as np
import torch
from torch.autograd import Variable


def recursiveTrace(obj):
  print(type(obj))
  if hasattr(obj, 'grad_fn'):
    print(obj.grad_fn)
    recursiveTrace(obj.grad_fn)
  elif hasattr(obj, 'saved_variables'):
    print(obj.requires_grad, len(obj.saved_tensors), len(obj.saved_variables))
    [print(v) for v in obj.saved_variables]
    [recursiveTrace(v.grad_fn) for v in obj.saved_variables]

def get_subsequent_mask(shape,gpu_id):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = shape
    if gpu_id>=0:
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=gpu_id, dtype=torch.uint8), diagonal=1)
    else:
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
def get_attn_key_pad_mask(pad_mask, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return pad_mask


def get_non_pad_mask(seq,pad_mask):
    if pad_mask is None:
        return None
    else:
        return pad_mask.ne(1).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def cuda(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(x, requires_grad=grad)
  else:
    return var(x.pin_memory(), requires_grad=grad).cuda(gpu_id, non_blocking=True)


def cudavec(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(T.from_numpy(x), requires_grad=grad)
  else:
    return var(T.from_numpy(x).pin_memory(), requires_grad=grad).cuda(gpu_id, non_blocking=True)


def cudalong(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(T.from_numpy(x.astype(np.long)), requires_grad=grad)
  else:
    return var(T.from_numpy(x.astype(np.long)).pin_memory(), requires_grad=grad).cuda(gpu_id, non_blocking=True)


def θ(a, b, dimA=2, dimB=2, normBy=2):
  """Batchwise Cosine distance

  Cosine distance

  Arguments:
      a {Tensor} -- A 3D Tensor (b * m * w)
      b {Tensor} -- A 3D Tensor (b * r * w)

  Keyword Arguments:
      dimA {number} -- exponent value of the norm for `a` (default: {2})
      dimB {number} -- exponent value of the norm for `b` (default: {1})

  Returns:
      Tensor -- Batchwise cosine distance (b * r * m)
  """
  a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
  b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

  x = T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
      T.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)
  # apply_dict(locals())
  return x


def σ(input, axis=1):
  """Softmax on an axis

  Softmax on an axis

  Arguments:
      input {Tensor} -- input Tensor

  Keyword Arguments:
      axis {number} -- axis on which to take softmax on (default: {1})

  Returns:
      Tensor -- Softmax output Tensor
  """
  input_size = input.size()

  trans_input = input.transpose(axis, len(input_size) - 1)
  trans_size = trans_input.size()

  input_2d = trans_input.contiguous().view(-1, trans_size[-1])
  if '0.3' in T.__version__:
    soft_max_2d = F.softmax(input_2d, -1)
  else:
    soft_max_2d = F.softmax(input_2d)
  soft_max_nd = soft_max_2d.view(*trans_size)
  return soft_max_nd.transpose(axis, len(input_size) - 1)

δ = 1e-6


def register_nan_checks(model):
  def check_grad(module, grad_input, grad_output):
    # print(module) you can add this to see that the hook is called
    # print('hook called for ' + str(type(module)))
    if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
      print('NaN gradient in grad_input ' + type(module).__name__)

  model.apply(lambda module: module.register_backward_hook(check_grad))


def apply_dict(dic):
  for k, v in dic.items():
    apply_var(v, k)
    if isinstance(v, nn.Module):
      key_list = [a for a in dir(v) if not a.startswith('__')]
      for key in key_list:
        apply_var(getattr(v, key), key)
      for pk, pv in v._parameters.items():
        apply_var(pv, pk)


def apply_var(v, k):
  if isinstance(v, Variable) and v.requires_grad:
    v.register_hook(check_nan_gradient(k))


def check_nan_gradient(name=''):
  def f(tensor):
    if np.isnan(T.mean(tensor).data.cpu().numpy()):
      print('\nnan gradient of {} :'.format(name))
      # print(tensor)
      # assert 0, 'nan gradient'
      return tensor
  return f

def ptr(tensor):
  if T.is_tensor(tensor):
    return tensor.storage().data_ptr()
  elif hasattr(tensor, 'data'):
    return tensor.clone().data.storage().data_ptr()
  else:
    return tensor

# TODO: EWW change this shit
def ensure_gpu(tensor, gpu_id):
  if "cuda" in str(type(tensor)) and gpu_id != -1:
    return tensor.cuda(gpu_id)
  elif "cuda" in str(type(tensor)):
    return tensor.cpu()
  elif "Tensor" in str(type(tensor)) and gpu_id != -1:
    return tensor.cuda(gpu_id)
  elif "Tensor" in str(type(tensor)):
    return tensor
  elif type(tensor) is np.ndarray:
    return cudavec(tensor, gpu_id=gpu_id).data
  else:
    return tensor


def print_gradient(x, name):
  s = "Gradient of " + name + " ----------------------------------"
  x.register_hook(lambda y: print(s, y.squeeze()))

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps,n_current_steps=0,init_lr=0.1,max_lr=None):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = init_lr
        self.hidden_size=d_model
        self.max_lr=max_lr
    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        lw=np.min([1.0, self.n_current_steps / self.n_warmup_steps])
        rsqrt_decay=np.power((np.max([self.n_current_steps, self.n_warmup_steps])),-0.5)
        return lw*rsqrt_decay

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        if self.max_lr is not None:
            lr = min(self.init_lr * self._get_lr_scale(),self.max_lr)
        else:
            lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
