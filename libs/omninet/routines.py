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

OmniNet routines for various tasks described in the paper.

"""

import torch
import torch.nn as nn
import numpy as np



def hmdb(omninet,videos,targets=None,mode='train',return_str_preds=False,num_steps=1):
    batch_size = videos.shape[0]
    #Reset OmniNet state
    omninet.reset(batch_size)
    #Encode video files
    omninet.encode_videos(videos,domain='IMAGE')
    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('HMDB',targets=targets)
    elif mode =='predict':
        predictions = omninet.decode_greedy('HMDB', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc = calc_nll_loss_and_acc(predictions,targets)
    else:
        loss,acc=None,None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc

def vqa(omninet,images,questions,targets=None,mode='train',return_str_preds=False,num_steps=1):
    # Reset the cnp memory
    batch_size = images.shape[0]
    omninet.reset(batch_size)
    # Encode and store images
    omninet.encode_images(images,domain='IMAGE')
    # Encode and store questions
    omninet.encode_englishtexts(questions)
    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('VQA', targets=targets)
    elif mode=='predict':
        predictions = omninet.decode_greedy('VQA', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc = calc_nll_loss_and_acc(predictions,targets)
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc

def image_caption(omninet,images,targets=None,mode='train',return_str_preds=False,num_steps=100):
    # Reset the cnp memory
    batch_size=images.shape[0]
    omninet.reset(batch_size)
    # Encode and store images
    omninet.encode_images(images,domain='IMAGE')
    #Calculate pad mask using the inbuilt tokenizer
    if targets is not None:
        targets,target_pad_mask = omninet.english_language_perph.tokenize_sentences(targets)
    if mode  in ['train','val']:
        predictions = omninet.decode_from_targets('IMAGE_CAPTION', targets=targets,target_pad_mask=target_pad_mask)
    elif mode=='predict':
        predictions = omninet.decode_greedy('IMAGE_CAPTION', num_steps=num_steps)
    # Calculate loss if targets is provided
    loss = None
    acc = None
    if targets is not None:
        pad_token = omninet.english_language_perph.id_PAD
        loss,acc=calc_nll_loss_and_acc(predictions,targets,pad_id=pad_token,target_pad_mask=target_pad_mask)
    else:
        loss, acc= None, None

    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
        predictions = omninet.english_language_perph.decode_tokens(predictions)
    return predictions,loss,acc

def penn(omninet,texts,pad_id=None,targets=None,target_pad_mask=None,mode='train',return_str_preds=False,num_steps=100):
    batch_size=len(texts)
    omninet.reset(batch_size)
    #Store the sentences
    omninet.encode_englishtexts(texts, domain='ENGLISH')
    #Get the tokenized targets
    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('PENN', targets=targets,target_pad_mask=target_pad_mask)
    elif mode=='predict':
        predictions = omninet.decode_greedy('PENN', num_steps=num_steps)
    #Calculate loss if targets is provided
    loss=None
    acc=None
    if targets is not None:
        loss, acc = calc_nll_loss_and_acc(predictions,targets,pad_id=pad_id,target_pad_mask=target_pad_mask)
    else:
        loss, acc = None, None
    if return_str_preds:
    # Return predictions in detokenized string format
        predictions=predictions.argmax(-1)
    return predictions,loss,acc
 

def calc_nll_loss_and_acc(predictions, targets, pad_id=None, target_pad_mask=None):
    #Calculate loss
    pr = torch.reshape(predictions, [-1, predictions.shape[2]])
    if pad_id is not None:
        loss_fn = nn.NLLLoss(ignore_index=pad_id)
    else:
        loss_fn = nn.NLLLoss()
    targets = torch.reshape(targets, [-1])
    loss = loss_fn(pr, targets)
    #Calculate accuracy
    preds=predictions.argmax(-1)
    preds=torch.reshape(preds,[-1])
    if target_pad_mask is not None:
        target_pad_mask=torch.reshape(target_pad_mask,[-1])
        preds=preds+(target_pad_mask*1000000).to(dtype=torch.long)
        acc=(torch.sum(targets==preds).sum().cpu().numpy()/(targets.shape[0]-target_pad_mask.sum().cpu().numpy()))*100
    else:
        acc=(torch.sum(targets==preds).sum().cpu().numpy()/(targets.shape[0]))*100
    return loss, acc