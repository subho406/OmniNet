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

OmniNet training script. 

"""
import argparse
import os
import torch
import time
import glob
import numpy as np
import libs.omninet as omninet
from libs.utils import dataloaders as dl
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import libs.omninet.routines as r
from libs.omninet.util import ScheduledOptim
from torch.optim.adam import Adam
import random
import sys
from tqdm import tqdm
from libs.utils.train_util import *


coco_images = 'data/coco/train_val'
caption_dir = 'data/coco'
vqa_dir = 'data/vqa'
model_save_path = 'checkpoints'
hmdb_data_dir='data/hmdb'
hmdb_process_dir='data/hmdbprocess'
penn_data_dir='data/penn'


def train(shared_model, task, batch_size, train_steps, gpu_id, start,  restore,  counter, barrier=None, save_interval=None,
          eval_interval=None, log=True):
    log_dir = 'logs/%s' % task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (log == True):
        summary_writer = SummaryWriter(log_dir)
    # Create local model
     
    torch.manual_seed(int(random.random() * 1000))
    if gpu_id>0:
        model = omninet.OmniNet(gpu_id=gpu_id)
        model=model.cuda(gpu_id)
    else:
        #For GPU 0, use the shared model always
        model=shared_model

    if task == 'caption':
        DL,val_dl = dl.coco_cap_batchgen(caption_dir=caption_dir, image_dir=coco_images,
                                  num_workers=8,
                                  batch_size=batch_size)
        
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=0.02)
    elif task == 'vqa':
        DL,val_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=8, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'hmdb':
        DL,val_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=8,batch_size=batch_size,
                                   test_batch_size=int(batch_size/4),
                                   clip_len=16)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'penn':
        DL,val_dl,test_dl=dl.penn_dataloader(penn_data_dir,batch_size=batch_size,
                                             test_batch_size=int(batch_size/2),num_workers=4,vocab_file='conf/penn_vocab.json')
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=0.02)
        
    model=model.train()

    for i in range(start, train_steps):
        model.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()
        if task == 'caption':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    if gpu_id>=0:
                        imgs=imgs.cuda(device=gpu_id)
                    captions = b['cap']
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    _,loss,acc = r.image_caption(model, imgs, targets=captions, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, COCO validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
            captions = batch['cap']
            _, loss,acc = r.image_caption(model, imgs, targets=captions)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, Caption Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'vqa':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    answers=b['ans']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        answers=answers.cuda(device=gpu_id)
                    questions= b['ques']
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    pred, loss,acc = r.vqa(model, imgs, questions,targets=answers, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, VQA validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            batch = next(DL)
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
                answers = batch['ans'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
                answers = batch['ans']
            questions = batch['ques']                       
            _, loss,acc = r.vqa(model, imgs, questions, targets=answers)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, VQA Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
        elif task=='hmdb':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    vid,labels = b
                    if gpu_id >= 0:
                        vid = vid.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)    
                    _, loss,acc = r.hmdb(model, vid,targets=labels, mode='val')
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, HMDB validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            vid,labels = next(DL)
            if gpu_id >= 0:
                vid = vid.cuda(device=gpu_id)
                labels = labels.cuda(device=gpu_id)    
            _, loss,acc = r.hmdb(model, vid,targets=labels,return_str_preds=True)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, HMDB Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'penn':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(test_dl):
                    en = b['text']
                    targets = b['tokens']
                    pad_id=b['pad_id']
                    pad_mask=b['pad_mask']
                    if gpu_id>=0:
                        targets=targets.to(gpu_id)
                        pad_mask=pad_mask.to(gpu_id)
                    _,loss,acc = r.penn(model, en, target_pad_mask=pad_mask,
                                        pad_id=pad_id,targets=targets, mode='val',return_str_preds=True)
                    loss=loss.detach()
                    val_loss += float(loss.cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, PENN validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            en = batch['text']
            targets = batch['tokens']
            pad_id=batch['pad_id']
            pad_mask=batch['pad_mask']
            if gpu_id>=0:
                targets=targets.to(gpu_id)
                pad_mask=pad_mask.to(gpu_id)
            _, loss,acc = r.penn(model, en, pad_id=pad_id, targets=targets,target_pad_mask=pad_mask)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, PENN Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        # End Calculate loss
        if gpu_id>0:
            ensure_shared_grads(model, shared_model, gpu_id)
        optimizer.step()
        # Save model
        if (save_interval != None and (i + 1) % save_interval == 0):
            shared_model.save(model_save_path, step)
        sys.stdout.flush()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmniNet training script.')
    parser.add_argument('n_iters', help='Number of iterations to train.')
    parser.add_argument('tasks', help='List of tasks seperated by comma.')
    parser.add_argument('batch_sizes', help='List of batch size for each task seperated by comma')
    parser.add_argument('--n_jobs', default=1, help='Number of asynchronous jobs to run for each task.')
    parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
    parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
    parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
    parser.add_argument('--eval_interval', help='Interval after which to evaluate on the test/val set.', default=1000)
    
    args = parser.parse_args()
    torch.manual_seed(47)
    mp.set_start_method('spawn',force=True)
    n_iters = int(args.n_iters)
    n_jobs = int(args.n_jobs)
    tasks=args.tasks
    batch_sizes=args.batch_sizes
    save_interval = int(int(args.save_interval) / n_jobs)
    eval_interval = int(int(args.eval_interval) / n_jobs)

    if args.restore_last == True:
        ckpts = glob.glob(os.path.join(model_save_path, '*'))
        iters = [int(os.path.basename(c)) for c in ckpts]
        if len(iters) != 0:
            restore = max(iters)
        else:
            restore = 0
    else:
        restore = int(args.restore)
    tasks=tasks.split(',')
    tasks=[t.strip() for t in tasks]
    batch_sizes=batch_sizes.split(',')
    batch_sizes=[int(b.strip()) for b in batch_sizes]

    if len(tasks)!=len(batch_sizes):
        raise Exception('Number of tasks provided does not match the number of batch sizes provided.')

    n_gpus = int(args.n_gpus)
    n_tasks = len(tasks) * n_jobs

    shared_model = omninet.OmniNet(gpu_id=0)
    if restore != -1:
        shared_model.restore(model_save_path, restore)
    else:
        restore=0
        
    shared_model=shared_model.to(0)
    shared_model.share_memory()
    counters = [Counter(restore) for i in range(len(tasks))]
    barrier = mp.Barrier(n_tasks)
    start = int(restore / n_jobs)
    # Declare training processes for multi-gpu hogwild training
    processes = []
    for i in range(n_tasks):
        #If more than one GPU is used, use first GPU only for model sharing
        if n_gpus>1:
            gpu_id=i%n_gpus
        else:
            gpu_id=0
        process = mp.Process(target=train, args=(shared_model, tasks[i % len(tasks)], batch_sizes[i % len(tasks)],
                                                 int(n_iters / n_jobs),
                                                 gpu_id, start, restore, counters[i % len(tasks)], barrier,
                                                 (save_interval if i == 0 else None),
                                                 (eval_interval if i < len(tasks) else None),
                                                 (True if i < len(tasks) else False)))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
