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

OmniNet evalution script.

"""
import argparse
import os
import torch
import pickle
import time
import json
import numpy as np
import glob
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
from nltk.tokenize import word_tokenize
import sys
from tqdm import tqdm
from PIL import Image
from libs.utils.train_util import *
from libs.utils.cocoapi.coco import COCO
from libs.utils.vqa.vqa import VQA
from libs.utils.vqa.vqaEval import VQAEval
from libs.utils.cocoapi.eval import COCOEvalCap
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from libs.utils.bleu import compute_bleu

coco_images = 'data/coco/train_val'
caption_dir = 'data/coco'
vqa_dir = 'data/vqa'
hmdb_data_dir='data/hmdb'
hmdb_process_dir='data/hmdbprocess'
model_save_path = 'checkpoints'


dropout = 0.5
image_height = 224
image_width = 224


class ImageDataset(Dataset):
    def __init__(self, img_list,transform):
        self.img_list=img_list
        self.N=len(self.img_list)
        self.transform=transform
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img
    

def vqa_evaluate(model,batch_size):
    predictions=[]
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
    vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
    questions=[]
    img_list=[]
    ques_ids=[]
    answers=[]
    #Multiple choice answer evaluation
    
    for q in vqa.questions['questions']:
        img_list.append(os.path.join(coco_images, '%012d.jpg' % (q['image_id'])))
        questions.append(q['question'])
        ques_ids.append(q['question_id'])
        answer = vqa.loadQA(q['question_id'])
        answers.append(answer[0]['multiple_choice_answer'])
    answers=np.array(answers)
    #Validation transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
    
    val_dataset = ImageDataset(img_list,val_tfms)
    val_dl= DataLoader(val_dataset, num_workers=8, batch_size=batch_size,
                                     drop_last=False,pin_memory=True,shuffle=False)
    counter=0
    result_json=[]
    
    for b in tqdm(val_dl):
            imgs = b.cuda(0)
            ques=questions[counter:counter+imgs.shape[0]]
            preds,_,_ = r.vqa(model, imgs, ques, mode='predict',return_str_preds=True,num_steps=1)
            preds=preds.reshape([-1]).cpu().numpy()
            for p in preds:
                result_json.append({'question_id':ques_ids[counter],'answer':id_to_ans[p]})
                counter+=1
    with open('results/vqa_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)
    #Evaluate the predictions 
    predictions=[]
    vqaRes=vqa.loadRes('results/vqa_prediction.json',vqa_val_ques)
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    with open('results/vqa_prediction.json', 'r') as f:
        json_ans = json.load(f)
    for j in json_ans:
        predictions.append(j['answer'])
    predictions=np.array(predictions)
    print(np.sum(predictions==answers)/predictions.shape[0])
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()
    print('\n\nVQA Evaluation results')
    print('-'*50)

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")
   
        
        
def coco_evaluate(model,batch_size):
    predictions=[]
    val_ann_file=os.path.join(caption_dir,'captions_val2017.json')
    coco = COCO(val_ann_file)
    img_ids=coco.getImgIds()
    img_list=[]
    for i in img_ids:
        img_list.append(os.path.join(coco_images, '%012d.jpg' % (i)))
    #Validation transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
    
    val_dataset = ImageDataset(img_list,val_tfms)
    val_dl= DataLoader(val_dataset, num_workers=8, batch_size=batch_size,
                                     drop_last=False,pin_memory=True,shuffle=False)
    counter=0
    result_json=[]
    for b in tqdm(val_dl):
            imgs = b.cuda(0)
            preds,_,_ = r.image_caption(model, imgs, mode='predict',return_str_preds=True)
            for p in preds:
                result_json.append({'image_id':img_ids[counter],'caption':p})
                counter+=1
                
    with open('results/caption_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)
    #Evaluate the predictions 
    cocoRes=coco.loadRes('results/caption_prediction.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    
    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    print('\n\nCOCO Evaluation results')
    print('-'*50)
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
                
def hmdb_evaluate(model,batch_size):
    _,test_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=8,batch_size=batch_size,
                                         test_batch_size=batch_size,
                                           clip_len=16)
    correct=0
    total=0
    for b in tqdm(test_dl):
            vid,labels = b
            vid = vid.cuda(device=0)
            preds,_,_ = r.hmdb(model, vid,mode='predict',return_str_preds=True,num_steps=1)
            preds=preds.reshape([-1]).cpu().numpy()
            labels=labels.reshape([-1]).cpu().numpy()
            correct+=np.sum(preds==labels)
            total+=labels.shape[0]
    accuracy=(correct/total)*100
    print('\n\nHMDB Evaluation results')
    print('-'*50)
    print('Accuracy: %f%%'%(accuracy))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmniNet Evaluation script.')
    parser.add_argument('task', help='Task for which to evaluate for.')
    parser.add_argument('model_file', help='Model file to evaluate with.')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    args = parser.parse_args()
    torch.manual_seed(47)
    task=args.task
    batch_size=int(args.batch_size)
    model_file=args.model_file
    
    model = omninet.OmniNet(gpu_id=0)
    model.restore_file(model_file)
   
    model=model.to(0)
    model=model.eval()
    if task=='caption':
        coco_evaluate(model,batch_size)
    elif task=='vqa':
        vqa_evaluate(model,batch_size)
    elif task=='hmdb':
        hmdb_evaluate(model, batch_size)
    else:
        print('Invalid task provided')
    