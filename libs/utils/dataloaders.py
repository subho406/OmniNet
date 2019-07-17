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
Authors: Subhojeet Pramanik, Priyanka Agrawal, Aman Hussain, Sayan Dutta

Dataloaders for standard datasets used in the paper

"""
import os
import torch
import pickle
import cv2
import numpy as np
import json
import tqdm
import random
from sklearn.model_selection import train_test_split
from PIL import Image
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# the following is required for the lib to work in terminal env
import matplotlib

matplotlib.use("agg", warn=False, force=True)
from .cocoapi.coco import COCO


from .vqa.vqa import VQA



class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, data_dir, output_dir, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = data_dir, output_dir
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.resize_height = 300
        self.resize_width = 300
        self.crop_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # The following three parameters are chosen as described in the paper section 4.1

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists('conf/hmdblabels.txt'):
            with open('conf/hmdblabels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels).unsqueeze(0)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True
    
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    
    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in tqdm.tqdm(os.listdir(self.root_dir)):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            split_file=os.path.join('conf/hmdb','%s_test_split1.txt'%file)
            train_files=[]
            test_files=[]
            with open(split_file,'r') as f:
                lines=f.readlines()
                for l in lines:
                    f_name,split=l.strip().split(' ')
                    if split=='1' or split=='0':
                        train_files.append(f_name)
                    elif split=='2':
                        test_files.append(f_name)
            train_dir = os.path.join(self.output_dir, 'train', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
  
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train_files:
                self.process_video(video, file, train_dir)

            for video in test_files:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')
        
        
    def normalize(self, buffer):
        buffer=buffer/255
        for i, frame in enumerate(buffer):
            frame -= np.array([[[0.485, 0.456, 0.406]]])
            frame /= np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer
    
    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] - clip_len>0 and self.split=='train':
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        else:
            time_index=0
        # Randomly select start indices in order to crop the video
        if self.split=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index=0
            width_index=0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer
    
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()


    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer


def hmdb_batchgen(data_dir,process_dir,num_workers=1,batch_size=1,test_batch_size=1,clip_len=16):
        dataset=VideoDataset(data_dir, process_dir, split='train',dataset='hmdb',clip_len=clip_len)
        test_dataset=VideoDataset(data_dir, process_dir, split='test',dataset='hmdb',clip_len=clip_len)
        dataloader=DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                   )
        itr = iter(cycle(dataloader))
        test_dl= DataLoader(test_dataset, num_workers=num_workers, batch_size=test_batch_size,
                                      drop_last=False)
        return itr,test_dl

class coco_cap_dataset(Dataset):
    def __init__(self, ann_file, image_dir,transforms=None,max_words=40):
        caption = COCO(ann_file)
        self.inputs = []
        self.outputs = []
        ann_ids = caption.getAnnIds()
        for idx, a in tqdm.tqdm(enumerate(ann_ids),'Loading MSCOCO to memory'):
            c = caption.loadAnns(a)[0]
            words = c['caption']
            if len(words.split(' '))<=max_words:
                img_file = os.path.join(image_dir, '%012d.jpg' % (c['image_id']))
                self.inputs.append(img_file)
                self.outputs.append(words)
        self.transform = transforms
        self.N=len(self.outputs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.inputs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        cap = self.outputs[idx]
        # returns the dictionary of images and captions
        return {'img': img, 'cap': cap}


def coco_cap_batchgen(caption_dir, image_dir,num_workers=1, batch_size=1):
        # transformations for the images
        train_ann=os.path.join(caption_dir,'captions_train2017.json')
        val_ann=os.path.join(caption_dir,'captions_val2017.json')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = coco_cap_dataset(train_ann, image_dir, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn=coco_collate_fn, drop_last=True,pin_memory=True)
       
        # the iterator over data loader
        itr = iter(cycle(dataloader))
        
        val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
        val_dataset = coco_cap_dataset( val_ann, image_dir, transforms=val_tfms,max_words=5000)
        val_dl= DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2),
                                     collate_fn=coco_collate_fn, drop_last=False,pin_memory=True)
        return itr, val_dl
                    

    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def coco_collate_fn(data):
    # collate function for the data loader
    collate_images = []
    collate_cap = []
    max_len = 0
    for d in data:
        collate_images.append(d['img'])
        collate_cap.append(d['cap'])
    collate_images = torch.stack(collate_images, dim=0)
    # return a dictionary of images and captions
    return {
        'img': collate_images,
        'cap': collate_cap
    }



class vqa_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        ques = self.ques[idx]
        ans = self.ans[idx]
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'ques': ques, 'ans': ans}


def vqa_batchgen(vqa_dir, image_dir, num_workers=1, batch_size=1):
        # a transformation for the images
        vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
        vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
        vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)
        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader


def vqa_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ans=[]
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ans.append((d['ans']))
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans
    }

class penn_dataset(Dataset):
    ''' Pytorch Penn Treebank Dataset '''

    def __init__(self, text_file,max_len=150):
        self.X = list()
        self.Y = list()
        with open(text_file) as f:
            # first line ignored as header
            f = f.readlines()[1:]
            for i in range(0,len(f),2):
                if len(f[i].split(' ', maxsplit=1)[1].split(' '))<max_len:
                    self.X.append(f[i])
                    self.Y.append(f[i+1])      
            assert len(self.X) == len(self.Y),\
            "mismatch in number of sentences & associated POS tags"
            self.count = len(self.X)
            del(f)
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return (self.X[idx].split(' ', maxsplit=1)[1],
                self.Y[idx].split()[1:])

    
class PennCollate:
    def __init__(self,vocab_file):
        with open(vocab_file,'r') as f:
            data=json.loads(f.read())
        self.tag_to_id=data['tag_to_id']
        self.id_to_tag=data['id_to_tag']
        
        
    def __call__(self,batch):
        pad_token=self.tag_to_id['<PAD>']
        text=[]
        tokens=[]
        max_len=0
        for b in batch:
            text.append(b[0].strip())
            tok=[self.tag_to_id[tag] for tag in b[1]]
            max_len=max(max_len,len(tok))
            tokens.append(tok)
        for i in range(len(tokens)):
            for j in range(max_len-len(tokens[i])):
                tokens[i].append(pad_token)
        tokens=torch.tensor(np.array(tokens))
        pad_mask=tokens.eq(pad_token)
        #Add padding to the tokens
        return {'text':text,'tokens':tokens,'pad_mask':pad_mask,'pad_id':pad_token}
    
def penn_dataloader(data_dir, batch_size=1, test_batch_size=1,num_workers=8,vocab_file='conf/penn_vocab.json'):
        train_file=os.path.join(data_dir,'train.txt')
        val_file=os.path.join(data_dir,'dev.txt')
        test_file=os.path.join(data_dir,'test.txt')
        collate_class=PennCollate(vocab_file)
        dataset=penn_dataset(train_file)
        val_dataset=penn_dataset(val_file)
        test_dataset=penn_dataset(test_file)
        train_dl=DataLoader(dataset,num_workers=num_workers,batch_size=batch_size,collate_fn=collate_class)
        val_dl=DataLoader(val_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        test_dl=DataLoader(test_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        train_dl=iter(cycle(train_dl))
        return train_dl,val_dl,test_dl
    




