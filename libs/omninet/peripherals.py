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

OmniNet standard input peripherals

"""

from bpemb import BPEmb
from torch.nn.functional import relu, log_softmax
from .base_models.resnet import resnet50, resnet152
from .util import *

class base_peripheral(nn.Module):
    """
        The base standard non recursive perpheral
        All base peripherals must implement the following functions:
            __init__()
            run_cycle()

    """
    def __init__(self):
        super(base_peripheral,self).__init__()


##############################################################
##############################################################
# Definition of standard peripherals for most common tasks   #
##############################################################


class ImageInputPeripheral(base_peripheral):
    def __init__(self,output_dim,dropout=0,weights_preload=True,freeze_layers=True):
        self.feature_dim=2048  
        super(ImageInputPeripheral,self).__init__()
        self.image_model=resnet152(pretrained=weights_preload)
        if freeze_layers:
            self.image_model=self.image_model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.image_model.train=self.empty_fun
            self.image_model.eval=self.empty_fun
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(self.feature_dim,output_dim)

    def encode(self,image_tensor):
        shape=image_tensor.shape
        if len(shape)==5:
            t_dim=image_tensor.shape[1]
            image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4]))    
        batch_size=image_tensor.shape[0]
        image_enc=self.image_model(image_tensor)
        enc_reshape=torch.reshape(image_enc,[batch_size,self.feature_dim,-1])
        enc_transposed=torch.transpose(enc_reshape,1,2)
        drp_enc=self.enc_dropout(enc_transposed)
        output_enc=self.output_fc(drp_enc)
        if len(shape)==5:
            output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2]))
        else:
            output_enc=output_enc.unsqueeze(1)
        return output_enc

    def empty_fun(self,mode):
        pass



class LanguagePeripheral(base_peripheral):
    def __init__(self,output_dim,vocab_size=10000,embed_dim=50,lang='en',embedding_preload=True,gpu_id=-1,dropout=0):
        super(LanguagePeripheral,self).__init__()
        self.gpu_id=gpu_id
        self.pad_char = vocab_size
        self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True)
        # Add an extra padding character
        self.embed_layer=nn.Embedding(vocab_size+1,embed_dim,padding_idx=self.pad_char)
        if(embedding_preload==True):
            self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
            print("Loading pretrained word embeddings.")
        self.enc_dropout = nn.Dropout(dropout)
        self.output=nn.Linear(embed_dim,output_dim)
        
    def forward(self,tokens):        
        pad_mask=tokens.eq(self.id_PAD)
        embeddings=self.embed_layer(tokens)
        embeddings=self.enc_dropout(embeddings)
        output=self.output(embeddings)
        return output.unsqueeze(2)

    def embed_sentences(self,sentences):
        # Generate the tokens using BPEmb
        tokens,pad_mask=self.tokenize_sentences(sentences)
        return self.forward(tokens),pad_mask

    def decode_tokens(self,tokens):
        if isinstance(tokens,torch.Tensor):
            tokens=tokens.cpu().numpy().astype(int).tolist()
        elif isinstance(tokens,np.ndarray):
            tokens=tokens.astype(int).tolist()
        #Filter out all tokens which have values larger than vocab_size and filter all elements after EOS
        filtered_tokens=[]
        for t in tokens:
            values=[]
            for i in t:
                if i==self.id_EOS:
                    break
                elif i<self.id_PAD:
                    values.append(i)
            filtered_tokens.append(values)
        #Remove all the padding characters in a list
        return self.bpe_encoder.decode_ids(filtered_tokens)


    def tokenize_sentences(self,sentences):
        tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)
        # Pad the tokens with the pad_char
        max_len = 0
        
        for t in tokens:
            max_len = max(max_len, len(t))
        for i in range(len(tokens)):
            tok_len = len(tokens[i])
            tokens[i].extend([self.pad_char]*(max_len-tok_len))
        tokens = torch.tensor(np.array(tokens))
        if self.gpu_id > -1:
            tokens = tokens.cuda(self.gpu_id)
        pad_mask=tokens.eq(self.id_PAD)
        return tokens,pad_mask

    @property
    def id_PAD(self):
        return self.pad_char

    @property
    def id_GO(self):
        return 1

    @property
    def id_EOS(self):
        return 2
