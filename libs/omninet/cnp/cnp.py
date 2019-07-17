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

OmniNet Central Neural Processor implementation

"""

from .Layers import *
from ..util import *
from torch.nn.functional import log_softmax, softmax


class CNP(nn.Module):

    def __init__(self,tasks,conf=None,domains=['EMPTY'],gpu_id=-1):
        super(CNP, self).__init__()
        default_conf=self.__defaultconf__()
        if(conf!=None):
            for k in conf.keys():
                if k not in conf:
                    raise ValueError("The provided configuration does not contain %s"%k)
        else:
            conf=default_conf
        #Load the Confurigation
        self.gpu_id=gpu_id
        self.input_dim=conf['input_dim']
        self.control_dim=conf['control_dim']
        self.output_dim=conf['output_dim']
        self.spatial_dim=conf['spatial_dim']
        self.temporal_dim=conf['temporal_dim']
        self.temporal_n_layers=conf['temporal_n_layers']
        self.temporal_n_heads=conf['temporal_n_heads']
        self.temporal_d_k=conf['temporal_d_k']
        self.temporal_d_v=conf['temporal_d_v']
        self.temporal_hidden_dim=conf['temporal_hidden_dim']
        self.decoder_dim=conf['decoder_dim']
        self.decoder_n_layers=conf['decoder_n_layers']
        self.decoder_n_heads=conf['decoder_n_heads']
        self.decoder_d_k=conf['decoder_d_k']
        self.decoder_d_v=conf['decoder_d_v']
        self.decoder_hidden_dim=conf['decoder_hidden_dim']
        self.max_seq_len=conf['max_seq_len']
        self.output_embedding_dim=conf['output_embedding_dim']
        self.dropout=conf['dropout']
        self.batch_size=-1 #Uninitilized CNP memory

        # Prepare the task lists and various output classifiers and embeddings
        if isinstance(tasks, dict):
            self.task_clflen = list(tasks.values())
            self.task_dict = {t: i for i, t in enumerate(tasks.keys())}
        else:
            raise ValueError('Tasks must be of type dict containing the tasks and output classifier dimension')

        self.output_clfs = nn.ModuleList([nn.Linear(self.output_dim, t) for t in self.task_clflen])
        #Use one extra to define padding
        self.output_embs = nn.ModuleList([nn.Embedding(t+1,self.output_embedding_dim,padding_idx=t) for t in self.task_clflen])

        #Initialize the various sublayers of the CNP
        control_states=domains+list(tasks.keys())
        self.control_peripheral=ControlPeripheral(self.control_dim,control_states,gpu_id=gpu_id)
        self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len,self.temporal_n_layers,
                                                     self.temporal_n_heads,self.temporal_d_k,self.temporal_d_v,
                                                    self.temporal_dim,self.temporal_hidden_dim,dropout=self.dropout,
                                                     gpu_id=self.gpu_id)
        self.decoder=Decoder(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        #Initialize the various CNP caches as empty
        self.spatial_cache=None
        self.temporal_cache=None
        self.decoder_cache=None
        self.temporal_spatial_link=[]
        self.pad_cache=None    #Used to store the padding values so that it can be used later in enc dec attn


        #Various projection layers
        self.spatial_pool=nn.AdaptiveAvgPool1d(1)
        self.inpcont_input_proj=nn.Linear(self.input_dim+self.control_dim,self.input_dim)
        self.input_spatial_proj=nn.Linear(self.input_dim,self.spatial_dim)
        self.input_temporal_proj=nn.Linear(self.input_dim,self.temporal_dim)
        self.emb_decoder_proj=nn.Linear(self.output_embedding_dim,self.decoder_dim)
        self.cont_decoder_proj=nn.Linear(self.control_dim,self.decoder_dim)
        
        #freeze layers

        
        
    def decode(self,task,targets=None,num_steps=100,recurrent_steps=1,pad_mask=None,beam_width=1):
        if targets is not None:
            b,t=targets.shape
            #Use teacher forcing to generate predictions. the graph is kept in memory during this operation.
            if (len(targets.shape) != 2 or targets.shape[0] != self.batch_size):
                raise ValueError(
                    "Target tensor must be of shape (batch_size,length of sequence).")
            if task not in self.task_dict.keys():
                raise ValueError('Invalid task %s'%task)
            dec_inputs=self.output_embs[self.task_dict[task]](targets)
            dec_inputs=self.emb_decoder_proj(dec_inputs)
            control=self.control_peripheral(task,(self.batch_size))
            control=control.unsqueeze(1)
            control=self.cont_decoder_proj(control)
            dec_inputs=torch.cat([control,dec_inputs],1)
            # Get output from decoder
            #Increase the length of the pad_mask to match the size after adding the control vector
            if pad_mask is not None:
                pad_extra=torch.zeros((b,1),device=self.gpu_id,dtype=pad_mask.dtype)
                pad_mask=torch.cat([pad_extra,pad_mask],1)
            logits,=self.decoder(dec_inputs,self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                 self.pad_cache,
                                 recurrent_steps=recurrent_steps,pad_mask=pad_mask)
            #Predict using the task specific classfier
            predictions=self.output_clfs[self.task_dict[task]](logits)
            predictions=predictions[:,0:t,:]
            return log_softmax(predictions,dim=2)
        else:
            control = self.control_peripheral(task, (self.batch_size))
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)
            dec_inputs=control
            
            for i in range(num_steps-1):
                logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                       self.pad_cache,
                                       recurrent_steps=recurrent_steps)
                prediction = self.output_clfs[self.task_dict[task]](logits)
                prediction=prediction[:,-1,:].unsqueeze(1)
                prediction=log_softmax(prediction,dim=2).argmax(-1)
                prediction=self.output_embs[self.task_dict[task]](prediction)
                prediction = self.emb_decoder_proj(prediction).detach()
                if beam_width>1:
                    p=torch.topk(softmax(prediction),beam_width)
                    
                dec_inputs=torch.cat([dec_inputs,prediction],1)
            logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,                                     self.pad_cache,recurrent_steps=recurrent_steps)
            predictions = self.output_clfs[self.task_dict[task]](logits)
            return log_softmax(predictions,dim=2)

        

    def encode(self,input,pad_mask=None,domain='EMPTY',recurrent_steps=1):
        if (len(input.shape)!=4):
            raise Exception('Invalid input dimensions.')
        b,t,s,f=list(input.size())
        self.temporal_spatial_link.append((t,s))
        if b != self.batch_size:
            raise Exception('Input batch size does not match.')
        #Spatial encode. Spatial encodes encodes both spatial and time dimension features together
        control_vecs = self.control_peripheral(domain, (b, t, s))
        input = torch.cat([input, control_vecs], 3)
        input=self.inpcont_input_proj(input)
        #Project the spatial data, into the query dimension and add it to the existing cache
        if s>1:
            spatial_f=torch.reshape(input,[b,t*s,f])
            spatial_f=self.input_spatial_proj(spatial_f)
            if self.spatial_cache is None:
                self.spatial_cache=spatial_f
            else:
                self.spatial_cache=torch.cat([self.spatial_cache,spatial_f],1)

        #Feed the time features. First AVGPool the spatial features.
        temp_data=input.transpose(2,3).reshape(b*t,f,s)
        temp_data=self.spatial_pool(temp_data).reshape(b,t,f)
        temp_data=self.input_temporal_proj(temp_data)
        #Create a control state and concat with the temporal data
        #Add data to temporal cache
        temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)
       
        if self.temporal_cache is None:
            self.temporal_cache=temp_data
        else:
            self.temporal_cache=torch.cat([self.temporal_cache,temp_data],1)

        #Add pad data to pad cache
        if pad_mask is None:
            pad_mask=torch.zeros((b,t),device=self.gpu_id,dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache=pad_mask
        else:
            self.pad_cache=torch.cat([self.pad_cache,pad_mask],1)
            
    def clear_spatial_cache(self):
        self.spatial_cache=None

    def clear_temporal_cache(self):
        self.temporal_raw_cache=None
        self.temporal_cache=None

    def reset(self,batch_size=1):
        self.attn_scores=[]
        self.batch_size=batch_size
        self.temporal_spatial_link=[]
        self.pad_cache=None
        self.clear_spatial_cache()
        self.clear_temporal_cache()
    
    @staticmethod
    def __defaultconf__():
        conf={
            'input_dim':128,
            'control_dim':32,
            'output_dim':128,
            'spatial_dim':128,
            'temporal_dim':512,
            'temporal_n_layers':6,
            'temporal_n_heads':8,
            'temporal_d_k':64,
            'temporal_d_v':64,
            'temporal_hidden_dim':2048,
            'decoder_dim':512,
            'decoder_n_layers':6,
            'decoder_n_heads':8,
            'decoder_d_k':64,
            'decoder_d_v':64,
            'decoder_hidden_dim':2048,
            'max_seq_len':1000,
            'output_embedding_dim':300,
            'dropout':0.1
        }
        return conf

class ControlPeripheral(nn.Module):
    """
        A special peripheral used to help the CNP identify the data domain or specify the context of
        the current operation.

    """
    def __init__(self, control_dim, control_states, gpu_id=-1):
        """
            Accepts as input control states as list of string. The control states are sorted before id's
            are assigned
        """
        super(ControlPeripheral, self).__init__()
        self.control_dim = control_dim
        self.gpu_id = gpu_id
        self.control_dict = {}
        for i, c in enumerate(control_states):
            self.control_dict[c] = i
        self.control_embeddings=nn.Embedding(len(control_states)+1,self.control_dim)

    def forward(self, control_state, shape=()):
        if self.gpu_id>=0:
            control_ids = torch.ones(shape, dtype=torch.long,device=self.gpu_id)*self.control_dict[control_state]
        else:
            control_ids = torch.ones(shape, dtype=torch.long)*self.control_dict[control_state]
        return self.control_embeddings(control_ids)

class TemporalCacheEncoder(nn.Module):
    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,gpu_id=-1):

        super().__init__()

        n_position = len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.gpu_id=gpu_id

    def forward(self, src_seq, return_attns=False,recurrent_steps=1, pad_mask=None):

        enc_slf_attn_list = []
        b,t,_=src_seq.shape

        if self.gpu_id >= 0:
            src_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            src_pos = torch.arange(1, t + 1).repeat(b, 1)
        # -- Forward
        enc_output = src_seq + self.position_enc(src_pos)
        enc_output = self.dropout_emb(enc_output)
        if pad_mask is not None:
            slf_attn_mask=get_attn_key_pad_mask(pad_mask,src_seq)
        else:
            slf_attn_mask=None
        non_pad_mask=get_non_pad_mask(src_seq,pad_mask)
        for i in range(recurrent_steps):
            for enc_layer in self.layer_stack:
                enc_output, enc_slf_attn = enc_layer(
                    enc_output,non_pad_mask,slf_attn_mask=slf_attn_mask)

                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, temporal_dim, spatial_dim,output_dim,dropout=0.1,gpu_id=-1):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, temporal_dim,
                         spatial_dim,dropout=dropout,gpu_id=gpu_id)
            for _ in range(n_layers)])
        self.output_fc=nn.Linear(d_model,output_dim)
        self.gpu_id=gpu_id

    def forward(self, dec_inputs, spatial_cache, temporal_cache,temporal_spatial_link,
                pad_cache,
                pad_mask=None,return_attns=False,recurrent_steps=1):

        # -- Forward
        b,t,_=dec_inputs.shape
        if self.gpu_id >= 0:
            dec_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            dec_pos = torch.arange(1, t + 1).repeat(b, 1)
        dec_outputs = dec_inputs + self.position_enc(dec_pos)
        slf_attn_mask_subseq=get_subsequent_mask((b,t),self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad=get_attn_key_pad_mask(pad_mask,dec_inputs)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask=slf_attn_mask_subseq
        #Run all the layers of the decoder building the prediction graph
        dec_enc_attn_mask=get_attn_key_pad_mask(pad_cache, dec_inputs)
        non_pad_mask=get_non_pad_mask(dec_inputs,pad_mask)
        for i in range(recurrent_steps):
            for dec_layer in self.layer_stack:
                dec_outputs, attns = dec_layer(dec_outputs,temporal_cache, spatial_cache,temporal_spatial_link,
                                               non_pad_mask,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_mask)
        dec_outputs=self.output_fc(dec_outputs)
        if return_attns:
            return dec_outputs,attns
        return dec_outputs,


