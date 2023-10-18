
import torch
import torch.nn as nn
import math
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from spacy.lang.en import English
from spacy.lang.fr import French
import re
import random
import sys

## Defining all the transformer classes



# class AttentionHead(nn.Module):
#     # enc_dev signifies if its a encoder-decoder attention head
#     def __init__(self,input_size,query_size,value_size,self_regress=False,enc_dec=False):
#         super().__init__()
#         self.wq=nn.Linear(input_size,query_size,bias=False) # W_q matrix
#         self.wk=nn.Linear(input_size,query_size,bias=False) # W_k matrix
#         self.wv=nn.Linear(input_size,value_size,bias=False) # W_v matrix
#         self.ec=enc_dec # indicates whether this attention head is doing encoder-decoder attention
#         self.self_regress=self_regress


#     # computes the final vectors of each token
#     # N -> Batch Size
#     # L -> Sequence Lengtj
#     # Q -> (N,L,eq)
#     # K -> (N,L,ek)
#     # V -> (N,L,ev)
#     # mask -> (N,L,L)
#     # out -> (N,L,ev)
#     def SelfAttention(self,Q,K,V,mask):
#         key_size=K.shape[-1]
#         out=torch.matmul(Q,torch.transpose(K,1,2))
#         # out=torch.div(out,math.sqrt(key_size))
#         sft=nn.Softmax(dim=2)
#         attention_weights=sft(torch.div(torch.add(out,mask),math.sqrt(key_size)))
#         out=torch.matmul(attention_weights,V)
#         return out




#     # padding mask given in the form of [0s and 1s] 0-pay attention 1-donot pay attention
#     # padding mask -> (N,L)
#     # input -> (N,L,input_size)
#     # self_regress: Boolean
#     def forward(self,input,padding_mask,K_inp=None,V_inp=None):

#         if not self.ec:
#             K_inp=input
#             V_inp=input
#         # calculating the Q,K,V matrices
#         Q=self.wq(input)
#         K=self.wk(K_inp)
#         V=self.wv(V_inp)

#         # making the attention mask
#         batch_size=input.shape[0]
#         seqlen=input.shape[1]
#         mask=torch.unsqueeze(padding_mask,1).repeat(1,input.shape[1],1)*float('-inf') # padding mask
#         mask=torch.nan_to_num(mask,nan=0,neginf=float('-inf'))
#         if self.self_regress:
#             # self-regress mask
#             selfRegressMask=torch.triu(torch.ones(batch_size,seqlen, seqlen) * float('-inf'), diagonal=1)
#             mask=torch.add(mask,selfRegressMask)

#         # computing self attention
#         out=self.SelfAttention(Q,K,V,mask)
#         return out,Q,K,V

# class Multi_HeadAttention(nn.Module):
#     # enc_dev signifies if its a encoder-decoder multi-head attention
#     def __init__(self,head_count,input_size,query_size,value_size,self_regress=False,enc_dec=False):
#         super().__init__()
#         self.finLinear=nn.Linear(head_count*value_size,value_size)
#         self.ec=enc_dec
#         self.heads=[]
#         for h in head_count:
#             self.heads.append(AttentionHead(input_size,query_size,value_size,self_regress,enc_dec))



#     # padding mask given in the form of [0s and 1s] 0-pay attention 1-donot pay attention
#     # padding mask -> (N,L)
#     # input -> (N,L,input_size)
#     # self_regress: Boolean
#     # returns ((N,L,ev),list of ks,list of vs)
#     def forward(self,input,padding_mask,K_inp=None,V_inp=None):
#         out_matrices=[]
#         # if return_k_v:
#         #     ks=[]
#         #     vs=[]
#         for head_id,head in enumerate(self.heads):
#             headout=head(input,padding_mask,K_inp,V_inp)
#             out_matrices.append(headout[0])

#         # concatenating and feeding through linear layer
#         mh_out=self.finLinear(torch.cat(tuple(out_matrices),dim=2))

#         return mh_out

class Multi_HeadAttention(nn.Module):
    # enc_dev signifies if its a encoder-decoder multi-head attention
    def __init__(self,head_count,input_size,query_size,value_size,self_regress=False,enc_dec=False):
        super().__init__()
        self.ec=enc_dec
        self.input_size=input_size
        self.query_size=query_size
        self.value_size=value_size
        self.head_count=head_count
        self.self_regress=self_regress

        # calculate all Qs,Ks, and Vs for all heads at once
        # bias=False?
        self.wq=nn.Linear(input_size,head_count*query_size) # W_q matrices for all heads
        self.wk=nn.Linear(input_size,head_count*query_size) # W_k matrices for all heads
        self.wv=nn.Linear(input_size,head_count*value_size) # W_v matrices for all heads
        self.finLinear=nn.Linear(head_count*value_size,input_size)



    # computes the final vectors of each token
    # N -> Batch Size
    # L -> Sequence Length
    # H -> head count
    # Q -> (N,H,L,eq)
    # K -> (N,H,L,ek)
    # V -> (N,H,L,ev)
    # mask -> (N,L,L)
    # out -> (N,H,L,ev)
    def SelfAttention(self,Q,K,V,mask):
        key_size=self.query_size
        out=torch.matmul(Q,torch.transpose(K,2,3))
        # out=torch.div(out,math.sqrt(key_size))
        sft=nn.Softmax(dim=3)
        mask=torch.unsqueeze(mask,1)
        attention_weights=sft(torch.div(torch.add(out,mask),math.sqrt(key_size)))
        out=torch.matmul(attention_weights,V)
        return out



    # padding mask given in the form of [0s and 1s] 0-pay attention 1-donot pay attention
    # padding mask -> (N,L) , pass the encoding padding mask if its encoder-decoder attention
    # input -> (N,L,input_size)
    # self_regress: Boolean
    # returns (N,L,input_size)
    def forward(self,input,padding_mask,K_inp=None,V_inp=None):
        batchSize=input.shape[0]
        seqLen=input.shape[1]
        if not self.ec:
            K_inp=input
            V_inp=input


        # calculating the Q,K,V matrices for all the heads
        Q=self.wq(input).view(batchSize,seqLen,self.head_count,self.query_size) # (N,seqLen,headCount,query_size)
        Q=torch.transpose(Q,1,2) # after transpose, of shape, (N,head_count,seqLen,query_size)
        K=self.wk(K_inp).view(batchSize,K_inp.shape[1],self.head_count,self.query_size)
        K=torch.transpose(K,1,2)
        V=self.wv(V_inp).view(batchSize,K_inp.shape[1],self.head_count,self.value_size)
        V=torch.transpose(V,1,2)

        # generating a mask( maybe do this in higher classes )
        mask=torch.unsqueeze(padding_mask,1).repeat(1,input.shape[1],1)*float('-inf') # padding mask
        mask=torch.nan_to_num(mask,nan=0,neginf=float('-inf'))
        if self.self_regress:
            # self-regress mask
            selfRegressMask=torch.triu(torch.ones(batchSize,seqLen, seqLen) * float('-inf'), diagonal=1).to('cuda')
            mask=torch.add(mask,selfRegressMask)


        # converting into (N,L,head_count*value_size)
        out=self.SelfAttention(Q,K,V,mask)
        out=torch.transpose(out,1,2).contiguous().view(batchSize,seqLen,self.input_size)
        mh_out=self.finLinear(out)


        return mh_out

# one encoder block
# take care of passing ks and vs to decoder
class EncoderBlock(nn.Module):
    def __init__(self,input_size,head_count):
        super().__init__()
        self.LN1=nn.LayerNorm(input_size)
        self.LN2=nn.LayerNorm(input_size)
        self.feedForward=nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(),
            nn.Linear(1024,input_size),
            nn.Dropout(p=0.1),
        )
        value_query_size=int(input_size/head_count)
        self.multiHAttention=Multi_HeadAttention(head_count,input_size,value_query_size,value_query_size)

    # inputs -> (N,L,input_size) , these have to be positional encodings
    # padding mask -> (N,L)
    def forward(self,enc_inputs):
        (inputs,padding_mask)=enc_inputs

        out1=self.multiHAttention(inputs,padding_mask)
        out1=self.LN1(torch.add(inputs,out1))
        out=self.feedForward(out1)
        out=self.LN2(torch.add(out1,out))
        return (out,padding_mask)

class DecoderBlock(nn.Module):
    def __init__(self,input_size,head_count):
        super().__init__()
        self.LN1=nn.LayerNorm(input_size)
        self.LN2=nn.LayerNorm(input_size)
        self.LN3=nn.LayerNorm(input_size)
        self.feedForward=nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(),
            nn.Linear(1024,input_size),
            nn.Dropout(p=0.1),
        )
        value_query_size=int(input_size/head_count)
        self.multiHAttention=Multi_HeadAttention(head_count,input_size,value_query_size,value_query_size,self_regress=True)
        self.encdecAttention=Multi_HeadAttention(head_count,input_size,value_query_size,value_query_size,enc_dec=True)

    # inputs -> (N,L,input_size) , these have to be positional encodings
    # padding_mask_enc -> padding mask of encoder, needed in encoder decoder attention
    # padding mask -> (N,L)
    # K_inp,V_inp -> (N,L,input_size)
    def forward(self,dec_inputs):
        (inputs,padding_mask,K_inp,V_inp,padding_mask_enc)=dec_inputs


        out1=self.multiHAttention(inputs,padding_mask)
        out1=self.LN1(torch.add(inputs,out1))
        out2=self.encdecAttention(out1,padding_mask_enc,K_inp=K_inp,V_inp=V_inp)
        out2=self.LN2(torch.add(out1,out2))
        out=self.feedForward(out2)
        out=self.LN3(torch.add(out2,out))
        return (out,padding_mask,K_inp,V_inp,padding_mask_enc)

class EncoderStack(nn.Module):
    def __init__(self,layers,input_size,head_count):
        super().__init__()
        # using sequential
        encoderStack=nn.Sequential()
        for i in range(layers):
            encoderStack.append(EncoderBlock(input_size,head_count))
        self.es=encoderStack

    # inputs -> (N,L,input_size) , these have to be positional encodings
    # padding mask -> (N,L)
    def forward(self,inputs,padding_mask):
        out=self.es((inputs,padding_mask))
        return out

class DecoderStack(nn.Module):
    def __init__(self,layers,input_size,head_count):
        super().__init__()
        # using sequential
        decoderStack=nn.Sequential()
        for i in range(layers):
            decoderStack.append(DecoderBlock(input_size,head_count))
        self.ds=decoderStack

    # inputs -> (N,L,input_size) , these have to be positional encodings
    # padding mask -> (N,L)
    # padding_mask_enc -> (N,L)
    # enc_outputs -> (N,L,input_size)
    def forward(self,inputs,padding_mask,enc_outputs,padding_mask_enc):
        out=self.ds((inputs,padding_mask,enc_outputs,enc_outputs,padding_mask_enc))
        return out


# max sequence length is 750
class PosEncoding(nn.Module):
    def __init__(self, input_size, max_seq_length=750):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, input_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe=pe.unsqueeze(0).to('cuda')

    # x -> (N,L,input_size)    
    def forward(self, x):
        c=torch.add(x,self.pe[:, :x.size(1)])
        return c





class Transformer_custom(nn.Module):
    def __init__(self,layers,embedding_size,head_count,inp_vocab_size,out_vocab_size):
        super().__init__()
        # embedding layer for both encoder and decoder
        self.embeddingsEnc=nn.Embedding(inp_vocab_size,embedding_size,0) # pad token is at index 0
        self.embeddingsDec=nn.Embedding(out_vocab_size,embedding_size,0)
        # positional embedding layer
        self.PosEnc=PosEncoding(embedding_size)
        # encoder layer
        self.encoder=EncoderStack(layers,embedding_size,head_count)
        # decoder layer
        self.decoder=DecoderStack(layers,embedding_size,head_count)
        self.toVocab=nn.Linear(embedding_size,out_vocab_size)
        # self.sft=nn.Softmax(dim=2)

    # inputs,outputs -> (N,L,input_size)
    # inp_padding,out_padding -> (N,L)
    # returns out -> (N,L,out_vocab_size)
    def forward(self,inputs,inp_padding,outputs,out_padding):
        enc_embeddings=self.embeddingsEnc(inputs)
        # add positional embedding
        enc_embeddings=self.PosEnc(enc_embeddings)
        enc_outputs=self.encoder(enc_embeddings,inp_padding)

        dec_embeddings=self.embeddingsDec(outputs)
        dec_embeddings=self.PosEnc(dec_embeddings)
        # add positional embeddings
        out=self.decoder(dec_embeddings,out_padding,enc_outputs[0],enc_outputs[1])
        out=self.toVocab(out[0])
        # out=self.sft(out)
        return out

## Data Preprocessing ------------------------------------



# using spacy to tokenize sentences into tokens, nltk was running into some caveats
en_nlp = English()
fr_nlp=French()
en_tokenizer=en_nlp.tokenizer
fr_tokenizer=fr_nlp.tokenizer


# Pre-cleaning the text before splitting into sentences
# This will clean a piece of text
def clean(t):
    # cleaning
    t = re.sub(r'(((http|https):\/\/)|www\.)([a-zA-Z0-9]+\.){0,2}[a-zA-Z0-9]+([a-zA-Z0-9\/#%&=\?_\.\-\+]+)', "", t)
    t = re.sub(r'(@[a-zA-Z0-9_]+)', "", t)
    t = re.sub(r'(#[a-zA-Z0-9_]+\b)', "", t)
    t = re.sub(r'\d+', "",t)
    t = re.sub(r'--'," ",t)
    # special characters
    t = re.sub(r'[\_\$\*\^\(\)\[\]\{\}\=\+\<\>",\&\%\-\—\”\“\–\\\.\?\!;]'," ",t)
    t=re.sub(r'\n'," ",t)
    t=t.lower()
    return t


def formatFiles(english_file_path,french_pile_path,output_arr):
    with open(english_file_path) as en:
        with open(french_pile_path) as fr:
            for en_line,fr_line in zip(en,fr):
                output_arr.append((en_line,fr_line))

# getting the test,train,val inputs,labels
train_sentences=[]
test_sentences=[]
val_sentences=[]

formatFiles('/home2/raghavd0/transformer/en-fr_dataset/train.en','/home2/raghavd0/transformer/en-fr_dataset/train.fr',train_sentences)
formatFiles('/home2/raghavd0/transformer/en-fr_dataset/test.en','/home2/raghavd0/transformer/en-fr_dataset/test.fr',test_sentences)
formatFiles('/home2/raghavd0/transformer/en-fr_dataset/dev.en','/home2/raghavd0/transformer/en-fr_dataset/dev.fr',val_sentences)


print('Train Sentences',len(train_sentences))
print('Val Sentences',len(val_sentences))
print('Test Sentences',len(test_sentences))

# data -> [[(input,label)],]
# cleans,splits into words,and replaces less frequent tokens with unknown tokens
def cleanData(data,min_ocuurences):
    en_vocab_count={}
    fr_vocab_count={}

    # splitting into tokens and counting occurences of words
    # tokenized data -> [[([],[]),],]
    en_total_tokens=0
    tokenized_data=[]
    for data_pack in data:
        tokenized_data_pack=[]
        for sample in data_pack:
            en_tokens=[]
            fr_tokens=[]

            for en_token in en_tokenizer(clean(sample[0])):
                en_total_tokens+=1
                en_token=str(en_token)
                en_tokens.append(en_token)
                en_vocab_count[en_token]=en_vocab_count.get(en_token,0)+1
            for fr_token in fr_tokenizer(clean(sample[1])):
                fr_token=str(fr_token)
                fr_tokens.append(fr_token)
                fr_vocab_count[fr_token]=fr_vocab_count.get(fr_token,0)+1

            tokenized_data_pack.append((en_tokens,fr_tokens))
        tokenized_data.append(tokenized_data_pack)




    # replacing low occuring words in english with <UNK> token
    en_unk=set()
    fr_unk=set()
    for data_pack in tokenized_data:
        for sample in data_pack: # sample = ([],[])

            for i,en_token in enumerate(sample[0]):
                if en_vocab_count[en_token]<min_ocuurences:
                    sample[0][i]='<UNK>'
                    en_unk.add(en_token)
            # for i,fr_token in enumerate(sample[1]):
            #     if fr_vocab_count[fr_token]<min_ocuurences:
            #         sample[1][i]='<UNK>'
            #         fr_unk.add(fr_token)

    print('English Unknown Tokens',len(en_unk))
    print('French Unknown Tokens',len(fr_unk))

    return tokenized_data


cleaned_tok_data=cleanData([train_sentences,val_sentences,test_sentences],2)

# 22k unique tokens in english
# 29k unique tokens in french

# print(cleaned_tok_data[0][4])

en_vocab=['<PAD>','<SOS>','<EOS>']
en_wordToIdx={'<PAD>':0,'<SOS>':1,'<EOS>':2} # word:index

fr_vocab=['<PAD>','<SOS>','<EOS>']
fr_wordToIdx={'<PAD>':0,'<SOS>':1,'<EOS>':2}

# indexing all the words and constructing vocabulary
for data_pack in cleaned_tok_data:
    for sample in data_pack:
        for en_token in sample[0]:
            if en_wordToIdx.get(en_token,0)==0:
                en_wordToIdx[en_token]=len(en_vocab)
                en_vocab.append(en_token)
        for fr_token in sample[1]:
            if fr_wordToIdx.get(fr_token,0)==0:
                fr_wordToIdx[fr_token]=len(fr_vocab)
                fr_vocab.append(fr_token)

print(len(en_vocab),en_vocab[:10])
print(len(fr_vocab),fr_vocab[:10])
en_vocab_len=len(en_vocab)
fr_vocab_len=len(fr_vocab)

def tokToIdx(wordToIdx,tokens,add_sos=False,add_eos=False):
    indexes=[]
    if add_sos:
        indexes.append(wordToIdx['<SOS>'])
    for tok in tokens:
        indexes.append(wordToIdx[tok])
    if add_eos:
        indexes.append(wordToIdx['<EOS>'])
    return indexes

def IdxToTok(vocab,tokens):
    toks=[]
    for idx in tokens:
        toks.append(vocab[idx])
    return toks




# converting tokens to indices
indexed_data=[]

for data_pack in cleaned_tok_data:
    indexed_data_pack=[]
    for sample in data_pack:
        indexed_en=tokToIdx(en_wordToIdx,sample[0])
        indexed_fr=tokToIdx(fr_wordToIdx,sample[1],True,True)
        indexed_data_pack.append((indexed_en,indexed_fr))
    indexed_data.append(indexed_data_pack)

# print(indexed_data[0][1])

from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot

# making dataset and dataloader-----------------------------------


class en_fr_Dataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    # returns input_sequence,decoder_input_sequence,decoder_target
    def __getitem__(self,idx):
        return (self.data[idx][0],self.data[idx][1][:-1].copy(),self.data[idx][1][1:])


# returns (enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets)
def collater(data):
    enc_inputs=[]
    dec_inputs=[]
    dec_targets=[]
    for m in data:
        enc_inputs.append(torch.tensor(m[0],dtype=torch.int))
        dec_inputs.append(torch.tensor(m[1],dtype=torch.int))
        dec_targets.append(torch.tensor(m[2],dtype=torch.int))



    enc_inputs=pad_sequence(enc_inputs,batch_first=True)
    dec_inputs=pad_sequence(dec_inputs,batch_first=True)
    dec_targets=one_hot(pad_sequence(dec_targets,batch_first=True).long(),num_classes=fr_vocab_len).to(torch.float32)

    enc_inputs_mask=(enc_inputs==0).int()
    dec_inputs_mask=(dec_inputs==0).int() # 0 - pay attention, 1 - no attention

    return (enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets)

batchSize=int(sys.argv[2])
train_DL=DataLoader(en_fr_Dataset(indexed_data[0]),batch_size=batchSize,collate_fn=collater)
val_DL=DataLoader(en_fr_Dataset(indexed_data[1]),batch_size=batchSize,collate_fn=collater)

# count=0
# for batch in train_DL:
#     print('Encoder inputs',batch[0])
#     print('Encoder inputs mask',batch[1])
#     print('Decoder inputs',batch[2])
#     print('Decoder inputs mask',batch[3])
#     print('Decoder targets',batch[4])
#     count+=1
#     if count==3:
#         break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


from nltk.translate.bleu_score import sentence_bleu
import os

epochs=int(sys.argv[1])
lr=3
heads=int(sys.argv[3])
dropout=float(sys.argv[4])
embedSize=int(sys.argv[5])
baseDir=f'./transformerModels/model_epoch{epochs}_heads{heads}_lr{lr}_dropout{dropout}_batchSize{batchSize}_embedSize{embedSize}/'
os.mkdir(baseDir)


# Load model,optimizer and loss function and get device
model=Transformer_custom(2,embedSize,heads,en_vocab_len,fr_vocab_len).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(model,DL,loss_fn,optimizer):
    model.train()
    total_loss=0
    count=0
    for enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets in DL:

        enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets=enc_inputs.to(device),enc_inputs_mask.to(device),dec_inputs.to(device),dec_inputs_mask.to(device),dec_targets.to(device)
        logits=model(enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask)


        logits,dec_targets=torch.flatten(logits,start_dim=0,end_dim=1),torch.flatten(dec_targets,start_dim=0,end_dim=1)

        # calculating loss
        loss=loss_fn(logits,dec_targets)
        # print(loss.item())
        total_loss+=loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count+=1
        with torch.no_grad():
            print(f'Train-----Batch {count} Loss ----> {loss.item()}\r',end="")


    avgLoss=(total_loss/len(DL))
    print(f'Train Loss: {avgLoss}\n')
    return avgLoss

# test
def test(model,DL,loss_fn,optimizer,bleu_scores=False,bleu_score_file=None):
    model.eval()
    total_loss=0
    count=0
    total_bleu=0
    with torch.no_grad():
        for enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets in DL:

            enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask,dec_targets=enc_inputs.to(device),enc_inputs_mask.to(device),dec_inputs.to(device),dec_inputs_mask.to(device),dec_targets.to(device)
            logits=model(enc_inputs,enc_inputs_mask,dec_inputs,dec_inputs_mask)

            # calculating bleu scores
            if bleu_scores:
                with torch.no_grad():
                    candidates=torch.argmax(logits,dim=2).tolist()
                    references=torch.argmax(dec_targets,dim=2).tolist()
                    for r,c in zip(references,candidates):
                        b_score=sentence_bleu([r],c)
                        total_bleu+=b_score
                        bleu_score_file.write(f'{b_score}\n')


            logits,dec_targets=torch.flatten(logits,start_dim=0,end_dim=1),torch.flatten(dec_targets,start_dim=0,end_dim=1)

            # calculating loss
            loss=loss_fn(logits,dec_targets)
            # print(loss.item())
            total_loss+=loss.item()

            count+=1
            print(f'Test-----Batch {count} Loss ----> {loss.item()}\r',end="")


        avgLoss=(total_loss/len(DL))
        print(f'Test Loss: {avgLoss}\n')
    ret=avgLoss
    if bleu_scores:
        ret=(avgLoss,total_bleu/len(DL.dataset))
    return ret



import csv


# calculate train/test bleu scores, train/val loss scores,test loss and save models

headers=['EpochNumber','TrainAverageLoss','ValAverageLoss']

with open(f'{baseDir}Stats.csv','w') as csvh:
    csvwriter = csv.writer(csvh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(headers)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_stats=train(model,train_DL,loss_fn,optimizer)
        val_stats=test(model,val_DL,loss_fn,optimizer)

        row=[t+1]
        row.append(train_stats)
        row.append(val_stats)

        csvwriter.writerow(row)

        # saving model after 3 epochs
        if (t+1)%3==0:
            torch.save(model.state_dict(),f'{baseDir}transformer{str(t+1)}epoch.pth')
    print("Done!")

# saving the final model
torch.save(model.state_dict(),f'{baseDir}transformer{str(epochs)}epoch.pth')


# Testing----------------------------------------

test_DL=DataLoader(en_fr_Dataset(indexed_data[2]),batch_size=batchSize,collate_fn=collater)




# getting train and test bleu scores
with open(f'{baseDir}TestScore.txt','w') as testScore:
    with open(f'{baseDir}TrainBleu.txt','w') as train_b:
        with open(f'{baseDir}TestBleu.txt','w') as test_b:
            print("Calculating Train Bleu scores")
            trainS=test(model,train_DL,loss_fn,optimizer,True,train_b)
            train_b.write(f'{trainS[1]}\n')


            print("Calculating Test Bleu scores")
            testStats=test(model,test_DL,loss_fn,optimizer,True,test_b)
            testScore.write(f'{testStats[0]}\n')
            test_b.write(f'{testStats[1]}\n')















# are input_size==query_size==value_size???
# add activations after linear layers
# add multiple layer norms
# add mask to encoder states to in enc-decoder side?? two masks? self-regress? -- got it
# keys and value matrices how propagated into decoder?? -- got it
# is the encoder decoder attention self regressed -- got it

# testing-------------------------------
# N -> Batch Size
    # L -> Sequence Lengtj
    # Q -> (N,L,eq)
    # K -> (N,L,ek)
    # V -> (N,L,ev)
    # mask -> (N,L,L)
    # out -> (N,L,ev)

# a=torch.tensor([[0,0,0,0,1,1],
#                 [0,0,0,1,1,1]],dtype=torch.float32)
# c=torch.tensor([[0,0,1,1,1,1],
#                 [0,0,0,1,1,1]],dtype=torch.float32)
# a=torch.unsqueeze(a,1)
# c=torch.unsqueeze(c,1)
# print(a.shape)
# b=torch.nan_to_num(a.repeat(1,4,1)*float('-inf'),nan=0,neginf=float('-inf'))
# d=torch.nan_to_num(c.repeat(1,4,1)*float('-inf'),nan=0,neginf=float('-inf'))
# f=torch.add(b,d)
# print(f)
# sft=nn.Softmax(dim=2)
# print(sft(f))
# a=[1,2,3]
# for k,v in enumerate(a):
#     print(k,v)
# import torch

# a=torch.ones(2,4,10)
# print(a)
# b=torch.flatten(a,start_dim=0,end_dim=1)
# print(b)

# from torch.nn.utils.rnn import pad_sequence
# a=[torch.ones(100),torch.ones(50),torch.ones(26)]
# b=pad_sequence(a,batch_first=True)
# print(b)
# print(b==0)
# print((b==0).int())
# b=6
# c=10

# a=(b,c)

# (d,m)=a
# print(d,m)

# from torch.nn.functional import one_hot

# a=torch.tensor([[1,2,3,5],
#                [2,3,5,6]])
# print(one_hot(a).dtype)

