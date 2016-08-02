# First Research Tempt.
# add memory write/evolution in MemN2N framework.
# by yyq, 2016-05-27

from collections import OrderedDict
import re
import os
import copy

import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params
def preProcessBabi():
    fileList = os.listdir('./babi_task_20/en')
    trainFileList = []
    testFileList = []
    for fileName in fileList:
        if re.search(r'[0-9a-zA-Z]*_train.txt', fileName) != None:
            trainFileList.append(fileName)
        else:
            testFileList.append(fileName)
    return trainFileList, testFileList

def parseBabiTaskTrain(fileList, dict, randomize_time, memSize):
    qCount = 0
    story_out = []
    q_out = []
    ans_out = [] # these three contains all data.
    story = [] # buffer for one story.
    for fileName in fileList:
        # print fileName
        fileStoryLens = []
        fileNameFull = './babi_task_20/en/' + fileName
        fp = open(fileNameFull, 'r')
        line = fp.readline().strip().lower()
        while line:
            if re.search(r'\?', line) == None: # a sentence in a story
                line = line.strip('.').split(' ')
                if line[0] == '1':
                    story = []
                line = line[1:]
                sent = []
                for word in line:
                    if word not in dict:
                        dict[word] = len(dict) + 1 # word id starts from 0
                    sent.append(dict[word])
                story.append(sent)
            else: # a question and support
                line = line.strip().split()
                line = line[1:]
                sent = []
                ansFlag = 0
                for word in line:
                    if word[-1] == '?':
                        word = word[:-1]
                        if word not in dict:
                            dict[word] = len(dict) + 1 # word id starts from 1 to distinct from 0 for paddings
                        sent.append(dict[word])
                        story_out.append(copy.deepcopy(story))
                        fileStoryLens.append(len(story))
                        q_out.append(sent)
                        qCount = qCount + 1
                        ansFlag = 1
                    elif ansFlag == 1:
                        if word not in dict:
                            dict[word] = len(dict) + 1
                        ans_out.append(dict[word])
                        ansFlag = 0
                        break
                    else:
                        if word not in dict:
                            dict[word] = len(dict)+ 1
                        sent.append(dict[word])
            line = fp.readline().strip().lower()
        
        # print max(fileStoryLens)
        fp.close() 
    print 'total_questions in train: %d' %qCount
    
    dict['nil'] = len(dict) + 1
    
    # time embedding preprocess and random noise inserting
    
    # get max length of story.
    storyLens = [len(x) for x in story_out]
    maxStoryLen = max(storyLens)
    if maxStoryLen > memSize:
        maxStoryLen = memSize
    # get max length of sentence.
    sentLens = []
    for s in story_out:
        sentLens.extend([len(x) for x in s])
    maxSentLen = max(sentLens)
    
    qLens = []
    for s in q_out:
        qLens.append(len(s))
    maxQLen = max(qLens)

    maxSentLen = max(maxQLen, maxSentLen)
    
    
    for i in range(len(story_out)):
        # cut
        if len(story_out[i]) > maxStoryLen:
            story_out[i] = copy.deepcopy(story_out[i][(-maxStoryLen):])
        
        tempStyLen = len(story_out[i])
        nblank = numpy.random.randint(0, int(numpy.ceil(tempStyLen * randomize_time)))
        nblank = 0
        rt = numpy.random.permutation(tempStyLen + nblank)
        for tt in range(rt.size):
            if rt[tt] > maxStoryLen - 1:
                rt[tt] = maxStoryLen - 1
        rt = rt[:tempStyLen]
        rt.sort()
        rt = rt[::-1]
        for mm in range(tempStyLen):
            story_out[i][mm].append(len(dict) + 1 + copy.deepcopy(rt[mm]))
    return story_out, q_out, ans_out, dict, maxStoryLen, maxSentLen

def parseBabiTaskTest(fileList, dict, memSize):
    qCount = 0
    story_out = []
    q_out = []
    ans_out = [] # these three contains all data.
    story = [] # buffer for one story.
    # dict['nil'] = 1 # unks
    for fileName in fileList:
        # print fileName
        fileNameFull = './babi_task_20/en/' + fileName
        fp = open(fileNameFull, 'r')
        line = fp.readline().strip().lower()
        while line:
            if re.search(r'\?', line) == None: # a sentence in a story
                line = line.strip('.').split(' ')
                if line[0] == '1':
                    story = []
                line = line[1:]
                sent = []
                for word in line:
                    if word not in dict:
                        sent.append(dict['nil']) # unks
                    else:
                        sent.append(dict[word])
                story.append(sent)
            else: # a question and support
                line = line.strip().split()
                line = line[1:]
                sent = []
                ansFlag = 0
                for word in line:
                    if word[-1] == '?':
                        word = word[:-1]
                        if word not in dict:
                            sent.append(dict['nil'])
                        else:
                            sent.append(dict[word])
                        story_out.append(copy.deepcopy(story))
                        q_out.append(sent)
                        qCount = qCount + 1
                        ansFlag = 1
                    elif ansFlag == 1:
                        if word not in dict:
                            ans_out.append(dict['nil'])
                        else:
                            ans_out.append(dict[word])
                        ansFlag = 0
                        break
                    else:
                        if word not in dict:
                            sent.append(dict['nil'])
                        else:
                            sent.append(dict[word])
            line = fp.readline().strip().lower()
        fp.close() 
    # print 'total_questions in train: %d' %qCount
    
    maxStoryLen = memSize
    # time embedding preprocess
    # still reverse the story time notes but dont add random noise.
    
    for i in range(len(story_out)):
        # cut
        if len(story_out[i]) > maxStoryLen:
            story_out[i] = copy.deepcopy(story_out[i][(-maxStoryLen):])
            
        tempStyLen = len(story_out[i])
        rt = range(tempStyLen)
        rt = rt[::-1]
        for mm in range(tempStyLen):
            story_out[i][mm].append(len(dict) + 1 + copy.deepcopy(rt[mm]))
    return story_out, q_out, ans_out

def prepareData(storyList, qList, ansList, maxStoryLen, maxSentLen): # remember to add THEANO.CONFIG.FLOATX after debugging!!!!!
    # input are both 32 samples(batch_size)
    nSamples = len(storyList)
     
    qLens = [len(x) for x in qList]
    maxQLen = max(qLens)
    x = numpy.zeros([maxStoryLen, maxSentLen + 1, nSamples]).astype('int64') # maxSentLen + 1 for time embedding.
    xmask = numpy.zeros([maxStoryLen, maxSentLen + 1, nSamples]).astype(config.floatX)
    q = numpy.zeros([maxSentLen + 1, nSamples]).astype('int64')
    qmask = numpy.zeros([maxSentLen + 1, nSamples]).astype(config.floatX)
    y = ansList
    
    # print maxStoryLen,maxSentLen
    
    
    for i in range(nSamples):

        
        for j in range(maxStoryLen):
            if j < len(storyList[i]):
                x[j,0:len(storyList[i][j]) - 1,i] = storyList[i][j][:-1]
                x[j,-1,i] = storyList[i][j][-1]
                xmask[j,:(len(storyList[i][j]) - 1),i] = 1.
                xmask[j,-1,i] = 1

                
        
        q[:len(qList[i]),i] = qList[i]
        qmask[:len(qList[i]),i] = 1.
    
    return x, xmask, q, qmask, y
        

def computeTimeMat(maxSentLen, dim_word):
    matLen = maxSentLen + 1
    timeMat = numpy.ones([dim_word, matLen])
    for i in range(dim_word):
        for j in range(matLen):
            timeMat[i][j] = (i + 1 - (dim_word + 1.) / 2.) * (j + 1 - (matLen + 1.) / 2.)
    timeMat = 1 + 4 * timeMat / matLen / dim_word
    print 'time mat shape: dim_word * (maxSentLen + 1)'
    print timeMat.shape
    return timeMat.astype(config.floatX)
        
def param_init(options):
    params = OrderedDict()
    params = init_embed(params, options)
    params = init_evo_GRU(params, options)
    return params

def init_embed(params, options):
    randA = numpy.random.randn(options['vocab_size'], options['dim_word'])  # row 0 corresponds to empty space. always masked though.
    params['Wemb_A'] = (0.1 * randA).astype(config.floatX)
    randB = numpy.random.randn(options['vocab_size'], options['dim_word'])
    params['Wemb_B'] = (0.1 * randA).astype(config.floatX)
    
    Wclass = numpy.random.randn(options['vocab_size'], options['dim_hidden']) # this is the final matrix to predict answer.
    params['Wclass'] = (0.1 * Wclass).astype(config.floatX)
    bclass = numpy.zeros((options['vocab_size'],))
    params['bclass'] = bclass.astype(config.floatX)
    return params
    
def ortho_weight(ndim):
    W = numpy.random.randn(ndim,ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def random_weight(ndim,mdim):
    W = numpy.random.randn(ndim,mdim)
    if ndim == mdim:
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)
    return W.astype(config.floatX)
    
def init_evo_GRU(params, options):
    wrt_emb_i_1 = numpy.concatenate([ortho_weight(options['dim_word']), ortho_weight(options['dim_word'])], 
                                    axis=1)
    wrt_emb_j_1 = numpy.concatenate([ortho_weight(options['dim_word']), ortho_weight(options['dim_word']), 
                                    ],axis=1)
    wrt_u_1 = numpy.concatenate([random_weight(options['dim_hidden'], options['dim_word']), 
                                        random_weight(options['dim_hidden'], options['dim_word']), 
                                        ],axis=1)
    WGRU_A = numpy.concatenate([wrt_emb_i_1, wrt_emb_j_1, wrt_u_1], axis=0) # ai, aj, u to r, z
    Whai_A = ortho_weight(options['dim_word']) # ai to h~(t), ai gated by rt
    Whaj_A = ortho_weight(options['dim_word']) # aj to h~(t)
    Whu_A = random_weight(options['dim_hidden'], options['dim_word']) # u to h~(t)
    
    bGRU_A = (numpy.zeros((2 * options['dim_word'],)) + 2.7).astype(config.floatX)
    bGRUh_A = numpy.zeros((options['dim_word'],)).astype(config.floatX)
    
    wrt_emb_b_2 = numpy.concatenate([ortho_weight(options['dim_word']), ortho_weight(options['dim_word']), 
                                    ],axis=1)
    wrt_emb_a_2 = numpy.concatenate([ortho_weight(options['dim_word']), ortho_weight(options['dim_word']), 
                                    ],axis=1)
    wrt_u_2 = numpy.concatenate([random_weight(options['dim_hidden'], options['dim_word']), 
                                        random_weight(options['dim_hidden'], options['dim_word']), 
                                        ],axis=1)
    wrt_q_2 = numpy.concatenate([random_weight(options['dim_hidden'], options['dim_word']), 
                                        random_weight(options['dim_hidden'], options['dim_word']), 
                                        ],axis=1)
    WGRU_B = numpy.concatenate([wrt_emb_b_2, wrt_emb_a_2, wrt_u_2, wrt_q_2], axis=0)
    Whbi_B = ortho_weight(options['dim_word']) # bi to h~(t), bi gated by rt
    Whai_B = ortho_weight(options['dim_word']) # ai to h~(t)
    Whu_B = random_weight(options['dim_hidden'], options['dim_word']) # u to h~(t)
    Whq_B = random_weight(options['dim_hidden'], options['dim_word']) # q to h~(t)
    bGRU_B = (numpy.zeros((2 * options['dim_word'],)) + 2.7).astype(config.floatX)
    bGRUh_B = numpy.zeros((options['dim_word'],)).astype(config.floatX)
    
    params['WGRU_A'] = WGRU_A
    params['Whai_A'] = Whai_A
    params['Whaj_A'] = Whaj_A
    params['Whu_A'] = Whu_A
    
    params['WGRU_B'] = WGRU_B
    params['Whbi_B'] = Whbi_B
    params['Whai_B'] = Whai_B
    params['Whu_B'] = Whu_B
    params['Whq_B'] = Whq_B
        
    params['bGRU_A'] = bGRU_A
    params['bGRU_B'] = bGRU_B
    params['bGRUh_A'] = bGRUh_A
    params['bGRUh_B'] = bGRUh_B
    return params
    
def param_shared(params):
    tparams = OrderedDict()
    for k,v in params.iteritems():
        tparams[k] = theano.shared(params[k], name=k)
    return tparams

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

def memLayers(tparams, options, x, xmask, q, qmask, nhops, wmat):

    assert xmask is not None
    assert qmask is not None
    
    # look up table
    emb_A = tparams['Wemb_A'][x.flatten()].reshape([x.shape[0], x.shape[1], x.shape[2], options['dim_word']]) # storyLen * sentLen * nsamples * dimword
    emb_A_m = emb_A * xmask.dimshuffle(0,1,2,'x').repeat(emb_A.shape[3], axis=3)
    emb_B = tparams['Wemb_B'][x.flatten()].reshape([x.shape[0], x.shape[1], x.shape[2], options['dim_word']])
    emb_B_m = emb_B * xmask.dimshuffle(0,1,2,'x').repeat(emb_B.shape[3], axis=3)
    emb_q = tparams['Wemb_A'][q.flatten()].reshape([q.shape[0], q.shape[1], options['dim_word']]) # sentLen * nsamples * dimword
    emb_q_m = emb_q * qmask.dimshuffle(0,1,'x').repeat(emb_q.shape[2], axis=2)
    
    
    # create sentence embedding from word and time embedding
    emb_A_sf = emb_A_m.dimshuffle(0,2,3,1)# storyLen * nsamples * dim_word
    emb_A_w = emb_A_sf * wmat
    emb_A_s = emb_A_w.sum(axis=3)
    emb_B_s = (emb_B_m.dimshuffle(0,2,3,1) * wmat).sum(axis=3)
    emb_q_s = (emb_q_m.dimshuffle(1,2,0) * wmat).sum(axis=2) # nsamples * dim_word
    
    xmask_story = xmask.max(axis=1).dimshuffle(1,0) # nsamples * storyLen
    
    # define memory hops
    def step(a_emb, b_emb, u_vec, q_vec, xmask_story):
        # a_emb, b_emb: storyLen * nsamples * dim_word
        # u_vec, q_vec: nsamples * dim_word
        # xmask_story: nsamples * storyLen
        
        # u attention over a:
        att_u_a = tensor.batched_dot(a_emb.dimshuffle(1,0,2), u_vec) # nsamples * storyLen
        att_u_a_sm = tensor.nnet.softmax(att_u_a) # nsamples * storyLen
        retrieve_u_a = ((a_emb.dimshuffle(2,1,0) * att_u_a_sm).sum(axis=2)).dimshuffle(1,0) # nsamples * dim_word
        u_vec_n = u_vec + retrieve_u_a # nsamples * dim_word (nsamples * dim_hidden in fact. but in this paradigm it is equal.)
        
        # use b to retrieve a; use a to retrieve b.
        # q attention over a:
        att_q_a = tensor.batched_dot(a_emb.dimshuffle(1,0,2), q_vec) # nsamples * storyLen
        att_q_a_sm = tensor.nnet.softmax(att_q_a) # nsamples * storyLen
        retrieve_q_a = ((b_emb.dimshuffle(2,1,0) * att_q_a_sm).sum(axis=2)).dimshuffle(1,0) # nsamples * dim_word
        
        # q attention over b:
        att_q_b = tensor.batched_dot(b_emb.dimshuffle(1,0,2), q_vec) # nsamples * storyLen
        att_q_b_sm = tensor.nnet.softmax(att_q_b) # nsamples * storyLen
        retrieve_q_b = ((a_emb.dimshuffle(2,1,0) * att_q_b_sm).sum(axis=2)).dimshuffle(1,0) # nsamples * dim_word
        
        q_vec_n = q_vec + retrieve_q_a + retrieve_q_b # nsamples * dim_word
        
        # memory self attentions:
        att_cov = tensor.batched_dot(a_emb.dimshuffle(1,0,2), a_emb.dimshuffle(1,2,0)) # nsamples * storyLen * storyLen
        def in_step(mat):
            return tensor.nnet.softmax(mat)
        att_cov_sm, update = theano.scan(in_step, sequences=[att_cov]) # only one output variable, so softcovs is not a list.
        # att_cov = tensor.stacklists(softcovs) # nsamples * storyLen * storyLen
        retrieve_cov_a = tensor.batched_dot(att_cov_sm,a_emb.dimshuffle(1,0,2)) # nsamples * storyLen * dim_word
        
        # GRU memory evolution:
        a_emb_sf = a_emb.dimshuffle(1,0,2) # nsamples * storyLen * dim_word
        u_vec_spam = u_vec.dimshuffle(0,'x',1).repeat(a_emb_sf.shape[1],axis=1) # nsamples * storyLen * dim_hidden
        u_vec_spam_m = u_vec_spam * xmask_story.dimshuffle(0,1,'x').repeat(u_vec_spam.shape[2],axis=2)
        networkInputs_A = tensor.concatenate([a_emb_sf, retrieve_cov_a, u_vec_spam_m], axis=2) # nsamples * storyLen * (2 * dim_word + dim_hidden)
        preactA = tensor.dot(networkInputs_A, tparams['WGRU_A']) + tparams['bGRU_A'] # nsamples * storyLen * (2 * dim_word)
        r_a = tensor.nnet.sigmoid(_slice(preactA, 0, options['dim_word'])) # nsamples * storyLen * dim_word
        z_a = tensor.nnet.sigmoid(_slice(preactA, 1, options['dim_word']))
        h_mid_a = tensor.dot(retrieve_cov_a, tparams['Whaj_A']) + tensor.dot(u_vec_spam_m, tparams['Whu_A'])
        h_mid_a_full = tensor.tanh(h_mid_a + tensor.dot(a_emb_sf * r_a, tparams['Whai_A']) + tparams['bGRUh_A'])
        a_emb_n_pre = (a_emb_sf * z_a + h_mid_a_full * (1. - z_a)) * xmask_story.dimshuffle(0,1,'x').repeat(r_a.shape[2],axis=2)
        a_emb_n = a_emb_n_pre.dimshuffle(1,0,2) # storyLen * nsamples * dim_word
        
        b_emb_sf = b_emb.dimshuffle(1,0,2) # nsamples * storyLen * dim_word
        # u_vec has already been preprocessed and masked
        q_vec_spam = q_vec.dimshuffle(0,'x',1).repeat(a_emb_sf.shape[1],axis=1) # nsamples * storyLen * dim_hidden
        q_vec_spam_m = q_vec_spam * xmask_story.dimshuffle(0,1,'x').repeat(u_vec_spam.shape[2],axis=2)
        networkInputs_B = tensor.concatenate([b_emb_sf, a_emb_sf, u_vec_spam_m, q_vec_spam_m], axis=2) # nsamples * storyLen * (2 * dim_word + 2 * dim_hidden)
        preactB = tensor.dot(networkInputs_B, tparams['WGRU_B']) + tparams['bGRU_B'] # nsamples * storyLen * (2 * dim_word)
        r_b = tensor.nnet.sigmoid(_slice(preactB, 0, options['dim_word'])) # nsamples * storyLen * dim_word
        z_b = tensor.nnet.sigmoid(_slice(preactB, 1, options['dim_word']))
        h_mid_b = tensor.dot(a_emb_sf, tparams['Whai_B']) + tensor.dot(u_vec_spam_m, tparams['Whu_B']) + tensor.dot(q_vec_spam_m, tparams['Whq_B'])
        h_mid_b_full = tensor.tanh(h_mid_b + tensor.dot(b_emb_sf * r_b, tparams['Whbi_B']) + tparams['bGRUh_B'])
        b_emb_n_pre = (b_emb_sf * z_b + h_mid_b_full * (1. - z_b)) * xmask_story.dimshuffle(0,1,'x').repeat(r_a.shape[2],axis=2)
        b_emb_n = b_emb_n_pre.dimshuffle(1,0,2) # storyLen * nsamples * dim_word
        
        return a_emb_n, b_emb_n, u_vec_n, q_vec_n
        
    states, updates = theano.scan(step, outputs_info=[emb_A_s, emb_B_s, emb_q_s, emb_q_s], non_sequences=[xmask_story], n_steps=nhops)
    return states[0], states[1], states[2], states[3]
                       
def build_model_EvoMN(options, tparams):
    trng = RandomStreams(SEED)
    use_noise = theano.shared(numpy_floatX(0.))
    
    x = tensor.tensor3('x', dtype='int64') # x is n_sent * n_word * n_samples
    xmask = tensor.tensor3('xmask', dtype=config.floatX) # same as x
    q = tensor.matrix('q', dtype='int64') # q is nword * n_samples
    qmask = tensor.matrix('qmask', dtype=config.floatX)
    y = tensor.vector('y',dtype='int64') # nsamples * 1
    nhops = tensor.scalar('nhops',dtype='int64') # nhops, used to loop.
    wmat = tensor.matrix('wmat',dtype=config.floatX) # dim_word * (maxSentLen+1)
    
    aEmbSeq, bEmbSeq, uSeq, qSeq = memLayers(tparams, options, x, xmask, q, qmask, nhops, wmat)
    proj = qSeq[-1] # nsamples * dim_hidden
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['Wclass'].T) + tparams['bclass']) # nsamples * vocab_size
    pred_ans = pred.argmax(axis=1) # nsamples vector
    off = 1e-8
    cost = -tensor.log(pred[tensor.arange(y.shape[0]), y] + off).sum()
    
    f_debug = theano.function([x,xmask,q,qmask,y,nhops,wmat], [x,xmask,q,qmask,y,nhops,wmat,proj,pred,pred_ans,aEmbSeq,bEmbSeq,uSeq,qSeq],name='f_debug')
    print 'f_debug complete~'
    f_pred = theano.function([x,xmask,q,qmask,nhops,wmat], pred, name='f_pred')
    print 'f_pred complete~'
    f_ans = theano.function([x,xmask,q,qmask,nhops,wmat], pred_ans, name='f_ans')
    print 'f_ans complete~'
    
    return use_noise, x, xmask, q, qmask, y, nhops, wmat, proj, pred, pred_ans, cost, f_debug, f_pred, f_ans


def adadelta(lr, tparams, grads, x, xmask, q, qmask, y, nhops, wmat, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()] # grad values wrt each weight. shared variables inited by all zeros.
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()] # accumulated delta-x E(delta-x ^ 2) from t=1 to now. shared varialbes inited by all zeros.
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()] # accumulated grad value E(g ^ 2)

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2,g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([x, xmask, q, qmask, y, nhops, wmat], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared') 
                                    # f_grad_shared input x,mask,y,q, output cost, and execute the two updates above with real values.

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)] # this is the delta-x result
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)] # update E(x ^ 2) according to x.
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] # update tparams, counting the new weights.

    f_update = theano.function([lr], [], updates=ru2up + param_up, # f_update take as input the learning rate but ignore it. then execute the above two updates.
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update    
    
def TVSplit(story_out, q_out, ans_out, valid_num):
    totalNum = len(q_out)
    trainNum = int(totalNum * (1. - valid_num))
    idx = numpy.random.permutation(totalNum)
    story_train = [story_out[i] for i in idx[:trainNum]]
    story_valid = [story_out[i] for i in idx[trainNum:]]
    q_train = [q_out[i] for i in idx[:trainNum]]
    q_valid = [q_out[i] for i in idx[trainNum:]]
    ans_train = [ans_out[i] for i in idx[:trainNum]]
    ans_valid = [ans_out[i] for i in idx[trainNum:]]    
    # dataValid = dataTrain
    return story_train, story_valid, q_train, q_valid, ans_train, ans_valid

def pred_error(f_pred, prepareData, x_data, q_data, y_data, nhops, wmat, maxStoryLen, maxSentLen, print_flag, iterator):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepareData: usual prepareData for that dataset.
    """
    valid_err = 0
    targets = 0
    preds = 0
    for _, valid_index in iterator:
        x, xmask, q, qmask, y = prepareData([x_data[t] for t in valid_index],
                                  [q_data[t] for t in valid_index],
                                  [y_data[t] for t in valid_index],
                                  maxStoryLen, maxSentLen)
        preds = f_pred(x, xmask, q, qmask, nhops, wmat)
        targets = numpy.array(y_data)[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(x_data)
    if print_flag:
        print preds[-10:]
        print targets[-10:]

    return valid_err

def pred_probs(f_pred, prepareData, x_data, q_data, y_data, nhops, wmat, maxStoryLen, maxSentLen, iterator):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepareData: usual prepareData for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, xmask, q, qmask, y = prepareData([x_data[t] for t in valid_index],
                                  [q_data[t] for t in valid_index],
                                  [y_data[t] for t in valid_index],
                                  maxStoryLen, maxSentLen)
        preds = f_pred(x, xmask, q, qmask, nhops, wmat)
        print preds[1,:]
        print preds.shape

def debug_print(f_debug, f_grad, prepareData, x_data, q_data, y_data, nhops, wmat, maxStoryLen, maxSentLen, tparams, sliced_iterator):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepareData: usual prepareData for that dataset.
    """
    _, valid_index = sliced_iterator
    x, xmask, q, qmask, y = prepareData([x_data[t] for t in valid_index],
                                  [q_data[t] for t in valid_index],
                                  [y_data[t] for t in valid_index],
                                  maxStoryLen, maxSentLen)
    [x,xmask,q,qmask,y,nhops,wmat,proj,pred,pred_ans,aEmbSeq,bEmbSeq, uSeq, qSeq] = f_debug(x, xmask, q, qmask, y, nhops, wmat)
    fd = open('debug_print.txt', 'w')
    old = sys.stdout
    sys.stdout = fd
    
    print x_data[3]
    print('x: ')
    print(x[:,:,3])
    print('xmask: ')
    print(xmask[:,:,3])
    print('q: ')
    print(q[:,3])
    print('qmask: ')
    print(qmask[:,3])
    print('y: ')
    print(y[3])
    
    print('hop1 a: ')
    print(aEmbSeq[0,:,3,:])
    print('hop1 b: ')
    print(bEmbSeq[0,:,3,:])
    print('hop1 q: ')
    print(qSeq[0,3,:])
    
    print('hop2 a: ')
    print(aEmbSeq[1,:,3,:])
    print('hop2 b: ')
    print(bEmbSeq[1,:,3,:])
    print('hop2 q: ')
    print(qSeq[1,3,:])
    
    print('hop3 a: ')
    print(aEmbSeq[2,:,3,:])
    print('hop3 b: ')
    print(bEmbSeq[2,:,3,:])
    print('hop3 q: ')
    print(qSeq[2,3,:])
    
    print('proj: ')
    print(proj[3,:])
    print('pred: ')
    print(pred[3,:])
    print('pred_ans: ')
    print(pred_ans[3])
    
    temp_grads = f_grad(x, xmask, q, qmask, y, nhops, wmat)
    print 'grad_len:'
    print len(temp_grads)
    print('grads: ')
    for k,t in zip(tparams.keys(),temp_grads):
        print k
        print t
    print 'parameters:'
    for k,t in tparams.iteritems():
        print k
        print t.get_value()
    
    sys.stdout = old
    fd.close()

'''['Wemb_A', 'Wemb_B', 'Wclass', 'bclass', 'WGRU_A', 'Whai_A', 'Whaj_A', 'Whu_A', 'WGRU_B', 'Whbi_B', 'Whai_B', 'Whu_B', 'Whq_B', 'bGRU_A', 'bGRU_B', 'bGRUh_A', 'bGRUh_B']'''

def pred_error_20(f_pred, parseBabiTaskTest, prepareData, dict, test_list, nhops, wmat, maxStoryLen, maxSentLen, mem_size):
    err_list = []
    for fileName in test_list:
        story_out, q_out, ans_out = parseBabiTaskTest([fileName], dict, mem_size)
        x, xmask, q, qmask, y = prepareData(story_out, q_out, ans_out, maxStoryLen, maxSentLen)
        # print x.shape, q.shape
        preds = f_pred(x, xmask, q, qmask, nhops, wmat)
        targets = numpy.array(y)
        # print preds, targets
        valid_err = (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(y)
        print 'the test error rate of task %s is:' %fileName
        print valid_err
        err_list.append(valid_err)
    
    return err_list
    
def train_EvoMN(
    dim_word=100, # dim for word embedding, which is equivalent to GRU hidden layer of evolution.
    dim_hidden=100, # dim for reasoning, which is equivalent to the system status.
    vocab_size=200,
    nhops=3,
    mem_size=50,
    sent_size=100,
    batch_size=32,
    valid_batch_size=32,
    use_dropout=0,
    randomize_time=0.1,
    decay_c=0.,
    optimizer=adadelta,
    valid_num=0.1,
    validFreq=500,
    saveFreq=1000,
    dispFreq=300,
    max_epochs=100000,
    maxGrad=40,
    lrate=0.001,
    saveto='EvoMN_model.npz',
    patience=10
    
):
    model_options = locals().copy()
    
    train_list, test_list = preProcessBabi()
    dict = {}
    story_out, q_out, ans_out, dict, maxStoryLen, maxSentLen = parseBabiTaskTrain(train_list, dict, randomize_time, mem_size)
    story_train, story_valid, q_train, q_valid, ans_train, ans_valid = TVSplit(story_out, q_out, ans_out, valid_num)
    story_test, q_test, ans_test = parseBabiTaskTest(test_list, dict, mem_size)
    
    # get data size. get max lengths from the whole train set.
    model_options['vocab_size'] = len(dict) + maxStoryLen + 1
    model_options['mem_size'] = maxStoryLen
    model_options['sent_size'] = maxSentLen

    # print dict

    print 'vocab_size is %d; mem_size is %d; sent_size is %d' %(model_options['vocab_size'], model_options['mem_size'], model_options['sent_size'])
    
    # init params
    params = param_init(model_options)
    tparams = param_shared(params)
    
    # compute word time weight matrix
    wordTimeMat = computeTimeMat(maxSentLen, dim_word) # dim_word * (maxSentLen + 1)
    
    # build model and get containers
    use_noise, x, xmask, q, qmask, y, nhops_v, wmat, proj, pred, pred_ans, cost, f_debug, f_pred, f_ans = build_model_EvoMN(model_options, tparams)
    # do L2 nomalization
    if decay_c > 0.:
        if decay_c > 0.:
            decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
            weight_decay = 0.
            weight_decay += (tparams['Wclass'] ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay
    
    # cost and grad
    f_cost = theano.function([x, xmask, q, qmask, y, nhops_v, wmat], cost, name='f_cost')
    print 'f_cost complete~'
    grads_unclipped = tensor.grad(cost, wrt=tparams.values())
    grads = [(g / ((g ** 2).sum() ** 0.5)) * tensor.minimum((g ** 2).sum() ** 0.5, model_options['maxGrad']) for g in grads_unclipped]
    f_grad = theano.function([x, xmask, q, qmask, y, nhops_v, wmat], grads, name='f_grad')
    print 'f_grad complete~'
    
    print tparams.keys()
    
    # optimization
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, xmask, q, qmask, y, nhops_v, wmat, cost)

    # valid and test batches are fixed.
    kf_valid = get_minibatches_idx(len(q_valid), valid_batch_size)  # kf_valid: a list of tuples. [(id, batch_index)]
    kf_test = get_minibatches_idx(len(q_test), valid_batch_size)  # these two include all batch indices.
    print "%d train examples" % len(q_train)
    print "%d valid examples" % len(q_valid)
    print "%d test examples" % len(q_test)
    
    history_errs = []
    best_p = None
    bad_count = 0
    
    if validFreq == -1:
        validFreq = len(q_train) / batch_size
    if saveFreq == -1:
        saveFreq = len(q_train) / batch_size
        
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(q_train), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [ans_train[t] for t in train_index]
                x = [story_train[t] for t in train_index]
                q = [q_train[t] for t in train_index]

                
                x, xmask, q, qmask, y = prepareData(x, q, y, maxStoryLen, maxSentLen)
                n_samples += x.shape[2]
                # print x.shape, xmask.shape, q.shape, qmask.shape, len(y)
                # print wordTimeMat.shape

                cost = f_grad_shared(x, xmask, q, qmask, y, nhops, wordTimeMat)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'
                    # save mediate values with freq equal to saving params.
                    use_noise.set_value(0.)
                    debug_print(f_debug, f_grad, prepareData, story_test, q_test, ans_test, nhops, wordTimeMat, maxStoryLen, maxSentLen, tparams, kf_valid[0])
                    print 'Mediate values saved'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_ans, prepareData, story_train, q_train, ans_train, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0, kf)
                    valid_err = pred_error(f_ans, prepareData, story_valid, q_valid, ans_valid, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0,
                                           kf_valid)
                    # pred_probs(f_pred, prepareData, story_valid, q_valid, ans_valid, nhops, wordTimeMat, maxStoryLen, maxSentLen,
                    #                        kf_valid)
                    test_err = pred_error(f_ans, prepareData, story_test, q_test, ans_test, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"
    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(q_train), batch_size)
    train_err = pred_error(f_ans, prepareData, story_train, q_train, ans_train, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0, kf_train_sorted)
    valid_err = pred_error(f_ans, prepareData, story_valid, q_valid, ans_valid, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0, kf_valid)
    test_err = pred_error(f_ans, prepareData, story_test, q_test, ans_test, nhops, wordTimeMat, maxStoryLen, maxSentLen, 0, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    
    # test for each single task:
    test_err_20 = pred_error_20(f_ans, parseBabiTaskTest, prepareData, dict, test_list, nhops, wordTimeMat, maxStoryLen, maxSentLen, mem_size)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_EvoMN(
        max_epochs=10000,
    )
