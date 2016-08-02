# for unit test in EvoMemNN, mainly to debug memLayer module
# by yyq, 2016-06-03

import numpy
import theano
from theano import config
import theano.tensor as tensor

def memLayers(tparams, x, xmask, wmat):

    # assert xmask is not None
    
    # look up table
    emb_A = tparams[x.flatten()].reshape([x.shape[0], x.shape[1], x.shape[2], 50]) # storyLen * sentLen * nsamples * dimword
    emb_A_m = emb_A * xmask.dimshuffle(0,1,2,'x').repeat(50)
    
    
    # create sentence embedding from word and time embedding
    emb_A_sf = emb_A_m.dimshuffle(0,2,3,1)# storyLen * nsamples * dim_word
    emb_A_w = emb_A_sf * wmat
    emb_A_s = emb_A_w.sum(axis=3)
    return emb_A_s
    
    
    # define memory hops
    '''def step(a_emb, b_emb, u_vec, q_vec, xmask_story):
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
    return states[0], states[1], states[2], states[3]'''

def build_model_EvoMN():
    tparams = tensor.matrix('tparams', dtype=config.floatX)
    
    x = tensor.tensor3('x', dtype='int64') # x is n_sent * n_word * n_samples
    xmask = tensor.tensor3('xmask', dtype=config.floatX) # same as x
    wmat = tensor.matrix('wmat',dtype=config.floatX) # dim_word * (maxSentLen+1)
    
    emb_A_s = memLayers(tparams, x, xmask,  wmat)
    f = theano.function([tparams,x,xmask,wmat], emb_A_s, name='f')
    return f
    
    '''proj = qSeq[-1] # nsamples * dim_hidden
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['Wclass']) + tparams['bclass']) # nsamples * vocab_size
    pred_ans = pred.argmax(axis=1) # nsamples vector
    off = 1e-8
    cost = -tensor.log(pred[tensor.arange(y.shape[0]), y] + off).mean()
    
    f_debug = theano.function([x,xmask,q,qmask,y,nhops,wmat], [x,xmask,q,qmask,y,nhops,wmat,proj,pred,pred_ans,aEmbSeq,bEmbSeq,uSeq,qSeq],name='f_debug')
    print 'f_debug complete~'
    f_pred = theano.function([x,xmask,q,qmask,nhops,wmat], pred, name='f_pred')
    print 'f_pred complete~'
    f_ans = theano.function([x,xmask,q,qmask,nhops,wmat], pred_ans, name='f_ans')
    print 'f_ans complete~'
    
    return use_noise, x, xmask, q, qmask, y, nhops, wmat, proj, pred, pred_ans, cost, f_debug, f_pred, f_ans'''
    
    
f = build_model_EvoMN()
tparams = numpy.random.randn(225,50).astype(config.floatX)
x = numpy.random.randint(low=0,high=224,size=(50,12,32)).astype('int64')
xmask = numpy.random.randn(50,12,32).astype(config.floatX)
wmat = numpy.random.randn(50,12).astype(config.floatX)
result = f(tparams, x, xmask,wmat)
print result
print result.shape
    
