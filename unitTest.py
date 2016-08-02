# for unit test in EvolutionMemNN.py
# yyq, 2016-06-02

from collections import OrderedDict
import re
import os
import copy
import numpy

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
        print fileName
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
        
        print max(fileStoryLens)
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
    
    
    for i in range(len(story_out)):
        # cut
        if len(story_out[i]) > maxStoryLen:
            story_out[i] = copy.deepcopy(story_out[i][(-maxStoryLen):])
        
        tempStyLen = len(story_out[i])
        nblank = numpy.random.randint(0, int(numpy.ceil(tempStyLen * randomize_time)))
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
    dict['nil'] = 1 # unks
    for fileName in fileList:
        print fileName
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
    print 'total_questions in train: %d' %qCount
    
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
    x = numpy.zeros([maxStoryLen, maxSentLen + 1, nSamples]) # maxSentLen + 1 for time embedding.
    xmask = numpy.zeros([maxStoryLen, maxSentLen + 1, nSamples])
    q = numpy.zeros([maxSentLen, nSamples])
    qmask = numpy.zeros([maxSentLen, nSamples])
    y = ansList
    
    print maxStoryLen,maxSentLen
    
    
    for i in range(nSamples):

        
        for j in range(maxStoryLen):
            if j < len(storyList[i]) - 1:
                x[j,0:len(storyList[i][j]) - 1,i] = storyList[i][j][:-1]
                x[j,-1,i] = storyList[i][j][-1]
                xmask[j,:(len(storyList[i][j]) - 1),i] = 1.
                xmask[j,-1,i] = 1

                
        
        q[:len(qList[i]),i] = qList[i]
        qmask[:len(qList[i]),i] = 1.
    
    return x, xmask, q, qmask, y
    
def TVSplit(dataIn, valid_num):
    totalNum = len(dataIn)
    trainNum = int(totalNum * (1. - valid_num))
    idx = numpy.random.permutation(totalNum)
    dataTrain = [dataIn[i] for i in idx[:trainNum]]
    dataValid = [dataIn[i] for i in idx[trainNum:]]
    return dataTrain, dataValid



train_list, test_list = preProcessBabi()
print 'train_list:'
print train_list
print 'test_list:'
print test_list
dict = {}
randomize_time = 0.1
memSize = 50
story_out, q_out, ans_out, dict, maxStoryLen, maxSentLen = parseBabiTaskTrain(train_list, dict, randomize_time, memSize)
print 'the second story in train_10:'
print story_out[1]
print 'the second q in train_10:'
print q_out[1]
print 'the second ans in train_10:'
print ans_out[1]
print 'len(dict) is %d; maxStoryLen is %d; maxSentLen is %d' %(len(dict), maxStoryLen, maxSentLen)
valid_num = 0.1
story_train, story_valid = TVSplit(story_out, valid_num)
print '%d train and %d valid after split' %(len(story_train), len(story_valid))
q_train, q_valid = TVSplit(q_out, valid_num)
ans_train, ans_valid = TVSplit(ans_out, valid_num)
story_test, q_test, ans_test = parseBabiTaskTest(test_list, dict, memSize)
print 'len(dict) after test parse: %d' %(len(dict))
print 'the second story in test_10:'
print story_test[1]
print 'the second q in test_10:'
print q_test[1]
print 'the second ans in test_10:'    
print ans_out[1]
x, xmask, q, qmask, y = prepareData(story_out, q_out, ans_out, maxStoryLen, maxSentLen)
print 'train shape:'
print x.shape, xmask.shape, q.shape, qmask.shape
print 'train second array:'
print x[2,:,1]
print xmask[2,:,1]
print 'train second q:'
print q[:,1]
print qmask[:,1]
x, xmask, q, qmask, y = prepareData(story_test, q_test, ans_test, maxStoryLen, maxSentLen)   
print 'test second array:'
print x[2,:,1]
print xmask[2,:,1]
print 'test second q:'
print q[:,1]
print qmask[:,1]
print 'test second y:'
print y[1]