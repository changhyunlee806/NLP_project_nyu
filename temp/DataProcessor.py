import vocab
import json
import pandas as pd
import torch
from transformers import AutoTokenizer

from tqdm import tqdm

import Constants

'''
would be removed after checking
'''

class DataProcessor:

    def __init__(self, bertPath):
        self.tokenizer = AutoTokenizer.from_pretrained(bertPath)
        specialTokenIds = self.tokenizer('')['input_ids']
        self.CLS = specialTokenIds[0]
        self.SEP = specialTokenIds[1]


    def getVocabs(self, train, val, test, additional): # dev == validation?, additional data?
        speakerNames = vocab.UnkVocab() # names of speaker > Unk
        emotionVocab = vocab.Vocab()
        sentimentVocab = vocab.Vocab()

        # index of 'neutral': 0
        emotionVocab.word2index('neutral', train = True)

        # train data
        trainData = pd.read_csv(train)
        for row in tqdm(trainData.iterrows()):
            emotion = row[1]['Emotion'].lower()
            emotionVocab.word2index(emotion, train=True)

        # val data
        valData = pd.read_csv(val)
        for row in tqdm(valData.iterrows()):
            emotion = row[1]['Emotion'].lower()
            emotionVocab.word2index(emotion, train=True)

        # test data
        testData = pd.read_csv(test)
        for row in tqdm(testData.iterrows()):
            emotion = row[1]['Emotion'].lower()
            emotionVocab.word2index(emotion, train=True)

        # additional
        additionalData = json.load(open(additional, 'r'))
        for episodeId in additionalData:
            for scene in additionalData.get(episodeId):
                for utterence in scene['utterances']:
                    speakerName = utterence['speakers'][0].lower()
                    speakerNames.word2index(speakerName, train=True)
        speakers = list(speakerNames.prune_by_count(1000).counts.keys())
        speakerNames = vocab.UnkVocab()
        for speaker in speakers:
            speakerNames.word2index(speaker, train=True)

        torch.save(emotionVocab.to_dict(), Constants.DataPaths['emotion_vocab_path'])
        torch.save(speakerNames.to_dict(), Constants.DataPaths['speaker_vocab_path'])
        torch.save(sentimentVocab.to_dict(), Constants.DataPaths['sentiment_vocab_path'])


    def getMELDdata(self, filePath):
        speakerNames = vocab.UnkVocab.from_dict(torch.load(Constants.DataPaths['speaker_vocab_path']))
        emotionVocab = vocab.Vocab.from_dict(torch.load(Constants.DataPaths['emotion_vocab_path']))
        meldData = pd.read_csv(filePath)

        utterances, fullContents = [], []
        speakerIds, emtionIdxes = [], []
        utterancesAll, speakerIdsAll, emtionIdxesAll = [], [], []

        maxLen = 0
        prevDialogueId = meldData[0][1]['Dialogue_ID']
        for row in tqdm(meldData.iterrows()):
            meta = row[1]
            utterance = meta['Utterance'].replace('’', '\'').replace("\"", '')
            speakerName = meta['Speaker']
            utterance = speakerName + ' says:, ' + utterance
            emotion = meta['Emotion'].lower()
            dialogueId, utteranceId = meta['Dialogue_ID'], meta['Utterance_ID']

            if dialogueId != prevDialogueId:
                utterancesAll.append(fullContents)
                fullContents = []
                speakerIdsAll.append(speakerIds)
                speakerIds = []
                emtionIdxesAll.append(emtionIdxes)
                emtionIdxes = []
                maxLen = max(maxLen, len(utterances))
                utterances = []


            speakerNameIdx = speakerNames.word2index(speakerName)
            emotionIdx = emotionVocab.word2index(emotion)
            tokenIds = self.tokenizer(utterance, add_special_tokens=False)['input_ids'] + self.SEP

            fullContents = []
            if len(utterances):
                # TODO check preUttr
                fullContents += utterances[-3:]
                # for preUttr in utterances[-3:]:
                #     fullContents.extend(preUttr)
            fullContents.extend(tokenIds)

            query = 'Now ' + speakerName + ' feels <mask>'
            queryIds = self.tokenizer(query, add_special_tokens=False)['input_ids'].extend(self.SEP)
            fullContents.extend(queryIds)

            fullContents.append()
            utterances.append(tokenIds)
            speakerIds.append(speakerNameIdx)
            emtionIdxes.append(emotionIdx)

            prevDialogueId = dialogueId

        padUtterance = self.SEP + self.tokenizer()['input_idx'] + self.SEP









