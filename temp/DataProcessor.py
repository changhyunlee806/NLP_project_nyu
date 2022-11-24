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


    def padByLength(self, sentence, maxLen):
        sentence = sentence[-maxLen:]
        sentence += [Constants.PAD] * (maxLen - len(sentence))
        return sentence


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


    # mask: 없으면 1, 있으면 0 > 없는 곳을 알려주는
    def getMELDdata(self, filePath):

        allUtterances, allSpeakerIds, allEmotionIdxes, allMask, allLastTurns = [], [], [], [], []

        speakerNames = vocab.UnkVocab.from_dict(torch.load(Constants.DataPaths['speaker_vocab_path']))
        emotionVocab = vocab.Vocab.from_dict(torch.load(Constants.DataPaths['emotion_vocab_path']))
        meldData = pd.read_csv(filePath)

        utterances, fullContents = [], []
        speakerIds, emtionIdxes = [], []

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
                allUtterances.append(fullContents)
                fullContents = []
                allSpeakerIds.append(speakerIds)
                speakerIds = []
                allEmotionIdxes.append(emtionIdxes)
                emtionIdxes = []
                maxLen = max(maxLen, len(utterances))
                utterances = []

            speakerNameIdx = speakerNames.word2index(speakerName)
            emotionIdx = emotionVocab.word2index(emotion)
            tokenIds = self.tokenizer(utterance, add_special_tokens=False)['input_ids'] + self.SEP

            fullContent = []
            if len(utterances):
                # TODO check preUttr
                fullContent += utterances[-3:]
                # for preUttr in utterances[-3:]:
                #     fullContents.extend(preUttr)
            fullContent.extend(tokenIds)

            # TODO change to question
            query = 'Now ' + speakerName + ' feels <mask>'
            queryIds = self.tokenizer(query, add_special_tokens=False)['input_ids'].extend(self.SEP)
            fullContent.extend(queryIds)

            # TODO pad to len
            fullContent = self.padByLength(fullContent, maxLen)
            fullContents.append(fullContent)
            utterances.append(tokenIds)
            speakerIds.append(speakerNameIdx)
            emtionIdxes.append(emotionIdx)

            prevDialogueId = dialogueId

        # TODO add pad to len
        padUtterance = self.SEP + self.tokenizer("1", add_special_tokens=False)['input_idx'] + self.SEP
        padUtterance = self.padByLength(padUtterance, maxLen)

        dialogueId = 0
        while dialogueId < len(allUtterances):
            utterances = allUtterances[dialogueId]
            mask = [1 for _ in range(len(utterances))]

            # padding
            length = len(utterances)
            while length < maxLen:
                utterances.append(padUtterance)
                length += 1
                mask.append(0)
                allEmotionIdxes[dialogueId].append(-1)
                allSpeakerIds[dialogueId].append(0)
            allUtterances[dialogueId] = utterances
            allMask.append(mask)

            lastTurns = [-1 for _ in range(maxLen)]
            for turnId in range(0, maxLen):
                curSpeaker = allSpeakerIds[dialogueId][turnId]
                if curSpeaker == 0:
                    break
                for idx in range(turnId-1, -1, -1):
                    if curSpeaker == allSpeakerIds[dialogueId][idx]:
                        lastTurns[turnId] = idx
                        break
            allLastTurns.append(lastTurns)

        return TensorDataset(
            torch.LongTensor(allUtterances),
            torch.LongTensor(allSpeakerIds),
            torch.LongTensor(allEmotionIdxes),
            torch.ByteTensor(allMask),
            torch.LongTensor(allLastTurns)
        )