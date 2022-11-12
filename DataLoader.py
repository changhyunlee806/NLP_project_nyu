import vocab
import json
import pandas as pd
import torch

from tqdm import tqdm

class DataLoader:

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
        file = open(additional)
        for episodeId in json.load(file):
            for scene in json.load(file).get(episodeId):
                for utterence in scene['utterances']:
                    speakerName = utterence['speakers'][0].lower()
                    speakerNames.word2index(speakerName, train=True)
        speakers = list(speakerNames.prune_by_count(1000).counts.keys())
        for speaker in speakers:
            speakerNames.word2index(speaker, train=True)

        torch.save(emotionVocab.to_dict(), 'path')
        torch.save(speakerNames.to_dict(), 'path')
        torch.save(sentimentVocab.to_dict(), 'path')





