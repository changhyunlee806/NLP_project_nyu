import vocab
import json
import pandas as pd
import torch

from tqdm import tqdm

import Constants

'''
would be removed after checking
'''

class DataProcessor:

    def getVocabs(train, val, test, additional): # dev == validation?, additional data?
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


    def getMELDdata(filePath):
        speakerNames = vocab.UnkVocab.from_dict(torch.load(Constants.DataPaths['speaker_vocab_path']))
        emotionVocab = vocab.Vocab.from_dict(torch.load(Constants.DataPaths['emotion_vocab_path']))
        meldData = pd.read_csv(filePath)

        prevDialogueId = meldData[0][1]['Dialogue_ID']
        for row in tqdm(meldData.iterrows()):
            meta = row[1]
            utterance = meta['Utterance'].replace('’', '\'').replace("\"", '')
            speakerName = meta['Speaker']
            utterance = speakerName + ' says:, ' + utterance
            emotion = meta['Emotion'].lower()
            dialogueId, utteranceId = meta['Dialogue_ID'], meta['Utterance_ID']

            if dialogueId != prevDialogueId:
                pass

            speakerNameIdx = speakerNames.word2index(speakerName)
            emotionIdx = emotionVocab.word2index(emotion)







