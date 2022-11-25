import vocab
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
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
        pads = [Constants.PAD] * (maxLen - len(sentence))
        sentence.extend(pads)
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
    def getMELDdata(self, file_path, train=False):

        allUtterances, allSpeakerIds, allEmotionIdxes, allMasks, allLastTurns = [], [], [], [], []

        speakerNames = vocab.UnkVocab.from_dict(torch.load(Constants.DataPaths['speaker_vocab_path']))
        emotionVocab = vocab.Vocab.from_dict(torch.load(Constants.DataPaths['emotion_vocab_path']))

        meldData = pd.read_csv(file_path)

        utterances, fullContents = [], []
        speakerIds, emotionIdxes = [], []
        prevDialogueId = meldData.iloc[0]['Dialogue_ID']
        maxTurns = 0
        for row in tqdm(meldData.iterrows()):
            meta = row[1]
            utterance = meta['Utterance'].replace('’', '\'').replace("\"", '')
            speaker = meta['Speaker']
            utterance = speaker + ' says:, ' + utterance
            emotion = meta['Emotion'].lower()
            dialogueId, utteranceId = meta['Dialogue_ID'], meta['Utterance_ID']

            # if prevDialogueId == -1:
            #     prevDialogueId = dialogueId
            if dialogueId != prevDialogueId:
                allUtterances.append(fullContents)
                fullContents = []
                allSpeakerIds.append(speakerIds)
                speakerIds = []
                allEmotionIdxes.append(emotionIdxes)
                emotionIdxes = []
                maxTurns = max(maxTurns, len(utterances))
                utterances = []
            prevDialogueId = dialogueId

            speakerId = speakerNames.word2index(speaker)
            emotionIdx = emotionVocab.word2index(emotion)
            tokenIds = self.tokenizer(utterance, add_special_tokens=False)['input_ids'] + [self.SEP]
            fullContent = []
            if len(utterances) > 0:
                context = utterances[-3:]
                #fullContent.extend(context)
                for pre_uttr in context:
                    fullContent += pre_uttr
            fullContent += tokenIds
            # query
            query = 'Now ' + speaker + ' feels <mask>'
            queryIds = self.tokenizer(query, add_special_tokens=False)['input_ids'] + [self.SEP]
            fullContent += queryIds

            fullContent = self.padByLength(fullContent, Constants.MAX_LEN)
            utterances.append(tokenIds)
            fullContents.append(fullContent)
            speakerIds.append(speakerId)
            emotionIdxes.append(emotionIdx)

        pad_utterance = [self.SEP] + self.tokenizer("1", add_special_tokens=False)['input_ids'] + [self.SEP]
        pad_utterance = self.padByLength(pad_utterance, Constants.MAX_LEN)

        for dialId in range(0, len(allUtterances)):
            utterances = allUtterances[dialId]
            mask = [1 for _ in range(len(utterances))]
        #for dialId, utterances in tqdm(enumerate(allUtterances), desc='build dataset'):
        #    mask = [1] * len(utterances)

            length = len(utterances)
            while length < maxTurns:
                utterances.append(pad_utterance)
                length += 1
                mask.append(0)
                allEmotionIdxes[dialId].append(-1)
                allSpeakerIds[dialId].append(0)

            allMasks.append(mask)
            allUtterances[dialId] = utterances

            last_turns = [-1] * maxTurns
            for turn_id in range(maxTurns):
                curr_spk = allSpeakerIds[dialId][turn_id]
                if curr_spk == 0:
                    break
                for idx in range(0, turn_id):
                    if curr_spk == allSpeakerIds[dialId][idx]:
                        last_turns[turn_id] = idx
            allLastTurns.append(last_turns)

        return TensorDataset(
            torch.LongTensor(allUtterances),
            torch.LongTensor(allSpeakerIds),
            torch.LongTensor(allEmotionIdxes),
            torch.ByteTensor(allMasks),
            torch.LongTensor(allLastTurns)
        )
