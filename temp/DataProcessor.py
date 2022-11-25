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

    def pad_to_len(self, list_data, max_len, pad_value):
        list_data = list_data[-max_len:]
        len_to_pad = max_len - len(list_data)
        pads = [pad_value] * len_to_pad
        list_data.extend(pads)
        return list_data


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
        speaker_vocab = vocab.UnkVocab.from_dict(torch.load(
            Constants.DataPaths['speaker_vocab_path']
        ))
        emotion_vocab = vocab.Vocab.from_dict(torch.load(
            Constants.DataPaths['emotion_vocab_path']
        ))

        data = pd.read_csv(file_path)
        #ret_utterances = []
        #ret_speaker_ids = []
        #ret_emotion_idxs = []
        utterances = []
        full_contexts = []
        speaker_ids = []
        emotion_idxs = []
        pre_dial_id = -1
        max_turns = 0
        for row in tqdm(data.iterrows(), desc='processing file {}'.format(file_path)):
            meta = row[1]
            utterance = meta['Utterance'].replace(
                '’', '\'').replace("\"", '')
            speaker = meta['Speaker']
            utterance = speaker + ' says:, ' + utterance
            emotion = meta['Emotion'].lower()
            dialogue_id = meta['Dialogue_ID']
            utterance_id = meta['Utterance_ID']
            if pre_dial_id == -1:
                pre_dial_id = dialogue_id
            if dialogue_id != pre_dial_id:
                allUtterances.append(full_contexts)
                allSpeakerIds.append(speaker_ids)
                allEmotionIdxes.append(emotion_idxs)
                max_turns = max(max_turns, len(utterances))
                utterances = []
                full_contexts = []
                speaker_ids = []
                emotion_idxs = []
            pre_dial_id = dialogue_id
            speaker_id = speaker_vocab.word2index(speaker)
            emotion_idx = emotion_vocab.word2index(emotion)
            token_ids = self.tokenizer(utterance, add_special_tokens=False)[
                            'input_ids'] + [self.SEP]
            full_context = []
            if len(utterances) > 0:
                context = utterances[-3:]
                for pre_uttr in context:
                    full_context += pre_uttr
            full_context += token_ids
            # query
            query = 'Now ' + speaker + ' feels <mask>'
            query_ids = self.tokenizer(query, add_special_tokens=False)['input_ids'] + [self.SEP]
            full_context += query_ids

            full_context = self.padByLength(
                full_context, Constants.MAX_LEN, Constants.PAD)
            # + CONFIG['shift']
            utterances.append(token_ids)
            full_contexts.append(full_context)
            speaker_ids.append(speaker_id)
            emotion_idxs.append(emotion_idx)

        pad_utterance = [self.SEP] + self.tokenizer(
            "1",
            add_special_tokens=False
        )['input_ids'] + [self.SEP]
        pad_utterance = self.padByLength(
            pad_utterance, Constants.MAX_LEN, Constants.PAD)
        # for CRF
        #ret_mask = []
        #ret_last_turns = []
        for dial_id, utterances in tqdm(enumerate(allUtterances), desc='build dataset'):
            mask = [1] * len(utterances)
            while len(utterances) < max_turns:
                utterances.append(pad_utterance)
                allEmotionIdxes[dial_id].append(-1)
                allSpeakerIds[dial_id].append(0)
                mask.append(0)
            allMasks.append(mask)
            allUtterances[dial_id] = utterances

            last_turns = [-1] * max_turns
            for turn_id in range(max_turns):
                curr_spk = allSpeakerIds[dial_id][turn_id]
                if curr_spk == 0:
                    break
                for idx in range(0, turn_id):
                    if curr_spk == allSpeakerIds[dial_id][idx]:
                        last_turns[turn_id] = idx
            allLastTurns.append(last_turns)
        dataset = TensorDataset(
            torch.LongTensor(allUtterances),
            torch.LongTensor(allSpeakerIds),
            torch.LongTensor(allEmotionIdxes),
            torch.ByteTensor(allMasks),
            torch.LongTensor(allLastTurns)
        )
        return dataset