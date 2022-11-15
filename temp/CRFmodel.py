from transformers import AutoTokenizer, AutoModel
from CRF import *

class CRFModel(nn.Module):

    def __init__(self, numClasses, dropout, bert_path):
        super().__init__()
        self.numClasses = numClasses
        self.dropout = dropout
        self.padValue = 1 # pad value
        # CLS
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.CLS = tokenizer('')['input_ids'][0]
        self.encoder = AutoModel.from_pretrained(bert_path)
        self.dimension = self.encoder.embeddings.word_embeddings.weight.data.shape[-1]
        self.spkEmbeddings = nn.Embedding(300, self.dimension)
        self.CRFlayer = CRF(self.numClasses)
        self.emission = nn.Linear(self.dimension, self.numClasses)
        self.lossFunc = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def device(self):
        return self.encoder.device

    def forward(self, sentences, sentencesMask, speakerIds, lastTurns, emotionIdxes=None):
    #def forward(self, sentences, sentencesMask, speakerIds, emotionIdxes):
        batchSize, maxTurns = sentences.shape[0], sentences.shape[1]

        speakerIdsReshaped = speakerIds.reshape(batchSize * maxTurns, -1)
        sentencesReshaped = sentences.reshape(batchSize * maxTurns, -1)

        clsId = torch.ones_like(speakerIdsReshaped) * self.CLS
        inputIds = torch.cat((clsId, sentencesReshaped), 1)
        mask = 1 - (inputIds == (self.padValue)).long()

        utteranceEncoded = self.encoder(
            input_ids=inputIds,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']

        maskPos = mask.sum(1)-2
        features = utteranceEncoded[torch.arange(maskPos.shape[0]), maskPos, :]
        emissions = self.emission(features)
        crfEmissions = emissions.reshape(batchSize, maxTurns, -1).transpose(0, 1)
        sentencesMask = sentencesMask.transpose(0, 1)
        speakerIds = speakerIds.reshape(batchSize, maxTurns).transpose(0, 1)
        lastTurns = lastTurns.transpose(0, 1)

        # train
        if emotionIdxes is not None:
            emotionIdxes = emotionIdxes.transpose(0, 1)
            return -self.CRFlayer(crfEmissions, emotionIdxes, mask=sentencesMask) + self.lossFunc(emissions.view(-1, self.numClasses), emotionIdxes.view(-1))
        else:
            return self.CRFlayer.decode(crfEmissions, mask=sentencesMask)
