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
        # batchSize, maxTurns = sentences.shape[0], sentences.shape[1]

        # speakerIdsReshaped = speakerIds.reshape(batchSize * maxTurns, -1)
        # sentencesReshaped = sentences.reshape(batchSize * maxTurns, -1)

        #works
        sentBatchSize, sentMaxTurns, sentMaxLen = sentences.shape[0], sentences.shape[1], sentences.shape[2]
        speakerBatchSize, speakerMaxTurns = speakerIds.shape[0], speakerIds.shape[1]

        sentInputRowNum = sentBatchSize*sentMaxTurns
        speakerInputRowNum = speakerBatchSize*speakerMaxTurns

        sentencesReshaped = sentences.reshape(sentInputRowNum, -1)
        speakerIdsReshaped = speakerIds.reshape(speakerInputRowNum, -1) #changed 6:42pm
        #works

        #changed
        clsId = torch.ones(speakerIdsReshaped.size(), dtype=speakerIdsReshaped.dtype, \
                            layout=speakerIdsReshaped.layout, device=speakerIdsReshaped.device) * self.CLS
        inputIds = torch.concat(tensors=(clsId, sentencesReshaped), dim=1)
        #inputIds = torch.cat((clsId, sentencesReshaped), 1)
        #mask = 1 - (inputIds == (self.padValue)).long()
        
        # mask is used to avoid/ignore padded values of the input tensor
        # masking indices should be {0: if padded, 1: if not padded}
        attentionMask = torch.where(inputIds == self.padValue, 0, 1)
        #changed 7:29pm
        

        utteranceEncoded = self.encoder(
            input_ids=inputIds,
            attention_mask=attentionMask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']

        maskPos = torch.sum(input=attentionMask, dim=1, keepdim=False) - 2
        # change below
        features = utteranceEncoded[torch.arange(maskPos.shape[0]), maskPos, :]
        emissions = self.emission(features)
        crfEmissions = torch.transpose(emissions.reshape(sentBatchSize, sentMaxTurns, -1), dim0=0, dim1=1)


        sentencesMask = torch.transpose(sentencesMask, dim0=0, dim1=1)
        # check if it runs, if not it may mean speaker and sentence batch size are different
        speakerIds = torch.transpose(speakerIds.reshape(speakerBatchSize, speakerMaxTurns), dim0=0, dim1=1) # 6:55pm changed from speakerBatchSize to sentBatchSize
        lastTurns = torch.transpose(lastTurns, dim0=0, dim1=1)

        # train
        if emotionIdxes is not None:
            emotionIdxes = torch.transpose(emotionIdxes, dim0=0, dim1=1)
            return -self.CRFlayer(crfEmissions, emotionIdxes, mask=sentencesMask) + self.lossFunc(emissions.view(-1, self.numClasses), emotionIdxes.view(-1))
        else:
            return self.CRFlayer.decode(crfEmissions, mask=sentencesMask)