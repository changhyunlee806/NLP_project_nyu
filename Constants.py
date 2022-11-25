
DataPaths = {
    'speaker_vocab_path': 'vocabs/speaker_vocab.pkl',
    'emotion_vocab_path': 'vocabs/emtion_vocab.pkl',
    'sentiment_vocab_path': 'vocabs/sentiment_vocab.pkl',
    'additional_vocab': 'friends_transcript.json'
}

CONFIG = {
    'bert_path': 'roberta-base',
    'epochs' : 20,
    'lr' : 1e-4,
    'ptmlr' : 5e-6,
    'batch_size' : 1,
    'max_len' : 256,
    'max_value_list' : 16,
    'bert_dim' : 1024,
    'pad_value' : 1,
    'shift' : 1024,
    'dropout' : 0.3,
    'p_unk': 0.1,
    'data_splits' : 20,
    'num_classes' : 7,
    'wp' : 1,
    'wp_pretrain' : 5,
    'data_path' : '../MELD/data/MELD/',
    'accumulation_steps' : 8,
    'rnn_layers' : 2,
    'tf_rate': 0.8,
    'aux_loss_weight': 0.3,
    'ngpus' : torch.cuda.device_count(),
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}