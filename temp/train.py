from config import *
from CRFmodel import CRFModel
import itertools
from DataProcessor import DataProcessor
import Constants
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

speaker_vocab_dict_path = 'vocabs/speaker_vocab.pkl'
emotion_vocab_dict_path = 'vocabs/emotion_vocab.pkl'
sentiment_vocab_dict_path = 'vocabs/sentiment_vocab.pkl'


def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = CONFIG['ptmlr']
    '''
    frozen_params = []
    frozen_layers = [3,4,5,6,7,8]
    for layer_idx in frozen_layers:
        frozen_params.extend(
            list(map(id, model.context_encoder.encoder.layer[layer_idx].parameters()))
        )
    '''
    bert_params = list(map(id, model.encoder.parameters()))
    crf_params = list(map(id, model.CRFlayer.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        # if id(param) in frozen_params:
        #     continue
        lr = CONFIG['lr']
        weight_decay = 0
        if id(param) in bert_params:
            lr = pre_train_lr
        if id(param) in crf_params:
            lr = CONFIG['lr'] * 10
        if not any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append(
            {
                'params': param,
                'lr': lr,
                'weight_decay': weight_decay
            }
        )

        warmup_params.append(
            {
                'params': param,
                'lr': 0 if id(param) in bert_params else lr,
                'weight_decay': weight_decay
            }
        )
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'], reverse=True)
    return params


def train_epoch(model, optimizer, data, epoch_num=0, max_step=-1):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=RandomSampler(data),
        num_workers=0  # multiprocessing.cpu_count()
    )

    for batch in tqdm(dataloader):
        for i in range(len(batch)):
            if i == 0:
                sentences = batch[i].to(model.device())
                continue
            elif i == 1:
                speaker_ids = batch[i].to(model.device())
                continue
            elif i == 2:
                emotion_idxs = batch[i].to(model.device())
                continue
            elif i == 3:
                mask = batch[i].to(model.device())
                continue
            elif i == 4:
                last_turns = batch[i].to(model.device())
                continue
        loss = model(sentences, mask, speaker_ids, last_turns, emotion_idxs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, data):
    yPred = []
    yTrue = []
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=0)

    for batch in dataloader:
        for i in range(len(batch)):
            if i == 0:
                sentences = batch[i].to(model.device())
                continue
            elif i == 1:
                speaker_ids = batch[i].to(model.device())
                continue
            elif i == 2:
                emotion_idxs = batch[i].to(model.device())
                continue
            elif i == 3:
                mask = batch[i].to(model.device())
                continue
            elif i == 4:
                last_turns = batch[i].to(model.device())
                continue
        out = model(sentences, mask, speaker_ids, last_turns, None)

        maskBatch = torch.arange(0, mask.shape[0])
        maskSequence = torch.arange(0, mask.shape[1])

        for batch1, sequence1 in torch.cartesian_prod(maskBatch, maskSequence):
            # Only if not padded (aka. there is information) -> mask==1 (True), APPEND
            if bool(mask[batch1][sequence1]) != True:
                continue
            else:
                yPred.append(out[batch1][sequence1])
                yTrue.append(emotion_idxs[batch1][sequence1].cpu())

    score = f1_score(y_pred=yPred, y_true=yTrue, average='weighted')
    print('confusion matrix')
    print(confusion_matrix(yTrue, yPred))
    print("classification_report")
    print(classification_report(yTrue, yPred))
    model.train()
    return score

'''
# create confusion matrix on test and valid data
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test_enc, y_test_pred_xgb), 'test')
print(confusion_matrix(y_valid_enc, y_valid_pred_xgb), 'valid')

# show classification report on test and valid data
from sklearn.metrics import classification_report
print("Test")
print(classification_report(y_test_enc, y_test_pred_xgb))
print("Valid")
print(classification_report(y_valid_enc, y_valid_pred_xgb))
'''

def train(model, train_data_path, dev_data_path, test_data_path, dataProcessor):
    devset = dataProcessor.getMELDdata(dev_data_path)
    testset = dataProcessor.getMELDdata(test_data_path)
    trainset = dataProcessor.getMELDdata(train_data_path)

    optimizer = torch.optim.AdamW(get_paramsgroup(model, warmup=True))
    for epoch in range(CONFIG['wp']):
        train_epoch(model, optimizer, trainset, epoch_num=epoch)
        torch.cuda.empty_cache()
        f1 = test(model, devset)
        torch.cuda.empty_cache()
        print('f1 on dev @ warmup epoch {} is {:.4f}'.format(
            epoch, f1), flush=True)
    # train
    optimizer = torch.optim.AdamW(get_paramsgroup(model))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    best_f1 = -1
    tq_epoch = tqdm(total=CONFIG['epochs'], position=0)
    for epoch in range(CONFIG['epochs']):
        tq_epoch.set_description('training on epoch {}'.format(epoch))
        tq_epoch.update()
        train_epoch(model, optimizer, trainset, epoch_num=epoch)
        torch.cuda.empty_cache()
        f1 = test(model, devset)
        torch.cuda.empty_cache()
        print('f1 on dev @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # '''
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model,
                       'models/f1_{:.4f}_@epoch{}.pkl'
                       .format(best_f1, epoch))
        if lr_scheduler.get_last_lr()[0] > 1e-5:
            lr_scheduler.step()
        f1 = test(model, testset)
        print('f1 on test @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # f1 = test(model, test_on_trainset)
        # print('f1 on train @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # '''
    tq_epoch.close()
    lst = os.listdir('./models')
    lst = list(filter(lambda item: item.endswith('.pkl'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)))
    model = torch.load(os.path.join('models', lst[-1]))
    f1 = test(model, testset)
    print('best f1 on test is {:.4f}'.format(f1), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-te', '--test', action='store_true',
                        help='run test', default=False)
    parser.add_argument('-tr', '--train', action='store_true',
                        help='run train', default=False)
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='fine tune base the best model', default=False)
    parser.add_argument('-pr', '--print_error', action='store_true',
                        help='print error case', default=False)
    parser.add_argument('-bsz', '--batch', help='Batch_size',
                        required=False, default=CONFIG['batch_size'], type=int)
    parser.add_argument('-epochs', '--epochs', help='epochs',
                        required=False, default=CONFIG['epochs'], type=int)
    parser.add_argument('-lr', '--lr', help='learning rate',
                        required=False, default=CONFIG['lr'], type=float)
    parser.add_argument('-p_unk', '--p_unk', help='prob to generate unk speaker',
                        required=False, default=CONFIG['p_unk'], type=float)
    parser.add_argument('-ptmlr', '--ptm_lr', help='ptm learning rate',
                        required=False, default=CONFIG['ptmlr'], type=float)
    parser.add_argument('-tsk', '--task_name', default='meld', type=str)
    parser.add_argument('-fp16', '--fp_16', action='store_true',
                        help='use fp 16', default=False)
    parser.add_argument('-wp', '--warm_up', default=CONFIG['wp'],
                        type=int, required=False)
    parser.add_argument('-dpt', '--dropout', default=CONFIG['dropout'],
                        type=float, required=False)
    parser.add_argument('-e_stop', '--eval_stop',
                        default=500, type=int, required=False)
    parser.add_argument('-bert_path', '--bert_path',
                        default=CONFIG['bert_path'], type=str, required=False)
    parser.add_argument('-data_path', '--data_path',
                        default=CONFIG['data_path'], type=str, required=False)
    parser.add_argument('-acc_step', '--accumulation_steps',
                        default=CONFIG['accumulation_steps'], type=int, required=False)

    # read the arguments from commandline
    args = parser.parse_args()
    CONFIG['data_path'] = args.data_path
    CONFIG['lr'] = args.lr
    CONFIG['ptmlr'] = args.ptm_lr
    CONFIG['epochs'] = args.epochs
    CONFIG['bert_path'] = args.bert_path
    CONFIG['batch_size'] = args.batch
    CONFIG['dropout'] = args.dropout
    CONFIG['wp'] = args.warm_up
    CONFIG['p_unk'] = args.p_unk
    CONFIG['accumulation_steps'] = args.accumulation_steps
    CONFIG['task_name'] = args.task_name
    train_data_path = os.path.join(CONFIG['ext_train_path'], 'original_5000.csv')
    test_data_path = os.path.join(CONFIG['data_path'], 'test_sent_emo.csv')
    dev_data_path = os.path.join(CONFIG['data_path'], 'dev_sent_emo.csv')

    os.makedirs('vocabs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    dataProcessor = DataProcessor('roberta-base')
    dataProcessor.getVocabs(train_data_path, dev_data_path, test_data_path, Constants.DataPaths['additional_vocab'])
    # model = PortraitModel(CONFIG)
    model = CRFModel(numClasses=7, dropout=0.3, bert_path='roberta-base')
    device = CONFIG['device']
    model.to(device)
    print('---config---')
    for k, v in CONFIG.items():
        print(k, '\t\t\t', v, flush=True)
    if args.finetune:
        lst = os.listdir('./models')
        lst = list(filter(lambda item: item.endswith('.pkl'), lst))
        lst.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)))
        model = torch.load(os.path.join('models', lst[-1]))
        print('checkpoint {} is loaded'.format(
            os.path.join('models', lst[-1])), flush=True)
    if args.train:
        train(model, train_data_path, dev_data_path, test_data_path, dataProcessor)
    if args.test:
        # testset = load_meld_and_builddataset(dev_data_path)
        if args.task_name == 'meld':
            testset = load_meld_and_builddataset(test_data_path)
        best_f1 = test(model, testset)
        print(best_f1)
