import os, argparse
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from BAFLineDP import *
from my_util import *

class InputFeatures(object):
    def __init__(self, input_ids, label):
        self.input_ids = input_ids
        self.label = label

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, datasets, labels):
        self.examples = []
        labels = torch.FloatTensor(labels)
        for dataset, label in zip(datasets, labels):
            dataset_ids = [convert_examples_to_features(item, tokenizer, args) for item in dataset]
            self.examples.append(InputFeatures(dataset_ids, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), self.examples[i].label

def convert_examples_to_features(item, tokenizer, args):
    code = ' '.join(item)
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return source_ids

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def predict_defective_files_in_releases(args, dataset_name, all_eval_rels_cross_projects):
    actual_save_model_dir = args.save_model_dir + dataset_name + '/'
    actual_prediction_dir = args.prediction_dir + dataset_name + '/'

    if not os.path.exists(actual_prediction_dir):
        os.makedirs(actual_prediction_dir)

    train_rel = all_train_releases[dataset_name]
    test_rel = all_eval_rels_cross_projects[dataset_name]

    # load the pre-trained CodeBERT model
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        codebert = model_class.from_pretrained(args.model_name_or_path,
                                               from_tf=bool('.ckpt' in args.model_name_or_path),
                                               config=config,
                                               cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        codebert = model_class(config)

    model = BAFLineDP(
        embed_dim=args.embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        bafn_output_dim=args.bafn_hidden_dim,
        dropout=args.dropout,
        device=args.device
    )

    checkpoint = torch.load(actual_save_model_dir + 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    sig = nn.Sigmoid()

    codebert.to(args.device)
    model.to(args.device)
    model.eval()

    for rel in test_rel:
        print('using model from {} to generate prediction of {}'.format(train_rel, rel))

        test_df = get_df(rel)
        row_list = []

        for filename, df in tqdm(test_df.groupby('filename')):
            df = df[df['code_line'].ne('')]

            file_label = bool(df['file-label'].unique())
            line_label = df['line-label'].tolist()
            line_number = df['line_number'].tolist()
            is_comments = df['is_comment'].tolist()

            code = df['code_line'].tolist()

            # avoid memory overflow
            drop_length = 35000
            if len(code) > drop_length:
                continue

            code2d = prepare_code2d(code, True)
            code3d = [code2d]
            codevec = TextDataset(tokenizer, args, code3d, [file_label])

            with torch.no_grad():
                input = torch.tensor(codevec.examples[0].input_ids)

                limit_length = 1000
                input = input.split(limit_length, 0)

                cov_input = [codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1)).pooler_output
                             for item in input]

                cov_input = torch.cat(cov_input, dim=0)

                output, line_att_weight = model([cov_input])
                file_prob = sig(output).item()
                prediction = bool(round(file_prob))

            torch.cuda.empty_cache()

            numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

            for i in range(0, len(code)):
                cur_line = code[i]
                cur_line_label = line_label[i]
                cur_line_number = line_number[i]
                cur_is_comment = is_comments[i]
                cur_line_attn = numpy_line_attn[i]

                row_dict = {
                    'project': dataset_name,
                    'train': train_rel,
                    'test': rel,
                    'filename': filename,
                    'file-level-ground-truth': file_label,
                    'prediction-prob': file_prob,
                    'prediction-label': prediction,
                    'line-number': cur_line_number,
                    'line-level-ground-truth': cur_line_label,
                    'is-comment-line': cur_is_comment,
                    'code-line': cur_line,
                    'line-attention-score': cur_line_attn
                }
                row_list.append(row_dict)

        df = pd.DataFrame(row_list)
        df.to_csv(actual_prediction_dir + train_rel + '-' + rel + '.csv', index=False)
        print('finished release', rel)

def main():
    all_eval_rels_cross_projects = {
        'activemq': ['camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2',
                     'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
        'camel': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2',
                  'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1',
                  'wicket-1.5.3'],
        'derby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                  'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
                  'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
        'groovy': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                   'derby-10.5.1.1', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
                   'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
        'hbase': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                  'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
                  'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
        'hive': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1',
                 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0',
                 'lucene-3.1', 'wicket-1.5.3'],
        'jruby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                  'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'lucene-3.0.0', 'lucene-3.1',
                  'wicket-1.5.3'],
        'lucene': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                   'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0',
                   'jruby-1.7.0.preview1', 'wicket-1.5.3'],
        'wicket': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0',
                   'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0',
                   'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1']
    }

    arg = argparse.ArgumentParser()

    arg.add_argument('-file_lvl_gt', type=str, default='datasets/preprocessed_data/',
                     help='the directory of preprocessed data')
    arg.add_argument('-save_model_dir', type=str, default='output/model/BAFLineDP/',
                     help='the save directory of model')
    arg.add_argument('-prediction_dir', type=str, default='output/prediction/BAFLineDP/cross-release/',
                     help='the results directory of prediction')

    arg.add_argument('-embed_dim', type=int, default=768, help='the input dimension of Bi-GRU')
    arg.add_argument('-gru_hidden_dim', type=int, default=64, help='hidden size of GRU')
    arg.add_argument('-gru_num_layers', type=int, default=1, help='number of GRU layer')
    arg.add_argument('-bafn_hidden_dim', type=int, default=256, help='output dimension of BAFN')
    arg.add_argument('-max_grad_norm', type=int, default=5, help='max gradient norm')
    arg.add_argument('-use_layer_norm', type=bool, default=True, help='weather to use layer normalization')
    arg.add_argument('-seed', type=int, default=0, help='random seed for initialization')
    arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')

    arg.add_argument('-model_type', type=str, default='roberta', help='the token embedding model')
    arg.add_argument('-model_name_or_path', type=str, default='microsoft/codebert-base',
                     help='the model checkpoint for weights initialization')
    arg.add_argument('-config_name', type=str, default=None,
                     help='optional pretrained config name or path if not the same as model_name_or_path')
    arg.add_argument('-tokenizer_name', type=str, default='microsoft/codebert-base',
                     help='optional pretrained tokenizer name or path if not the same as model_name_or_path')
    arg.add_argument('-cache_dir', type=str, default=None,
                     help='optional directory to store the pre-trained models')
    arg.add_argument('-block_size', type=int, default=75,
                     help='the training dataset will be truncated in block of this size for training')
    arg.add_argument('-do_lower_case', action='store_true', help='set this flag if you are using an uncased model')

    args = arg.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    dataset_names = list(all_releases.keys())
    for dataset_name in dataset_names:
        predict_defective_files_in_releases(args, dataset_name, all_eval_rels_cross_projects)

if __name__ == "__main__":
    main()