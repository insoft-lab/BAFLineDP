import re
import torch
import pandas as pd

all_train_releases = {'activemq': 'activemq-5.0.0',
                      'camel': 'camel-1.4.0',
                      'derby': 'derby-10.2.1.6',
                      'groovy': 'groovy-1_5_7',
                      'hbase': 'hbase-0.94.0',
                      'hive': 'hive-0.9.0',
                      'jruby': 'jruby-1.1',
                      'lucene': 'lucene-2.3.0',
                      'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'],
                     'hive': ['hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
                'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
                'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
                'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

file_lvl_gt = 'datasets/preprocessed_data/'

def get_df(rel, is_baseline=False):
    if is_baseline:
        df = pd.read_csv('../' + file_lvl_gt + rel + ".csv")
    else:
        df = pd.read_csv(file_lvl_gt + rel + ".csv")

    df = df.fillna('')

    df = df[df['is_blank'] == False]
    df = df[df['is_test_file'] == False]

    return df

def prepare_code2d(code_list, to_lowercase=False):
    code2d = []

    for c in code_list:
        c = re.sub('\\s+', ' ', c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        code2d.append(token_list)

    return code2d
    
def get_code3d_and_label(df, to_lowercase=False, max_sent_length=None):
    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):
        group_df = group_df[group_df['code_line'].ne('')]
        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        if max_sent_length:
            code2d = prepare_code2d(code, to_lowercase)[:max_sent_length]
        else:
            code2d = prepare_code2d(code, to_lowercase)

        code3d.append(code2d)
        all_file_label.append(file_label)

    return code3d, all_file_label