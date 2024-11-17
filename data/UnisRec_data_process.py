import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
import pandas as pd
from tqdm import tqdm

from utils import check_path, set_device, load_plm, amazon_dataset2fullname


def load_ratings(df):
    users, items, inters = set(), set(), set()
    # with open(file, 'r') as fp:

    for row in tqdm(df.itertuples(), desc='Load ratings'):
        # if i >= max_lines:  # 限制读取的行数
        #     break
        try:
            user, item, rating, time = row.userId, row.itemId, row.rate, row.date
            users.add(user)
            items.add(item)
            inters.add((user, item, float(rating), time))
        except ValueError:
            print(row)
    return users, items, inters
def parse(path_tmp):
    g = gzip.open(path_tmp, "rb")
    for line in g:
        yield json.loads(line)

def getDF(path_tmp, format):
    if format == "json":
        i = 0
        df = {}
        for d in parse(path_tmp):
            df[i] = d
            i += 1
        df = pd.DataFrame.from_dict(df, orient="index")
        df = df[["asin", "title", "description"]]
        return df
    elif format == "txt":
        df = pd.read_csv(path_tmp, compression="gzip", header=None, sep=" ", names=["reviewerID", "asin", "ratings", "date"] )
        return df

def load_meta_items(file,max_lines):
    items = set()
    with gzip.open(file, 'r') as fp:
        for i, line in enumerate(tqdm(fp, desc='Load metas')):
        # for line in tqdm(fp, desc='Load metas'):
            if i >= max_lines:  # 限制读取的行数
                break
            data = json.loads(line)
            items.add(data['asin'])
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args,m):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)
    # load ratings
    # txt_file_path = os.path.join(args.input_path)
    df = pd.read_csv(f'{m}'+'_5core.txt', sep=' ')
    # rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(df)

    # load item IDs with meta data
    # meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    # meta_df = getDF(meta_file_path, 'json')
    # # meta_items = load_meta_items(meta_file_path,max_lines)
    # meta_items=meta_df[meta_df['asin'].isin(itemset)]

    # txt_file_path = os.path.join(args.input_path)
    # itemset = set()
    # df = pd.read_csv('../raw/ca'+'_5core.txt', sep=' ')
    # itemset = itemset.union(set(df.itemId.unique()))
    # selected_meta_items = {item for item in meta_items if item in itemset}
    # # 1. Filter items w/o meta data;
    # # 2. K-core filtering;
    # print('The number of raw inters: ', len(rating_inters))
    # rating_inters = filter_inters(rating_inters, can_items=meta_items,
    #                               user_k_core_threshold=args.user_k,
    #                               item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for i, line in enumerate(ratings):
        user, item, rating, time = line
        users.add(user)
        items.add(item)
    return users, items


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def generate_text(args, items, features):
    item_text_list = []
    already_items = set()

    dataset_full_name = amazon_dataset2fullname[args.dataset]
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    with gzip.open(meta_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Generate text'):
            data = json.loads(line)
            item = data['asin']
            if item in items and item not in already_items:
                already_items.add(item)
                text = ''
                for meta_key in features:
                    if meta_key in data:
                        meta_value = clean_text(data[meta_key])
                        text += meta_value + ' '
                item_text_list.append([item, text])
    return item_text_list


def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items, ['title', 'category', 'brand'])
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        if len(inters)>=3:
            train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
            valid_inters[u_index] = [str(inters[-2])]
            test_inters[u_index] = [str(inters[-1])]
            assert len(user2items[u_index]) == len(train_inters[u_index]) + \
                len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def generate_item_embedding(args, item_text_list, item2index, tokenizer, model,m, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)


    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    output_dir = os.path.join(args.output_path, m)
    os.makedirs(output_dir, exist_ok=True)
    file = os.path.join(args.output_path, m,
                        m + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


def convert_to_atomic_files(args, train_data, valid_data, test_data,market_list):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, m, f'{m}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, m, f'{m}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, m, f'{m}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Scientific', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='./raw/')
    parser.add_argument('--output_path', type=str, default='./downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='/workspaces/HKUST_Docker/bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    market_list = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk', 'us']
    # market_list = [ 'uk', 'us']
    max_lines=5000
    # load interactions from raw rating file
    for m in market_list:
        rating_inters = preprocess_rating(args,m)

        # load item text from raw meta data file
        item_text_list = preprocess_text(args, rating_inters)

        # split train/valid/test
        train_inters, valid_inters, test_inters, user2index, item2index = \
            generate_training_data(args, rating_inters)

        # device & plm initialization
        device = set_device(args.gpu_id)
        args.device = device
        plm_tokenizer, plm_model = load_plm(args.plm_name)
        plm_model = plm_model.to(device)

        # create output dir
        check_path(os.path.join(args.output_path, m))

        # generate PLM emb and save to file
        generate_item_embedding(args, item_text_list, item2index,
                                plm_tokenizer, plm_model,m, word_drop_ratio=-1)
        # pre-stored word drop PLM embs
        if args.word_drop_ratio > 0:
            generate_item_embedding(args, item_text_list, item2index,
                                    plm_tokenizer, plm_model,m, word_drop_ratio=args.word_drop_ratio)

        # # save interaction sequences into atomic files
        # convert_to_atomic_files(args, train_inters, valid_inters, test_inters,m)


        # item_text_list_sub=[(item, text) for item, text in item_text_list if item in itemset]
        write_text_file(item_text_list, os.path.join(args.output_path, m, f'{m}.text'))
        # user2index_sub = [(user, index) for user, index in user2index if user in Userset]
        write_remap_index(user2index, os.path.join(args.output_path, m, f'{m}.user2index'))
        # item2index_sub=[(item, index) for item, index in item2index if item in itemset]
        write_remap_index(item2index, os.path.join(args.output_path, m, f'{m}.item2index'))