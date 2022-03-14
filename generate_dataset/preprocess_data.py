from tensorflow.keras.preprocessing.text import Tokenizer
import os
import sys
import ast
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import *
from transformers import AutoTokenizer
from params import BERT_PRETRAINED, PKL_PATH

tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED)

parser = argparse.ArgumentParser()
parser.add_argument('--file_csv', type=str, required=False, default='autocorrection/dataset/Data/VNTC/sample.csv',
                    help='what is name for file text?')
arg = parser.parse_args()

def mask_word_correction(data, correct_token_id):
    label_errors = data.label_error.values.tolist()
    correct_token_id = data.correct_token_id.values.tolist()
    # replace the correct word = mask token id
    for i in range(len(label_errors)):
        for j in range(len(label_errors[i])):
            if label_errors[i][j] == 0:
                correct_token_id[i][j] = 0
    return correct_token_id


def calculate_ids(error_sentence):
    size_subwords = []
    input_ids = []
    size_words = []

    for sentence in error_sentence:
        temp = split_token(tokenizer, sentence)
        size_subwords.append(temp)
        input_ids.append(tokenizer.encode(sentence))
        size_words.append(get_size_word_in_sentence(sentence))
    return size_subwords, input_ids, size_words


def save_data(correct_token_id,
              error_token_id,
              char_error_token_id,
              input_ids,
              label_errors,
              size_subwords,
              size_words,
              n_words,
              n_chars,
              ):
    save_picke_file(
        PKL_PATH + 'phobert.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': input_ids,
            'label_errors': label_errors,
            'size_subwords': size_subwords,
            'n_words': n_words
        }
    )

    save_picke_file(
        PKL_PATH + 'trans.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': error_token_id,
            'label_errors': label_errors,
            'char_token_ids': char_error_token_id,
            'size_word': size_words,
            'n_words': n_words,
            'n_chars': n_chars
        }
    )

    save_picke_file(
        PKL_PATH + 'char_trans.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': error_token_id,
            'label_errors': label_errors,
            'char_token_ids': char_error_token_id,
            'size_word': size_words,
            'n_words': n_words,
            'n_chars': n_chars
        }
    )
    print("Save done")


def tokenize_sentence(data):
    ori_sentence = data.original_sentence.values.tolist()
    error_sentence = data.error_sentence.values.tolist()
    all_texts = ori_sentence + error_sentence

    print("Start tokenize")
    word_tokenizer = Tokenizer(oov_token='<unk>', lower=True)
    word_tokenizer.fit_on_texts(all_texts)
    char_tokenizer = Tokenizer(char_level=True, oov_token='<unk>', lower=True)
    char_tokenizer.fit_on_texts(all_texts)

    word_tokenizer.index_word[0] = '<mask>'
    word_tokenizer.word_index['<mask>'] = 0
    char_tokenizer.word_index['<mask>'] = 0
    char_tokenizer.index_word[0] = '<mask>'
    print("End tokenize")
    save_picke_file(PKL_PATH+'word_tokenizer.pkl', word_tokenizer)
    save_picke_file(PKL_PATH+'char_tokenizer.pkl', char_tokenizer)
    return word_tokenizer, char_tokenizer


def calculate_token_id(word_tokenizer, char_tokenizer, data):
    print("Start calculate token id")
    ori_sentence = data.original_sentence.values.tolist()
    error_sentence = data.error_sentence.values.tolist()
    correct_token_id = [word_tokenizer.texts_to_sequences([text])[0] for text in ori_sentence]
    error_token_id = [word_tokenizer.texts_to_sequences([text])[0] for text in error_sentence]
    char_error_token_id = [char_tokenizer.texts_to_sequences([text])[0] for text in error_sentence]

    data['correct_token_id'] = correct_token_id
    data['label_error'] = data['label_error'].apply(lambda x: ast.literal_eval(x))
    data['error_token_id'] = error_token_id
    data['char_error_token_id'] = char_error_token_id

    label_errors = data['label_error'].values.tolist()
    correct_token_id = data['correct_token_id'].values.tolist()
    error_token_id = data['error_token_id'].values.tolist()

    idx_errors = []
    for i in range(len(data)):
        if (len(label_errors[i]) != len(correct_token_id[i])) or (len(label_errors[i]) != len(error_token_id[i])):
            idx_errors.append(i)
    data = data.drop(idx_errors, axis=0)
    data = data.reset_index(drop=True)

    correct_token_id = mask_word_correction(data, correct_token_id)
    print("End calculate token id")
    # correct_token_id = data.correct_token_id.values.tolist()

    return data

# def remove_error_tokens():
#     data = load_file_picke(PKL_PATH+'trans.pkl')
#     trans = pd.DataFrame()
#     trans['correct_token_ids'] = data['correct_token_ids']
#     trans['error_ids'] = data['error_ids']
#     trans['label_errors'] = data['label_errors']
#     trans['label_errors'] = trans['label_errors'].apply(lambda x: ast.literal_eval(x))

#     idx_errors = []
#     for i in range(len(trans)):
#         if len(trans['correct_token_ids'][i]) != len(trans['label_errors'][i]):
#             idx_errors.append(i)
#     trans = trans.drop(idx_errors, axis=0)

#     data_trans = trans.to_dict('list')
#     data_trans['n_words'] = data['n_words']
#     save_picke_file(PKL_PATH+'trans.pkl', data_trans)
#     print('Remove error tokens successfully!')
#     print(f'Seen {len(trans)} sentences')

def main():
    data = pd.read_csv(arg.file_csv)
    word_tokenizer, char_tokenizer = tokenize_sentence(data)
    data = calculate_token_id(word_tokenizer, char_tokenizer, data)
    

    _ , input_ids, size_words = calculate_ids(data.error_sentence)
    save_data(
        data.correct_token_id.tolist(), 
        data.error_token_id.tolist(),
        data.char_error_token_id.tolist(),
        input_ids, 
        data.label_error.tolist(),
        size_words, 
        size_words,
        len(word_tokenizer.word_index),
        len(char_tokenizer.word_index)
              )

if __name__ == '__main__':
    main()
    # remove_error_tokens()
