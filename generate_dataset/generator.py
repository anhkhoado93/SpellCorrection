import numpy as np
import unidecode
from nltk.tokenize import word_tokenize
import string
import re
import random
import json
import os
from random import shuffle

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class GeneratorDataset:
    def __init__(self):
        self.tokenizer = word_tokenize
        self.words_dict = self.load_file_text('autocorrection/dataset/words_dict.txt')

        with open('autocorrection/dataset/words_similar.json', 'r') as file:
            self.words_similar = json.load(file)
        
        with open('autocorrection/dataset/syllable_similar.json', 'r') as file:
            self.words_homophone = json.load(file)
        
        with open('autocorrection/dataset/pair_last_character.json','r') as file:
            self.pair_character = json.load(file)
         
        with open('autocorrection/dataset/pair_syllable.json','r') as file:
            self.pair_syllabel = json.load(file)
        
        with open('autocorrection/dataset/teencode_dict_2.json','r') as file:
            self.teencode_dict = json.load(file)
        
        with open('autocorrection/dataset/typos.json','r') as file:
            self.typos = json.load(file)

        with open('autocorrection/dataset/typo.json','r') as file:
            self.typo = json.load(file)
        
        with open('autocorrection/dataset/key_neighbors.json','r') as file:
            self.key_neighbors = json.load(file)

        self.word_couples = [['sương', 'xương'], ['sĩ', 'sỹ'], ['sẽ', 'sẻ'], ['sã', 'sả'], ['sả', 'xả'], ['sẽ', 'sẻ'],
                             ['mùi', 'muồi'],
                             ['chỉnh', 'chỉn'], ['sữa', 'sửa'], ['chuẩn', 'chẩn'], ['lẻ', 'lẽ'], ['chẳng', 'chẵng'],
                             ['cổ', 'cỗ'],
                             ['sát', 'xát'], ['cập', 'cặp'], ['truyện', 'chuyện'], ['xá', 'sá'], ['giả', 'dả'],
                             ['đỡ', 'đở'],
                             ['giữ', 'dữ'], ['giã', 'dã'], ['xảo', 'sảo'], ['kiểm', 'kiễm'], ['cuộc', 'cục'],
                             ['dạng', 'dạn'],
                             ['tản', 'tảng'], ['ngành', 'nghành'], ['nghề', 'ngề'], ['nổ', 'nỗ'], ['rảnh', 'rãnh'],
                             ['sẵn', 'sẳn'],
                             ['sáng', 'xán'], ['xuất', 'suất'], ['suôn', 'suông'], ['sử', 'xử'], ['sắc', 'xắc'],
                             ['chữa', 'chửa'],
                             ['thắn', 'thắng'], ['dỡ', 'dở'], ['trải', 'trãi'], ['trao', 'trau'], ['trung', 'chung'],
                             ['thăm', 'tham'],
                             ['sét', 'xét'], ['dục', 'giục'], ['tả', 'tã'], ['sông', 'xông'], ['sáo', 'xáo'],
                             ['sang', 'xang'],
                             ['ngã', 'ngả'], ['xuống', 'suống'], ['xuồng', 'suồng']]
        self.all_word_candidates = self.get_all_word_candidates(self.word_couples)

        self.vn_alphabet = ['a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô',
                            'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y']

        self.bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ'],
                               ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
                               ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
                               ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
                               ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
                               ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
                               ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
                               ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
                               ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
                               ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
                               ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
                               ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']]
        self.nguyen_am_to_ids = {}
        self.list_i = ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị']
        self.list_y = ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']
        self.list_aa = ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ']
        for i in range(len(self.nguyen_am_to_ids)):
            for j in range(len(self.bang_nguyen_am[i])):
                self.nguyen_am_to_ids[self.bang_nguyen_am[i][j]] = (i, j)
        self.nguyen_am = []
        for list_nguyen_am in self.bang_nguyen_am:
            self.nguyen_am.extend(list_nguyen_am)

        # self.all_chars_candidate = self.get_all_char_candidate()

    def is_number(self, token):
        if token.isnumeric():
            return True
        return bool(re.match('(\d+[\.,])+\d', token))

    def is_date(self, token):
        return bool(re.match('(\d+[-.\/])+\d+', token))

    def is_special_token(self, token):
        return bool(re.match(
            '([a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+[\+\*\^\@\#\.\&\/-])+[a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+',
            token))

    def check_special_token(self, sentence: str):
        tokens = self.tokenizer(sentence)
        index_special = {}
        for i in range(len(tokens)):
            if self.is_number(tokens[i]):
                index_special[tokens[i]] = 'numberic'
                tokens[i] = 'numberic'
            elif self.is_date(tokens[i]):
                index_special[tokens[i]] = 'date'
                tokens[i] = 'date'
            elif self.is_special_token(tokens[i]):
                index_special[tokens[i]] = 'specialw'  # mark differ 'special' word
                tokens[i] = 'specialw'
        return " ".join(tokens), index_special

    def get_index(self, word):
        for i, char in enumerate(word):
            if char == 'i' and i > 0 and word[i - 1] == 'g':
                if i != len(word) - 1:
                    return i + 1
            elif char == 'u' and i > 0 and word[i - 1] == 'q':
                return i + 1
            elif char in self.nguyen_am:
                return i
        return -1


    def get_all_word_candidates(self, word_couples):

        all_word_candidates = []
        for couple in self.word_couples:
            all_word_candidates.extend(couple)
        return all_word_candidates
    
    def replace_teencode(self, word):
        candidates = self.teencode_dict.get(word, None)
        if candidates is not None:
            chosen_one = 0
            if len(candidates) > 1:
                chosen_one = np.random.randint(0, len(candidates))
            return candidates[chosen_one]

    def replace_word_candidate(self, word):
        """
        Return a homophone word of the input word.
        """
        capital_flag = word[0].isupper()
        word = word.lower()
        if capital_flag and word in self.teencode_dict:
            return self.replace_teencode(word).capitalize()
        elif word in self.teencode_dict:
            return self.replace_teencode(word)

        for couple in self.word_couples:
            for i in range(2):
                if couple[i] == word:
                    if i == 0:
                        if capital_flag:
                            return couple[1].capitalize()
                        else:
                            return couple[1]
                    else:
                        if capital_flag:
                            return couple[0].capitalize()
                        else:
                            return couple[0]

    def replace_with_homophone_word(self, text, onehot_label):
        """
        Replace a candidate word (if exist in the word_couple) with its homophone. if successful, return True, else False
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, text, onehot_label
        """
        # account for the case that the word in the text is upper case but its lowercase match the candidates list
        candidates = []
        for i in range(len(text)):
            if text[i].lower() in self.all_word_candidates or text[i].lower() in self.teencode_dict.keys():
                candidates.append((i, text[i]))

        if len(candidates) == 0:
            return text, onehot_label

        idx = np.random.randint(0, len(candidates))
        prevent_loop = 0
        while onehot_label[candidates[idx][0]] == 1:
            idx = np.random.choice(np.arange(0, len(candidates)))
            prevent_loop += 1
            if prevent_loop > 5:
                return text, onehot_label
        text[candidates[idx][0]] = self.replace_word_candidate(candidates[idx][1])
        onehot_label[candidates[idx][0]] = 1
        return text, onehot_label

    def replace_char_candidate_typo(self, char):
        """
        return a homophone char/subword of the input char.
        """

        i = np.random.randint(0, len(self.typo[char]))

        return self.typo[char][i]

    def replace_with_typo_letter_2(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        # find index noise
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.add_noiseadd_noiseadd_noisetuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        index_noise = idx
        onehot_label[index_noise] = 1

        word_noise = text[index_noise]
        for j in range(0, len(word_noise)):
            char = word_noise[j]

            if char in self.typo:
                replaced = self.replace_char_candidate_typo(char)
                word_noise = word_noise[: j] + replaced + word_noise[j + 1:]
                text[index_noise] = word_noise
                return True, text, onehot_label
        return True, text, onehot_label
    # def get_all_char_candidate(self):
    #     list_char_homophones = []
    #     for pair_char in self.char_homophones:
    #         list_char_homophones.extend(pair_char)
    #     return list_char_homophones

    # def replaced_char_candidate(self, char_candidate):
    #     for pair_char in self.char_homophones:
    #         for i in range(2):
    #             if pair_char[i] == char_candidate:
    #                 return pair_char[(i + 1) % 2]

    # def is_valid_vietnam_word(self, word):
    #     chars = list(word)
    #     nguyen_am_index = -1

    #     for index, char in enumerate(chars):
    #         x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
    #         if x != -1:
    #             if nguyen_am_index == -1:
    #                 nguyen_am_index = index
    #             else:
    #                 if index - nguyen_am_index != 1:
    #                     return False
    #                 nguyen_am_index = index
    #     return True

    # def gen_word(self, word):
    #     chars = list(word)
    #     qu_or_gi = False
    #     dau_cau_index = -1
    #     temp = {}
    #     for index, char in enumerate(chars):
    #         x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
    #         if x != -1:
    #             if (x == 9 and index != 0 and chars[index - 1] == 'q') or (
    #                     x == 5 and index != 0 and chars[index - 1] == 'g'):
    #                 qu_or_gi = True
    #             else:
    #                 if y != 0:
    #                     dau_cau_index = index
    #                 temp[index] = (x, y)
    #     if len(temp) > 1 and dau_cau_index != -1:
    #         x_dau, y_dau = temp[dau_cau_index]
    #         del temp[dau_cau_index]
    #         random_index = random.choice(list(temp.keys()))
    #         chars[random_index] = self.bang_nguyen_am[temp[random_index][0]][y_dau]
    #         chars[dau_cau_index] = self.bang_nguyen_am[x_dau][0]

    #     return "".join(chars)

    # def generate_unicode_error(self, word):
    #     if self.is_valid_vietnam_word(word) is False:
    #         return word
    #     new_word = self.gen_word(word)
    #     return new_word
    def load_file_text(self, file_name):
        with open(file_name, 'r') as file:
            data = file.readlines()
            for idx,word in enumerate(data):
                data[idx] = word.replace('\n',"")
        return data

    def replace_teencode_word(self, word_candidate):
        list_results = self.teencode_dict[word_candidate]
        return random.choice(list_results)

    def generate_word_similar(self, word_candidate):
        # if random.random() < 0.3:
        #     return self.generate_unicode_error(word_candidate)
        # else:
        #     list_words_replace = self.word_homophones.get(word_candidate,None)
        #     if list_words_replace is not None:
        #         word_replace = random.choice(list_words_replace)
        #         return word_replace
        #     else:
        #         return word_candidate
        list_words_replace = self.words_similar.get(word_candidate, None)
        if list_words_replace is not None and len(list_words_replace) > 0:
            shuffle(list_words_replace)
            word_replace = random.choice(list_words_replace)
            return word_replace
        else:
            return word_candidate

    def replaced_word_candidate(self, word_candidate):
        word_candidate = word_candidate.lower()
        if word_candidate in self.teencode_dict and random.random() < 0.4:
            return self.replace_teencode_word(word_candidate)
        else:
            return self.generate_word_similar(word_candidate)

    def replace_with_typo_letter(self, tokens: list, onehot_labels: list):
        idx = np.random.randint(0, len(onehot_labels))
        prevent_loop = 0
        while onehot_labels[idx] == 1 or tokens[idx].isnumeric() or tokens[idx] in string.punctuation:
            idx = np.random.randint(0, len(tokens))
            prevent_loop += 1
            if prevent_loop >= 10:
                break
        if onehot_labels[idx] == 0:
            word_modify = tokens[idx]
            candidates = []
            for i in range(len(word_modify)):
                char = word_modify[i]
                if char in self.typos:
                    candidates.append((i, char))
            if len(candidates) > 0:
                shuffle(candidates)
                index, char_modify = random.choice(candidates)
                char_replace = self.typos[char_modify]
                flag = False
                end_character = ('f', 'r', 'j', 's', 'x')
                if char_replace.endswith(end_character):
                    flag = True
                if flag:
                    word_modify = word_modify[:index] + char_replace[:-1] + word_modify[index + 1:] + char_replace[-1]
                else:
                    word_modify = word_modify[:index] + char_replace + word_modify[index + 1:]

                tokens[idx] = word_modify
                onehot_labels[idx] = 1

        return tokens, onehot_labels

    def replace_with_typo_close_letter(self, tokens: list, onehot_labels: list):
        idx = np.random.randint(0, len(onehot_labels))
        prevent_loop = 0
        while onehot_labels[idx] == 1 or tokens[idx].isnumeric() or tokens[idx] in string.punctuation:
            idx = np.random.randint(0, len(tokens))
            prevent_loop += 1
            if prevent_loop >= 10:
                break
        if onehot_labels[idx] == 0:
            word_modify = tokens[idx]
            candidates = []
            for i in range(len(word_modify)):
                char = word_modify[i]
                if char in self.typos:
                    candidates.append((i, char))
            if len(candidates) > 0:
                shuffle(candidates)
                index, char_modify = random.choice(candidates)
                char_replace = self.typos[char_modify]
                flag = False
                end_character = ('f', 'r', 'j', 's', 'x', 'w')
                if char_replace.endswith(end_character):
                    flag = True
                if flag:
                    word_modify = word_modify[:index] + char_replace[:-1] + word_modify[index + 1:] + char_replace[-1]
                    neighbor_replace = random.choice(self.key_neighbors.get(char_replace[-1], char_replace[-1]))
                    word_modify = re.sub(char_replace[-1], neighbor_replace, word_modify)
                else:
                    word_modify = word_modify[:index] + char_replace + word_modify[index + 1:]

                tokens[idx] = word_modify
                
                onehot_labels[idx] = 1

        return tokens, onehot_labels

    def replace_with_homophone_syllable(self, tokens: list, onehot_labels: list):
        candidates = []
        for i in range(len(tokens)):
            if onehot_labels[i] == 0:
                if tokens[i].lower() in self.words_homophone.keys():
                    candidates.append((i, tokens[i]))
        if len(candidates) > 0:
            idx = np.random.randint(0, len(candidates))
            list_word_replaces = self.words_homophone.get(candidates[idx][1], None)
            shuffle(list_word_replaces)
            if list_word_replaces is not None and len(list_word_replaces) > 0:
                word_replace = random.choice(list_word_replaces)
                index_modify = candidates[idx][0]
                tokens[index_modify] = word_replace
                onehot_labels[index_modify] = 1

        return tokens, onehot_labels

    def replace_with_word(self, tokens: list, onehot_labels: list):
        candidates = []
        for i in range(len(tokens)):
            if onehot_labels[i] == 0:
                if tokens[i].lower() in self.words_similar.keys() or tokens[i] in self.teencode_dict.keys():
                    candidates.append((i, tokens[i]))

        if len(candidates) != 0:
            shuffle(candidates)
            idx = np.random.randint(0, len(candidates))
            index_modify = candidates[idx][0]
            word_replace = self.replaced_word_candidate(candidates[idx][1])
            tokens[index_modify] = word_replace
            onehot_labels[index_modify] = 1

        return tokens, onehot_labels

    def replace_with_random_letter(self, tokens: list, onehot_labels: list):
        """Replace, add or delete a random letter from a random word"""
        idx = np.random.randint(0, len(tokens))
        prevent_loop = 0
        while onehot_labels[idx] != 0 or tokens[idx].isnumeric() or tokens[idx] in string.punctuation:
            idx = np.random.randint(0, len(tokens))
            prevent_loop += 1
            if prevent_loop >= 10:
                break

        if onehot_labels[idx] == 0 and len(tokens[idx]) > 1:
            dice = random.choice([0, 1, 2])
            chosen_letter = tokens[idx][np.random.randint(0, len(tokens[idx]))]
            # replace
            if dice == 0:
                replaced_letter = random.choice(self.vn_alphabet)
                if replaced_letter != chosen_letter:
                    onehot_labels[idx] = 1
                    tokens[idx] = re.sub(chosen_letter, replaced_letter, tokens[idx])
            elif dice == 1:  # add
                replaced_letter = chosen_letter + random.choice(self.vn_alphabet)
                tokens[idx] = re.sub(chosen_letter, replaced_letter, tokens[idx])
                onehot_labels[idx] = 1
            else:  # delete.
                tokens[idx] = re.sub(chosen_letter, '', tokens[idx])
                onehot_labels[idx] = 1

        return tokens, onehot_labels

    def remove_last_character(self, tokens: list, onehot_labels: list):
        candidates = []
        for i, token in enumerate(tokens):
            if len(token) >= 3 and onehot_labels[i] == 0:
                candidates.append((i, token))
        if len(candidates) > 0:
            shuffle(candidates)
            idx = np.random.randint(0, len(candidates))
            index_modified = candidates[idx][0]
            if tokens[index_modified][:-1] in self.words_dict:
                tokens[index_modified] = tokens[index_modified][:-1]
                onehot_labels[index_modified] = 1
        return tokens, onehot_labels

    def add_last_character(self, tokens: list, onehot_labels):
        candidates = []
        for i, token in enumerate(tokens):
            if onehot_labels[i] == 0:
                candidates.append((i, token))
        if len(candidates) > 0:
            idx = np.random.randint(0, len(candidates))
            prevent_loop = 0
            index_modified = candidates[idx][0]
            word = candidates[idx][1]
            list_character_can_add = self.pair_character.get(token[-1])
            if list_character_can_add is not None:
                shuffle(list_character_can_add)
                while prevent_loop < 10:
                    random.shuffle(list_character_can_add)
                    char_random = random.choice(list_character_can_add)
                    prevent_loop += 1
                    if (word + char_random) in self.words_dict:
                        tokens[index_modified] = word + char_random
                        onehot_labels[index_modified] = 1
                        break

        return tokens, onehot_labels

    def add_or_remove_last_character(self, tokens: list, onehot_labels: list):
        if random.random() < 0.5:
            return self.add_last_character(tokens, onehot_labels)
        return self.remove_last_character(tokens, onehot_labels)

    def remove_diacritics(self, tokens: list, onehot_labels: list):
        idx = np.random.randint(0, len(tokens))
        prevent_loop = 0
        while onehot_labels[idx] != 0 or tokens[idx].isnumeric() or tokens[idx] in string.punctuation:
            idx = np.random.randint(0, len(tokens))
            prevent_loop += 1
            if prevent_loop >= 10:
                break

        if onehot_labels[idx] == 0:
            temp = unidecode.unidecode(tokens[idx])
            if temp != tokens[idx]:
                tokens[idx] = temp
                onehot_labels[idx] = 1

        return tokens, onehot_labels

    def replace_syllable(self, tokens: list, onehot_lables: list):
        candidates = []
        for i, token in enumerate(tokens):
            if onehot_lables[i] == 0:
                index = self.get_index(token)
                if index != -1:
                    candidates.append((i, token[index:]))

        if len(candidates) > 0:
            idx = np.random.randint(0, len(candidates))
            index_modified = candidates[idx][0]
            syllable_replace = candidates[idx][1]
            if self.pair_syllabel.get(syllable_replace, None) != None:
                syllabel_random_replace = random.choice(self.pair_syllabel[syllable_replace])
                tokens[index_modified] = re.sub(syllable_replace, syllabel_random_replace, tokens[index_modified])
                onehot_lables[index_modified] = 1
        return tokens, onehot_lables

    def exist_y(self, word):
        for i, char in enumerate(self.list_y):
            if char == word[-1]:
                return i
        return -1
        # O, ô, ơ, ư, g, b, d, đ, h, k, r, s, t, ph, th, gh

    def exist_i(self, word):
        for i, char in enumerate(self.list_i):
            if char == word[-1]:
                return i
        return -1

    def replace_special_character(self, tokens: list, onehot_labels: list):
        indexs_temp = [i for i in range(len(tokens))]
        random.shuffle(indexs_temp)
        cnt = 0
        for i in indexs_temp:
            if onehot_labels[i] == 0 and cnt < 2 and len(tokens[i]) > 1:
                # replace 's' to 'x' or 'x' to 's'
                word_temp = tokens[i].lower()
                if word_temp[0] in ['s', 'x'] and random.random() < 0.4:  # repalce 's' to 'x' or 'x' to 's'
                    if word_temp[0] == 's':
                        word_temp = 'x' + word_temp[1:]
                    elif word_temp[0] == 'x':
                        word_temp = 's' + word_temp[1:]
                    onehot_labels[i] = 1
                    tokens[i] = word_temp
                    cnt += 1
                # replace 'd' to 'gi' or 'gi' to 'd'
                elif 'gi' in word_temp and word_temp[0] == 'g' and random.random() < 0.4:
                    if random.random() < 0.5:
                        tokens[i] = 'd' + word_temp[2:]
                    else:
                        tokens[i] = 'đ' + word_temp[2:]
                    onehot_labels[i] = 1
                    cnt += 1
                elif word_temp[0] == 'd' and random.random() < 0.3:
                    tokens[i] = 'gi' + word_temp[1:]
                    onehot_labels[i] = 1
                    cnt += 1
                elif word_temp[0] == 'đ' and random.random() < 0.3:
                    tokens[i] = 'gi' + word_temp[1:]
                    onehot_labels[i] = 1
                    cnt += 1
                # replace 'l' to 'n' or 'n' to 'l'
                elif word_temp[0] in ['l', 'n'] and\
                    len(word_temp) > 1 and \
                    word_temp[1] in self.nguyen_am and \
                    random.random() < 0.05:

                    if word_temp[0] == 'n':
                        word_temp = 'l' + word_temp[1:]
                    elif word_temp[0] == 'l':
                        word_temp = 'n' + word_temp[1:]
                    if word_temp in self.words_dict:
                        tokens[i] = word_temp
                        onehot_labels[i] = 1
                        cnt += 1
                # replace 'i' to 'y'
                # elif self.exist_i(word_temp) != -1 and random.random() < 0.3:
                #     index_i = self.exist_i(word_temp)
                #     word_temp = word_temp[:-1] + self.list_y[index_i]
                #     tokens[i] = word_temp
                #     onehot_labels[i] = 1
                #     cnt += 1
                # replace 'y' to 'i'
                # elif self.exist_y(word_temp) != -1 and word_temp[-2] not in self.list_aa and random.random() < 0.5: # because don't exist 'âi'
                #     index_y = self.exist_y(word_temp)
                #     word_temp = word_temp[:-1] + self.list_i[index_y]
                #     tokens[i] = word_temp
                #     onehot_labels[i] = 1
                #     cnt += 1

        return tokens, onehot_labels

    def add_noise(self, sentence: str, percent_error: float):
        sentence, index_special = self.check_special_token(sentence)
        sentence = re.sub(r'[^\w\s]', " ", sentence)  # remove punctuation
        original_tokens = self.tokenizer(sentence)
        tokens = original_tokens.copy()
        onehot_labels = [0] * len(tokens)
        # if original_tokens[-1] in ['k', 'ko']:
        #     original_tokens[-1] = 'không'
        #     onehot_labels[-1] = 1

        """"
        onehot_labels: onehot array indicate the position of word that has already modify, so we only choose the word which has onehot_label=0
        """
        for i in range(len(onehot_labels)):
            if tokens[i] in ['numberic', 'specialw', 'date']:
                onehot_labels[i] = 10

        num_wrong = int(np.ceil(len(tokens) * percent_error))
        if np.random.rand() < 0.05:
            num_wrong = 0
            percent_error = 0
        if num_wrong == 0:
            num_wrong = 1
        for _ in range(num_wrong):
            error_th = np.random.randint(0, 240 + 1)
            if 0 <= error_th <= 20: # 20
                tokens, onehot_labels = self.replace_with_typo_letter(tokens, onehot_labels) # Telex
            elif 20 < error_th <= 70:# 50
                tokens, onehot_labels = self.replace_with_homophone_word(tokens, onehot_labels) # Chữ đồng âm
            elif 70 < error_th <= 120:# 50
                tokens, onehot_labels = self.replace_with_word(tokens, onehot_labels) # Viết tắt/teencode
            elif 120 < error_th <= 170:# 50
                tokens, onehot_labels = self.remove_diacritics(tokens, onehot_labels) # Dấu phụ
            elif 170 < error_th <= 190:# 20
                tokens, onehot_labels = self.replace_special_character(tokens, onehot_labels) # s and x, d to gi...
            # elif 240 < error_th <= 250:# 10
                # tokens, onehot_labels = self.add_or_remove_last_character(tokens, onehot_labels) # Thêm xóa chữ
            # elif 250 < error_th <= 260:# 10
            #     tokens, onehot_labels = self.replace_syllable(tokens, onehot_labels) # Vần đồng âm
            # elif 260 < error_th <= 270: # 10
            #     tokens, onehot_labels = self.replace_with_random_letter(tokens, onehot_labels) # Thay chữ
            elif 190 < error_th <= 240: # 50
                tokens, onehot_labels = self.replace_with_typo_close_letter(tokens, onehot_labels) # Kết hợp telex và phím gần

        # restore non error
        for i in range(len(onehot_labels)):
            if tokens[i] in ['numberic', 'specialw', 'date']:
                onehot_labels[i] = 0
        if len(original_tokens) == len(tokens):
            return " ".join(original_tokens), " ".join(tokens), onehot_labels, index_special, percent_error
        else:
            print("Not equal")
            print(original_tokens)
            print(tokens)
            return None, None, None, None

