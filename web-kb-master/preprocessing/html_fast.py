import glob
import json
import os
import re
from collections import Counter

import bs4
import nltk
import pandas as pd
from tqdm import tqdm


rootdir = 'D:\\Developer\\P05\\'
data_dir = os.path.join(rootdir, 'data', 'webkb')

tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'body', 'a', 'b', 'u', 'i']
stop_words = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

pronoun_1 = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
pronoun_2 = ['you', 'your', 'yours', 'yourself', 'yourselves']
pronoun_3 = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
             'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']

weights = json.load(open(os.path.join(rootdir, 'preprocessing', 'tab_weights.json'), 'r'))


def get_tag_contents(html):
    parser = bs4.BeautifulSoup(html, 'html.parser')
    text = {}
    for tag in weights.keys():
        for i in parser.find_all(tag):
            text[tag] = i.get_text()
    return text


if __name__ == '__main__':
    pd.options.display.max_columns = None
    df = pd.DataFrame(data={'d_type':   pd.Series([], dtype='str'),
                            'd_school': pd.Series([], dtype='str'),
                            'p_1':      pd.Series([], dtype='int'),
                            'p_2':      pd.Series([], dtype='int'),
                            'p_3':      pd.Series([], dtype='int')})

    for file in tqdm(glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True)):
        attr = os.path.dirname(file).split(os.sep)
        d_school = attr[-1]
        d_type = attr[-2]
        tag_contents = get_tag_contents(open(file, 'rb').read().lower())
        reserved_col = {'d_type': d_type, 'd_school': d_school, 'p_1': 0, 'p_2': 0, 'p_3': 0}
        csv_row = reserved_col
        for tag, contents in tag_contents.items():
            valid_tokens = []
            for token in set(nltk.word_tokenize(contents)):
                if token in pronoun_1:
                    reserved_col['p_1'] += 1
                elif token in pronoun_2:
                    reserved_col['p_2'] += 1
                elif token in pronoun_3:
                    reserved_col['p_3'] += 1
                elif token not in stop_words:
                    # ignore non-alphabetic characters
                    if re.compile(r'^[a-z]+$').match(token):
                        valid_tokens.append(stemmer.stem(token))

            word_count = dict(Counter(valid_tokens))
            if tag != 'body':
                for word in word_count:
                    # increase weight by factor
                    word_count[word] = round(word_count[word] * weights[tag], 2)
            csv_row.update(word_count)

        df = df.append(csv_row, ignore_index=True)

    df.fillna(0).to_csv(os.path.join(rootdir, 'data', 'tokens_no_other_weighted.csv'), header=True,
                        index=False)
