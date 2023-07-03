import re
import openai

from sklearn.model_selection import train_test_split


def inspect_nulls(df):
    nulls = df.isna().sum()
    return nulls[nulls > 0]

def split_train_test(X, y, *args, **kwargs):
    X_train, X_test = train_test_split(X, *args, **kwargs)
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

    return X_train, X_test, y_train, y_test


def get_completion(prompt, model='gpt-3.5-turbo', temperature=0):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message['content']


def get_dict_from_data(sentences):
    data = []

    # patterns
    sentence_pattern = r'\<\w+\>|\</\w+\>'
    patterns = {
        'subject': r'(?<=\<e1\>)(.+)(?=\</e1\>)',
        'object': r'(?<=\<e2\>)(.+)(?=\</e2\>)',
        'event': r'(?<=\<rel\>)(.+)(?=\</rel\>)',
        'complement': r'(?<=\<comp\>)(.+)(?=\</comp\>)',
        'relationSignature': r'(?<=RelationSignature:\s)(.+)(?=\s\()',
        'relationType': r'(?<=RelationType:\s)(.+)(?=\s\()'
    }

    _sentences = re.split(r'\d+\t', sentences)
    for raw_sentence in _sentences[1:]:
        sentence = raw_sentence.split('\n')[0]
        sentence_data = {}

        sentence_data['text'] = ''.join(re.split(sentence_pattern, sentence))

        for role, pattern in patterns.items():
            role_text = re.search(pattern, raw_sentence)
            role_text = role_text and role_text.group(0)
            if role != 'relationSignature':
                sentence_data[role] = role_text
            else:
                relation_signature = (
                    role_text.split('-')
                    if role_text else
                    [None, None]
                )
                sentence_data['subjectLabel'] = relation_signature[0]
                sentence_data['objectLabel'] = relation_signature[1]

        data.append(sentence_data)

    return data
