import re
import openai

from sklearn.model_selection import train_test_split
from random import sample
from itertools import groupby, chain

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


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


def get_dict_from_data(sentences):
    data = []

    # patterns
    sentence_pattern = r'\<\w+\>|\</\w+\>'
    patterns = {
        'subject': r'(?<=\<e1\>)(.+)(?=\</e1\>)',
        'object': r'(?<=\<e2\>)(.*?)(?=\</e2\>)',
        'event': r'(?<=\<rel\>)(.+)(?=\</rel\>)',
        'complement': r'(?<=\<comp\>)(.*?)(?=\</comp\>)',
        'relationSignature': r'(?<=RelationSignature:\s)(.+)(?=\s\()',
        'relationType': r'(?<=RelationType:\s)(.+)(?=\s\()'
    }

    _sentences = re.split(r'\d+\t', sentences)
    for raw_sentence in _sentences[1:]:
        sentence = raw_sentence.split('\n')[0]
        sentence_data = {}

        sentence_data['text'] = ''.join(re.split(sentence_pattern, sentence))

        for role, pattern in patterns.items():
            role_text = re.findall(pattern, raw_sentence)
            role_text = (
                (role_text[0] if role_text else None)
                if role not in ('complement', 'object') else
                (role_text or [])
            )
            if role != 'relationSignature':
                sentence_data[role] = role_text
            else:
                relation_signature = (
                    role_text.split('-')
                    if role_text else
                    [None, None]
                )
                if sentence_data['subject']:
                    sentence_data['subjectLabel'] = relation_signature[0]
                if sentence_data['object']:
                    sentence_data['objectLabel'] = relation_signature[1]

        data.append(sentence_data)

    return data


def prepare_data_for_fine_tune(data):
    fine_tune_data = []

    for sentence in data:
        text = sentence['text']

        fine_tune_prompt = {
            'prompt': text + '\n\n###\n\n'
        }

        completion_data = []
        if sentence["subject"]:
            subject = sentence["subject"]
            subject_item = f'{subject} / subject / {sentence["subjectLabel"]}'
            completion_data.append((subject_item, text.index(subject)))

        if sentence["object"]:
            for object in sentence["object"]:
                object_item = f'{object} / object / {sentence["objectLabel"]}'
                completion_data.append((object_item, text.index(object)))

        if sentence["event"]:
            event = sentence["event"]
            event_item = f'{event} / trigger / {sentence["relationType"]}'
            completion_data.append((event_item, text.index(event)))

        if sentence["complement"]:
            for complement in sentence["complement"]:
                complement_item = f'{complement} / complement / '
                completion_data.append(
                    (complement_item, text.index(complement)))

        completion_data = sorted(completion_data, key=lambda x: x[1])

        fine_tune_prompt['completion'] = ' ' + '\n'.join(
            entity for entity, _ in completion_data
        ) + '\n###'

        fine_tune_data.append(fine_tune_prompt)

    return fine_tune_data


def get_examples(data):
    examples = {
        'role': [],
        'subject-type': [],
        'object-type': [],
        'relation-type': []
    }

    for i, sentence in enumerate(data):
        examples['role'].append(
            f'Example {i + 1}:\n'
            f'Input: {sentence["text"]}\n'
            f'Output:\n'
            f'    * subject: {sentence["subject"] or "null"}\n'
            f'    * object: {sentence["object"] or "[]"}\n'
            f'    * event: {sentence["event"] or "null"}\n'
            f'    * complement: {sentence["complement"] or "[]"}\n'
        )

        examples['subject-type'].append(
            f'Example {i + 1}:\n'
            f'Input:\n'
            f'    * sentence: {sentence["text"]}\n'
            f'    * subject: {sentence["subject"]}\n'
            f'Output: {sentence["subjectLabel"] if "subjectLabel" in sentence else "null"}\n'
        )

        examples['object-type'].append(
            f'Example {i + 1}:\n'
            f'Input:\n'
            f'    * sentence: {sentence["text"]}\n'
            f'    * object: {sentence["object"] or "[]"}\n'
            f'Output: {sentence["objectLabel"] if "objectLabel" in sentence else "null"}\n'
        )

        examples['relation-type'].append(
            f'Example {i + 1}:\n'
            f'Input:\n'
            f'    * sentence: {sentence["text"]}\n'
            f'    * subject: {sentence["subject"] or "null"}\n'
            f'    * event: {sentence["event"] or "null"}\n'
            f'    * object: {sentence["object"] or "[]"}\n'
            f'Output: {sentence["relationType"]}\n'
        )

    return examples


def example_validation_split(data, k, stratum=None):
    if not stratum:
        sampled_data = sample(list(enumerate(data)), k)
    else:
        _data = sorted(
            enumerate(data),
            key=lambda x, stratum=stratum: x[1][stratum])
        grouped_data = [
            list(strata)
            for _, strata in groupby(
                _data, lambda x, stratum=stratum: x[1][stratum])
        ]
        _k = k
        if isinstance(k, int):
            _k = [k] * len(grouped_data)
        sampled_data = list(chain.from_iterable(
            [sample(strata, i) for i, strata in zip(_k, grouped_data)]))
    sampled_indexes = [example_data[0] for example_data in sampled_data]
    return (
        [example_data[1] for example_data in sampled_data],
        [data[i] for i in range(len(data)) if i not in sampled_indexes]
    )


def fix_model_output(output):
    _output = output.copy()
    for sentence in _output:
        if not isinstance(sentence['object'], list) and sentence['object']:
            sentence['object'] = [sentence['object']]
        if not isinstance(sentence['complement'], list) and sentence['complement']:
            sentence['complement'] = [sentence['complement']]
        if isinstance(sentence['subject'], list):
            sentence['subject'] = sentence['subject'][0] if sentence['subject'] else None
    return _output
