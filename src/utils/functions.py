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
