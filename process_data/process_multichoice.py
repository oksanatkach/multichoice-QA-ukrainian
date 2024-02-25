import os
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer

letter2ind = {
    'А': 0,
    'Б': 1,
    'В': 2,
    'Г': 3,
    'Д': 4
}


def fits_format(data_line, subject):
  if data_line['correct_answers'][0] in letter2ind.keys() and data_line['answers'] and data_line['subject'] == subject:
    return True
  return False


def preprocess_line(data_line):
    context = data_line['question'].replace('\xa0', ' ')
    possible_responses = [el['text'].replace('\xa0', ' ') for el in data_line['answers']]
    label = letter2ind[data_line['correct_answers'][0]]

    return context, possible_responses, label


def main_process_data(data_path, zno_train_file, subject, MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def custom_tokenize(data_line):
        context = [data_line['context']] * 5
        if len(data_line['responses']) == 4:
            resp_choice = set(range(4)) - set([data_line['label']])
            data_line['responses'].append(data_line['responses'][random.choice(list(resp_choice))])

        tokenized = tokenizer(context, data_line['responses'], truncation=True)
        return tokenized

    contexts = []
    responses = []
    labels = []

    data = open(os.path.join(data_path, zno_train_file), 'r').read().strip().split('\n')

    for str_line in data:
      data_line = json.loads(str_line)
      if fits_format(data_line, subject):
        context, possible_responses, label = preprocess_line(data_line)
        contexts.append(context)
        responses.append(possible_responses)
        labels.append(label)

    data = {'context': contexts,
         'responses': responses,
         'label': labels
        }

    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(custom_tokenize, batched=False)
    return tokenized_dataset.train_test_split(test_size=0.3)
