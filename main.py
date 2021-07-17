import json
from pathlib import Path

import torch
from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering, Trainer, TrainingArguments

DATA_DIR = './data'
TRAIN_PATH = f'{DATA_DIR}/train.json'
DEV_PATH = f'{DATA_DIR}/dev.json'

model_checkpoint = "monologg/koelectra-base-v3-discriminator"


def read_korquad_v2(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        context = group['context']

        for qa in group['qas']:
            question = qa['question']
            answer = qa['answer']

            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_korquad_v2(TRAIN_PATH)
val_contexts, val_questions, val_answers = read_korquad_v2(DEV_PATH)


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            # When the gold label is off by one character
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            # When the gold label is off by two characters
            answer['answer_end'] = end_idx - 2


add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)


print(f'Start load tokenizer : {model_checkpoint}')
tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint)
print(f'Finish load tokenizer : {model_checkpoint}')

train_encodings = tokenizer(
    train_contexts,
    train_questions,
    truncation=True,
    padding=True,
)
val_encodings = tokenizer(
    val_contexts,
    val_questions,
    truncation=True,
    padding=True,
)


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(
            i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(
            i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions,
    })


add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)


class KorquadLongQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


print(f'Start build datasets')
train_dataset = KorquadLongQADataset(train_encodings)
val_dataset = KorquadLongQADataset(val_encodings)
print(f'Finish build datasets')


training_args = TrainingArguments(
    output_dir='./koelectra-v3-long-qa',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

print(f'Start load model : {model_checkpoint}')
model = ElectraForQuestionAnswering.from_pretrained(model_checkpoint)
print(f'Finish load model : {model_checkpoint}')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
