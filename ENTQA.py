from transformers import AutoTokenizer, AutoModelForQuestionAnswering
ckpt = "mrm8488/longformer-base-4096-finetuned-squadv2"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(ckpt)

import pickle

with open('examples.pickle', 'rb') as f:
    examples = pickle.load(f)
print("Length ",len(examples))

from random import sample  
examples_subset = sample(examples,300000)

del examples

import pandas as pd
from datasets import Dataset 

df = pd.DataFrame.from_records(examples_subset)
dataset = Dataset.from_pandas(df).train_test_split(test_size=.02)

with open('testset.pickle', 'wb') as handle:
    pickle.dump(dataset["test"], handle, protocol=pickle.HIGHEST_PROTOCOL)

from datasets import set_caching_enabled
set_caching_enabled(False)

def preprocess_function(examples):

    # questions = [q.strip() for q in examples["question"]]   #Surfaceform + Description
    question = examples["candidate"]["description"]
    context = examples["text"]
    input_pairs = [question, context]
    encodings = tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=1024)
    context_encodings = tokenizer.encode_plus(context)
    sep_idx = encodings['input_ids'].index(tokenizer.sep_token_id)
    try:
      if(examples["result"] == True):
        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx = examples["candidate"]["start"]
        end_idx = examples["candidate"]["end"]
        
        
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx-1)

        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example 
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1


        if end_positions > 1024:
          start_positions, end_positions = 0,0
      else:
        start_positions, end_positions = 0,0
    except:
      start_positions, end_positions = 0,0

    encodings.update({'start_positions': start_positions,
                      'end_positions': end_positions,
                      'attention_mask': encodings['attention_mask']})
    return encodings

tokenized_data = dataset.map(preprocess_function, batched=False)
print(tokenized_data)


from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer

data_collator = DefaultDataCollator()


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=7e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps = 6,
    warmup_ratio= 0.2,
    num_train_epochs=10,
    save_total_limit=1,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()