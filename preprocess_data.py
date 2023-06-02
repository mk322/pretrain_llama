# install datasets
import torch
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader
from math import floor
# tokenization
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# sentiment analysis
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

#Batch processing & Tokenization
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = "<unk>"

def encode(examples):
    return tokenizer(examples['text'], truncation=True, max_length=513, padding='max_length')

def load_data(batch_size=128, num_train=100000):
    # load shuffled train, validation and test sets 
    train = load_dataset("EleutherAI/pile", split='train', streaming=True).take(num_train)
    validation = load_dataset("EleutherAI/pile", split='validation', streaming=True).take(10000)
    test = load_dataset("EleutherAI/pile", split='test', streaming=True).take(100)
    # {'text': 'ACTUAL TEXT', 'meta':{'pile_set_name': 'LABEL'}}

    # sid = SentimentIntensityAnalyzer()

    # Filter only Neutral sentiment, 'neu' > 0.7
    # train = train.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    # validation = validation.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    # test = test.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    
    #train = itertools.islice(train, 10)    # default batch size = 1000
    train = train.map(encode, batched=True, remove_columns=["text", "meta","token_type_ids","attention_mask"])
    validation = validation.map(encode, batched=True, remove_columns=["text", "meta","token_type_ids","attention_mask"])
    test = test.map(encode, batched=True, remove_columns=["text", "meta","token_type_ids","attention_mask"])
    # {'text': 'ACTUAL TEXT', 'input_ids':[#]}
    def collate_fn(data):
        #print(data[0]['input_ids'][:-1])
        input_ids = torch.stack([torch.IntTensor(item['input_ids'][:-1]) for item in data], dim=0)
        labels = torch.stack([torch.LongTensor(item['input_ids'][1:]) for item in data], dim=0)
        return input_ids, labels
    train = train.with_format("torch")
    validation = validation.with_format("torch")
    test = test.with_format("torch")
    #train = Subset(train.with_format("torch"), range(100))
    # Convert to torch.utils.data.Dataset
    train = DataLoader(train,collate_fn=collate_fn, batch_size=batch_size)
    validation = DataLoader(validation, collate_fn=collate_fn, batch_size=batch_size)
    test = DataLoader(test, collate_fn=collate_fn, batch_size=batch_size)
    
    #dataloader_iterator = iter(train)
    #first_batch = next(dataloader_iterator)
    train_len = floor(num_train/batch_size) + 1
    validation_len = floor(10000/batch_size) + 1
    
    return train, validation, test, train_len, validation_len


