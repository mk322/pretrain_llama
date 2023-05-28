# install datasets
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader
# tokenization
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

#Batch processing & Tokenization
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = "<unk>"

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

def load_data():
    # load shuffled train, validation and test sets 
    train = load_dataset("EleutherAI/pile", split='train', streaming=True).take(100)
    validation = load_dataset("EleutherAI/pile", split='validation', streaming=True).take(100)
    test = load_dataset("EleutherAI/pile", split='test', streaming=True).take(100)
    # {'text': 'ACTUAL TEXT', 'meta':{'pile_set_name': 'LABEL'}}

    sid = SentimentIntensityAnalyzer()

    # Filter only Neutral sentiment, 'neu' > 0.7
    train = train.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    validation = validation.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    test = test.filter(lambda example: sid.polarity_scores(example['text'])['neu'] > 0.7)
    
    #train = itertools.islice(train, 10)    # default batch size = 1000
    train = train.map(encode, batched=True, remove_columns=["meta","token_type_ids","attention_mask"])
    validation = validation.map(encode, batched=True, remove_columns=["meta","token_type_ids","attention_mask"])
    test = test.map(encode, batched=True, remove_columns=["meta","token_type_ids","attention_mask"])
    # {'text': 'ACTUAL TEXT', 'input_ids':[#]}

    #train = Subset(train.with_format("torch"), range(100))
    # Convert to torch.utils.data.Dataset
    train = DataLoader(train.with_format("torch"), batch_size=8)
    validation = DataLoader(validation.with_format("torch"), batch_size=8)
    test = DataLoader(test.with_format("torch"), batch_size=8)
    return train, validation, test


