import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn
import torch.nn.functional
import torch.optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pdb

class DisneyReviewsDataset(Dataset):
    def __init__(self, csv_file, bert_model_name):
        self.model_name = bert_model_name
        self.csv_encoding = 'iso-8859-1'
        self.csv_encoding = None
        self.df = pd.read_csv(csv_file, encoding=self.csv_encoding, na_values='missing')
        if 'Review_Tokens' not in self.df.columns:
            self.__tokenize()

        self.df = self.df[['Review_Text','Rating', 'Review_Tokens']]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        review = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]
        return review, label

    def __tokenize(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        print('Tokenizing dataset...')
        self.df['Review_Tokens'] = self.df['Review_Text'].progress_apply(lambda text: tokenizer.encode(text,
                                                                            add_special_tokens=True,
                                                                            padding='max_length',
                                                                            truncation=True,
                                                                            max_length=1024))

class BertDisneyReviews(torch.nn.Module):
    def __init__(self, bert_model_name):
        super(BertDisneyReviews, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create the model. Output label is a single scalar (the review score)
        self.encoder = BertForSequenceClassification.from_pretrained(bert_model_name)

    def forward(self):
        pass

def main():
    batch_size = 3
    num_workers = 8
    model_name = 'bert-base-uncased'
    csv_file = './DisneylandReviews.csv'
    csv_file = './DisneylandReviews_Tokenized.csv'
    dataset = DisneyReviewsDataset(csv_file, model_name)
    loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size,
                num_workers=num_workers)
    model = BertDisneyReviews(model_name)

    pdb.set_trace()

if __name__ == "__main__":
    main()