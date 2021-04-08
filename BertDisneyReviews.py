import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pdb

class DisneyReviewsDataset(Dataset):
    def __init__(self, in_file, bert_model_name, debug=False, num_debug_rows=None):
        self.bert_model_name = bert_model_name
        in_file_ob = Path(in_file)
        if in_file_ob.suffix == '.pkl':
            # Pickle file already contains tokens
            with open(in_file, 'rb') as file:
                print('woops! pickle file not implemented yet')
                pass
                # df = pickle.load(file)

        elif in_file_ob.suffix == '.csv':
            df = pd.read_csv(in_file, encoding='iso-8859-1', na_values='missing', index_col=0)
            if debug:
                df = df.iloc[0:num_debug_rows]

            # Normalize rating by dividing by max
            df['Rating'] = df['Rating'].astype(float) / 5
            print('Tokenizing model...')
            tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, do_lower_case=True)
            self.encodings = {}
            encodings_list = []
            chunksize=1000
            with tqdm(total=len(df)) as pbar:
                for idx, cdf in enumerate(self.__chunker(df, chunksize)):
                    curr_review_texts = cdf['Review_Text'].tolist()
                    curr_encodings = tokenizer(curr_review_texts,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=512,
                        truncation=True)
                    
                    # Add current chunk of encodings to the full dictionary
                    for key, value in curr_encodings.items():
                        if key in self.encodings.keys():
                            self.encodings[key] = self.encodings[key] + value
                        else:
                            self.encodings[key] = value

                    pbar.update(chunksize)

            self.ratings = df['Rating'].tolist()

    def __chunker(self, seq, size):
        """
        https://stackoverflow.com/a/39495229
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        """
        https://huggingface.co/transformers/custom_datasets.html#seq-imdb
        """
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['ratings'] = torch.tensor(self.ratings[index])
        return item

class BertDisneyReviews(torch.nn.Module):
    def __init__(self, device, model_name):
        super(BertDisneyReviews, self).__init__()
        # Output dimensions. In this case it's a single scalar, the rating
        self.num_outputs = 1

        # Create the model. Output label is a single scalar (the review score)
        self.model = AutoModel.from_pretrained(model_name)

        self.drop_out = nn.Dropout(p=0.3)
        self.output = nn.Linear(self.model.config.hidden_size, self.num_outputs)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids, attention_mask)
        x = self.drop_out(pooled_output)
        return self.output(x)

def main():
    debug = False
    num_debug_rows = 100
    batch_size = 1
    num_workers = 8
    NUM_EPOCHES = 20
    learning_rate = 0.0001
    PRINT_FREQUENCY = 200
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name = 'bert-base-uncased'
    model_name = 'distilbert-base-uncased'
    in_file = './DisneylandReviews.csv'

    dataset = DisneyReviewsDataset(in_file, model_name, debug=debug, num_debug_rows=num_debug_rows)
    dataset_split_lengths = [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)]
    # Fix any small errors due to float multiplication and rounding. The splits need to add up to the full dataset
    dataset_split_lengths[0] = dataset_split_lengths[0] + (len(dataset) - dataset_split_lengths[0] - dataset_split_lengths[1] - dataset_split_lengths[2])
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, dataset_split_lengths)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=batch_size,
                num_workers=num_workers)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size=batch_size,
                num_workers=num_workers)
                
    model = BertDisneyReviews(device, model_name)
    # model = BertForSequenceClassification.from_pretrained(model_name)
    model.train()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    loss_func = torch.nn.MSELoss()
    print('\nstarting training loop...')
    loss_history = []
    # Saves for which mini batches a loss was saved
    loss_history_mini_batch_count = []
    for epoch in range(NUM_EPOCHES):
        print('starting epoch {}'.format(epoch))
        running_loss = 0.0
        num_mini_batches = len(train_loader)
        for count, data in enumerate(train_loader):
            input_ids, ratings, attention_mask = data['input_ids'].to(device), data['ratings'].to(device), data['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            optimizer.zero_grad()
            outputs = torch.squeeze(outputs)
            loss = loss_func(outputs, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (count % PRINT_FREQUENCY) == 0:
                loss_average = running_loss / PRINT_FREQUENCY
                print('epoch {}, mini-batch {} of {}, loss = {}'.format(epoch, count+1, num_mini_batches, loss_average))
                loss_history_mini_batch_count.append(count+1)
                loss_history.append(loss_average)
                running_loss = 0.0

    print('testing the model...')
    with torch.no_grad():
        running_mse = torch.tensor(0.0)
        running_mae = torch.tensor(0.0)
        for count, data in enumerate(test_loader):
            input_ids, ratings, attention_mask = data['input_ids'].to(device), data['ratings'].to(device), data['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
            running_mse = torch.add(running_mse, torch.sum(torch.pow(torch.subtract(outputs, ratings), 2)))
            running_mae = torch.add(running_mae, torch.sum(torch.abs(torch.subtract(outputs, ratings))))
        
        total_mse = running_mse.cpu().item() / len(test_dataset)
        total_mae = running_mae.cpu().item() / len(test_dataset)
        print('total_mse = {}\ntotal_mae = {}'.format(total_mse, total_mae))

    fig, ax = plt.subplots()
    ax.plot(loss_history_mini_batch_count, loss_history)
    ax.set(xlabel='Mini Batch Number', ylabel='Mean Squared Error', title='Loss Versus Iterations')
    plt.show()
    pdb.set_trace()  


if __name__ == "__main__":
    main()