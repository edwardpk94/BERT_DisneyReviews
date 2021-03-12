import pandas as pd
import torch
import torch.nn
import torch.nn.functional
import torch.optim
from torch.utils.data import Dataset, DataLoader
# from torchtext.data import TabularDataset
import torchtext.data as tt_data
import pdb

class DisneyReviewsDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='iso-8859-1', na_values='missing')
        self.df = self.df[['Review_Text','Rating']]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        review = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]
        return review, label

class BertDisneyReviews(torch.nn.Module):
    def __init__(self):
        super(BertDisneyReviews, self).__init__()

    def forward(self):
        pass

def main():
    batch_size = 3
    num_workers = 8
    # dataset = DisneyReviewsDataset('./DisneylandReviews.csv')
    # loader = torch.utils.data.DataLoader(dataset,
    #             batch_size=batch_size,
    #             num_workers=num_workers)
    print('hello world!')
    dataset = tt_data.TabularDataset(path='./DisneylandReviews.csv', format='.csv',
                fields=[('Review_Text', tt_data.Field(sequential=True),
                            'Rating', tt_data.Field(sequential=False))])
    # test = next(iter(loader))
    pdb.set_trace()

if __name__ == "__main__":
    main()