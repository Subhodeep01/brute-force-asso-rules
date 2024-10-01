import pandas as pd

class CreateDataset:
    def __init__(self, filepath:str) -> None:
        self.data = pd.read_csv(filepath,)

    def extract_items(self):
        items = self.data.iloc[:10]
        # print(items)
        return items
    
    def extract_transac(self):
        transactions = self.data.iloc[10:]
        transactions.columns = transactions.iloc[0]
        transactions = transactions[1:]
        transactions.reset_index(drop=True, inplace=True)
        # print(transactions)
        return transactions


