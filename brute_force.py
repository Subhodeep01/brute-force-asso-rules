from read_files import CreateDataset
import itertools
from copy import deepcopy
from efficient_apriori import apriori
from fpgrowth_py import fpgrowth
import time

AMAZON = "Amazon.csv"
BESTBUY = "BestBuy.csv"
KMART = "Kmart.csv"
NIKE = "Nike.csv"
WALMART = "Walmart.csv"

class BruteForce:
    def __init__(self, database, support = 0.5, confidence = 0.5) -> None:
        self.database = database
        cd = CreateDataset(self.database)
        self.items = cd.extract_items()
        self.transactions = cd.extract_transac()
        self.support = support
        self.confidence = confidence
        self.prepare_itemsets_recs()
        
    def show_items_transacs(self):    
        print("\n\nThe individual items present in the database:")
        for i in range(self.items.shape[0]):
            print(f"{i+1}. {self.items.iloc[i,1]}")
        print("\nAll the transactions in the database:")
        for i in range(self.transactions.shape[0]):
            print(f"TID {self.transactions.iloc[i, 0]}: {self.transactions.iloc[i,1]}")
        print("\n\n")

    def prepare_itemsets_recs(self):  
        '''Preparing single itemsets and transactions record for each individual transaction from the pandas dataframe'''
        self.itemsets = self.items.iloc[:, 1].to_list()
        self.transac_rec = {}
        for i in range(self.transactions.shape[0]):
            id, trans = self.transactions.iloc[i, 0], self.transactions.iloc[i,1]
            self.transac_rec[id] = trans.split(', ')
        # print(self.itemsets)

    def get_support(self, combos=None):
        '''returning a dictionary containing items whose frequency is greater than or equal to the min support defined by user'''
        freq = {}
        l = self.transactions.shape[0]
        if combos:
            for combo in combos:
                for vals in self.transac_rec.values():
                    if all(item in vals for item in combo):
                        freq[combo] = (freq.get(combo, 0) + 1)
                freq[combo] = round(freq.get(combo, 0)/l,2)
        else:
            for i in self.itemsets:
                for vals in self.transac_rec.values():
                    if i in vals:
                        freq[(i,)] = (freq.get((i,), 0) + 1)
                freq[(i,)] = round(freq.get((i,), 0)/l,2)
        temp = deepcopy(freq)
        for i in temp.keys():
            if freq[i] < self.support:
                freq.pop(i)   # Popping out the itemsets whose support is lower than the user defined min support
        return freq

    def gen_freq_sets(self):
        '''Generating all frequent itemsets'''
        freq = self.get_support()
        max_freq = max(freq.values()) if len(freq.values()) >0 else 0
        self.freq_itemsets = freq
        k = 1
        while(max_freq >= self.support):
            combos = list(itertools.combinations(self.itemsets, k+1))
            freq = self.get_support(combos)
            if not freq: 
                max_freq = 0
                break
            max_freq = max(freq.values())
            self.freq_itemsets.update(freq)
            k += 1

    def gen_asso_rules(self):
        self.gen_freq_sets()
        all_combos = {key:val for key, val in self.freq_itemsets.items() if len(key) > 1}
        asso_rules = {}
        for combo in all_combos:
            asso_list = []
            for l in range(len(combo)-1):
                asso_list.extend(itertools.combinations(combo, l+1)) 
            combo_supp = all_combos[combo]
            for i in asso_list:
                others = tuple(set(combo)- set(i))
                if i in self.freq_itemsets.keys():
                    conf = round(combo_supp/self.freq_itemsets[i],2)
                    if conf >= self.confidence:
                        asso_rules[(i,combo)] = [others, combo_supp, conf]
                if others in self.freq_itemsets.keys():
                    rev_conf = round(combo_supp/self.freq_itemsets[others],2)
                    if rev_conf >= self.confidence:
                        asso_rules[(others, combo)] = [i, combo_supp, rev_conf]
        return self.freq_itemsets, asso_rules
    
    def testing(self):
        transacs = [tuple(val) for val in self.transac_rec.values()]
        start_time_ap = time.time()
        itemsets, rules = apriori(transacs, min_support=self.support, min_confidence=self.confidence)
        end_time_ap = time.time()
        counter = 0
        print("-----LIBRARY APRIORI ALGORITHM-----")
        print(f"\nUsing library apriori algorithm we get itemsets --> {itemsets} \n")
        print("Using Library Apriori the association rules are as follows (with support and confidence) using Brute Force:")
        for rule in rules:
             counter += 1
             print(f"{counter}. {rule}")
        print("\n\n")
        print("-----LIBRARY FPT ALGORITHM-----")
        start_time_fp = time.time()
        freqItemSet, rules = fpgrowth(transacs, self.support, self.confidence)
        end_time_fp = time.time()
        print(f"\nUsing library FPT algorithm we get itemsets --> {freqItemSet} \n")
        print("Using Library FPT the association rules are as follows (with support and confidence) using Brute Force:")
        counter = 0
        for rule in rules:
             counter += 1
             print(f"{counter}. {rule}")
        print("\n\n")
        exec_time_ap = end_time_ap - start_time_ap
        exec_time_fp = end_time_fp - start_time_fp
        return exec_time_ap, exec_time_fp

n = 0
while n != "exit":
    print("Welcome to Association rule mining. Please select one of the following databases to find association rules from:")
    database = [AMAZON, BESTBUY, KMART, NIKE, WALMART]
    for i in range(len(database)):
        print(f"{i}.: {database[i]}")
    n = input(f"Your choice (type and enter 'exit' to exit): ")
    if n == 'exit': break
    try:
        n = int(n)
        if n < 0 or n > 4:
            print("Invalid choice. Please try again!")
            continue
    except ValueError:
        print("Invalid input. Please input a number!")
        continue
    db_of_choice = database[n]
    try:
        supp = float(input("Choose a support from 0.01 to 1 (e.g. 0.5) where 0.01 means 1 percent and 1 means 100 percent support: "))
        conf = float(input("Choose a confidence from 0.01 to 1 (e.g. 0.5) where 0.01 means 1 percent and 1 means 100 percent confidence: "))
        if (supp > 1 or supp<=0) or (conf>1 or conf<=0): 
            print("Wrong support or confidence. Please try again.")
            continue
    except ValueError:
        print("Invalid input. Please input a number!")
        continue
    bf = BruteForce(db_of_choice, supp, conf)
    start_time = time.time()
    freq_itemsets, asso_rules = bf.gen_asso_rules()
    end_time = time.time()
    bf.show_items_transacs()
    print("-----BRUTE FORCE ALGORITHM-----")
    print(f"\nUsing Brute Force we get itemsets --> {freq_itemsets}\n")
    print("Using Brute Force the association rules are as follows (with support and confidence) using Brute Force:")
    counter = 0
    for key, val in asso_rules.items():
        counter += 1
        print(f"{counter}. [{key[0]} --> {val[0]}] : Support = {val[1]*100}%; Confidence = {val[2]*100}%")
    print("\n\n")
    exec_time = end_time - start_time
    exec_time_ap, exec_time_fp = bf.testing()
    print(f"Execution time for Brute Force: {exec_time} \nExecution time for Library Apriori: {exec_time_ap} \nExecution time for Library FPT: {exec_time_fp}")
    print("\n\n")
print("Thank you, have a nice day!")

