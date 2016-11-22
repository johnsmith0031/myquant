import numpy as np

class rand_strategy():
    
    def __init__(self,price):
        self.price = price
        self.hlen = len(price)