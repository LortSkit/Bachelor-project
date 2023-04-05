class Buyer:
    def __init__(self, name, price, grid_price, demand):
        self.name = name
        self.price = price
        self.grid_price = grid_price
        self.demand = demand
        
        self.total_purchase = 0
        self.peer_purchase = 0
        self.grid_purchase = 0
    
    def purchase(self, quantity):
        self.peer_purchase += quantity
        self.total_purchase += quantity
    
    def unfulfilled_demand(self):
        return self.demand - self.total_purchase
    
    def purchase_from_grid(self):
        demand_gap = self.unfulfilled_demand()
        cost = demand_gap * self.grid_price
        self.grid_purchase += demand_gap
        self.total_purchase += demand_gap
        return cost
    
    def purchase_from_peers(self):
        return self.peer_purchase*self.price
    
    def total_cost(self):
        return self.purchase_from_grid()+self.purchase_from_peers()

class Seller:
    def __init__(self, name, price, sbr, supply):
        self.name = name
        self.price = price
        self.sbr = sbr
        self.supply = supply
        
        self.total_sale = 0
        self.peer_sale = 0
        self.grid_sale = 0
    
    def sell(self, quantity):
        self.total_sale += quantity
        self.peer_sale += quantity
    
    def unsold_supply(self):
        return self.supply - self.total_sale
    
    def sell_to_grid(self):
        unsold = self.unsold_supply()
        revenue = unsold * self.price * self.sbr
        self.total_sale += unsold
        return revenue
    
    def sell_to_peers(self):
        return self.peer_sale*self.price
    
    def total_cost(self):
        return self.sell_to_peers() + self.sell_to_grid()

    
def energy_exchange(buyers, sellers):
    if (not buyers) or (not sellers):
        return buyers

    buyer = max(buyers, key=lambda b: b.demand) # Find the buyer with the highest demand
        
    seller = max(sellers, key=lambda s: s.supply) # Find the seller with the highest supply
    
    if buyer.demand > seller.supply:
        buyer.purchase(seller.supply)
        seller.sell(seller.supply)
        sellers.remove(seller)
        return [buyer] + energy_exchange(buyers, sellers)
    
    elif seller.supply > buyer.demand:
        seller.sell(buyer.demand)
        buyer.purchase(buyer.demand)
        buyers.remove(buyer)
        return [seller] + energy_exchange(buyers, sellers)
    
    else:
        buyer.purchase(buyer.demand)
        seller.sell(seller.supply)
        buyers.remove(buyer)
        sellers.remove(seller)
        return [buyer, seller] + energy_exchange(buyers, sellers)

class EnergyMarket:
    def __init__(self, participants, price, grid_price):
        self.buyers = {}
        self.sellers = {}
        self.participants = participants
        self.price = price
        self.grid_price = grid_price
        
        for name, value in self.participants.items():
            if value < 0:
                buyer = Buyer(name, self.price, self.grid_price, -value)
                self.buyers[name] = buyer
            else:
                seller = Seller(name, self.price, 0.10, value)
                self.sellers[name] = seller
        
        self.buyers_list = list(self.buyers.values())
        self.sellers_list = list(self.sellers.values())
        
    def get_buyers(self):
        return self.buyers
    
    def get_sellers(self):
        return self.sellers

    def cal_costs(self):
        results = energy_exchange(self.buyers_list,self.sellers_list)
        return self.buyers, self.sellers
    
    def get_total_costs(self):
        buy_dict, sell_dict = self.cal_costs()
        costs = {}
        for name, buyer in buy_dict.items():
            costs[name] = buyer.total_cost()
        for name, seller in sell_dict.items():
            costs[name] = -seller.total_cost()
        return costs
    
if __name__ == "__main__":
    print("This file is meant to be imported")