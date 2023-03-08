import numpy as np

class Battery():
    def __init__(self, max_capacity = 13.5, current_capacity = 0, surplus = 0, max_charge = 7, degrade_rate = 0.05, 
                 actions=None): 
        self.max_capacity = max_capacity
        self.current_capacity = current_capacity
        self.surplus = surplus
        self.charge_list = np.array([])
        self.max_charge = max_charge
        self.rate = degrade_rate
        self.previous_capacity = 0
        self.actions = actions
        
        #Require max_charge <= max_capacity
        if self.max_charge > self.max_capacity:
            import warnings
            warnings.warn(f"max_charge must be less than or equal to max_capcity! Setting max_charge = {self.max_capacity}")
            self.max_charge = self.max_capacity

    def charge(self, amount, degrade=True):
        #Degrade battery by 1 hour
        if degrade:
            self.degrade(1)
            
        #Shorten to two decimals
        self.current_capacity = int(self.current_capacity*10)/10 
            
        capacity = self.get_current_capacity() #capacity after degrade, before charge/discharge
        self.previous_capacity = capacity
        extra_amount = 0
        
        if amount <=0:
            #Check for overcharging
            if amount < -self.max_charge:
                extra_amount = amount + self.max_charge
                amount = -self.max_charge
            
            #Check for overdraining
            if capacity + amount < 0:
                overdrain = capacity + amount
                extra_amount += overdrain
                amount -= overdrain
            
        else:
            #Check for overcharging
            if amount > self.max_charge:
                extra_amount = amount - self.max_charge
                amount = self.max_charge
            
            #Check for overfilling
            if capacity + amount > self.max_capacity:
                overfill = capacity + amount - self.max_capacity
                extra_amount += overfill
                amount -= overfill
        
        #Shorten to two decimals
        og_amount = amount+extra_amount
        amount = int(amount*10)/10 
        extra_amount = og_amount-amount
        
        #Charge/discharge battery by amount and shorten to two decimals (edge case)
        self.current_capacity += amount
        self.current_capacity = int(self.current_capacity*10)/10 
        
        self.surplus = extra_amount
        self.charge_list = np.append(self.charge_list,[amount])
        
    def degrade(self, hours):
        self.current_capacity -= self.current_capacity * self.rate * hours
    
    def get_surplus(self):
        return self.surplus
    
    def get_previous_capacity(self):
        return self.previous_capacity
    
    def get_current_capacity(self):
        return self.current_capacity
    
    def get_max_capacity(self):
        return self.max_capacity
    
    def get_percentage(self):
        return self.current_capacity / self.max_capacity * 100
    
    def __str__(self):
        return f"Capacity: {self.current_capacity}, Surplus: {self.surplus}"

if __name__ == "__main__":
    print("This file is meant to be imported")