import numpy as np
import warnings

class Battery():
    '''
    A simple simulated battery
    
    Usage: Used to simulate batteries in any optimization model, logic model, or rollout
    
    
    Input:
    
    max_capacity: float, The maximum amount of capacity the battery can contain (should non-negative)
                  
                  Is by standard set to 13.0
                  
    current_capacity: float, The amount currently in the battery (should non-negative <= max_capacity)
                  
                      Is by standard set to 0.0             
                  
    max_charge: float, The maximum amount that can be charged/discharged at once (should non-negative <= max_capacity)
                  
                Is by standard set to 7.0
                
    degrade_rate: float, The amount the battery degrades by (should be between 0 and 1)
                  
                  Is by standard set to 0.05, however, when charging degrading is turned off by default
   
    
    Example: bat = Battery(max_capacity = 13.0, max_charge = 7.0)
    
             bat.charge(15.0)
             
             capacity = bat.get_current_capacity() #=7.0
             
             surplus = bat.get_surplus() #=8.0
    
    
    Additional attributes:
    
    surplus: float, How much was leftover after charging (if abs(charge amount) > max_charge)
                  
             Gets updated in charge function
             
    charge_list: float list, history of what has been charged/discharged
                  
                 Gets updated in charge function
                 
    previous_capacity: float, The previous capacity before degrading and charging
                  
                       Gets updated in charge function
                       
    previous_degraded_capacity: float, The previous capacity after degrading, before charging
                  
                                Gets updated in charge function
                                
                                
    Functions (check their docs):
    
    charge:     float -> None
                      or
            float * bool -> None
    
    degrade: float -> None
    
    round_one_decimal: float -> float
    
    get_surplus: -> float
    
    get_previous_capacity: -> float
    
    get_previous_degraded_capacity: -> float
    
    get_current_capacity: -> float
    
    get_max_capacity: -> float
    
    get_percentage: -> float
    '''
    def __init__(self, max_capacity = 13.0, current_capacity = 0.0, max_charge = 7.0, degrade_rate = 0.05): 
        self.max_capacity = max_capacity
        self.current_capacity = current_capacity
        self.surplus = 0.0
        self.charge_list = np.array([])
        self.max_charge = max_charge
        self.degrade_rate = degrade_rate
        self.previous_capacity = 0.0
        self.previous_degraded_capacity = 0.0
        
        #Require max_capacity >= 0
        if self.max_capacity < 0.0:
            warnings.warn(f"max_capacity must non-negative! Setting max_capacity = 0.0")
            self.max_capacity = 0.0
        
        #Require current_capacity <= max_capacity
        if self.current_capacity > self.max_capacity:
            warnings.warn(f"current_capacity = {self.current_capacity} must be less than or equal to max_capcity! Setting current_capacity = {self.max_capacity}")
            self.current_capacity = self.max_capacity
            
        #Require current_capacity >= 0
        if self.current_capacity < 0.0:
            warnings.warn(f"current_capacity = {self.current_capacity} must be non-negative! Setting current_capacity = 0.0")
            self.current_capacity = 0.0
        
        #Require max_charge <= max_capacity
        if self.max_charge > self.max_capacity:
            warnings.warn(f"max_charge = {self.max_charge} must be less than or equal to max_capcity! Setting max_charge = {self.max_capacity}")
            self.max_charge = self.max_capacity
            
        #Require max_charge >= 0
        if self.max_charge < 0.0:
            warnings.warn(f"max_charge = {self.current_capacity} must be non-negative! Setting max_charge = 0.0")
            self.max_charge = 0.0

    def charge(self, amount, degrade=False):
        """
        Charges the battery when amount is postive and otherwise discharges it
        
        Return type: None

        Usage: Used whenever the battery is charged or discharged in rollouts


        Input:

        amount: float, Amount of kWh to charge (positive), or discharge (negative)

                The battery is dynamic with this input, so if charged or discharged more than
                possible, the "surplus" attribute will be updated, so that it is known how
                much of the input did or did not enter/leave the battery

        degrade: bool, Whether battery should be degraded before charging

                 By default is False, meaning no degrading


        Example: bat = Battery(max_capacity = 13.0, max_charge = 7.0)
    
                 bat.charge(15.0)

                 capacity = bat.get_current_capacity() #=7.0

                 surplus = bat.get_surplus() #=8.0
        """
        self.previous_capacity = self.current_capacity
        
        #Degrade battery by 1 hour
        if degrade:
            self.degrade(1)
            
        #Shorten to one decimal
        self.current_capacity = self.round_one_decimal(current_capacity) 
            
        capacity = self.get_current_capacity() #capacity after degrade, before charge/discharge
        self.previous_degraded_capacity = capacity
        extra_amount = 0.0
        
        if amount <= 0.0:
            #Check for overcharging
            if amount < -self.max_charge:
                extra_amount = amount + self.max_charge
                amount = -self.max_charge
            
            #Check for overdraining
            if capacity + amount < 0.0:
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
        
        #Shorten to one decimal
        og_amount = amount+extra_amount
        amount = self.round_one_decimal(amount) 
        extra_amount = og_amount-amount
        
        #Charge/discharge battery by amount and shorten to two decimals (edge case)
        self.current_capacity += amount
        self.current_capacity = self.round_one_decimal(self.current_capacity) 
        
        self.surplus = extra_amount
        self.charge_list = np.append(self.charge_list,[amount])
        
    def degrade(self, hours):
        """
        Degrades the battery by the degrade_rate for input amount of hours. Using this in charge,
        the output should be rounded, as to keep the number within one decimal
        
        Return type: None

        Usage: Used in charge function if degrade parameter is set to True


        Input:

        hours: float, how many hours the battery should be degraded for
        
               Is usually hours = 1, since the data is hourly, and so one action per hour,
               meaning one degration per hour


        Example: bat = Battery(max_capacity = 13.0, current_capacity=3.0, max_charge = 7.0, degrade_rate = 0.05)
                 
                 bat.degrade(1)
                 
                 capacity = bat.get_current_capacity() #=2.85
        """
        self.current_capacity -= self.current_capacity * self.degrade_rate * hours
    
    def round_one_decimal(self, number):
        """
        Returns the float that ignores everything beyond the first decimal
        
        Return type: float
        """
        return int(number*10)/10
    
    def get_surplus(self):
        """
        Returns the class attribute "surplus"
        
        Return type: float
        """
        return self.surplus
    
    def get_previous_capacity(self):
        """
        Returns the class attribute "previous_capacity"
        
        Return type: float
        """
        return self.previous_capacity
    
    def get_previous_degraded_capacity(self):
        """
        Returns the class attribute "previous_degraded_capacity"
        
        Return type: float
        """
        return self.previous_degraded_capacity
    
    def get_current_capacity(self):
        """
        Returns the class attribute "current_capacity"
        
        Return type: float
        """
        return self.current_capacity
    
    def get_max_capacity(self):
        """
        Returns the class attribute "max_capacity"
        
        Return type: float
        """
        return self.max_capacity
    
    def get_percentage(self):
        """
        Returns how full the battery is in %
        
        Return type: float
        """
        return self.current_capacity / self.max_capacity * 100
    
    def __str__(self):
        return f"Capacity: {self.get_current_capacity()}"

if __name__ == "__main__":
    print("This file is meant to be imported")