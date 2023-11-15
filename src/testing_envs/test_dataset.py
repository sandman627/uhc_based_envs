import os
import pickle
from collections import deque

import numpy as np
import torch

# import types

# # List all the types defined in the types module
# type_list = [getattr(types, a) for a in dir(types) if not a.startswith('__')]
# # Filter out modules, functions, and other non-type entries
# type_list = [t for t in type_list if isinstance(t, type)]
class Dataset_Checker():
    def __init__(self, queue_size:int=5) -> None:
        self.queue_size = queue_size
        
        self.queue = deque(iterable=[], maxlen=self.queue_size)
        pass

    def get_dataset_structure(self, dataset_path:str="/home/nhstest/uhc_based_envs/data/stretching/sa_seq.pkl"):
        with open(dataset_path, 'rb') as pkl_file:
            dataset = pickle.load(pkl_file)
        print(f"dataset path : {dataset_path}")
        print(f"dataset type : {type(dataset)}")
        
        # for key, val in dataset.items():
        #     print(f"{key} : {type(val)}, {len(val)}")
        #     print(f"{val[0].shape}")
        
        
        print("\n=========================")
        self.get_matching_type_info(dataset)
        print("\n=========================")
    

    def print_with_check(self, *args, **kwargs):
        cur_input = [args, kwargs]
        self.queue.append(cur_input)
        
        if (len(self.queue) == self.queue_size):
            if self.queue[0] == cur_input:
                # print(*args, **kwargs)
                return
            else:
                self.queue.clear()         
        print(*args, **kwargs)
        


    def get_matching_type_info(self, data, data_level:int=0, indent:int=1):
        check_prefix = "  "*data_level*indent + f"{type(data)} : "
        if isinstance(data, (str, int)):
            self.print_with_check(f"{check_prefix}{data}")
        if isinstance(data, (np.ndarray, torch.Tensor)):
            self.print_with_check(f"{check_prefix}{data.shape}")
        elif isinstance(data, list):
            print(f"{check_prefix}len({len(data)})")
            for element in data:
                self.get_matching_type_info(element, data_level+1)
        elif isinstance(data, dict):
            print(f"{check_prefix}len({len(data)})")
            for key, val in data.items():
                print("  "*(data_level + 1) + f"{key} : ", end="")
                self.get_matching_type_info(val, data_level + 2, indent=0)
        else:
            print("Unsupported data type.")
        
        return 




test_example = {
    "action": [np.array([1,1,1,1]), np.array([1,1,1,1])],
    "states": [np.array([2,2,2]), np.array([2,2,2]), np.array([2,2,2])]
}

if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))
    
    checker = Dataset_Checker()
    checker.get_dataset_structure()
    
    # def check_if_same(data1, data2):
    #     if data1 == data2:
    #         print("The data have the same values.")
    #     else:
    #         print("The data have different values.")
        
    #     if data1 is data2:
    #         print("The data are the exact same object in memory.")
    #     else:
    #         print("The data are not the exact same object in memory.")

    # # Example usage:
    # data_a = [1, 2, 3]
    # data_b = [1, 2, 3]
    # data_c = data_a

    # check_if_same(data_a, data_b) # They have the same values but are different objects.
    # check_if_same(data_a, data_c) # They are the same object.    
    # exit()
    
    
