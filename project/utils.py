import pickle
import json
import config
import numpy as np

class MedicalInsurnace():
    def __init__(self,age,bmi,children,smoker,region):
        self.age = age
        self.bmi = bmi
        self.children = children
        self.smoker= smoker
        self.region = region
    def load_model(self):
        ## Read Model 
        with open(config.model_path,"rb") as file:
            self.model = pickle.load(file)
        # read json file
        with open(config.json_path,"r") as file:
            self.json_data = json.load(file)
    def get_charges(self):
        self.load_model() # calling model
        test_array = np.zeros(len(self.json_data["columns"]),dtype=int)
        test_array[0]= self.age
        test_array[1] = self.bmi
        test_array[2] = self.children
        test_array[3] = self.json_data["smoker"][self.smoker] #  no
        test_array[4] =  self.json_data["region"][self.region] #  no
        predict_charges = self.model.predict([test_array])[0]
        return predict_charges

