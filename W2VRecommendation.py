from pymongo import MongoClient
import pandas as pd
import datetime
import numpy as np
from gensim.models import Word2Vec
import pickle
import glob
import os

from .W2VModel import W2VModel
from connections.Connections import Connections
from queries.LogQueries import LogQueries

    
class W2VRecommendation:
    def __init__(self):
        self.models = self.load_models()
        self.dbConnection = Connections()
        self.errorDB = LogQueries()
        
    def load_models(self):
        # get the names of model files from the directory
        os.chdir("w2v_util/models")
        file_names = [os.path.basename(x) for x in glob.glob('*_model.pkl')]     

        #create a dictionary where key = target name & value : loaded model file
        model_files = {}
        for name in file_names:
            with open(name, 'rb') as file:
                pickle_file = pickle.load(file)
                model_files = {f'{str(name).split("_")[0]}': pickle_file}
                
        os.chdir("../../")
        
        return model_files


    def check_previously_recommended_products(self, target, userID, database):
        
        previously_recommended = list(database.recommendations.aggregate([
          {
              "$match": 
                  {
                      "target": target,
                      "userID": userID,
                      "liked": True
                  }
          },
          {
               "$project":
                  {
                       "_id": 0,
                       "productID": 1
                  }
          }
        ]))
        

        previously_recommended = list(map(lambda d: d['productID'], previously_recommended))
        
        if len(previously_recommended) == 0:
            previously_recommended = []
            
        return previously_recommended



    def check_available_products(self, target, products, database):
        available_products = list(database.products.aggregate([
            {
                "$match": {
                    "target" : target,
                    "id" : {"$in": products},
                    "availability": {"$ne": "out of stock"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "id": 1
                }
            }
            ]))
        
        available_products = list(map(lambda d: d['id'], available_products))
                
        return available_products


    
    def recommend_products(self, target, recommendation_type = None, userID = None, productIDs = None):
        try:
            database = self.dbConnection.get_db(target)

            if recommendation_type == 'user':

                model = self.models[f'{target}']['user_based_model']

                user_data = self.models[f'{target}']['user_data']

                top_sold_products = self.models[f'{target}']['top_ten_products']

                recommendations = []

                never_recommended = []
                
                previously_recommended = []
                
                previously_recommended = self.check_previously_recommended_products(target, userID, database)
                
                # if the user is logged in
                if productIDs == None:
                    recommendations = map(lambda x: model.wv.most_similar(positive=x, topn=20), list(user_data[user_data['userID'] == userID]['id']))
                    recommendations = list(set(x[0] for x in (sorted([x for y in recommendations for x in y], key=lambda x:(-x[1], x[0])))))

                
                # if the user is not logged in
                if productIDs != None:
                    products = [x for x in productIDs if x in model.wv.vocab]
                    recommendations = map(lambda x: model.wv.most_similar(positive=x, topn=20), products)
                    recommendations = list(set(x[0] for x in (sorted([x for y in recommendations for x in y], key=lambda x:(-x[1], x[0])))))

                
                if len(recommendations) == 0:
                    recommendations = top_sold_products
                        
                available_products = self.check_available_products(target, recommendations, database)
                never_recommended = [x for x in available_products if x not in previously_recommended]
                
                if len(never_recommended) == 0:
                        recommendations = available_products
                else:
                    recommendations = [x for x in available_products if x in never_recommended]
                
                return recommendations

            elif recommendation_type == 'order':

                ### Continue with order based logic to recommend products

                model = self.models[f'{target}']['order_based_model']

                user_data = self.models[f'{target}']['user_data']

                top_sold_products = self.models[f'{target}']['top_ten_products']

                recommendations = []

                never_recommended = []

                previously_recommended = self.check_previously_recommended_products(target, userID, database)

                if productIDs == None:
                    recommendations = map(lambda x: model.wv.most_similar(positive=x, topn=20), list(user_data[user_data['userID'] == userID]['id']))
                    recommendations = list(set(x[0] for x in (sorted([x for y in recommendations for x in y], key=lambda x:(-x[1], x[0])))))

                if productIDs != None:
                    products = [x for x in productIDs if x in model.wv.vocab]
                    recommendations = map(lambda x: model.wv.most_similar(positive=x, topn=20), products)
                    recommendations = list(set(x[0] for x in (sorted([x for y in recommendations for x in y], key=lambda x:(-x[1], x[0])))))

                
                if len(recommendations) == 0:
                    recommendations = top_sold_products

                available_products = self.check_available_products(target, recommendations, database)
                never_recommended = [x for x in available_products if x not in previously_recommended]
                
                if len(never_recommended) == 0:
                        recommendations = available_products
                else:
                    recommendations = [x for x in available_products if x in never_recommended]
                
                return recommendations

            
        except Exception:
            self.errorDB.error("method: recommend_products", target)
