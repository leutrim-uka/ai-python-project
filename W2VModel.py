from pymongo import MongoClient
import datetime
import pandas as pd
import numpy as np
from connections.Connections import Connections
from queries.LogQueries import LogQueries
from gensim.models import Word2Vec
import pickle
from sklearn.model_selection import train_test_split
import itertools
import datetime as DT


class W2VModel:
    
    def __init__(self, target):
        obj = Connections()
        self.db = obj.get_db(target)
        self.target = target
        self.errorDB = LogQueries()

    def get_top_sold_products(self):
        try:
            last_day = DT.datetime.today() - DT.timedelta(days=1)

            checkout = pd.DataFrame(self.db.checkout.aggregate([
            {
                "$match": {"target" : self.target, "date": {"$gte": last_day}}
            },
            {
                "$project": {"_id": 0, "userID": 1, "cart": 1, "orderID": 1, "date": 1}
            },
            {
                "$unwind": "$cart"
            },
            {
                "$project": {
                    "_id": 0,
                    "userID": 1,
                    # "id": "$cart.id",
                    'id': "$cart.productID",
                    "amount": "$cart.amount",
                    "orderID": 1,
                    "date": 1
                }
            }]))

            checkout = checkout.groupby('id').sum().sort_values('amount', ascending = False).reset_index()
            top_products = list(checkout['id'][:10])
            return top_products

        except Exception:
            self.errorDB.error("method: get_top_sold_products", self.target)

    def create_model(self):
        try:

            checkout = self.get_checkout_data()

            top_sold_products = self.get_top_sold_products()

            products_by_user = self.create_products_by_user_dataset(checkout)
        
            third_quartile = np.percentile(products_by_user['sen_len'], 75)
            first_quartile = np.percentile(products_by_user['sen_len'], 25)
            outliers = third_quartile + (third_quartile - first_quartile) * 1.5
            model_data = products_by_user[products_by_user["sen_len"] <= outliers]
            
            data_for_model_training = list(model_data['id'])
            models = {}

            # split data into train and test
            training_data, test_data = train_test_split(data_for_model_training, train_size=0.8)
            w2v_model_train = Word2Vec(training_data, min_count=1)

            # store recall_at_k for each epoch
            accuracy = []

            # the right amount of epochs to use
            epochs_to_use = 0

            for epoch in range(1, 101):
                w2v_model_train.train(training_data, total_examples=len(training_data), epochs=epoch)
                recall_rate = self.calculate_recall_at_k(test_data, w2v_model_train)
                accuracy.append(recall_rate)
                models[f'{epoch}'] = w2v_model_train


            #print(accuracy)

            print("User based model finished")

            best_accuracy = max(accuracy)
            epochs_to_use = accuracy.index(best_accuracy) + 1

            # create the model with the right number of epochs
            user_based_w2v_model = models[f'{epochs_to_use}']

            products_by_user = (pd.DataFrame(products_by_user)).explode('id')
            products_by_user = products_by_user[products_by_user['id'].isin(user_based_w2v_model.wv.vocab)]

            

            # create the new model on order based
            products_by_order = self.create_products_by_order_dataset(checkout)

            third_quartile = np.percentile(products_by_order['sen_len'], 75)
            first_quartile = np.percentile(products_by_order['sen_len'], 25)
            outliers = third_quartile + (third_quartile - first_quartile) * 1.5
            model_data = products_by_order[products_by_order["sen_len"] <= outliers]
            
            data_for_model_training = list(model_data['id'])
            models = {}

            # split data into train and test
            training_data, test_data = train_test_split(data_for_model_training, train_size=0.8)
            w2v_model_train = Word2Vec(training_data, min_count=1)

            # store recall_at_k for each epoch
            accuracy = []

            # the right amount of epochs to use
            epochs_to_use = 0

            for epoch in range(1, 101):
                w2v_model_train.train(training_data, total_examples=len(training_data), epochs=epoch)
                recall_rate = self.calculate_recall_at_k(test_data, w2v_model_train)

                accuracy.append(recall_rate)
                models[f'{epoch}'] = w2v_model_train

            #print(accuracy)

            print("Order based model finished")
            
            best_accuracy = max(accuracy)
            epochs_to_use = accuracy.index(best_accuracy) + 1

            # create the model with the right number of epochs
            order_based_w2v_model = models[f'{epochs_to_use}']

            # store the model as a pickle file
            self.convert_to_pickle(order_based_w2v_model, user_based_w2v_model, products_by_user, top_sold_products)

        except Exception:
            self.errorDB.error("method: create_model", self.target)
        
    # def create_dataset(self):
    #     try:
    #         checkout_data = self.get_checkout_data()
    #         # prepared_data = self.prepare_data(checkout_data)
    #         #return prepared_data

    #     except Exception:
    #         self.errorDB.error("method: create_dataset", self.target)
    

    # get data from Checkout in MongoDB 
    def get_checkout_data(self):
        try:
            # check products' availability
            # available_products = list(self.db.products.aggregate([
            # {
            #     "$match": {
            #         "target" : self.target,
            #         "availability": {"$ne": "out of stock"}
            #     }
            # },
            # {
            #     "$project": {
            #         "_id": 0,
            #         "productID": 1
            #     }
            # }
            # ]))
            
            # remove key from 'key':'value' pair
            # available_products = list(set(map(lambda d: d['productID'], available_products)))


            available_products = pd.DataFrame(self.db.products.aggregate([
                {
                    "$match": {
                        "target" : self.target,
                        "availability": {"$ne": "out of stock"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "productID": 1
                    }
                }
            ]))

            available_products = list(available_products['productID'].unique())

            # get only available products from checkout collection
            checkout = pd.DataFrame(self.db.checkout.aggregate([
            {
                "$match": {"target" : self.target}
            },
            # {
            #     "$project": {"_id": 0, "userID": 1, "cart": 1, "orderID": 1, "date": 1}
            # },
            {
                "$unwind": "$cart"
            },
            {
                "$project": {
                    "_id": 0,
                    "userID": 1,
                    "productID": "$cart.productID",
                    "amount": "$cart.amount",
                    "orderID": 1,
                    "date": 1
                }
            },
            {
                "$match": {"productID": {"$in": available_products}}
            },
            {
                "$sort": {"date": 1}
            }
            ]))

            return checkout

        except Exception:
            self.errorDB.error("method: get_checkout_data", self.target)
                    

    # Clean data and prepare the final dataset
    def create_products_by_user_dataset(self, checkout_data):
        try:
            # Clean data and prepare the final dataset
            checkout_data.dropna(inplace=True)
            max_prd = checkout_data["amount"].mean() + 3*checkout_data["amount"].std()
            checkout_data = checkout_data[checkout_data["amount"]<max_prd]
            total_prds_order = checkout_data.groupby("orderID")["amount"].sum().reset_index()
            max_order = total_prds_order["amount"].mean() + 3*total_prds_order["amount"].std()
            total_prds_order = total_prds_order[total_prds_order["amount"]<max_order]
            checkout_data = pd.merge(checkout_data, total_prds_order[["orderID"]], on="orderID", how="inner")

            checkout_data["id"] = checkout_data["productID"].astype(str)
            products_by_user = checkout_data.groupby("userID")["id"].apply(' '.join).reset_index()

            products_by_user["id"] = products_by_user["id"].map(lambda x: x.split(" "))
            products_by_user["sen_len"] = products_by_user["id"].map(lambda x: len(x))
            products_by_user = products_by_user[products_by_user["sen_len"]>1]

            return products_by_user

        except Exception:
            self.errorDB.error("method: create_products_by_user_dataset", self.target)


    def create_products_by_order_dataset(self, checkout_data):
        try:
            checkout_data.dropna(inplace=True)
            max_prd = checkout_data["amount"].mean() + 3*checkout_data["amount"].std()
            checkout_data = checkout_data[checkout_data["amount"]<max_prd]
            total_prds_order = checkout_data.groupby("orderID")["amount"].sum().reset_index()
            max_order = total_prds_order["amount"].mean() + 3*total_prds_order["amount"].std()
            total_prds_order = total_prds_order[total_prds_order["amount"]<max_order]
            checkout_data = pd.merge(checkout_data, total_prds_order[["orderID"]], on="orderID", how="inner")

            checkout_data["id"] = checkout_data["productID"].astype(str)


            products_by_order = checkout_data.groupby("orderID")["id"].apply(list).reset_index()

            products_by_order["sen_len"] = products_by_order["id"].apply(lambda x: len(x))

            products_by_order = products_by_order[products_by_order["sen_len"]>1]

            return products_by_order

        except Exception:
            self.errorDB.error("method: create_products_by_order_dataset", self.target)


    def calculate_recall_at_k(self, test_data, w2v_model):
        try:
            recall_ratio_per_user = []
            for purchase_array in test_data:
                recall_at_k = 0
                recommendations = 0
                k = 1

                for elements in range(1, len(purchase_array)):
                    for subset in itertools.combinations(purchase_array, elements):
                        elements_not_in_subset = list(set(purchase_array)-set(subset))
                        if all(elem in w2v_model.wv.vocab for elem in list(subset)):
                            model_recommendations = [x[0] for x in w2v_model.wv.most_similar(positive=subset, topn=k)]
                            recommendations += 1
                            if any(elem in model_recommendations for elem in elements_not_in_subset):
                                recall_at_k += 1

                if recommendations == 0:
                    continue

                recall_ratio = float(recall_at_k) / float(recommendations)
                recall_ratio_per_user.append(recall_ratio)

            return (np.mean(recall_ratio_per_user))

        except Exception:
            self.errorDB.error("method: calculate_recall_at_k", self.target)

    
    def convert_to_pickle(self, order_based_model, user_based_model, products_by_user, top_10_products):
        try:
            configs = {
                "order_based_model" : order_based_model,
                "user_based_model": user_based_model,
                "user_data": products_by_user,
                "top_ten_products": top_10_products
            }
            
            with open(f"w2v_util/models/{self.target}_model.pkl", "wb") as file:
                pickle.dump(configs, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Model is saved to pickle!!!")

        except Exception:
            self.errorDB.error("method: convert_to_pickle", self.target)

   


