import tensorflow as tf
from data_loader import *

class TensorflowMF:
    """
    Biased matrix factorisation model using TensorFlow
    r_ui = b + b_u + b_i + < U_u, V_i >
    """
    def __init__(self, num_users, num_items, rank, reg):
        self.rank = rank
        self.num_users = num_users
        self.num_items = num_items
        self.reg = reg
        self.initialize_values()
        
    def initialize_values(self):
        self.b = tf.Variable(0.0, name="global_bias")
        self.b_u = tf.Variable(tf.truncated_normal([self.num_users, 1], stddev=0.01, mean=0), name="user_bias")
        self.b_i = tf.Variable(tf.truncated_normal([self.num_items, 1], stddev=0.01, mean=0), name="item_bias")
        self.U = tf.Variable(tf.truncated_normal([self.num_users, rank], stddev=0.01, mean=0), name="users")
        self.V = tf.Variable(tf.truncated_normal([self.num_items, rank], stddev=0.01, mean=0), name="items")
          
             
    def predict(self, users, items):
        U_ = tf.squeeze(tf.nn.embedding_lookup(self.U, users))
        V_ = tf.squeeze(tf.nn.embedding_lookup(self.V, items))
        prediction = tf.nn.sigmoid((tf.reduce_sum(tf.mul(U_, V_), reduction_indices=[1]))) 
        ubias = tf.squeeze(tf.nn.embedding_lookup(self.b_u, users))
        ibias = tf.squeeze(tf.nn.embedding_lookup(self.b_i, items))
        prediction =   self.b + ubias + ibias + tf.squeeze(prediction)
        return prediction

    def regLoss(self):
        reg_loss = 0
        reg_loss += tf.reduce_sum(tf.square(self.U))
        reg_loss += tf.reduce_sum(tf.square(self.V))
        reg_loss += tf.reduce_sum(tf.square(self.b_u))
        reg_loss += tf.reduce_sum(tf.square(self.b_i))
        return reg_loss * self.reg
    
    def loss(self, users_items_ratings):
        users, items, ratings = users_items_ratings
        prediction = self.predict(users, items)
        err_loss = tf.nn.l2_loss(prediction - ratings) 
        reg_loss = self.regLoss()
        self.total_loss = err_loss + reg_loss
        tf.scalar_summary("loss", self.total_loss)
        return self.total_loss
    
    def fit(self, users_items_ratings, test_users_items_ratings=None, n_iter=10):
        cost = self.loss(users_items_ratings)
        optimiser = tf.train.AdamOptimizer(0.01).minimize(cost)
        with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            users, items, ratings = users_items_ratings
            for i in range(n_iter):
                sess.run(optimiser)
                if i%20 == 0:
                    print self.evalTestError(test_users_items_ratings).eval()
                    
    def evalTestError(self, test_user_items_ratings):
        testusers, testitems, testratings = test_user_items_ratings
        testprediction = self.predict(testusers, testitems)
        return tf.sqrt(tf.nn.l2_loss(testprediction - testratings) * 2.0 / len(testusers))


train_path = "./data/train.1"
test_path = "./data/test.1"

import numpy as np
def sparseMatrix2UserItemRating(_mat):
    temp = _mat.tocoo()
    user = temp.row.reshape(-1,1)
    item = temp.col.reshape(-1,1)
    rating = temp.data
    return user, item, rating

parser = UserItemRatingParser("\t")
d = Data()
d.import_data(train_path, parser)
train = d.R
test, cold_start_user_item_ratings = loadDataset(test_path, d.users, d.items, parser)

num_users, num_items = train.shape
rank = 5
reg = 1.0
n_iter = 400
t = TensorflowMF(num_users, num_items, rank, reg)
users_items_ratings = sparseMatrix2UserItemRating(train)
test_users_items_ratings = sparseMatrix2UserItemRating(test)
t.fit(users_items_ratings, test_users_items_ratings, n_iter)

