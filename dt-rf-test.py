#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:49:51 2018

@author: anshhirani
"""

### Import ###

import unittest
from dt-rf-classifier import *



### Read data and create data types for testing ###

## 1) Create train PandasData structure
pdata = PandasData('train_df.csv', 'Rock')
del pdata.df['Unnamed: 0']

## 2) Create test PandasData structure
full_test_df = pd.read_csv('music_dev.csv')
for i in range(full_test_df.shape[0]):
    if full_test_df['Jazz'].iloc[i] == 'yes':
        full_test_df['Jazz'].iloc[i] = 1
    else:
        full_test_df['Jazz'].iloc[i] = 0
    if full_test_df['Rock'].iloc[i] == 'yes':
        full_test_df['Rock'].iloc[i] = 1
    else:
        full_test_df['Rock'].iloc[i] = 0

pdata_test = PandasData(full_test_df, 'Rock')

## 3) Create SQLData structure

connection = ps2.connect(host = 'blocked for github', database = 'blocked for github',
                         user = 'blocked for github', password = 'blocked for github')
sdata = SQLData('train', 'rock', conn = connection)


### Functions for testing ###

def samplePandas(data):
    d_sample = data.get_sample(100)
    return PandasData(d_sample, 'Rock')

def sampleSQL(data, conn):
    d_sample = data.get_sample(100)
    return SQLData(d_sample, 'rock', conn)

def same_impurity():
    c1 = [1]*10
    c2 = [1]*10
    label = [0]*10
    df = pd.DataFrame({'c1' : c1, 'c2' : c2, 'label' : label})
    return PandasData(df, 'label')

def same_value():
    c1 = [1]*10
    c2 = [1]*5 + [0]*5
    label = [1]*5 + [0]*5
    df = pd.DataFrame({'c1' : c1, 'c2' : c2, 'label' : label})
    return PandasData(df, 'label')

def build_bad_tree():
    tree = build_tree(PandasData, pdata, cross_entropy)
    tree.right_child.data = PandasData(pdata.get_sample(50, 0), 'Rock')
    return tree

def build_bad_tree2():
    tree = build_tree(PandasData, pdata, cross_entropy)
    tree.right_child = None
    return tree


### Testing ###

class TestDecisionTree(unittest.TestCase):
    '''
    Testing all main functions also implicitely tests all object methods
    since every function uses these methods. Thus, if the main function works, 
    the methods work as well.
    
    Furthermore, all data objects have the same methods. Thus all cases for
    different data structures are also accounted for.
    '''
    
    ## Build Tree
    
    def test_build_tree(self):
        # PandasData test
        self.assertEqual(is_valid(build_tree(PandasData, pdata, cross_entropy)), True)
        self.assertEqual(is_valid(build_tree(PandasData, pdata, gini_index)), True)
        self.assertEqual(is_valid(build_tree(PandasData, pdata, bayes_error)), True)
        
        # SQLData test
        self.assertEqual(is_valid(build_tree(SQLData, sdata, cross_entropy)), True)
        self.assertEqual(is_valid(build_tree(SQLData, sdata, gini_index)), True)
        self.assertEqual(is_valid(build_tree(SQLData, sdata, bayes_error)), True)
    
    def test_find_split(self):
        # Data structure test
        self.assertEqual(find_split(pdata, cross_entropy), ('Jazz', 1, 0.0676339740100263))
        self.assertEqual(find_split(pdata, gini_index), ('Jazz', 1, 0.018409994353472625))
        
        self.assertEqual(find_split(sdata, cross_entropy), ('jazz', 1, 0.0676339740100263))
        self.assertEqual(find_split(sdata, gini_index), ('jazz', 1, 0.018409994353472625))
        
        # Randomized test
        for i in range(20):
            # checks that impurity is always > 0 since we can't test for var & split
            self.assertGreater(find_split(samplePandas(pdata))[2], 0) # Pandas
            self.assertGreater(find_split(sampleSQL(sdata, connection))[2], 0) #SQL
        
        # Edge case 1: same impurity 
        self.assertEqual(find_split(same_impurity(), cross_entropy), ('c2', 1, 0.0))
        self.assertEqual(find_split(same_impurity(), gini_index), ('c2', 1, 0.0))
        # Edge case 2: same value for 1 column
        self.assertEqual(find_split(same_value(), cross_entropy), ('c2', 1, 1.0))
    
    def test_numeric_split(self):
        # Test on data with same impurity values
        pd_same_imp = same_impurity()
        self.assertEqual(numeric_split(pd_same_imp.get_variable_vector('c1'),
                                       pd_same_imp.get_variable_vector('label'),
                                       cross_entropy), (1, 0.0))
        # Test on data with 1 irrelevant column
        pd_same_vals = same_value()
        self.assertEqual(numeric_split(pd_same_vals.get_variable_vector('c2'),
                                       pd_same_vals.get_variable_vector('label'),
                                       cross_entropy), (1, 1.0))
        
    def test_cross_entropy(self):
        self.assertEqual(cross_entropy(pdata.get_variable_vector("Rock")), 0.7780113035465377)
        self.assertEqual(cross_entropy(sdata.get_variable_vector("rock")), 0.7780113035465377)
        self.assertEqual(cross_entropy(same_impurity().get_variable_vector("label")), 0)
    
    def test_gini(self):
        self.assertEqual(gini_index(pdata.get_variable_vector("Rock")), 0.1771)
        self.assertEqual(gini_index(sdata.get_variable_vector("rock")), 0.1771)
        self.assertEqual(gini_index(same_impurity().get_variable_vector("label")), 0)
    
    def test_bayes_error(self):
        self.assertEqual(bayes_error(pdata.get_variable_vector("Rock")), 0.23)
        self.assertEqual(bayes_error(sdata.get_variable_vector("rock")), 0.23)
        self.assertEqual(bayes_error(same_impurity().get_variable_vector("label")), 0)
    
    ## Is a Valid Tree
    
    def test_is_valid(self):
        # Test empty trees
        self.assertEqual(is_valid(TreeNode('a', 1, pdata), pdata), True)
        self.assertEqual(is_valid(TreeNode('a', 1, sdata), sdata), True)
        # Test valid trees
        t1 = build_tree(PandasData, pdata, cross_entropy)
        t2 = build_tree(SQLData, sdata, gini_index, connection)
        self.assertEqual(is_valid(t1, pdata), True) #Pandas
        self.assertEqual(is_valid(t2, sdata), True) #SQL
        # Test bad trees
        t3 = build_bad_tree()
        t4 = build_bad_tree2()
        self.assertEqual(is_valid(t3, pdata), False)
        self.assertEqual(is_valid(t4, pdata), False)
    
    ## Predict: This test encompasses the labelize function
    
    def test_predict_new_point(self):
        t1 = build_tree(PandasData, pdata, cross_entropy) # Pandas
        t2 = build_tree(PandasData, pdata, gini_index)
        ts1 = build_tree(SQLData, sdata, cross_entropy, sdata.conn) # SQL
        # Test on train set
        self.assertEqual(predict(t1, pdata), pdata.get_variable_vector(pdata.get_label_name()))
        self.assertEqual(predict(t2, pdata), pdata.get_variable_vector(pdata.get_label_name()))
        self.assertEqual(predict(ts1, sdata), sdata.get_variable_vector('rock'))
        # Test on test set
        self.assertEqual(len(predict(t1, pdata_test)), 26) # checks proper length
        self.assertEqual(len(set(predict(t1, pdata_test))), 2) # checks vals in [0,1]
    
    ## Prune
    
    def test_prune(self):
        t1 = build_tree(PandasData, pdata, cross_entropy) # Pandas
        prune(t1, alpha = .01)
        ts1 = build_tree(SQLData, sdata, cross_entropy, sdata.conn) # SQL
        prune(ts1, alpha = .01)
        # Checks that pruning still maintains tree structure
        self.assertEqual(is_valid(t1, pdata), True)
        self.assertEqual(is_valid(ts1, sdata), True)
        # Check heavy penalty is just root node
        t2 = build_tree(PandasData, pdata, gini_index)
        prune(t2, alpha = 1)
        self.assertEqual(t2.left_child, None)
        self.assertEqual(t2.right_child, None)
    
    def test_g(self):
        t1 = build_tree(PandasData, pdata, cross_entropy)
        parent_leaf_nodes = t1.get_leaf_parents()
        self.assertEqual(round(g(t1, parent_leaf_nodes[0]), 5), 0.01)
        self.assertEqual(round(g(t1, parent_leaf_nodes[1]), 5), 0.01)
        
        ts1 = build_tree(SQLData, sdata, cross_entropy, sdata.conn) # SQL
        parent_leaf_nodes_2 = ts1.get_leaf_parents()
        self.assertEqual(round(g(ts1, parent_leaf_nodes_2[0]), 5), 0.01)
        self.assertEqual(round(g(ts1, parent_leaf_nodes_2[1]), 5), 0.01)
    
    def test_R(self):
        t1 = build_tree(PandasData, pdata, cross_entropy)
        parent_leaf_nodes = t1.get_leaf_parents()
        self.assertEqual(round(R(t1, parent_leaf_nodes[0]), 3), .01)
        self.assertEqual(round(R(t1, parent_leaf_nodes[1]), 3), .01)
        t1_leaves =  t1.get_leaves()
        self.assertEqual(R(t1, t1_leaves[0]), 0)
        
        ts1 = build_tree(SQLData, sdata, cross_entropy, sdata.conn) # SQL
        parent_leaf_nodes_2 = ts1.get_leaf_parents()
        t2_leaves = ts1.get_leaves()
        self.assertEqual(round(R(ts1, parent_leaf_nodes_2[0]), 3), .01)
        self.assertEqual(round(R(ts1, parent_leaf_nodes_2[1]), 3), .01)
        self.assertEqual(R(ts1, t2_leaves[0]), 0)
    
    ## Cross Validation
    
    def test_cv_alpha(self):
        '''
        These serve as runtime tests and randomized tests since
        the function uses random sampling inside.
        
        We test to make sure the cross validated values are within the proper
        ranges. 
        '''
        self.assertGreater(get_alpha(PandasData, pdata, cross_entropy), 0)
        self.assertGreater(get_alpha(SQLData, sdata, cross_entropy, sdata.conn), 0)
        self.assertLess(get_alpha(PandasData, pdata, cross_entropy), .2)
        self.assertLess(get_alpha(SQLData, sdata, cross_entropy, sdata.conn), .2)
    
    def test_misclassification_rate(self):
        t1 = build_tree(PandasData, pdata, cross_entropy)
        preds = predict(t1, pdata_test)
        self.assertEqual(round(misclassification_rate(pdata_test, preds), 3), .269)
        t2 = build_tree(PandasData, pdata, gini_index)
        preds2 = predict(t2, pdata_test)
        self.assertEqual(round(misclassification_rate(pdata_test, preds2), 3), .231)
    
    ## Build Forest: This test encompasses building the tree for a forest
    
    def test_build_forest(self):
        f1 = build_forest(PandasData, pdata, cross_entropy, 7, 2)
        f2 = build_forest(SQLData, sdata, cross_entropy, 3, 2, connection)
        # Check that forest is of specified depth
        self.assertEqual(len(f1), 7)
        self.assertEqual(len(f2), 3)
        # Check for valid trees
        self.assertEqual([is_valid(x[0], x[1]) for x in f1], [True]*7)
        self.assertEqual([is_valid(x[0], x[1]) for x in f2], [True]*3)


if __name__ == '__main__':
    unittest.main()