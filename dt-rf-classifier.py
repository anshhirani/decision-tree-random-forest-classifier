#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:06:06 2018

@author: anshhirani
"""

### Import Modules

import pandas as pd
import numpy as np
import psycopg2 as ps2
from sklearn.model_selection import KFold
import math
import copy
import random


### Define Data Classes

class TabularData:
    
    def __init__(self):
        pass
    
    def get_covariate_names(self):
        pass
    
    def get_label_name(self):
        pass
    
    def get_variable_vector(self, var):
        pass
    
    def get_data_size(self):
        pass
    
    def left_query(self, split_var, val):
        pass
    
    def right_query(self, split_var, val):
        pass
    
    def get_row(self, i):
        pass
    
    def get_row_ids(self):
        pass
    
    def get_sample(self, pts, i):
        pass
    
    def get_train_subset(self, train_ids):
        pass
    
    def get_test_subset(self, test_ids):
        pass
    
    def get_subdata(self, sampled_cov):
        pass


class PandasData(TabularData):
    
    def __init__(self, data, label):
        '''
        We assume that all data that come from csv files will be read
        in as a pandas dataframe. 
        
        We also need the user to specify the data's label variable. 
        This input is a string. 
        
        If the input is not a csv, the input must be of type df.
        '''
        if type(data) is str:
            self.df = pd.read_csv(data)
        else:
            self.df = data
        self.label = label
    
    def get_covariate_names(self):
        covariates = [cov for cov in self.df.columns.get_values().tolist()
                      if cov != self.label]
        return covariates
    
    def get_label_name(self):
        return self.label
    
    def get_variable_vector(self, var):
        return self.df[var].tolist()
    
    def get_data_size(self):
        return self.df.shape[0]
    
    def left_query(self, split_var, val):
        return self.df.query('%s < %f' % (split_var, val))
    
    def right_query(self, split_var, val):
        return self.df.query('%s >= %f' % (split_var, val))
    
    def get_row(self, i):
        return self.df.iloc[i].tolist()
    
    def get_row_ids(self):
        return range(len(self.df))
    
    def get_sample(self, pts, i):
        return self.df.sample(pts, replace = True)
    
    def get_train_subset(self, train_ids):
        return self.df.iloc[train_ids]
    
    def get_test_subset(self, test_ids):
        return self.df.iloc[test_ids]
    
    def get_subdata(self, sampled_cov):
        sampled_cov.append(self.label)
        return self.df[sampled_cov]


class SQLData(TabularData):
    '''
    We assume this structure always has a serial primary key named 'id'.
    '''

    def __init__(self, data, label, conn):
        self.conn = conn
        self.cur = conn.cursor()
        self.label = label.lower()
        self.data_table = data
    
    def get_covariate_names(self):
        self.cur.execute('select * from %s limit 0 ;' % (self.data_table))
        names = [desc[0] for desc in self.cur.description]
        covariates = [cov for cov in names if (cov != self.label and cov != 'id')]
        return covariates

    def get_label_name(self):
        return self.label
    
    def get_variable_vector(self, var):
        self.cur.execute('select %s from %s;' % (var, self.data_table))
        vec = self.cur.fetchall()
        return [float(vec[i][0]) for i in range(len(vec))]
    
    def get_data_size(self):
        self.cur.execute('select count(*) from %s;' % (self.data_table))
        return self.cur.fetchall()[0][0]
    
    def left_query(self, split_var, val):
        # convert val to string
        val = str(val)
        # checks if 'where' clause in current select statement.
        if 'where' in self.data_table:
            new_table = self.data_table + ' and %s<%s' % (split_var, val)
        else: 
            new_table = self.data_table + ' where %s<%s' % (split_var, val)
        return new_table
    
    def right_query(self, split_var, val):
        val = str(val)
        if 'where' in self.data_table:
            new_table = self.data_table + ' and %s>=%s' % (split_var, val)
        else:
            new_table = self.data_table + ' where %s>=%s' % (split_var, val)
        return new_table
    
    def get_row(self, i):
        if 'where' in self.data_table:
            self.cur.execute('select %s from %s and id = %s;' % (','.join(self.get_covariate_names()), self.data_table, i))
        else:
            self.cur.execute('select %s from %s where id = %s' % (','.join(self.get_covariate_names()),self.data_table, i))
        row = self.cur.fetchall()
        row = list([row[i] for i in range(len(row))][0])
        return [float(row[i]) for i in range(len(row))]
    
    def get_row_ids(self):
        self.cur.execute('select id from %s;' % (self.data_table))
        ids = self.cur.fetchall()
        return [int(ids[i][0]) for i in range(len(ids))]
    
    def get_sample(self, pts, i):
        # create temp table named 'sample_table'
        self.cur.execute('drop table if exists sample_table_%s' % (i))
        self.cur.execute("create table sample_table_%s as "
                         "(select * from %s "
                         " right join (select ceil(random() * %s) as rowid "
                         "            from generate_series(1, %s)) as randomstuff"
                         " on %s.id = randomstuff.rowid)" % (i, self.data_table, pts, pts, self.data_table))
        self.conn.commit()
        return 'sample_table'
    
    def get_train_subset(self, train_ids):
        self.cur.execute('drop table if exists train_subset;')
        if 'where' in self.data_table:
            self.cur.execute("create table train_subset as (select * from %s and id in %s);"
                             % (self.data_table, tuple(train_ids)))
        else:
            self.cur.execute("create table train_subset as (select * from %s where id in %s);"
                             % (self.data_table, tuple(train_ids)))
        self.conn.commit()
        return 'train_subset'
    
    def get_test_subset(self, test_ids):
        self.cur.execute('drop table if exists test_subset;')
        if 'where' in self.data_table:
            self.cur.execute("create table test_subset as (select * from %s and id in %s);"
                             % (self.data_table, tuple(test_ids)))
        else:
            self.cur.execute("create table test_subset as (select * from %s where id in %s);"
                             % (self.data_table, tuple(test_ids)))
        self.conn.commit()
        return 'test_subset'
    
    def get_subdata(self, sampled_cov):
        sampled_cov.append(self.label)
        sampled_cov.append('id')
        # convert the string
        
        self.cur.execute('drop table if exists sub_table')
        self.cur.execute('create table sub_table as (select %s from %s)'
                         % (', '.join(sampled_cov), self.data_table))
        self.conn.commit()
        return 'sub_table'


### Define Tree Class    

class TreeNode:
    '''
    This class is that of the individual node. The node contains:
        1. covariate as its key/name
        2. split value of the covariate
        3. edges that follow to the child node
        4. data stored in the node
    '''

    def __init__(self, name, split, data):
        self.name = name
        self.split = split
        self.data = data
        self.left_child = None
        self.right_child = None
    
    def get_data(self):
        return self.data
    
    def get_leaves(self):
        if self.left_child == None and self.right_child == None:
            return [self]
        elif self.left_child == None:
            return self.right_child.get_leaves()
        elif self.right_child == None:
            return self.left_child.get_leaves()
        else:
            return self.left_child.get_leaves() + self.right_child.get_leaves()
    
    def is_leaf(self):
        if self.left_child is None and self.right_child is None:
            return True
        return False
    
    def get_leaf_parents(self):
        if self.left_child is None and self.right_child is None:
            return []
        left = self.left_child
        right = self.right_child
        if left.is_leaf() and right.is_leaf():
            return [self]
        else:
            return left.get_leaf_parents() + right.get_leaf_parents()


### Build Tree

def build_tree(data_type, train, impurity_fn, *data_args):
    '''
    This function builds a decision tree using the given impurity
    function on the training set. 
    
    The tree is built to the point where all leaves have the same label or
    until next splits are repeated.
    
    data_type specifies the type of tabular data we're using
    '''
    if train.get_data_size() == 0:
        return None
    label = train.get_label_name()
    # get variable and split value
    split_attr = find_split(train, impurity_fn)
    tree = TreeNode(name = split_attr[0], split = split_attr[1], data = train)
    # if impurity reduction is 0, don't split
    if (len(set(tree.get_data().get_variable_vector(label))) == 1) or (split_attr[2] == 0):
        return tree
    else:
        # splits
        left_split = data_type(train.left_query(split_attr[0], split_attr[1]), label, *data_args)
        right_split = data_type(train.right_query(split_attr[0], split_attr[1]), label, *data_args)
        assert train.get_data_size() > left_split.get_data_size(), "left split failure"
        assert train.get_data_size() > right_split.get_data_size(), "right split failure"
        tree.left_child = build_tree(data_type, left_split, impurity_fn, *data_args)
        tree.right_child = build_tree(data_type, right_split, impurity_fn, *data_args)
    return tree


def find_split(data, impurity_fn):
    '''
    This function finds the best covariate and
    split value of input dataset using
    the specified impurity function.
    
    Returns a 3-tuple of:
    (attribute, split value, information gained)
    '''
    # list of best impurities per covariate
    impurities = []
    covariates = data.get_covariate_names()    
    label = data.get_label_name()
    label_vals = data.get_variable_vector(label)
    # find split and impurity
    for i in range(len(covariates)):
        cov = covariates[i]
        cov_vals = data.get_variable_vector(cov)
        split_and_imp = numeric_split(cov_vals, label_vals, impurity_fn)
        impurities.append((cov, *split_and_imp))
    impurities.sort(key = lambda tup: tup[2])
    return impurities[-1]


def numeric_split(covariate_vals, label_vals, impurity_fn):
    '''
    This function finds the optimal split value of the given covariate
    along with the information gained.
    
    covariate_vals: The vector of values of the covariate
    label_vals: The vector of the label values
    impurity_fn: The function used to calculate impurity / information gained
    '''
    impurity_region = impurity_fn(label_vals)
    cov_vals_set = set(covariate_vals)
    n = len(covariate_vals)
    impurities = []
    for val in cov_vals_set:
        # create data splits
        left_split_indices = [i for i in range(len(covariate_vals)) if
                              covariate_vals[i] < val]
        
        right_split_indices = [i for i in range(len(covariate_vals)) if
                              covariate_vals[i] >= val]
        # impurity calculations
        pl = len(left_split_indices) / n
        pr = len(right_split_indices) / n
        # get label vectors from indices
        left_label_vals = [label_vals[i] for i in left_split_indices]
        right_label_vals = [label_vals[i] for i in right_split_indices]
        impurity_left = impurity_fn(left_label_vals)
        impurity_right = impurity_fn(right_label_vals)
        tot_imp_red = impurity_region - (pl*impurity_left) - (pr*impurity_right)
        impurities.append((tot_imp_red, val))
    return max(impurities)[::-1]


def cross_entropy(label_vec):
    '''
    Uses cross entropy as the impurity function
    '''
    p = 0
    n = len(label_vec)
    if n == 0: return 0
    for i in range(len(label_vec)):
        if label_vec[i] == 1:
            p += 1
    pr_p = p / n
    if pr_p == 0 or pr_p == 1: return 0
    ch = -pr_p*math.log(pr_p, 2) - (1-pr_p)*math.log(1-pr_p, 2)
    return ch

def gini_index(label_vec):
    '''
    Uses gini index as the impurity function
    '''
    p = 0
    n = len(label_vec)
    if n == 0: return 0
    for i in range(len(label_vec)):
        if label_vec[i] == 1:
            p += 1
    pr_p = p/n
    if pr_p == 0 or pr_p == 1: return 0
    gini = pr_p*(1-pr_p)
    return gini

def bayes_error(label_vec):
    '''
    Uses bayes error as the impurity function
    '''
    p = 0
    n = len(label_vec)
    if n == 0: return 0
    for i in range(len(label_vec)):
        if label_vec[i] == 1:
            p += 1
    pr_p = p/n
    if pr_p == 0 or pr_p == 1: return 0
    be = min(pr_p, (1-pr_p))
    return be


### Is a Valid Tree
    
def is_valid(tree, train):
    # check tree structure
    if type(tree.name) != str:
        return False
    tree_data = tree.get_data()
    tree_covs = tree_data.get_covariate_names()
    for tcov in tree_covs:
        if tree_data.get_variable_vector(tcov) != train.get_variable_vector(tcov):
            return False
    # check leaves of tree at every covariate
    leaves = tree.get_leaves()
    for tcov in tree_covs:
        full_data = []
        for leaf in leaves:
            leaf_data = leaf.get_data()
            leaf_data_vector = leaf_data.get_variable_vector(tcov)
            full_data.extend(leaf_data_vector)
        if sorted(full_data) != sorted(train.get_variable_vector(tcov)):
            return False
    
    attr = tree.name
    split = tree.split
    # left child
    if tree.left_child != None:
        data_left = tree.left_child.get_data()
        attr_data = data_left.get_variable_vector(attr)
        for i in range(len(attr_data)):
            if attr_data[i] >= split:
                return False
    # right child
    if tree.right_child != None:
        data_right = tree.right_child.get_data()
        right_attr_data = data_right.get_variable_vector(attr)
        for i in range(len(right_attr_data)):
            if right_attr_data[i] < split:
                return False
        return is_valid(tree.left_child, data_left) and is_valid(tree.right_child, data_right)
    return True


### Predict New Point

def predict(tree, test):
    '''
    This function will return a list of the labels for each entry in the 
    testing data. Testing data must be of same type as the training data.
    '''
    labels = []
    test_covs = test.get_covariate_names()
    ids = test.get_row_ids()
    for ID in ids:
        row = test.get_row(ID)
        row_mapping = dict()
        for j in range(len(test_covs)):
            cov = test_covs[j]
            val = row[j]
            row_mapping[cov] = val
        row_label = labelize(tree, row_mapping)
        labels.append(row_label)
    return labels


def labelize(t, row_mapping):
    '''
    This function takes a tree as input and a row of a dataframe and
    traverses through the tree to get the point's label.
    
    row_mapping is a dictionary of a row with keys being covariates and
    the values being that's row's values for the covariate.
    
    '''
    tree = t
    # traverse tree
    while((tree.left_child is not None) and (tree.right_child is not None)):
        attr = tree.name
        split = tree.split
        if row_mapping[attr] < split:
            tree = tree.left_child
        else:
            tree = tree.right_child
    # get label
    final_data = tree.get_data()
    tree_label = final_data.get_variable_vector(final_data.get_label_name())
    n = len(tree_label)
    num_votes = sum(tree_label)
    if num_votes/n >= .5:
        label = 1
    else:
        label = 0
    return label


### Pruning a Tree

def prune(tree, alpha):
    '''
    This function prunes the tree using tuning parameter alpha.
    '''
    # initial prune
    parent_leaf_nodes = tree.get_leaf_parents()
    gt_list = []
    for subtree in parent_leaf_nodes:
        gt = g(tree, subtree)
        gt_list.append((gt, subtree))
    gt_list.sort(key=lambda tup: tup[0])
    # next steps
    while gt_list[0][0] < alpha:
        # prune node
        node_to_prune = gt_list[0][1]
        node_to_prune.left_child = None
        node_to_prune.right_child = None
        # repeat cycle
        parent_leaf_nodes = tree.get_leaf_parents()
        gt_list = []
        for subtree in parent_leaf_nodes:
            gt = g(tree, subtree)
            gt_list.append((gt, subtree))
        if gt_list == []:
            break
        gt_list.sort(key=lambda tup: tup[0])
    return tree

def g(tree, subtree):
    '''
    Calculates g(t) for pruning the tree
    '''
    # calculate R(t)
    R_t = R(tree, subtree)
    # calculate R(T_t)
    RT_t = 0
    subtree_leaves = subtree.get_leaves()
    for subtree_leaf in subtree_leaves:
        rtt = R(tree, subtree_leaf)
        RT_t += rtt
    # calculate gt
    gt = (R_t - RT_t) / (len(subtree_leaves) - 1)
    return gt

def R(tree, subtree):
    sub_data = subtree.get_data()
    sub_label = sub_data.get_label_name()
    n = sub_data.get_data_size()
    num_votes = sum(sub_data.get_variable_vector(sub_label))
    if num_votes/n >= .5:
        rt = (n - num_votes)/ n
    else:
        rt = num_votes / n
    pt = n / tree.get_data().get_data_size()
    return rt*pt


## Cross Validation for Alpha

def get_alpha(data_type, data, impurity_fn, *data_args, k=5):
    '''
    Uses cross validation to get an alpha value
    k is the number of folds that we will use
    '''
    # set up values for cross validation
    label = data.get_label_name()
    alpha_vals = [round(x*.01, 2) for x in range(1,20)]
    kf = KFold(n_splits = k, shuffle = True, random_state = 123)
    alpha_error = []
    id_list = data.get_row_ids()
    # cross validate
    for alpha in alpha_vals:
        misclass_rate_lst = []
        for train_ids, test_ids in kf.split(id_list):
            # construct train & test data
            train_data = data.get_train_subset(train_ids)
            test_data = data.get_test_subset(test_ids)
            # build tree, prune, classify
            tree = build_tree(data_type,
                              data_type(train_data, label, *data_args),
                              impurity_fn,
                              *data_args)
            tree = prune(tree, alpha)
            preds = predict(tree, data_type(test_data, label, *data_args))
            misclass_rate = misclassification_rate(data_type(test_data, label, *data_args), preds)
            misclass_rate_lst.append(misclass_rate)
        avg_err = sum(misclass_rate_lst)/len(misclass_rate_lst)
        alpha_error.append((avg_err, alpha))
    min_err = min(alpha_error)
    final_alpha = min_err[1]
    return final_alpha

def misclassification_rate(test, preds):
    n = len(preds)
    labels = test.get_variable_vector(test.get_label_name())
    misclass = 0
    for i in range(n):
        true_l = labels[i]
        predicted = preds[i]
        if true_l != predicted:
            misclass += 1
    return misclass/n


### Build Forest

def build_forest(data_type, data, impurity_fn, nt, k_cov, *data_args):
    '''
    This function takes a data structure and builds nt trees. 
    
    We sample k<p covariates for each random tree
    '''
    forest = []
    pts = data.get_data_size()
    covariates = data.get_covariate_names()
    label = data.get_label_name()
    if k_cov > len(covariates):
        raise "k must be less than the number of covariates"
    for i in range(nt):
        samp = data.get_sample(pts, i)
        samp_data = data_type(samp, label, *data_args)
        t = build_forest_tree(data_type,
                              samp_data,
                              impurity_fn,
                              k_cov,
                              *data_args)
        forest.append((t, samp_data))
    return forest


def build_forest_tree(data_type, data, impurity_fn, k, *data_args):
    '''
    This function builds a tree with k random covariates chosen at each split
    '''
    if data.get_data_size() == 0:
        return None
    label = data.get_label_name()
    covariates = data.get_covariate_names()
    sampled_cov = random.sample(covariates, k)
    sub_data_df = data.get_subdata(sampled_cov)
    sub_data = data_type(sub_data_df, label, *data_args)
    split_attr = find_split(sub_data, impurity_fn)
    tree = TreeNode(split_attr[0], split_attr[1], data)
    if (len(set(tree.get_data().get_variable_vector(label))) == 1) or (split_attr[2] == 0):
        return tree
    else:
        # splits
        left_split = data_type(data.left_query(split_attr[0], split_attr[1]), label, *data_args)
        right_split = data_type(data.right_query(split_attr[0], split_attr[1]), label, *data_args)
        assert data.get_data_size() > left_split.get_data_size(), "left split failure"
        assert data.get_data_size() > right_split.get_data_size(), "right split failure"
        tree.left_child = build_forest_tree(data_type, left_split, impurity_fn, k, *data_args)
        tree.right_child = build_forest_tree(data_type, right_split, impurity_fn, k, *data_args)
    return tree