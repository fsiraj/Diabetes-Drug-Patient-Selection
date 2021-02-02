import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    new_df = df.copy()
    new_df = new_df.merge(
        ndc_df[['NDC_Code', 'Proprietary Name']], left_on='ndc_code', right_on='NDC_Code')
    new_df['generic_drug_name'] = new_df['Proprietary Name']
    del new_df["NDC_Code"]
    del new_df["Proprietary Name"]
    return new_df
    
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    enc_id = 'encounter_id'
    pat_id = 'patient_nbr'
    
    df = df.sort_values(by=[enc_id])
    first_encs = df.groupby(pat_id)[enc_id].head(1).values
    first_encounter_df = df[ df[enc_id].isin(first_encs) ]
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    N = len(df)
    p1 = int(0.6*N)
    p2 = int(0.8*N)
    
    indices = list(df.index)
    np.random.shuffle(indices)
    
    train = df.loc[indices[:p1]]
    validation = df.loc[indices[p1:p2]]
    test = df.loc[indices[p2:]]
    
    return train, validation, test


#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        if c == 'primary_diagnosis_code':
            cat_col = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=10)        
        else:
            cat_col = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(cat_col)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    thresh = 5
    student_binary_prediction = np.array(df[col].apply(lambda x: 1 if x > thresh else 0).values).astype(int)
    return student_binary_prediction
