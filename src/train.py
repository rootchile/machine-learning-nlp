import io
import torch
import time

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn import metrics

import config
import IMDBDataset as dataset
import engine
from LSTM import LSTM


def load_vectors(filename):
    f_in = io.open(filename, 'r', encoding='utf-8', newline='\n', errors = 'ignore')
    n, d = map(int, f_in.readline().split())
    data = {}
    
    for line in f_in:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    
    return data

def create_embedding(word_index, embedding_dict):
    """Function to create the embedding matrix

    Args:
        word_index ([dict]): dictionary with word:index_value
        embedding_dict ([dict]): dictionary with word:embedding_vector
    
    Return:
        Numpy array eith embedding vector for all know words
    """
    
    embedding_matrix = np.zeros((len(word_index)+1,300))
    
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
            
    return embedding_matrix

def run(df, fold):
    """Run training and validation for a given fold

    Args:
        df ([pandas dataframe]): with kfold column
        fold ([int]): current fold
    """
    
    df_train = df[df.kfold != fold ].reset_index(drop=True)
    df_valid = df[df.kfold == fold ].reset_index(drop=True)
    time_init = time.time()
    print('Keras tokenizer')
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df['review'].values.tolist())
    
    x_train = tokenizer.texts_to_sequences(df_train['review'].values)
    x_valid = tokenizer.texts_to_sequences(df_valid['review'].values)
    
    
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=config.MAX_LEN)
    x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=config.MAX_LEN)
    
    train_dataset = dataset.IMDBDataset(reviews=x_train, targets=df_train['sentiment'].values)
    valid_dataset = dataset.IMDBDataset(reviews=x_valid, targets=df_valid['sentiment'].values)

    #Torch data_loader
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2)
    
    print(f'Time: {np.round(time.time()-time_init,2)} secs.')
    time_init = time.time()
    print('Loading embeddings')
    
    embedding_dict = load_vectors(config.EMBEDDING_DICT)
    embedding_matrix = create_embedding(tokenizer.word_index, embedding_dict)
    
    device = torch.device('cpu')
    
    model = LSTM(embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f'Time: {np.round(time.time()-time_init,2)} secs.')
    print('Training model')
    
    best_accuracy = 0
    
    early_stopping_counter = 0
    
    for epoch in range(config.EPOCHS):
        time_init = time.time()
        print(f'Fold {fold} - Training epoch {epoch}')    
        engine.train(train_data_loader, model, optimizer, device)
        outputs, targets = engine.evaluate(valid_data_loader, model, device)
        outputs = np.array(outputs)
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'Fold: {fold}, Epoch: {epoch}, Accuracy: {accuracy}')
        print(f'Time: {np.round(time.time()-time_init,2)} secs.')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter=+1
        
        if early_stopping_counter > 2:
            break
        
    
if __name__ == '__main__':
    print('Loading folds')
    time_init = time.time()
    df = pd.read_csv(config.DATA_FOLDS)
    print(f'Time: {np.round(time.time()-time_init,2)} secs.')

    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)