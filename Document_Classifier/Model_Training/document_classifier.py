import os
import json

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Concatenate,Embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from multi_lstm_unit import Multi_LSTM_Unit


class Document_Classifier:
    def __init__(
        self,
        context_window,
        num_multi_lstm_units,
        multi_lstm_layer_config,
        dense_layer_config,
        optimizer_config,
        model_loss,
        embedding_dim,
        glove_embeddings_path,
        word_index
    ):
        
        self.context_window=context_window
        self.num_multi_lstm_units=num_multi_lstm_units
        self.multi_lstm_layer_config=multi_lstm_layer_config
        self.dense_layer_config=dense_layer_config
        self.optimizer_config=optimizer_config
        self.model_loss=model_loss
        self.embedding_dim=embedding_dim
        self.glove_embeddings_path=glove_embeddings_path
        self.word_index=word_index

        self.vocab_size=len(self.word_index)


    def embedding_matrix_generation(self):
        matrix=np.zeros(shape=(self.vocab_size+1,self.embedding_dim),dtype="float32")
        word_embed={}
        not_words=0

        with open(self.glove_embeddings_path,"r",encoding="utf-8") as file:
            for line in file:
                line=line.split()
                word=line[0]
                embeddings=np.asarray(line[1:],dtype="float32")

                word_embed[word]=embeddings


            for word in self.word_index.keys():
                if word in word_embed.keys():
                    matrix[self.word_index[word]]=word_embed[word]

                else:
                    not_words+=1


            self.embedding_matrix=matrix

            print(f"[INFO] Embedding Matrix Created...")
            print(f"[WARNING] Embeddings for {not_words} words not in Glove Embeddings!!!")


    def create_model(self):
        input_layers=[]
        for _ in range(self.num_multi_lstm_units):
            input_layers.append(
                Input(shape=(self.context_window,),dtype=tf.int32)
            )
            

        out=[]
        for inp in input_layers:
            x=Embedding(
                input_dim=self.vocab_size+1,
                output_dim=self.embedding_dim,
                embeddings_initializer=Constant(self.embedding_matrix),
                trainable=True
            )(inp)

            x=Multi_LSTM_Unit(layer_config=self.multi_lstm_layer_config)(x)
            out.append(x)


        x=Concatenate(axis=-1)(out)

        for config in self.dense_layer_config:
            x=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x)
            

        out=x

        model=Model(inputs=input_layers,outputs=out)

        self.model=model

        print(f"[INFO] Model Architecture Created Successfully...")


    def compile_model(self):
        optimizer=Adam(
            learning_rate=self.optimizer_config["learning_rate"],
            beta_1=self.optimizer_config["beta_1"],
            beta_2=self.optimizer_config["beta_2"],
            clipnorm=self.optimizer_config["clipnorm"]
        )

        self.model.compile(
            optimizer=optimizer,
            loss=self.model_loss,
            metrics=["accuracy"]
        )

        print(f"[INFO] Model Compiled Successfully...")


    def build_model(self):
        print(f"[INFO] Building Model..")

        self.embedding_matrix_generation()
        self.create_model()
        self.compile_model()

        print(f"[INFO] Model Architecture:")
        print("\n\n")
        self.model.summary()

        print(f"[INFO] Model Built Successfully...")


    def train_model(self,train_X,train_Y,val_X,val_Y,epochs,batch_size):
        train_X=np.reshape(train_X,(train_X.shape[0],self.num_multi_lstm_units,self.context_window))
        train_X_inp=[train_X[:,ind,:] for ind in range(self.num_multi_lstm_units)]
        print(f"[INFO] Training Data Prepared...")

        val_X=np.reshape(val_X,(val_X.shape[0],self.num_multi_lstm_units,self.context_window))
        val_X_inp=[val_X[:,ind,:] for ind in range(self.num_multi_lstm_units)]
        print(f"[INFO] Validation Data Prepared...")

        print(f"[INFO] Model Training Started..")

        self.model.fit(
            x=train_X_inp,
            y=train_Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_X_inp,val_Y)
        )

        print(f"[INFO] Model Training Completed..")

        return self.model

        


    
            
        
        
        

