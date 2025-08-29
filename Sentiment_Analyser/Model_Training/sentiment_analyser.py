import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","..","Document_Classifier","Model_Training"))

from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Input,
                                     LSTM,
                                     Dense,
                                     Concatenate,
                                     LayerNormalization,
                                     Activation,
                                     Embedding)

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

from multi_lstm_unit import Multi_LSTM_Unit

import warnings
warnings.filterwarnings("ignore")

data_store=Path(os.getcwd()).parent.parent/"Data_Store"


class Sentiment_Analyser:
    def __init__(
        self,
        num_multi_lstm_units,
        lstm_layers_config,
        dense_layers_config,
        optimizer_config,
        model_loss_function,
        word_index,
        context_window,
        embedding_dim
    ):
        
        self.num_multi_lstm_units=num_multi_lstm_units
        self.lstm_layers_config=lstm_layers_config
        self.dense_layers_config=dense_layers_config
        self.optimizer_config=optimizer_config
        self.model_loss_function=model_loss_function

        self.word_index=word_index
        self.vocab_size=len(word_index)
        self.context_window=context_window
        self.embedding_dim=embedding_dim
        self.glove_embedding_path=os.path.join(data_store,"glove.6B.100d.txt")

        self.model_h5_path=os.path.join(data_store,"sentiment_analyser_model.h5")
        self.model_keras_path=os.path.join(data_store,"sentiment_analyser_model.keras")
        
        self.dense_layers=[]
        for config in self.dense_layers_config:
            layer=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )
            self.dense_layers.append(layer)



    def embedding_matrix_generation(self):
        matrix=np.zeros(shape=(self.vocab_size+1,self.embedding_dim),dtype="float32")
        word_embedding={}
        no_embedding_word=0

        with open(self.glove_embedding_path,"r",encoding="utf-8") as file:
            for line in file:
                line=line.split(" ")
                word=line[0]
                vector=np.asarray(line[1:])

                word_embedding[word]=vector


        for word in self.word_index.keys():
            if word in word_embedding.keys():
                matrix[self.word_index[word]]=word_embedding[word]

            else:
                no_embedding_word+=1


        self.embedding_matrix=matrix
        print(f"[INFO] Embedding matrix generated")
        print(f"[WARNING] {no_embedding_word} words embedding not found!!!")


    def build_model(self):
        print(f"[INFO] Building Model")

        inputs=[]
        for _ in range(self.num_multi_lstm_units):
            inp=Input(shape=(self.context_window,),dtype=tf.int32)
            inputs.append(inp)

        outputs=[]
        for input_layer in inputs:
            x=Embedding(
                input_dim=self.vocab_size+1,
                output_dim=self.embedding_dim,
                embeddings_initializer=Constant(self.embedding_matrix),
                trainable=True
            )(input_layer)

            x=Multi_LSTM_Unit(layer_config=self.lstm_layers_config)(x)
            outputs.append(x)

        out=Concatenate(axis=-1)(outputs)

        
        for dense_layer in self.dense_layers:
            out=dense_layer(out)
            

        model=Model(inputs=inputs,outputs=out)
        self.model=model

        print(f"[INFO] Model Built")


    def compile_model(self):
        optimizer=Adam(
            learning_rate=self.optimizer_config["learning_rate"],
            beta_1=self.optimizer_config["beta_1"],
            beta_2=self.optimizer_config["beta_2"],
            clipnorm=self.optimizer_config["clipnorm"]
        )

        self.model.compile(
            optimizer=optimizer,
            loss=self.model_loss_function,
            metrics=["accuracy"]
        )

        print(f"[INFO] Model Compiled")


    def create_model(self):
        self.embedding_matrix_generation()
        self.build_model()
        self.compile_model()

        print(f"[INFO] Model Architecture: ")
        self.model.summary()



    def train_model(self,train_X,train_Y,val_X,val_Y,epochs,batch_size,callback):
        train_X=np.reshape(train_X,(train_X.shape[0],self.num_multi_lstm_units,self.context_window))
        train_X=[train_X[:,ind,:] for ind in range(self.num_multi_lstm_units)]

        val_X=np.reshape(val_X,(val_X.shape[0],self.num_multi_lstm_units,self.context_window))
        val_X=[val_X[:,ind,:] for ind in range(self.num_multi_lstm_units)]

        print(f"[INFO] Data Prepared for Training")
        
        print(f"[INFO] Model Training started")
        self.model.fit(
            x=train_X,
            y=train_Y,
            validation_data=(val_X,val_Y),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback]
        )

        print(f"[INFO] Model Training completed")



    def save_model(self):
        self.model.save(
            self.model_h5_path
        )
        print(f"[INFO] Model saved at path: {self.model_h5_path}")

        self.model.save(
            self.model_keras_path
        )
        print(f"[INFO] Model saved at path: {self.model_keras_path}")


