import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM,LayerNormalization,Activation

class Multi_LSTM_Unit(Layer):
    def __init__(self,layer_config,**kwargs):
        super().__init__(**kwargs)

        self.layer_config=layer_config
        
        self.lstm_layers=[]
        self.normalization_layers=[]
        self.activation_layers=[]

        for config in self.layer_config:
            lstm_layer=LSTM(
                units=config["units"],
                activation=None,
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )

            norm_layer=LayerNormalization()
            activation_layer=Activation(activation=config["activation"])

            self.lstm_layers.append(lstm_layer)
            self.normalization_layers.append(norm_layer)
            self.activation_layers.append(activation_layer)


    def call(self,inputs):
        x=inputs
        for lstm_layer,norm_layer,activation_layer in zip(self.lstm_layers,self.normalization_layers,self.activation_layers):
            x=lstm_layer(x)
            x=norm_layer(x)
            x=activation_layer(x)


        return x

    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.layer_config[-1]["units"])


    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "layer_config":self.layer_config
            }
        )

        return config
        