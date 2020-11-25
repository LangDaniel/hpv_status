# model.py
import tensorflow.keras as keras
import h5py
import metrics

class Model():
    def __init__(self, args):
        self.args = args

    def get_optimizer(self):
        args = self.args['optimizer']
        if args['name'].lower() == 'adam':
            optimizer = keras.optimizers.Adam
        elif args['name'].lower() == 'sgd':
            optimizer = keras.optimizers.SGD
        else:
            raise ValueError('optimizer not listed')
        names = optimizer.__init__.__code__.co_varnames[1:-1]
        values = list(optimizer.__init__.__defaults__)
        variables = dict(zip(names[:-len(values)], values))
        
        for ag in args.keys():
            if ag in names:
                variables[ag] = args[ag]
            else:
                raise ValueError(f'variable {ag} not in callback')
   
        return optimizer(**args)

    def get_loss(self):
        if self.args['loss']['type'].lower() == 'binary_crossentropy':
            loss = keras.losses.BinaryCrossentropy()
        else:
            raise ValueError('loss not listed')

        return loss

    def make_base(self):
        if self.args['trained']['path'] == 'None':
            weights = None
        else:
            weights = self.args['trained']['path'] 

        base_model = keras.applications.VGG16(
            include_top=self.args['trained']['include_top'],
            weights=weights,
            input_shape=self.args['size']['input_size'], 
            pooling=None)

        base_model.trainable = False

        return base_model

    def make_model(self):

        base_model = self.make_base()
        inputs = base_model.inputs

        x = inputs
        x = base_model(x)
        
        x = keras.layers.Flatten()(x)
    
        for units, drop in zip(self.args['size']['dense_units'], self.args['size']['dense_drop']):
            if self.args['size']['dense_bn']:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(units)(x)
            x = keras.layers.Activation(self.args['size']['activation'])(x)
            if drop:
                x = keras.layers.Dropout(drop)(x)
    
        x = keras.layers.Dense(self.args['size']['output_size'])(x)
        outputs = keras.layers.Activation(self.args['size']['out_activation'], name='out_layer')(x)
    
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_model(self):
        model = self.make_model()

        optimizer = self.get_optimizer()
        loss = self.get_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics.metrics)

        return model
