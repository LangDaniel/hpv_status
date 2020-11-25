# model.py
import tensorflow.keras as keras
import metrics

class Model():
    def __init__(self, args):
        self.args = args
        if len(args['size']['conv_filters']) != len(args['size']['pool_kernel']):
            raise ValueError('convolutional and pool stride size does not match')
        if len(args['size']['conv_filters']) != len(args['size']['conv_drop']):
            raise ValueError('convolutional and dropout size does not match')
        if len(args['size']['pool_kernel']) != len(args['size']['pool_stride']):
            raise ValueError('pool kernel size does not match pool stride size')
        if len(args['size']['dense_units']) != len(args['size']['dense_drop']):
            raise ValueError('dense units size does not match dropout size')

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

    
    def conv_layer(self, inputs, filters, pool_size, strides, drop):
        x = inputs

        print(x.shape, end=' -> ')
        if self.args['size']['conv_bn']: 
            x = keras.layers.BatchNormalization()(x)

        if type(filters) == list: 
            if len(filters) != len(drop):
                raise ValueError('length of filters and dropout must match')
            for ff, dd in zip(filters, drop):
                x = keras.layers.Conv3D(ff, self.args['size']['conv_kernel'], padding='same')(x)
                x = keras.layers.Activation(self.args['size']['activation'])(x)
                if dd:
                    x = keras.layers.Dropout(dd)(x)
        else:
            x = keras.layers.Conv3D(filters, self.args['size']['conv_kernel'], padding='same')(x)
            x = keras.layers.Activation(self.args['size']['activation'])(x)
            if drop:
                x = keras.layers.Dropout(drop)(x)

        x = keras.layers.MaxPooling3D(pool_size=pool_size, strides=strides, padding='valid')(x)

        print(x.shape)
    
        return x

    def dense_layer(self, inputs, units, drop):
        x = inputs

        if self.args['size']['dense_bn']:
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dense(units)(x)
        x = keras.layers.Activation(self.args['size']['activation'])(x)

        if drop:
            x = keras.layers.Dropout(drop)(x)

        return x


    def make_model(self):

        inputs = keras.layers.Input(shape=self.args['size']['input_size'])
        x = inputs

        # conv layers
        for ii in range(0, len(self.args['size']['conv_filters'])):
            x = self.conv_layer(x,
                    self.args['size']['conv_filters'][ii],
                    self.args['size']['pool_kernel'][ii],
                    self.args['size']['pool_stride'][ii],
                    self.args['size']['conv_drop'][ii]
                )

        # dense layers

        x = keras.layers.Flatten()(x)

        for ii in range(0, len(self.args['size']['dense_units'])):
            x = self.dense_layer(x,
                    self.args['size']['dense_units'][ii],
                    self.args['size']['dense_drop'][ii]
                )

        x = keras.layers.Dense(self.args['size']['output_size'])(x)
        outputs = keras.layers.Activation(self.args['size']['out_activation'], name='out')(x)

        return keras.Model(inputs=inputs, outputs=outputs)


    def get_model(self):
        model = self.make_model()

        optimizer = self.get_optimizer()
        loss = self.get_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics.metrics)

        return model
