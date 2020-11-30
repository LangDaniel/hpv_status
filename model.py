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
        inputs = keras.layers.Input(shape=self.args['size']['input_size'])
    
        conv1 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu',
                       padding='same', name='conv1a')(inputs)
        pool1 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                             padding='valid', name='pool1')(conv1)
    
        conv2 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu',
                       padding='same', name='conv2a')(pool1)
        pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding='valid', name='pool2')(conv2)
    
        conv3a = keras.layers.Conv3D(256, (3, 3, 3), activation='relu',
                        padding='same', name='conv3a')(pool2)
        conv3b = keras.layers.Conv3D(256, (3, 3, 3), activation='relu',
                        padding='same', name='conv3b')(conv3a)
        pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding='valid', name='pool3')(conv3b)
    
        conv4a = keras.layers.Conv3D(512, (3, 3, 3), activation='relu',
                        padding='same', name='conv4a')(pool3)
        conv4b = keras.layers.Conv3D(512, (3, 3, 3), activation='relu',
                        padding='same', name='conv4b')(conv4a)
        pool4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding='valid', name='pool4')(conv4b)
    
        conv5a = keras.layers.Conv3D(512, (3, 3, 3), activation='relu',
                        padding='same', name='conv5a')(pool4)
        conv5b = keras.layers.Conv3D(512, (3, 3, 3), activation='relu',
                        padding='same', name='conv5b')(conv5a)
        zeropad5 = keras.layers.ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                                 name='zeropad5')(conv5b)
        pool5 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding='valid', name='pool5')(zeropad5)
    
        flattened = keras.layers.Flatten()(pool5)
        
        if not self.args['trained']['include_top']:
            out = flattened
        else:
            fc6 = keras.layers.Dense(4096, activation='relu', name='fc6')(flattened)
            dropout1 = keras.layers.Dropout(rate=0.5)(fc6)
    
            fc7 = keras.layers.Dense(4096, activation='relu', name='fc7')(dropout1)
            dropout2 = keras.layers.Dropout(rate=0.5)(fc7)
    
            out = keras.layers.Dense(487, activation='softmax', name='fc8')(dropout2)
    
        base_model = keras.Model(inputs=inputs, outputs=out)

        if self.args['trained']['use']:
            print('using pretrained model')

            with h5py.File(self.args['trained']['path'], 'r') as ff: 
                layer_names = list(ff.keys())
    
                for layer in base_model.layers:
                    name = layer.name
                    if name in layer_names and (name.startswith('conv') or name.startswith('fc')):
                        print('using weights of layer ' + name)
                        layer.set_weights(
                            [ff[name]['kernel'], ff[name]['bias']])

            base_model.trainable = False

        return base_model

    def make_model(self):

        base_model = self.make_base()
        inputs = base_model.inputs

        x = inputs
        x = base_model(x)
    
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
