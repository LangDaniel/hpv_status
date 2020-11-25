## sequence.py

## the dicom reference coordinates system is used:
#   x direction: increases from the right side of the patient to the left
#   y direction: increases from the front of the patient to the back
#   z direction: increases from the feet to the head
# https://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.utils import Sequence

class CustomSequence(Sequence):

    def __init__(self, args):
        self.args = args
        self.meta_df = pd.read_csv(args['df_path']) 
        self.out_dict = dict(zip(self.meta_df['identifier'], self.meta_df['value']))

        # attributes to be set in prepare()
        self.bundle_df = None
        self.total_count = None

    ##############################################################################################
    ########################### functions to read and write bundles ##############################

    def get_indices(self, sgmt):
        ''' This function takes the segmenation as an argument and has to return
        the indices of all the bundles. For the paper only one bundle is cut from
        each input image, but for other tasks you may want to change this.
        Indices have to be returned as list in the order:
        [x dimensions], [y dimensions], [z dimensions]'''

        shape = sgmt.shape    
        x_contour, y_contour, z_contour = np.nonzero(sgmt)
    
        # select z indices
        zmin = self.indices_helper(
            z_contour,
            self.args['bundle_size'][-1],
            shape[-1]
        )
        zmax = zmin + self.args['bundle_size'][-1]
            
        # select x indices
        x_contour = x_contour[np.where(
            (z_contour > zmin) & (z_contour < zmax)
        )]
    
        xmin = self.indices_helper(
            x_contour,
            self.args['bundle_size'][0],
            shape[0]
        )
        
        # select y indices
        y_contour = y_contour[np.where(
            (z_contour > zmin) & (z_contour < zmax)
        )]
    
        ymin = self.indices_helper(
            y_contour,
            self.args['bundle_size'][1],
            shape[1]
        )
        
        return [xmin], [ymin], [zmin]

    def indices_helper(self, contour, size, maximum):
        '''helper function for get_indice() '''
        val_min = min(contour)
        val_max = max(contour)
    
        mid = (val_min + val_max) // 2
        low = max(0, mid - size // 2)
        if (low + size) > maximum:
            low = maximum - size
            
        return low

    def generate_bundles(self):
        ''' Reads the images and segmentations from the h5 input file and crops them
        around the region of interest (ROI).
        Bundle indices are selected via get_indices(). '''

        print(f'generating bundles with {len(self.meta_df)} cases')

        if Path(self.args['bundles_path']).exists():
            raise ValueError('file already exists')
        Path(self.args['bundles_path']).parent.mkdir(parents=True, exist_ok=True)


        with h5py.File(self.args['bundles_path'], 'w') as h5_out:
            img_grp = h5_out.create_group('images')
            sgmt_grp = h5_out.create_group('segmentation')

            with h5py.File(self.args['image_path'], 'r') as h5_in:

                ll = 0
                bundle_df = pd.DataFrame(columns=[
                    'identifier', 'bundle_n', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
                ])

                for pid in self.meta_df['identifier'].values: 

                    complete_image = self.image_preprocessing(
                        h5_in['ct_images'][pid][:]
                    )
                    complete_sgmt = h5_in['ct_sgmt'][pid][:]
                    
                    x_idx, y_idx, z_idx = self.get_indices(complete_sgmt)

                    for kk, (x_ii, y_ii, z_ii)  in enumerate(zip(x_idx, y_idx, z_idx)):

                        # bid = bundle ID
                        bid = pid + '_' + str(kk)

                        x_ff = x_ii + self.args['bundle_size'][0]
                        y_ff = y_ii + self.args['bundle_size'][1]
                        z_ff = z_ii + self.args['bundle_size'][2]

                        img = complete_image[x_ii:x_ff, y_ii:y_ff, z_ii:z_ff]
                        img_grp.create_dataset(bid, data=img)

                        sgmt = complete_sgmt[x_ii:x_ff, y_ii:y_ff, z_ii:z_ff]
                        sgmt_grp.create_dataset(bid, data=sgmt)

                        ## save the boarders of each bundle to a dataframe
                        bundle_df.loc[ll] = [pid, kk, x_ii, x_ff, y_ii, y_ff, z_ii, z_ff]
                        ll += 1

                # shuffle the df
                bundle_df = bundle_df.sample(frac=1).reset_index(drop=True)

                bundle_path = Path(self.args['bundles_path'])
                df_path = bundle_path.parent /  (bundle_path.stem + '.csv')
                bundle_df.to_csv(df_path, index=False)

        Path(self.args['bundles_path']).chmod(0o555)
        Path(df_path).chmod(0o555)

    def prepare(self):
        ''' Reads bundle_df from disk in order to be called by __getitem__()
        additionally a dictonary can be provided to be used in __getitem__() '''

        try:
            bundle_path = Path(self.args['bundles_path'])
            df_path = bundle_path.parent /  (bundle_path.stem + '.csv')
            self.bundle_df = pd.read_csv(df_path)
        except:
            raise ValueError(
                'bundles not found, to generate a new set call \'generate_bundles\' first'
            )

        self.total_count = len(self.bundle_df)

    ##############################################################################################
    ########################### preprocessing and data augmentation ##############################

    def scale(self, img, xlow, xhigh, ylow, yhigh):
        ''' rescale the input voxel value to a new range '''
        m = (yhigh - ylow) / (xhigh - xlow)
        b = ylow - m * xlow
        return m * img + b

    def image_preprocessing(self, image):
        ''' image preprocessing '''
        if 'rescale' in self.args.keys():
            in_min, in_max = self.args['rescale']['input_range']
            out_min, out_max = self.args['rescale']['output_range']
            img = self.scale(image, in_min, in_max, out_min, out_max)
            img = np.clip(img, out_min, out_max)

        return img 

    def on_epoch_end(self):
        ''' shuffle the batch_df after each epoch '''
        if self.args['training']:
            self.bundle_df = self.bundle_df.sample(frac=1).reset_index(drop=True)

    def augmentation(self, img):
        ''' takes a single image as input and returns the augmented image '''
        if self.args['augment']['flip']:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0)
            
            # flip on coronal plane
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)

        if self.args['augment']['rot']:
            # rotate k*90 degree
            rot_k = np.random.randint(0, 4)
            if rot_k:
                img = np.rot90(img, axes=(0, 1), k=rot_k)

        return img

    ##############################################################################################
    ########################### keras Sequence specific functions ################################

    def __len__(self):
        ''' cases involved '''
        return int(np.ceil(self.total_count / float(self.args['batch_size'])))

    def getitem_helper(self):
        ''' returns the final indices '''
        x_ii = np.random.randint(
            0, self.args['bundle_size'][0] - self.args['image_size'][0] + 1
        )
        x_ff = x_ii + self.args['image_size'][0]

        y_ii = np.random.randint(
            0, self.args['bundle_size'][1] - self.args['image_size'][1] + 1
        )
        y_ff = y_ii + self.args['image_size'][1]

        z_ii = np.random.randint(
            0, self.args['bundle_size'][2] - self.args['image_size'][2] + 1
        )
        z_ff = z_ii + self.args['image_size'][2]

        return x_ii, x_ff, y_ii, y_ff, z_ii, z_ff
        

    def __getitem__(self, idx):
        ''' idx \epsilon [0, self.__len__()[ '''

        start_idx = idx * self.args['batch_size']
        end_idx = (idx + 1) * self.args['batch_size']
        end_idx = min(end_idx, self.total_count)

        batch_size = end_idx - start_idx
        img_batch = np.empty((batch_size, 16, 112, 112, 3))

        batch_df = self.bundle_df.iloc[start_idx: end_idx].reset_index(drop=True)

        class_batch = []

        with h5py.File(self.args['bundles_path'], 'r') as ff:
            for ii, row in batch_df.iterrows():
                bid = row['identifier'] + '_' + str(row['bundle_n'])

                x_ii, x_ff, y_ii, y_ff, z_ii, z_ff = self.getitem_helper()
            
                img = ff['images'][bid][x_ii:x_ff, y_ii:y_ff, z_ii:z_ff]
                if self.args['training']:
                    img = self.augmentation(img)

                class_batch.append(self.out_dict[row['identifier']])
                for jj in range(0, 16):
                    img_batch[ii][jj] = img[:, :, jj*3:(jj+1)*3]  

        class_batch = np.array(class_batch)
        img_batch = img_batch.astype(np.float32)

        # set weights
        weights = np.array([self.args['weight_dict'][ii] for ii in class_batch])

        return img_batch.astype(np.float32), class_batch, weights
