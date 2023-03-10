import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
#import tensorflow.keras.Sequential as Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from tensorflow.train import Checkpoint
train = utils.image_dataset_from_directory(
    'monkeys',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (512, 512),
    seed = 230,
    validation_split = 0.2,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'monkeys',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (512, 512),
    seed = 230,
    validation_split = 0.2,
    subset = 'validation',
)

tf.autograph.set_verbosity(
    level=0, alsologtostdout=False
)

class_names = train.class_names

print("Class Names:")
print(class_names)

class Net():
    def __init__(self, image_shape):
        self.model = models.Sequential()
        self.model.add(layers.ZeroPadding2D(
            padding = ((1,1), (2,2))
        ))

        #minor data augmentation
        self.model.add(layers.RandomFlip(
            "horizontal_and_vertical"
        ))
        self.model.add(layers.RandomRotation(
            0.2
        ))

        self.model.add(layers.Conv2D(
            8, #filters
            25, #kernelsize
            strides = 7,
            activation = 'relu',
        ))
        #output: 71  x 71 x 8
        self.model.add(layers.ZeroPadding2D(
            padding = ((1,1),(0,0))
        ))
        #output: 72 x 72 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,
        ))
        #output: 36 x 36 x 8
        self.model.add(layers.ZeroPadding2D(
            padding =  ((1,1), (0,0))))

        #output: 37 x 37 x 8

        self.model.add(layers.Conv2D(
            8, #filters
            3, #kernelsize
            strides = 2,
            activation = 'relu',
        ))
        #output: 18 x 18 x 8
        #self.model.add(layers.ZeroPadding2D(
        #    padding = ((1,1), (0,0)) ))

        #output: 18 x 18 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,
        ))

        #9 x 9 x 8
        self.model.add(layers.Flatten(
        )) # Output: 648
        self.model.add(layers.Dense(
            512, 
            activation = 'relu'))
        self.model.add(layers.Dropout(.4))
        self.model.add(layers.Dense(
            128,
            activation = "relu",
        ))
        self.model.add(layers.Dense(
            32,
            activation = "relu",
        ))
        
        self.model.add(layers.Dense(
            10, # number of classes
            activation = "softmax", #Always softmax on last layer
        ))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam()
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ["accuracy"])




def __str__(self):
    self.model.summary()
    return ""

net = Net((512, 512, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 20,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)
#for person in class_names:
    # Get the first image of that person and set it up
    #img = cv2.imread(f'archive/{person}/img_0.jpeg')
    #img = cv2.resize(img, (250, 250))
    #img = utils.img_to_array(img)
    #img = img[tf.newaxis, ...]

    # Did checkpoints every 2 epochs up to 40.
    #for k in range(2, 42, 2):
        # Set up the architecture and load in the checkpoint weights
        #net = Net((250, 250, 3))
        # print(net)
        #checkpoint = Checkpoint(net.model)
        #checkpoint.restore(f'checkpoints/checkpoints_{k:02d}').expect_partial()
        # Get the first conv layer, feed the image and set it up for viewing
        #filters = net.model.layers[0](img)[0]
        #shape = filters.shape
        #filters = filters.numpy()
        # Put all filters in one big mosaic image with 2 rows, padded by 
        #   20px gray strips.  
        # Scaling up the filters by 3x to make them easier to see
        #cols = shape[2] // 2
        #mosaic = np.zeros(
         #   (6*shape[0] + 20, 3*cols*shape[1] + (cols - 1)*20)
        #)  
        # Print the filter max and average to screen so we can see how much
        #   the classification uses this filter.
        #print(f'{person:>12} Chkpt {k:02d} Maxes:', end = ' ')
        #second_str = '                      Avgs: '
        # Shape[2] = number of filters
        #for i in range(shape[2]):
            # Get just one filter
            #filter = filters[0:shape[0],0:shape[1],i]
            # Calculate and print max and avg
            #maxes = []
            #avgs = []
            #for j in range(shape[0]):
              #  maxes.append(max(filter[j]))
             #   avgs.append(sum(filter[j])/len(filter[j]))
            #print(f'{max(maxes):8.4f}', end = ' ')
            #second_str += f'{sum(avgs)/len(avgs):9.4f}'
            # Triple the filter size to make it easier to see
            #filter = cv2.resize(filter, (3*shape[0], 3*shape[1]))
            # Rescale so the grayscale is more useful
            #filter = filter / max(maxes) * 2
            # Locate the filter in the mosaic and copy the values in
           # cv2.imshow(f'{person} Checkpoint {k}', filter)
          #  if chr(cv2.waitKey(0)) == 'q':
         #       quit()
        #cv2.destroyAllWindows()
        #    
        #offset = ((i % 2)*(3*shape[0] + 20), (i // 2)*(3*shape[1] + 20))
        #    mosaic[
        #        offset[0]:offset[0] + 3*shape[0], 
        #        offset[1]:offset[1] + 3*shape[1]] = filter  
        #print()
        #print(f'{second_str}')
        # Make the gray stripes that separate the filters
        # Vertical Stripes
        #for i in range(1, cols):
        #    start_vert_stripe = 3*i*shape[1] + (i - 1)*20
        #    mosaic[
        #        0:mosaic.shape[0], 
        #        start_vert_stripe:start_vert_stripe + 20] = np.ones(
        #            (mosaic.shape[0], 20)) * 0.5  
        # Horizontal Stripe
        #mosaic[3*shape[0]:3*shape[0] + 20, 0:mosaic.shape[1]] = np.ones(
        #        (20, mosaic.shape[1])) * 0.5
        # Display the image
