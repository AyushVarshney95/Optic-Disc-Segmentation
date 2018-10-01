##################################################################################
# U-NET MODEL 



inputs = Input( (width,height,1))
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(inputs)
conv1 = Dropout(0.5)(conv1)
conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv1)
conv2 = Dropout(0.3)(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool1 = Dropout(0.3)(pool1)
up1 = UpSampling2D(size = (2,2))(pool1)
conv3 = Conv2D(64,3,activation= 'relu',padding = 'same', kernel_initializer = 'he_uniform')(up1)
conv3 = Dropout(0.3)(conv3)
conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv3)
conv4 = Dropout(0.3)(conv4)
conv5 = Conv2D(1, 1, activation = 'sigmoid')(conv4)
model = Model(input = inputs, output = conv5)
model.summary()


############################################################################################################
