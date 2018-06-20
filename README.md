# KerasCode
Some code using Lynda to work through Keras


# Loading Data
# Transform & Scale Data
# Model Training
# TensorBoard Syntax
# Testing the Model
# Saving/Loading Model
# PRODUCTIONIZING


```python
import pandas as pd
```


```python
filepath = '/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/03/sales_data_training.csv'
train_data = pd.read_csv(filepath, sep = ',')
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>critic_rating</th>
      <th>is_action</th>
      <th>is_exclusive_to_us</th>
      <th>is_portable</th>
      <th>is_role_playing</th>
      <th>is_sequel</th>
      <th>is_sports</th>
      <th>suitable_for_kids</th>
      <th>total_earnings</th>
      <th>unit_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>132717</td>
      <td>59.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>83407</td>
      <td>49.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>62423</td>
      <td>49.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>69889</td>
      <td>39.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>161382</td>
      <td>59.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
filepath = '/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/03/sales_data_test.csv'
test_data = pd.read_csv(filepath, sep = ',')
```


```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>critic_rating</th>
      <th>is_action</th>
      <th>is_exclusive_to_us</th>
      <th>is_portable</th>
      <th>is_role_playing</th>
      <th>is_sequel</th>
      <th>is_sports</th>
      <th>suitable_for_kids</th>
      <th>total_earnings</th>
      <th>unit_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>247537</td>
      <td>59.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>73960</td>
      <td>59.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>82671</td>
      <td>59.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>137456</td>
      <td>39.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>89639</td>
      <td>59.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
```


```python
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data) # Applys the same amount of scaling to the Test as was to the Train
```


```python
# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_train, columns=train_data.columns.values)
scaled_testing_df = pd.DataFrame(scaled_test, columns=test_data.columns.values)
```


```python
# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/03/sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/03/sales_data_testing_scaled.csv", index=False)
```

### Linear MODEL TRAINING


```python
import keras
from keras.models import Sequential
from keras.layers import *
```


```python
training_data = scaled_training_df
```


```python
# Separate Xs and Ys
X = training_data.drop('total_earnings', axis = 1).values
Y = training_data[['total_earnings']].values
n_cols = X.shape[1]
```


```python
RUN_NAME = 'run1 with 50 nodes' # For comparing different trainings in TensorBoard

# Create the model
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape = (n_cols, ), name = 'layer1'))
model.add(Dense(100, activation = 'relu', name = 'layer2'))
model.add(Dense(50, activation = 'relu', name = 'layer3'))
model.add(Dense(1, activation = 'linear', name = 'output_layer'))
model.compile(loss = 'mse', optimizer = 'adam')
# Naming is mostly for using TensorBoard for visualization

# Also create a tensorboard logger
logger = keras.callbacks.TensorBoard(
    log_dir='/Users/Mitch/Desktop/tensorboard/keras_linear_model/{}'.format(RUN_NAME),
    write_graph=True, # Makes logs much bigger tho
    histogram_freq=5 # Every 5 passes through the training data, the log will write out statistics
)

# Train the model
model.fit(X, Y,
         epochs = 50,
         shuffle = True,
         verbose = 2,
         callbacks = [logger]) # This is necessary for the TensorBoard
# To view the TensorBoard, find the log folder in dir, put it in tensorboard folder with name and then use Terminal to 'tensorboard --logdir=path directory'... Also cannot have spaces in the path!
```

    Epoch 1/50
     - 0s - loss: 0.0098



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-26-559182244e1e> in <module>()
         22          shuffle = True,
         23          verbose = 2,
    ---> 24          callbacks = [logger]) # This is necessary for the TensorBoard
         25 # To view the TensorBoard, find the log folder in dir, put it in tensorboard folder with name and then use Terminal to 'tensorboard --logdir=path directory'... Also cannot have spaces in the path!


    ~/anaconda3/lib/python3.5/site-packages/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
       1040                                         initial_epoch=initial_epoch,
       1041                                         steps_per_epoch=steps_per_epoch,
    -> 1042                                         validation_steps=validation_steps)
       1043 
       1044     def evaluate(self, x=None, y=None,


    ~/anaconda3/lib/python3.5/site-packages/keras/engine/training_arrays.py in fit_loop(model, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)
        217                         for l, o in zip(out_labels, val_outs):
        218                             epoch_logs['val_' + l] = o
    --> 219         callbacks.on_epoch_end(epoch, epoch_logs)
        220         if callback_model.stop_training:
        221             break


    ~/anaconda3/lib/python3.5/site-packages/keras/callbacks.py in on_epoch_end(self, epoch, logs)
         75         logs = logs or {}
         76         for callback in self.callbacks:
    ---> 77             callback.on_epoch_end(epoch, logs)
         78 
         79     def on_batch_begin(self, batch, logs=None):


    ~/anaconda3/lib/python3.5/site-packages/keras/callbacks.py in on_epoch_end(self, epoch, logs)
        863 
        864         if not self.validation_data and self.histogram_freq:
    --> 865             raise ValueError("If printing histograms, validation_data must be "
        866                              "provided, and cannot be a generator.")
        867         if self.embeddings_data is None and self.embeddings_freq:


    ValueError: If printing histograms, validation_data must be provided, and cannot be a generator.



```python
RUN_NAME = 'run2 with 5 nodes' # For comparing different trainings in TensorBoard

# Create the model
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape = (n_cols, ), name = 'layer1'))
model.add(Dense(100, activation = 'relu', name = 'layer2'))
model.add(Dense(50, activation = 'relu', name = 'layer3'))
model.add(Dense(1, activation = 'linear', name = 'output_layer'))
model.compile(loss = 'mse', optimizer = 'adam')
# Naming is mostly for using TensorBoard for visualization

# Also create a tensorboard logger
logger = keras.callbacks.TensorBoard(
    log_dir='/Users/Mitch/Desktop/tensorboard/keras_linear_model/{}'.format(RUN_NAME),
    write_graph=True, # Makes logs much bigger tho
    histogram_freq=5 # Every 5 passes through the training data, the log will write out statistics
)

# Train the model
model.fit(X, Y,
         epochs = 50,
         shuffle = True,
         verbose = 2,
         callbacks = [logger]) # This is necessary for the TensorBoard
# To view the TensorBoard, find the log folder in dir, put it in tensorboard folder with name and then use Terminal to 'tensorboard --logdir=path directory'... Also cannot have spaces in the path!
```

    Epoch 1/50
     - 0s - loss: 0.0352



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-27-1f1aa3dadaff> in <module>()
         22          shuffle = True,
         23          verbose = 2,
    ---> 24          callbacks = [logger]) # This is necessary for the TensorBoard
         25 # To view the TensorBoard, find the log folder in dir, put it in tensorboard folder with name and then use Terminal to 'tensorboard --logdir=path directory'... Also cannot have spaces in the path!


    ~/anaconda3/lib/python3.5/site-packages/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
       1040                                         initial_epoch=initial_epoch,
       1041                                         steps_per_epoch=steps_per_epoch,
    -> 1042                                         validation_steps=validation_steps)
       1043 
       1044     def evaluate(self, x=None, y=None,


    ~/anaconda3/lib/python3.5/site-packages/keras/engine/training_arrays.py in fit_loop(model, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)
        217                         for l, o in zip(out_labels, val_outs):
        218                             epoch_logs['val_' + l] = o
    --> 219         callbacks.on_epoch_end(epoch, epoch_logs)
        220         if callback_model.stop_training:
        221             break


    ~/anaconda3/lib/python3.5/site-packages/keras/callbacks.py in on_epoch_end(self, epoch, logs)
         75         logs = logs or {}
         76         for callback in self.callbacks:
    ---> 77             callback.on_epoch_end(epoch, logs)
         78 
         79     def on_batch_begin(self, batch, logs=None):


    ~/anaconda3/lib/python3.5/site-packages/keras/callbacks.py in on_epoch_end(self, epoch, logs)
        863 
        864         if not self.validation_data and self.histogram_freq:
    --> 865             raise ValueError("If printing histograms, validation_data must be "
        866                              "provided, and cannot be a generator.")
        867         if self.embeddings_data is None and self.embeddings_freq:


    ValueError: If printing histograms, validation_data must be provided, and cannot be a generator.


### Test the model


```python
# Load the separate test data set
test_data_df = scaled_testing_df

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values
```


```python
test_error_rate = model.evaluate(X_test, Y_test, verbose = 0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
```

    The mean squared error (MSE) for the test data set is: 0.004283269103616476


### Let's use the model with production data!


```python
# Load the data we make to use to make a prediction
X = pd.read_csv("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/04/proposed_new_product.csv").values
```


```python
print(X)

X[0][0]
```

    [[0.7 1.  1.  1.  0.  1.  0.  1.  0.8]]





    0.7




```python
# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]
```

### Re-scale the data from the 0-to-1 range back to dollars


```python
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
```

    Earnings Prediction for Proposed Product - $265186.11336806923


### Saving and Loading Models


```python
# Saving
model.save("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/04/trained_model.h5") # Convention is to use .h5 for models
print("Model saved to drive!")
```

    Model saved to drive!



```python
from keras.models import load_model
my_model = load_model("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/Exercise Files/04/trained_model.h5")
# This reloads the structure and weights of our model!
```


```python
prediction = my_model.predict(X)
prediction = prediction[0][0]
```


```python
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
```

    Earnings Prediction for Proposed Product - $265186.11336806923



```python
# The prediction values from both models match!
```

# Lastly, let's push this model to production for APIs


```python
import tensorflow as tf

# Need TF specific code
model_builder = tf.saved_model.builder.SavedModelBuilder("/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/exported_model")

inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input)
}
outputs = {
    'earnings': tf.saved_model.utils.build_tensor_info(model.output)
}

# TF looks for this def to know how to run the model
signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

# Tells TF to save the structure and weights of the model
model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()
```

    INFO:tensorflow:No assets to save.
    INFO:tensorflow:No assets to write.
    INFO:tensorflow:SavedModel written to: b'/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/exported_model/saved_model.pb'





    b'/Users/Mitch/Documents/DATA SCIENCE/Self Learning Not School/Ex_Files_Building_Deep_Learning_Apps/exported_model/saved_model.pb'


