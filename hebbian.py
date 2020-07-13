# Marisetti, Mohith
# 1001-669-337
# 2019-10-06
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
	    cell.set_height(1/number_of_rows)
	    cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.normal(size= [self.number_of_classes,self.input_dimensions+1])
    
    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1])

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """
        X = np.insert(X,0,np.ones(X.shape[1]) , 0); 
        predictions = np.dot(self.weights,X) 
        if (self.transfer_function == "Hard_limit"):
            return (predictions >= 0).astype(int)
        elif (self.transfer_function == "Linear"):
            return predictions
        elif (self.transfer_function == "Sigmoid"):        
            predictions_shape = predictions.shape
            temp = predictions.flatten()
            temp  = temp.tolist()
            temp_list = []
            for eachtemp in temp:
                val = round(eachtemp,2)
                if val < 0:
                    val = 1 - 1/(1 + math.exp(val))
                else:                    
                    val = 1/(1 + math.exp(-val))
                temp_list.append(val)
            predictions = np.array(temp_list).reshape(predictions_shape)            
            return predictions
            
            
            
        

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print("****** Model weights ******")
        print(self.weights)
        
    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        X = np.insert(X,0,np.ones(X.shape[1]) , 0)                  # INSERTING 1's as first row
        for _ in range(num_epochs):                                 # Run till specified num_epochs
            for i in range(math.ceil((X.shape[1]/batch_size))):     # Iterate till sample is used up in a batch
                p = (X[:,(i)*batch_size:(i+1)*batch_size])
                p_transpose = p.transpose()
                a = np.dot(self.weights,p)

                if self.transfer_function=="Hard_limit":                # We apply the hard limit function of a
                    a = (a>=0).astype(float)  
                elif self.transfer_function == "Linear":                # We can leave the 'a' value as is
                    pass
                elif (self.transfer_function == "Sigmoid"):             # We apply the sigmoid function for each value of a
                    a_shape = a.shape
                    temp = a.flatten()
                    temp  = temp.tolist()
                    temp_list = []
                    for eachtemp in temp:
                        val = round(eachtemp,2)    
                        if val < 0:
                            val = 1 - 1/(1 + math.exp(val))
                        else:                    
                            val = 1/(1 + math.exp(-val))
                        temp_list.append(val)
                        a = np.array(temp_list).reshape(a_shape)
                    
                t = y[(i)*batch_size:batch_size*(i+1)]                  # Take batches from t
                
                # One Hot conversion Logic
                t_list = t.tolist()
                temp_list = []
                for item in t_list:                         
                    oneHot = np.zeros(self.number_of_classes)
                    oneHot[item] = 1
                    temp_list.append(oneHot)
                t = (np.array(temp_list)).transpose()
                t_a = t - a

                # To apply W = W alpha*(t-a)pT (Delta Rule)
                if(learning=="Delta"):
                    self.weights += alpha*np.dot(t_a,p_transpose)
                # To apply W = (1-gamma)W + aplha*t*pT
                elif (learning == "Filtered"):
                    self.weights = (1-gamma)*self.weights + alpha*np.dot(t,p_transpose)
                # To apply W = W + aplha*a*pT
                elif (learning == "Unsupervised_hebb"):
                    self.weights += alpha*np.dot(a,p_transpose)
                
        
        


    def calculate_percent_error(self,X, y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        
        prediction = self.predict(X)        # First we get our model's prediction values
        error = 0
    
        predictions_list = []
        size = prediction.shape[1]          # Because we need to iterate over columns
        for i in range(size):               # Converting the one hot to a number again
            predicted_val = prediction[:,i].tolist()
            predicted_val = np.argmax(predicted_val)
            predictions_list.append(predicted_val)
        
        y = y.tolist()
        for i in range(len(y)):
            if( y[i] != predictions_list[i]):   # Compare each class label
                error+=1        # For every wrong prediction error gets incremented by 1.
        percent_error = (error/size)        # Basically columns in Y or columns in prediction represent samples. So the total number of samples is columns shape of that matrix (Y)
        return percent_error
    
        
        
        
    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        prediction = self.predict(X)
        confusion_matrix = np.zeros([self.number_of_classes,self.number_of_classes])
        predictions_list = []
        size = prediction.shape[1]          # Because we need to iterate over columns
        for i in range(size):               # Converting the one hot to a number again
            predicted_val = prediction[:,i].tolist()
            predicted_val = np.argmax(predicted_val)
            predictions_list.append(predicted_val)
        
        y = y.tolist()
        for i in range(len(y)):
            confusion_matrix[y[i],predictions_list[i]]+=1       # Indexed as Rows, Columns - (Target Value, Model's prediction)
        return confusion_matrix


if __name__ == "__main__":
    number_of_classes = 10
    number_of_training_samples_to_use = 1000
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized = ((X_train.reshape(X_train.shape[0], -1)).T)[:, 0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=5)
    model.initialize_all_weights_to_zeros()
    percent_error = []
    for k in range(10):
        model.train(X_train_vectorized, y_train, batch_size=300, num_epochs=2, alpha=0.1, gamma=0.1, learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized, y_test))

    print("\n\n\n\n\n PERCENT ERROR is")
    print(percent_error)

    