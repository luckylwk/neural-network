# Python Neural Network.

Some notes on this Neural Network setup.

~~~python
# To initialize the network.
NN = NeuralNetwork( sizes=[ X_train.shape[0], 50, 30, 10 ] )
~~~

To start training the network we need the **stochastic_gradient_descent** function.

~~~python
# To perform stochastic gradient descent.
NN.stochastic_gradient_descent( 
	X=X_train, Y=Y_train, 
	X_CV=X_test, Y_CV=Y_test, 
	epochs=10, batch_size=10, 
	eta=0.6, lmbda=0.05, dropout=False 
)


# To save the trained weights/parameters to a file.
NN.save_to_file( 
	X=X_train, Y=Y_train, 
	PATH='', FILE='test_save.json' 
)
~~~
