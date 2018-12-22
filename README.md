<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/README_1200x800.gif">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/coversmall_alt.png">
  <br/>
</p>

# data-science-ipython-notebooks

## Index

* [deep-learning](#deep-learning)
    * [tensorflow](#tensor-flow-tutorials)
    * [theano](#theano-tutorials)
    * [keras](#keras-tutorials)
    * [caffe](#deep-learning-misc)
* [scikit-learn](#scikit-learn)
* [statistical-inference-scipy](#statistical-inference-scipy)
* [pandas](#pandas)
* [matplotlib](#matplotlib)
* [numpy](#numpy)
* [python-data](#python-data)
* [kaggle-and-business-analyses](#kaggle-and-business-analyses)
* [spark](#spark)
* [mapreduce-python](#mapreduce-python)
* [amazon web services](#aws)
* [command lines](#commands)
* [misc](#misc)
* [notebook-installation](#notebook-installation)
* [credits](#credits)
* [contributing](#contributing)
* [contact-info](#contact-info)
* [license](#license)

<br/>
<p align="center">
  <img src="http://i.imgur.com/ZhKXrKZ.png">
</p>

## deep-learning

IPython Notebook(s) demonstrating deep learning functionality.

<br/>
<p align="center">
  <img src="https://avatars0.githubusercontent.com/u/15658638?v=3&s=100">
</p>

### tensor-flow-tutorials

Additional TensorFlow tutorials:

* [pkmital/tensorflow_tutorials](https://github.com/pkmital/tensorflow_tutorials)
* [nlintz/TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials)
* [alrojo/tensorflow-tutorial](https://github.com/alrojo/tensorflow-tutorial)
* [BinRoot/TensorFlow-Book](https://github.com/BinRoot/TensorFlow-Book)
* [tuanavu/tensorflow-basic-tutorials](https://github.com/tuanavu/tensorflow-basic-tutorials)

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [tsf-basics](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/1_intro/basic_operations.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F1_intro%2Fbasic_operations.ipynb) | Learn basic operations in TensorFlow, a library for various kinds of perceptual and language understanding tasks from Google. |
| [tsf-linear](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/2_basic_classifiers/linear_regression.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F2_basic_classifiers%2Flinear_regression.ipynb) | Implement linear regression in TensorFlow. |
| [tsf-logistic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/2_basic_classifiers/logistic_regression.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F2_basic_classifiers%2Flogistic_regression.ipynb) | Implement logistic regression in TensorFlow. |
| [tsf-nn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/2_basic_classifiers/nearest_neighbor.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F2_basic_classifiers%2Fnearest_neighbor.ipynb) | Implement nearest neighboars in TensorFlow. |
| [tsf-alex](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/3_neural_networks/alexnet.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F3_neural_networks%2Falexnet.ipynb) | Implement AlexNet in TensorFlow. |
| [tsf-cnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/3_neural_networks/convolutional_network.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F3_neural_networks%2Fconvolutional_network.ipynb) | Implement convolutional neural networks in TensorFlow. |
| [tsf-mlp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/3_neural_networks/multilayer_perceptron.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F3_neural_networks%2Fmultilayer_perceptron.ipynb) | Implement multilayer perceptrons in TensorFlow. |
| [tsf-rnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/3_neural_networks/recurrent_network.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F3_neural_networks%2Frecurrent_network.ipynb) | Implement recurrent neural networks in TensorFlow. |
| [tsf-gpu](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/4_multi_gpu/multigpu_basics.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F4_multi_gpu%2Fmultigpu_basics.ipynb) | Learn about basic multi-GPU computation in TensorFlow. |
| [tsf-gviz](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/5_ui/graph_visualization.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F5_ui%2Fgraph_visualization.ipynb) | Learn about graph visualization in TensorFlow. |
| [tsf-lviz](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/5_ui/loss_visualization.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-examples%2Fnotebooks%2F5_ui%2Floss_visualization.ipynb) | Learn about loss visualization in TensorFlow. |

### tensor-flow-exercises

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [tsf-not-mnist](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/1_notmnist.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F1_notmnist.ipynb) | Learn simple data curation by creating a pickle with formatted datasets for training, development and testing in TensorFlow. |
| [tsf-fully-connected](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/2_fullyconnected.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F2_fullyconnected.ipynb) | Progressively train deeper and more accurate models using logistic regression and neural networks in TensorFlow. |
| [tsf-regularization](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/3_regularization.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F3_regularization.ipynb) | Explore regularization techniques by training fully connected networks to classify notMNIST characters in TensorFlow. |
| [tsf-convolutions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/4_convolutions.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F4_convolutions.ipynb) | Create convolutional neural networks in TensorFlow. |
| [tsf-word2vec](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/5_word2vec.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F5_word2vec.ipynb) | Train a skip-gram model over Text8 data in TensorFlow. |
| [tsf-lstm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/6_lstm.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftensor-flow-exercises%2F6_lstm.ipynb) | Train a LSTM character model over Text8 data in TensorFlow. |

<br/>
<p align="center">
  <img src="http://www.deeplearning.net/software/theano/_static/theano_logo_allblue_200x46.png">
</p>

### theano-tutorials

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [theano-intro](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/intro_theano/intro_theano.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftheano-tutorial%2Fintro_theano%2Fintro_theano.ipynb) | Intro to Theano, which allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It can use GPUs and perform efficient symbolic differentiation. |
| [theano-scan](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/scan_tutorial/scan_tutorial.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftheano-tutorial%2Fscan_tutorial%2Fscan_tutorial.ipynb) | Learn scans, a mechanism to perform loops in a Theano graph. |
| [theano-logistic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/intro_theano/logistic_regression.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftheano-tutorial%2Fintro_theano%2Flogistic_regression.ipynb) | Implement logistic regression in Theano. |
| [theano-rnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/rnn_tutorial/simple_rnn.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftheano-tutorial%2Frnn_tutorial%2Fsimple_rnn.ipynb) | Implement recurrent neural networks in Theano. |
| [theano-mlp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/theano_mlp/theano_mlp.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Ftheano-tutorial%2Ftheano_mlp%2Ftheano_mlp.ipynb) | Implement multilayer perceptrons in Theano. |

<br/>
<p align="center">
  <img src="http://i.imgur.com/L45Q8c2.jpg">
</p>

### keras-tutorials

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| keras | | Keras is an open source neural network library written in Python. It is capable of running on top of either Tensorflow or Theano. 
| [setup](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/0.%20Preamble.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F0.%2520Preamble.ipynb) | Learn about the tutorial goals and how to set up your Keras environment. |
| [intro-deep-learning-ann](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/1.1%20Introduction%20-%20Deep%20Learning%20and%20ANN.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F1.1%2520Introduction%2520-%2520Deep%2520Learning%2520and%2520ANN.ipynb) | Get an intro to deep learning with Keras and Artificial Neural Networks (ANN). |
| [theano](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/1.2%20Introduction%20-%20Theano.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F1.2%2520Introduction%2520-%2520Theano.ipynb) | Learn about Theano by working with weights matrices and gradients. |
| [keras-otto](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/1.3%20Introduction%20-%20Keras.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F1.3%2520Introduction%2520-%2520Keras.ipynb) | Learn about Keras by looking at the Kaggle Otto challenge. |
| [ann-mnist](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/1.4%20%28Extra%29%20A%20Simple%20Implementation%20of%20ANN%20for%20MNIST.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F1.4%2520%2528Extra%2529%2520A%2520Simple%2520Implementation%2520of%2520ANN%2520for%2520MNIST.ipynb) | Review a simple implementation of ANN for MNIST using Keras. |
| [conv-nets](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/2.1%20Supervised%20Learning%20-%20ConvNets.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F2.1%2520Supervised%2520Learning%2520-%2520ConvNets.ipynb) | Learn about Convolutional Neural Networks (CNNs) with Keras. |
| [conv-net-1](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/2.2.1%20Supervised%20Learning%20-%20ConvNet%20HandsOn%20Part%20I.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F2.2.1%2520Supervised%2520Learning%2520-%2520ConvNet%2520HandsOn%2520Part%2520I.ipynb) | Recognize handwritten digits from MNIST using Keras - Part 1. |
| [conv-net-2](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/2.2.2%20Supervised%20Learning%20-%20ConvNet%20HandsOn%20Part%20II.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F2.2.2%2520Supervised%2520Learning%2520-%2520ConvNet%2520HandsOn%2520Part%2520II.ipynb) | Recognize handwritten digits from MNIST using Keras - Part 2. |
| [keras-models](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/2.3%20Supervised%20Learning%20-%20Famous%20Models%20with%20Keras.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F2.3%2520Supervised%2520Learning%2520-%2520Famous%2520Models%2520with%2520Keras.ipynb) | Use pre-trained models such as VGG16, VGG19, ResNet50, and Inception v3 with Keras. |
| [auto-encoders](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/3.1%20Unsupervised%20Learning%20-%20AutoEncoders%20and%20Embeddings.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F3.1%2520Unsupervised%2520Learning%2520-%2520AutoEncoders%2520and%2520Embeddings.ipynb) | Learn about Autoencoders with Keras. |
| [rnn-lstm](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/3.2%20RNN%20and%20LSTM.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F3.2%2520RNN%2520and%2520LSTM.ipynb) | Learn about Recurrent Neural Networks (RNNs) with Keras. |
| [lstm-sentence-gen](https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/keras-tutorial/3.3%20%28Extra%29%20LSTM%20for%20Sentence%20Generation.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fkeras-tutorial%2F3.3%2520%2528Extra%2529%2520LSTM%2520for%2520Sentence%2520Generation.ipynb) |  Learn about RNNs using Long Short Term Memory (LSTM) networks with Keras. |

### deep-learning-misc

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [deep-dream](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/deep-dream/dream.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fdeep-learning%2Fdeep-dream%2Fdream.ipynb) | Caffe-based computer vision program which uses a convolutional neural network to find and enhance patterns in images. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scikitlearn.png">
</p>

## scikit-learn

IPython Notebook(s) demonstrating scikit-learn functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [intro](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-intro.ipynb) | Intro notebook to scikit-learn.  Scikit-learn adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |
| [knn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb#K-Nearest-Neighbors-Classifier) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-intro.ipynb) | Implement k-nearest neighbors in scikit-learn. |
| [linear-reg](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-linear-reg.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-linear-reg.ipynb) | Implement linear regression in scikit-learn. |
| [svm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-svm.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-svm.ipynb) | Implement support vector machine classifiers with and without kernels in scikit-learn. |
| [random-forest](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-random-forest.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-random-forest.ipynb) | Implement random forest classifiers and regressors in scikit-learn. |
| [k-means](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-k-means.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-k-means.ipynb) | Implement k-means clustering in scikit-learn. |
| [pca](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-pca.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-pca.ipynb) | Implement principal component analysis in scikit-learn. |
| [gmm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-gmm.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-gmm.ipynb) | Implement Gaussian mixture models in scikit-learn. |
| [validation](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-validation.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscikit-learn%2Fscikit-learn-validation.ipynb) | Implement validation and model selection in scikit-learn. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scipy.png">
</p>

## statistical-inference-scipy

IPython Notebook(s) demonstrating statistical inference with SciPy functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scipy | | SciPy is a collection of mathematical algorithms and convenience functions built on the Numpy extension of Python. It adds significant power to the interactive Python session by providing the user with high-level commands and classes for manipulating and visualizing data.
| [effect-size](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/effect_size.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscipy%2Feffect_size.ipynb) | Explore statistics that quantify effect size by analyzing the difference in height between men and women.  Uses data from the Behavioral Risk Factor Surveillance System (BRFSS) to estimate the mean and standard deviation of height for adult women and men in the United States. |
| [sampling](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/sampling.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscipy%2Fsampling.ipynb) | Explore random sampling by analyzing the average weight of men and women in the United States using BRFSS data. |
| [hypothesis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/hypothesis.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fscipy%2Fhypothesis.ipynb) | Explore hypothesis testing by analyzing the difference of first-born babies compared with others. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/pandas.png">
</p>

## pandas

IPython Notebook(s) demonstrating pandas functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [pandas](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/pandas.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2Fpandas.ipynb) | Software library written for data manipulation and analysis in Python. Offers data structures and operations for manipulating numerical tables and time series. |
| [github-data-wrangling](https://github.com/donnemartin/viz/blob/master/githubstats/data_wrangling.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fviz%2Fblob%2Fmaster%2Fgithubstats%2Fdata_wrangling.ipynb) | Learn how to load, clean, merge, and feature engineer by analyzing GitHub data from the [`Viz`](https://github.com/donnemartin/viz) repo. 
| [Introduction-to-Pandas](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.00-Introduction-to-Pandas.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.00-Introduction-to-Pandas.ipynb) | Introduction to Pandas. |
| [Introducing-Pandas-Objects](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.01-Introducing-Pandas-Objects.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.01-Introducing-Pandas-Objects.ipynb) | Learn about Pandas objects. |
| [Data Indexing and Selection](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.02-Data-Indexing-and-Selection.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.02-Data-Indexing-and-Selection.ipynb) | Learn about data indexing and selection in Pandas. |
| [Operations-in-Pandas](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.03-Operations-in-Pandas.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.03-Operations-in-Pandas.ipynb) | Learn about operating on data in Pandas. |
| [Missing-Values](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.04-Missing-Values.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.04-Missing-Values.ipynb) | Learn about handling missing data in Pandas. |
| [Hierarchical-Indexing](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.05-Hierarchical-Indexing.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.05-Hierarchical-Indexing.ipynb) | Learn about hierarchical indexing in Pandas. |
| [Concat-And-Append](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.06-Concat-And-Append.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.06-Concat-And-Append.ipynb) | Learn about combining datasets: concat and append in Pandas. |
| [Merge-and-Join](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.07-Merge-and-Join.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.07-Merge-and-Join.ipynb) | Learn about combining datasets: merge and join in Pandas. |
| [Aggregation-and-Grouping](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.08-Aggregation-and-Grouping.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.08-Aggregation-and-Grouping.ipynb) | Learn about aggregation and grouping in Pandas. |
| [Pivot-Tables](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.09-Pivot-Tables.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.09-Pivot-Tables.ipynb) | Learn about pivot tables in Pandas. |
| [Working-With-Strings](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.10-Working-With-Strings.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.10-Working-With-Strings.ipynb) | Learn about vectorized string operations in Pandas. |
| [Working-with-Time-Series](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.11-Working-with-Time-Series.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.11-Working-with-Time-Series.ipynb) | Learn about working with time series in pandas. |
| [Performance-Eval-and-Query](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/03.12-Performance-Eval-and-Query.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpandas%2F03.12-Performance-Eval-and-Query.ipynb) | Learn about high-performance Pandas: eval() and query() in Pandas. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/matplotlib.png">
</p>

## matplotlib

IPython Notebook(s) demonstrating matplotlib functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [matplotlib](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/matplotlib.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2Fmatplotlib.ipynb) | Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. |
| [matplotlib-applied](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/matplotlib-applied.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2Fmatplotlib-applied.ipynb) | Apply matplotlib visualizations to Kaggle competitions for exploratory data analysis.  Learn how to create bar plots, histograms, subplot2grid, normalized plots, scatter plots, subplots, and kernel density estimation plots. |
| [Introduction-To-Matplotlib](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.00-Introduction-To-Matplotlib.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.00-Introduction-To-Matplotlib.ipynb) | Introduction to Matplotlib. |
| [Simple-Line-Plots](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.01-Simple-Line-Plots.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.01-Simple-Line-Plots.ipynb) | Learn about simple line plots in Matplotlib. |
| [Simple-Scatter-Plots](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.02-Simple-Scatter-Plots.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.02-Simple-Scatter-Plots.ipynb) | Learn about simple scatter plots in Matplotlib. |
| [Errorbars.ipynb](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.03-Errorbars.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.03-Errorbars.ipynb) | Learn about visualizing errors in Matplotlib. |
| [Density-and-Contour-Plots](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.04-Density-and-Contour-Plots.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.04-Density-and-Contour-Plots.ipynb) | Learn about density and contour plots in Matplotlib. |
| [Histograms-and-Binnings](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.05-Histograms-and-Binnings.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.05-Histograms-and-Binnings.ipynb) | Learn about histograms, binnings, and density in Matplotlib. |
| [Customizing-Legends](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.06-Customizing-Legends.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.06-Customizing-Legends.ipynb) | Learn about customizing plot legends in Matplotlib. |
| [Customizing-Colorbars](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.07-Customizing-Colorbars.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.07-Customizing-Colorbars.ipynb) | Learn about customizing colorbars in Matplotlib. |
| [Multiple-Subplots](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.08-Multiple-Subplots.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.08-Multiple-Subplots.ipynb) | Learn about multiple subplots in Matplotlib. |
| [Text-and-Annotation](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.09-Text-and-Annotation.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.09-Text-and-Annotation.ipynb) | Learn about text and annotation in Matplotlib. |
| [Customizing-Ticks](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.10-Customizing-Ticks.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.10-Customizing-Ticks.ipynb) | Learn about customizing ticks in Matplotlib. |
| [Settings-and-Stylesheets](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.11-Settings-and-Stylesheets.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.11-Settings-and-Stylesheets.ipynb) | Learn about customizing Matplotlib: configurations and stylesheets. |
| [Three-Dimensional-Plotting](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.12-Three-Dimensional-Plotting.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.12-Three-Dimensional-Plotting.ipynb) | Learn about three-dimensional plotting in Matplotlib. |
| [Geographic-Data-With-Basemap](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.13-Geographic-Data-With-Basemap.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.13-Geographic-Data-With-Basemap.ipynb) | Learn about geographic data with basemap in Matplotlib. |
| [Visualization-With-Seaborn](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/04.14-Visualization-With-Seaborn.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmatplotlib%2F04.14-Visualization-With-Seaborn.ipynb) | Learn about visualization with Seaborn. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/numpy.png">
</p>

## numpy

IPython Notebook(s) demonstrating NumPy functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [numpy](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/numpy.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2Fnumpy.ipynb) | Adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |
| [Introduction-to-NumPy](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.00-Introduction-to-NumPy.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.00-Introduction-to-NumPy.ipynb) | Introduction to NumPy. |
| [Understanding-Data-Types](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.01-Understanding-Data-Types.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.01-Understanding-Data-Types.ipynb) | Learn about data types in Python. |
| [The-Basics-Of-NumPy-Arrays](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.02-The-Basics-Of-NumPy-Arrays.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.02-The-Basics-Of-NumPy-Arrays.ipynb) | Learn about the basics of NumPy arrays. |
| [Computation-on-arrays-ufuncs](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.03-Computation-on-arrays-ufuncs.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.03-Computation-on-arrays-ufuncs.ipynb) | Learn about computations on NumPy arrays: universal functions. |
| [Computation-on-arrays-aggregates](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.04-Computation-on-arrays-aggregates.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.04-Computation-on-arrays-aggregates.ipynb) | Learn about aggregations: min, max, and everything in between in NumPy. |
| [Computation-on-arrays-broadcasting](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.05-Computation-on-arrays-broadcasting.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.05-Computation-on-arrays-broadcasting.ipynb) | Learn about computation on arrays: broadcasting in NumPy. |
| [Boolean-Arrays-and-Masks](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.06-Boolean-Arrays-and-Masks.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.06-Boolean-Arrays-and-Masks.ipynb) | Learn about comparisons, masks, and boolean logic in NumPy. |
| [Fancy-Indexing](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.07-Fancy-Indexing.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.07-Fancy-Indexing.ipynb) | Learn about fancy indexing in NumPy. |
| [Sorting](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.08-Sorting.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.08-Sorting.ipynb) | Learn about sorting arrays in NumPy. |
| [Structured-Data-NumPy](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/02.09-Structured-Data-NumPy.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fnumpy%2F02.09-Structured-Data-NumPy.ipynb) | Learn about structured data: NumPy's structured arrays. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/python.png">
</p>

## python-data

IPython Notebook(s) demonstrating Python functionality geared towards data analysis.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [data structures](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/structs.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Fstructs.ipynb) | Learn Python basics with tuples, lists, dicts, sets. |
| [data structure utilities](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/structs_utils.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Fstructs_utils.ipynb) | Learn Python operations such as slice, range, xrange, bisect, sort, sorted, reversed, enumerate, zip, list comprehensions. |
| [functions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/functions.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Ffunctions.ipynb) | Learn about more advanced Python features: Functions as objects, lambda functions, closures, *args, **kwargs currying, generators, generator expressions, itertools. |
| [datetime](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/datetime.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Fdatetime.ipynb) | Learn how to work with Python dates and times: datetime, strftime, strptime, timedelta. |
| [logging](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/logs.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Flogs.ipynb) | Learn about Python logging with RotatingFileHandler and TimedRotatingFileHandler. |
| [pdb](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/pdb.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Fpdb.ipynb) | Learn how to debug in Python with the interactive source code debugger. |
| [unit tests](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/unit_tests.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fpython-data%2Funit_tests.ipynb) | Learn how to test in Python with Nose unit tests. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/kaggle.png">
</p>

## kaggle-and-business-analyses

IPython Notebook(s) used in [kaggle](https://www.kaggle.com/) competitions and business analyses.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [titanic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/kaggle/titanic.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fkaggle%2Ftitanic.ipynb) | Predict survival on the Titanic.  Learn data cleaning, exploratory data analysis, and machine learning. |
| [churn-analysis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/analyses/churn.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fanalyses%2Fchurn.ipynb) | Predict customer churn.  Exercise logistic regression, gradient boosting classifers, support vector machines, random forests, and k-nearest-neighbors.  Includes discussions of confusion matrices, ROC plots, feature importances, prediction probabilities, and calibration/descrimination.|

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/spark.png">
</p>

## spark

IPython Notebook(s) demonstrating spark and HDFS functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [spark](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/spark/spark.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fspark%2Fspark.ipynb) | In-memory cluster computing framework, up to 100 times faster for certain applications and is well suited for machine learning algorithms. |
| [hdfs](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/spark/hdfs.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fspark%2Fhdfs.ipynb) | Reliably stores very large files across machines in a large cluster. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/mrjob.png">
</p>

## mapreduce-python

IPython Notebook(s) demonstrating Hadoop MapReduce with mrjob functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [mapreduce-python](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/mapreduce/mapreduce-python.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmapreduce%2Fmapreduce-python.ipynb) | Runs MapReduce jobs in Python, executing jobs locally or on Hadoop clusters. Demonstrates Hadoop Streaming in Python code with unit test and [mrjob](https://github.com/Yelp/mrjob) config file to analyze Amazon S3 bucket logs on Elastic MapReduce.  [Disco](https://github.com/discoproject/disco/) is another python-based alternative.|

<br/>

<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/aws.png">
</p>

## aws

IPython Notebook(s) demonstrating Amazon Web Services (AWS) and AWS tools functionality.


Also check out:

* [SAWS](https://github.com/donnemartin/saws): A Supercharged AWS command line interface (CLI).
* [Awesome AWS](https://github.com/donnemartin/awesome-aws): A curated list of libraries, open source repos, guides, blogs, and other resources.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [boto](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#Boto) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Official AWS SDK for Python. |
| [s3cmd](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3cmd) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Interacts with S3 through the command line. |
| [s3distcp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3distcp) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Combines smaller files and aggregates them together by taking in a pattern and target file.  S3DistCp can also be used to transfer large volumes of data from S3 to your Hadoop cluster. |
| [s3-parallel-put](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3-parallel-put) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Uploads multiple files to S3 in parallel. |
| [redshift](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#redshift) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb). | Acts as a fast data warehouse built on top of technology from massive parallel processing (MPP) |
| [kinesis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#kinesis) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Streams data in real time with the ability to process thousands of data streams per second. |
| [lambda](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#lambda) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Faws%2Faws.ipynb) | Runs code in response to events, automatically managing compute resources. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/commands.png">
</p>

## commands

IPython Notebook(s) demonstrating various command lines for Linux, Git, etc.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [linux](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/linux.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Flinux.ipynb) | Unix-like and mostly POSIX-compliant computer operating system.  Disk usage, splitting files, grep, sed, curl, viewing running processes, terminal syntax highlighting, and Vim.|
| [anaconda](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#anaconda) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. |
| [ipython notebook](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#ipython-notebook) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Web-based interactive computational environment where you can combine code execution, text, mathematics, plots and rich media into a single document. |
| [git](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#git) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Distributed revision control system with an emphasis on speed, data integrity, and support for distributed, non-linear workflows. |
| [ruby](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#ruby) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Used to interact with the AWS command line and for Jekyll, a blog framework that can be hosted on GitHub Pages. |
| [jekyll](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#jekyll) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Simple, blog-aware, static site generator for personal, project, or organization sites.  Renders Markdown or Textile and Liquid templates, and produces a complete, static website ready to be served by Apache HTTP Server, Nginx or another web server. |
| [pelican](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#pelican) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb) | Python-based alternative to Jekyll. |
| [django](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#django) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fcommands%2Fmisc.ipynb). | High-level Python Web framework that encourages rapid development and clean, pragmatic design. It can be useful to share reports/analyses and for blogging. Lighter-weight alternatives include [Pyramid](https://github.com/Pylons/pyramid), [Flask](https://github.com/pallets/flask), [Tornado](https://github.com/tornadoweb/tornado), and [Bottle](https://github.com/bottlepy/bottle)

## misc

IPython Notebook(s) demonstrating miscellaneous functionality.

| Notebook | Interactive Notebook | Description |
|-----------------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [regex](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/misc/regex.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmisc%2Fregex.ipynb) | Regular expression cheat sheet useful in data wrangling.|
[algorithmia](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/misc/Algorithmia.ipynb) | [<img height="24" src="https://beta.deepnote.org/buttons/launch-in-deepnote.svg">](https://beta.deepnote.org/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdonnemartin%2Fdata-science-ipython-notebooks%2Fblob%2Fmaster%2Fmisc%2FAlgorithmia.ipynb) | Algorithmia is a marketplace for algorithms. This notebook showcases 4 different algorithms: Face Detection, Content Summarizer, Latent Dirichlet Allocation and Optical Character Recognition.|

## notebook-installation

### anaconda

Anaconda is a free distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing that aims to simplify package management and deployment.

Follow instructions to install [Anaconda](https://docs.continuum.io/anaconda/install) or the more lightweight [miniconda](http://conda.pydata.org/miniconda.html).

### dev-setup

For detailed instructions, scripts, and tools to set up your development environment for data analysis, check out the [dev-setup](https://github.com/donnemartin/dev-setup) repo.

### running-notebooks

To view interactive content or to modify elements within the IPython notebooks, you must first clone or download the repository then run the notebook.  More information on IPython Notebooks can be found [here.](http://ipython.org/notebook.html)

    $ git clone https://github.com/donnemartin/data-science-ipython-notebooks.git
    $ cd data-science-ipython-notebooks
    $ jupyter notebook

Notebooks tested with Python 2.7.x.

## credits

* [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793) by Wes McKinney
* [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas
* [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) by Jake VanderPlas
* [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
* [Statistical Interference Using Computational Methods in Python](https://github.com/AllenDowney/CompStats) by Allen Downey
* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) by Aymeric Damien
* [TensorFlow Tutorials](https://github.com/pkmital/tensorflow_tutorials) by Parag K Mital
* [TensorFlow Tutorials](https://github.com/nlintz/TensorFlow-Tutorials) by Nathan Lintz
* [TensorFlow Tutorials](https://github.com/alrojo/tensorflow-tutorial) by Alexander R Johansen
* [TensorFlow Book](https://github.com/BinRoot/TensorFlow-Book) by Nishant Shukla
* [Summer School 2015](https://github.com/mila-udem/summerschool2015) by mila-udem
* [Keras tutorials](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio
* [Kaggle](https://www.kaggle.com/)
* [Yhat Blog](http://blog.yhat.com/)

## contributing

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/donnemartin/data-science-ipython-notebooks/issues).

## contact-info

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [donne.martin@gmail.com](mailto:donne.martin@gmail.com)
* Twitter: [@donne_martin](https://twitter.com/donne_martin)
* GitHub: [donnemartin](https://github.com/donnemartin)
* LinkedIn: [donnemartin](https://www.linkedin.com/in/donnemartin)
* Website: [donnemartin.com](http://donnemartin.com)

## license

This repository contains a variety of content; some developed by Donne Martin, and some from third-parties.  The third-party content is distributed under the license provided by those parties.

The content developed by Donne Martin is distributed under the following license:

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer (Facebook).*

    Copyright 2015 Donne Martin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
