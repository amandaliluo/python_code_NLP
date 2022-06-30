# Understanding Code in Python Notebooks
## About The Project

### Project Goal
In this project, we want to built a model to predict the correct ordering of the cells in a given notebook whose markdown cells have been shuffled. We hope the result can help us in understanding the relationships between code and markdown. We built a Pairwise and a Ranking model by leveraging Feed-Forward Neural Network with Pytorch.

### Data
The dataset is from a [Kaggle compitition](https://www.kaggle.com/competitions/AI4Code). The data consisted of 160,000 Jupyter Notebook, with 139,226 being the training notebooks. However, we only used 1,000 samples as our training data. 
All notebooks:
* Have been published publicly on Kaggle.
* Represent the most recently version of the notebook.
* Contain at least one code cell and markdown cell each.
* Have code written in the Python language.
* Have had empty cells removed.

 <p align="center"><img src="images/sample_data.png" width="1000" height="300"></p>

### Modeling

#### Pairwise Model Architecture

A classification problem: For each markdown cell, pair it with the code cell right after it and mark this pair as label equals to 1; random sampling 2 code cells in the same notebook for each markdown, and mark these as 0s; embedding the concatenation of markdown cell and code cell into 100 dimensions using BERT; put the output of embedding into a 2 layer neural network.

![paird_format](/Users/caoyanan/Desktop/USF/Deep Learning/untitled folder/python_code_NLP/images/paird_format.png)

#### Ranking Model Architecture
For the Ranking Model, we used a vanilla NN Feed-Forward Neural Network with one linear layer as our base. The model worked best with an embedding size of 10 and a learning rate of 0.001. We also tried adding a dropout layer in our model, however, it actually made the performance worst.

### Further Steps


## Getting Started

### Prerequisites

In this project, we need to use Pandas, Numpy, Spacy, and PyTorch with Python 3.5.0 or greater.


### Installation

* Pytorch
  ```sh
  pip3 install torch torchvision
  ```

## Usage and result


## Contributor

[Amanda Li Luo](https://www.linkedin.com/in/amanda-li-luo/)

[Yanan Cao](https://www.linkedin.com/in/yanancao21/) 
