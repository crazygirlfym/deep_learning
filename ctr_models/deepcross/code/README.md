# Deep & Cross

This is our implementation for the paper:

Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang. [Deep&Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)

We have additionally released our TensorFlow implementation of Factorization Machines under our proposed neural network framework. 

## Environments
* Tensorflow (version: 1.2.0)
* numpy
* sklearn
## Dataset
We use the same input format as the LibFM toolkit (http://www.libfm.org/). In this instruction, we use [MovieLens](grouplens.org/datasets/movielens/latest).
The MovieLens data has been used for personalized tag recommendation, which contains 668,953 tag applications of users on movies. We convert each tag application (user ID, movie ID and tag) to a feature vector using one-hot encoding and obtain 90,445 binary features. The following examples are based on this dataset and it will be referred as ***ml-tag*** wherever in the files' name or inside the code.
When the dataset is ready, the current directory should be like this:
* code
    - DCN.py
    - LoadData.py
* data
    - ml-tag
        - ml-tag.train.libfm
        - ml-tag.validation.libfm
        - ml-tag.test.libfm

## Quick Example with Optimal parameters
Use the following command to train the model with the optimal parameters:
```
# step into the code folder
cd code

python DCN.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 100  --lr 0.1
```
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

The current implementation supports regression classification, which optimizes RMSE. 

After the trainning processes finish, the trained models will be saved into the ***pretrain*** folder, which should be like this:
* pretrain
    - ml-tag_16
        - checkpoint
        - ml-tag_16.data-00000-of-00001
        - ml-tag_16.index
        - ml-tag_16.meta
    
### Evaluate
Now it's time to evaluate the pretrained models with the test datasets, which can be done by running ***DCN.py*** with ***--process evaluate*** as follows:
```


# evaluate the pretrained DeepFM model
python DCN.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 100  --lr 0.1 --process evaluate

```

