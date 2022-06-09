# DequantizerNet
First of all, download all the necessary libraries with:
```
pip install -r requirements.txt
```
## Dataset creation
To download the dataset, please follow the instructions of [Flickr](https://www.flickr.com/services/api/) to create an account and obtain authentication credentials. Then fill config.cfg (you can set the lastId to 0) and run 
```
python src/create_dataset.py
```

## Model training
You will need a wandb account. Fill up environment.sh with your directories and files and run

```
source environment.sh
```

Then configure your model in params.yaml. Set resume property to False if you are not using a checkpointed model. Then train your new model running:
 ```
 python src/train.py
 ```
