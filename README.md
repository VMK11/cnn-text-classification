**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

## Train:

```bash
./train.py 
```
## Export model for Tensorflow Serving
```bash
./export_model_tf_serving.py  --checkpoint_dir='./runs/1532862764/checkpoints/'
```

Replace the ckeckpoints ID number with the current ID. The protobuf and varaibles folder are going to be exported in the './runs' directory.

## Freeze model for local inference
```bash
./freeze_graph.py --model_name='cnn_freezed.pb' --checkpoint_meta_dir_file='./runs/1532862764/checkpoints/model-100.meta' --checkpoint_meta_dir='./runs/1532862764/checkpoints/model-100'
```

Replace the model_name string with your desired name. Also, replace checkpoint dir with the output from the training. 

## Inference Code for predicting locally. 
```bash
./inference_client.py  --eval_train -- server=12.0.0.1:8500 --checkpoint_dir="./runs/1533200979/checkpoints/"
```

In this case, I run the TF serving using docker locally binding it to the port 8500. Replace the IP and port if you've uploaded the model on the cloud. Replace checkpoint dir with the output from the training. The dir of the checkpoints from training is now imported as a parameter in order to load the model and then froze it in a form appropriate for TF serving.

## Inference Code Kubernetes
```bash
./inference_client.py --server=IP:PORT --checkpoint_dir='./runs/1533205895/checkpoints/'
```

Replace IP and PORT with kubernetes IP and port. Also, replace checkpoint dir with the output from the training. The dir of the checkpoints from training is imported as a parameter because the vocab file is essential for the inference.



## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
