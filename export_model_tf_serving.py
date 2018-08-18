#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


print("\Graph Loading...\n")

# Load Graph
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        node_names = [n.name for n in graph.as_graph_def().node]
        #for n in node_names:
        
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        
	
# Export Model
# ==================================================
	
        # Set export path
        builder = tf.saved_model.builder.SavedModelBuilder('./runs' + '/1')

        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        tensor_info_input = tf.saved_model.utils.build_tensor_info(input_x)
        tensor_info_drop = tf.saved_model.utils.build_tensor_info(dropout_keep_prob)
	#tensor_info_drop = tf.saved_model.utils.build_tensor_info(dropout_keep_prob)
        # output tensor info
        tensor_info_output = tf.saved_model.utils.build_tensor_info(predictions)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': tensor_info_input,'dropout_keep_prob' : tensor_info_drop},
                outputs={'output': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_cnn':
                    prediction_signature,
            })

        # export the model
        builder.save(as_text=True)
        print('Done exporting!')


