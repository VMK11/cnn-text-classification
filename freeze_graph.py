import os, argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util


tf.flags.DEFINE_string("checkpoint_meta_dir",'',"checkpoint directory")
tf.flags.DEFINE_string("model_name",'',"Model name")




saver = tf.train.import_meta_graph(checkpoint_meta_dir, clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, checkpoint_meta_dir)

output_node_names="output/predictions"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")  
)

output_graph="./" + model_name
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()