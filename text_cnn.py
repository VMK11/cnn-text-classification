import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    # Export model apropriate for tensorflow serving
    def export_for_serving(self, model_path):
        # SavedModelBuilder will perform model export for us
        builder = tf.saved_model.builder.SavedModelBuilder(model_path + '/1')

        # First, we need to describe the signature for our API.
        # It will consist of single prediction method with chopstck_length as 
        # an input and class probability as an output.
        # We build TensorInfo protos as a starting step. Those are needed to 
        # shape prediction method signature
        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.input_x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.predictions)

        prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tensor_info_x},
            outputs={'classes_prob': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        
        # Legacy calls to initialize tables
        # For more details on this see
        # https://stackoverflow.com/questions/45521499/legacy-init-op-in-tensorflow-serving
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        # Finally, let's export the model as servable
        sess = tf.get_default_session()
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                'prediction':
                prediction_signature,
            },
            legacy_init_op=legacy_init_op)
        builder.save()
        print('Done exporting')


    def export_protobuf(self, model_name, checkpoint_meta_dir):
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
        print('Done exporting protobuf')


        
        