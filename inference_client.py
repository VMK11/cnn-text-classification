from grpc.beta import implementations
import numpy
import tensorflow as tf
from datetime import datetime
import data_helpers
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import learn
import numpy as np
import sys 
import os



tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
 # Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("batch_size", 150, "Batch Size (default: 64)")


# Model Parameters
tf.flags.DEFINE_string("frozen_model_path","./cnn_export_last_2/1/cnn_freezed.pb","Load frozen model")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



def do_inference(hostport, test):
  """Tests PredictionService with concurrent requests.
  Args:
  hostport: Host:port address of the Prediction Service.
  Returns:
  pred values, ground truth label
  """
  # create connection
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
 
  # initialize a request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'cnn'
  request.model_spec.signature_name = 'predict_cnn'
 
  # Get test data
  request.inputs['input'].CopyFrom(
  tf.contrib.util.make_tensor_proto(test.astype(dtype=np.int32)))
  request.inputs['dropout_keep_prob'].CopyFrom(
  tf.contrib.util.make_tensor_proto(1.0))

  #print(request)
  # predict
  result = stub.Predict(request, 1000.0).outputs['output'] # 1000 seconds
  return result

def load_data():
  """Loads the data from directory.
  Args:
    None
  """
  FLAGS = tf.flags.FLAGS
  FLAGS(sys.argv)
  print("\nParameters:")
  for attr, value in sorted(FLAGS.__flags.items()):
      print("{}={}".format(attr.upper(), value))
  print("")

  # Load data. 
  if FLAGS.eval_train:
      x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
      y_test = np.argmax(y_test, axis=1)
  else:
      x_raw = ["a masterpiece four years in the making", "everything is off."]
      y_test = [1, 0]

  # Map data into vocabulary
  vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
  vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
  x_test = np.array(list(vocab_processor.transform(x_raw)))

  return x_test, y_test


def load_frozen_graph():
  """Loads the Tensorflow frozen graph into memory.
  Args:
  x_test, y_test: Test data.
  Returns:
  Input and Output tensros and Session
  """
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.frozen_model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name = '')
    sess = tf.Session(graph=detection_graph)

  # Define input and output tensors

  # Input tensor is the embeddings
  text_tensor = detection_graph.get_tensor_by_name('input_x:0')
  
  # Dropout tesnor 
  dropout_tensor = detection_graph.get_tensor_by_name('dropout_keep_prob:0')

  # Predcited class rearding the sentiment of the review
  text_classes = detection_graph.get_tensor_by_name("output/predictions:0")

  return [text_tensor,dropout_tensor,text_classes,sess]


def predcit(tens,x_test,y_test):
  """Predicts the labels for a pre-defined batch of observations.
  Args:
  x_test, y_test: Test data.
  tens: Tensorflow Session
  Returns:
  Array with predictions
  """

  # Generate batches for one epoch
  batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

  # Collect the predictions here
  all_predictions = []

  for x_test_batch in batches:
      batch_predictions = tens[3].run(tens[2],feed_dict={tens[0] : x_test_batch, tens[1] : 1.0})
      all_predictions = np.concatenate([all_predictions, batch_predictions])
  
  return all_predictions


def performance(predictions,ground_truth):
  """Evaluates the performance of the model.
  Args:
  x_test, y_test: Test data.
  tens: Tensorflow Session
  Returns:
  Array with predictions
  """
  arr = tf.make_ndarray(predictions)
  print(arr)
  if ground_truth is not None:
    correct_predictions = float(sum(arr == ground_truth))
    print("Total number of test examples: {}".format(len(ground_truth)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(ground_truth))))
    

def main(_):

  # Load the data
  x_test, y_test = load_data()

  # Check Server IP 
  if not FLAGS.server:
    print('You have not provided Server ip. Frozen graph is loaded...')
    tensors = load_frozen_graph()
    y_pred = predcit(tensors,x_test, y_test)  
    performance(y_pred,y_test)  
  else:
    result = do_inference(FLAGS.server, x_test)
    performance(result,y_test)
    
  
 
if __name__ == '__main__':
  tf.app.run()
