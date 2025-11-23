#!/usr/bin/env python
# coding: utf-8

# **Chapter 19 – Training and Deploying TensorFlow Models at Scale**

# _This notebook contains all the sample code and solutions to the exercises in chapter 19._

# <table align="left">
#   <td>
#     <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/19_training_and_deploying_at_scale.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   </td>
#   <td>
#     <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/19_training_and_deploying_at_scale.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
#   </td>
# </table>

# # Setup

# This project requires Python 3.7 or above:

import sys

assert sys.version_info >= (3, 7)


# **Warning**: the latest TensorFlow versions are based on Keras 3. For chapters 10-15, it wasn't too hard to update the code to support Keras 3, but unfortunately it's much harder for this chapter, so I've had to revert to Keras 2. To do that, I set the `TF_USE_LEGACY_KERAS` environment variable to `"1"` and import the `tf_keras` package. This ensures that `tf.keras` points to `tf_keras`, which is Keras 2.*.

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras


# And TensorFlow ≥ 2.8:

from packaging import version
import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")


# If running on Colab or Kaggle, you need to install the Google AI Platform client library, which will be used later in this notebook. You can ignore the warnings about version incompatibilities.
# 
# * **Warning**: On Colab, you must restart the Runtime after the installation, and continue with the next cells.

import sys
if "google.colab" in sys.modules or "kaggle_secrets" in sys.modules:
    get_ipython().run_line_magic('pip', 'install -q -U google-cloud-aiplatform')


# This chapter discusses how to run or train a model on one or more GPUs, so let's make sure there's at least one, or else issue a warning:

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. Neural nets can be very slow without a GPU.")
    if "google.colab" in sys.modules:
        print("Go to Runtime > Change runtime and select a GPU hardware "
              "accelerator.")
    if "kaggle_secrets" in sys.modules:
        print("Go to Settings > Accelerator and select GPU.")


# # Serving a TensorFlow Model

# Let's start by deploying a model using TF Serving, then we'll deploy to Google Vertex AI.

# ## Using TensorFlow Serving

# The first thing we need to do is to build and train a model, and export it to the SavedModel format.

# ### Exporting SavedModels

# Let's load the MNIST dataset, scale it, and split it.

from pathlib import Path
import tensorflow as tf

# extra code – load and split the MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# extra code – build & train an MNIST model (also handles image preprocessing)
tf.random.set_seed(42)
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8),
    tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_name = "my_mnist_model"
model_version = "0001"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")


# Let's take a look at the file tree (we've discussed what each of these file is used for in chapter 10):

sorted([str(path) for path in model_path.parent.glob("**/*")])  # extra code


# Let's inspect the SavedModel:

get_ipython().system("saved_model_cli show --dir '{model_path}'")


get_ipython().system("saved_model_cli show --dir '{model_path}' --tag_set serve")


get_ipython().system("saved_model_cli show --dir '{model_path}' --tag_set serve                        --signature_def serving_default")


# For even more details, you can run the following command:
# 
# ```ipython
# !saved_model_cli show --dir '{model_path}' --all
# ```

# ### Installing and Starting TensorFlow Serving

# If you are running this notebook in Colab or Kaggle, TensorFlow Server needs to be installed:

if "google.colab" in sys.modules or "kaggle_secrets" in sys.modules:
    url = "https://storage.googleapis.com/tensorflow-serving-apt"
    src = "stable tensorflow-model-server tensorflow-model-server-universal"
    get_ipython().system("echo 'deb {url} {src}' > /etc/apt/sources.list.d/tensorflow-serving.list")
    get_ipython().system("curl '{url}/tensorflow-serving.release.pub.gpg' | apt-key add -")
    get_ipython().system('apt update -q && apt-get install -y tensorflow-model-server')
    get_ipython().run_line_magic('pip', 'install -q -U tensorflow-serving-api')


# If `tensorflow_model_server` is installed (e.g., if you are running this notebook in Colab), then the following 2 cells will start the server. If your OS is Windows, you may need to run the `tensorflow_model_server` command in a terminal, and replace `${MODEL_DIR}` with the full path to the `my_mnist_model` directory.

import os

os.environ["MODEL_DIR"] = str(model_path.parent.absolute())


get_ipython().run_cell_magic('bash', '--bg', 'tensorflow_model_server \\\n    --port=8500 \\\n    --rest_api_port=8501 \\\n    --model_name=my_mnist_model \\\n    --model_base_path="${MODEL_DIR}" >my_server.log 2>&1\n')


import time

time.sleep(2) # let's wait a couple seconds for the server to start


# If you are running this notebook on your own machine, and you prefer to install TF Serving using Docker, first make sure [Docker](https://docs.docker.com/install/) is installed, then run the following commands in a terminal. You must replace `/path/to/my_mnist_model` with the appropriate absolute path to the `my_mnist_model` directory, but do not modify the container path `/models/my_mnist_model`.
# 
# ```bash
# docker pull tensorflow/serving  # downloads the latest TF Serving image
# 
# docker run -it --rm -v "/path/to/my_mnist_model:/models/my_mnist_model" \
#     -p 8500:8500 -p 8501:8501 -e MODEL_NAME=my_mnist_model tensorflow/serving
# ```

# ### Querying TF Serving through the REST API

# Next, let's send a REST query to TF Serving:

import json

X_new = X_test[:3]  # pretend we have 3 new digit images to classify
request_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})


request_json[:100] + "..." + request_json[-10:]


# Now let's use TensorFlow Serving's REST API to make predictions:

import requests

server_url = "http://localhost:8501/v1/models/my_mnist_model:predict"
response = requests.post(server_url, data=request_json)
response.raise_for_status()  # raise an exception in case of error
response = response.json()


import numpy as np

y_proba = np.array(response["predictions"])
y_proba.round(2)


# ### Querying TF Serving through the gRPC API

from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]  # == "flatten_input"
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))


import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)


# Convert the response to a tensor:

output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)


y_proba.round(2)


# If your client does not include the TensorFlow library, you can convert the response to a NumPy array like this:

# extra code – shows how to avoid using tf.make_ndarray()
output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
shape = [dim.size for dim in outputs_proto.tensor_shape.dim]
y_proba = np.array(outputs_proto.float_val).reshape(shape)
y_proba.round(2)


# ### Deploying a new model version

# extra code – build and train a new MNIST model version
np.random.seed(42)
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8),
    tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


model_version = "0002"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")


# Let's take a look at the file tree again:

sorted([str(path) for path in model_path.parent.glob("**/*")])  # extra code


# **Warning**: You may need to wait a minute before the new model is loaded by TensorFlow Serving.

import requests

server_url = "http://localhost:8501/v1/models/my_mnist_model:predict"
            
response = requests.post(server_url, data=request_json)
response.raise_for_status()
response = response.json()


response.keys()


y_proba = np.array(response["predictions"])
y_proba.round(2)


# ## Creating a Prediction Service on Vertex AI

# Follow the instructions in the book to create a Google Cloud Platform account and activate the Vertex AI and Cloud Storage APIs. Then, if you're running this notebook in Colab, you can run the following cell to authenticate using the same Google account as you used with Google Cloud Platform, and authorize this Colab to access your data.
# 
# **WARNING: only do this if you trust this notebook!**
# * Be extra careful if this is not the official notebook from https://github.com/ageron/handson-ml3: the Colab URL should start with https://colab.research.google.com/github/ageron/handson-ml3. Or else, the code could do whatever it wants with your data.
# 
# If you are not running this notebook in Colab, you must follow the instructions in the book to create a service account and generate a key for it, download it to this notebook's directory, and name it `my_service_account_key.json` (or make sure the `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to your key).

project_id = "my_project"  ##### CHANGE THIS TO YOUR PROJECT ID #####

if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()
elif "kaggle_secrets" in sys.modules:
    from kaggle_secrets import UserSecretsClient
    UserSecretsClient().set_gcloud_credentials(project=project_id)
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_service_account_key.json"


from google.cloud import storage

bucket_name = "my_bucket"  ##### CHANGE THIS TO A UNIQUE BUCKET NAME #####
location = "us-central1"

storage_client = storage.Client(project=project_id)
bucket = storage_client.create_bucket(bucket_name, location=location)
#bucket = storage_client.bucket(bucket_name)  # to reuse a bucket instead


def upload_directory(bucket, dirpath):
    dirpath = Path(dirpath)
    for filepath in dirpath.glob("**/*"):
        if filepath.is_file():
            blob = bucket.blob(filepath.relative_to(dirpath.parent).as_posix())
            blob.upload_from_filename(filepath)

upload_directory(bucket, "my_mnist_model")


# extra code – a much faster multithreaded implementation of upload_directory()
#              which also accepts a prefix for the target path, and prints stuff

from concurrent import futures

def upload_file(bucket, filepath, blob_path):
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(filepath)

def upload_directory(bucket, dirpath, prefix=None, max_workers=50):
    dirpath = Path(dirpath)
    prefix = prefix or dirpath.name
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filepath = {
            executor.submit(
                upload_file,
                bucket, filepath,
                f"{prefix}/{filepath.relative_to(dirpath).as_posix()}"
            ): filepath
            for filepath in sorted(dirpath.glob("**/*"))
            if filepath.is_file()
        }
        for future in futures.as_completed(future_to_filepath):
            filepath = future_to_filepath[future]
            try:
                result = future.result()
            except Exception as ex:
                print(f"Error uploading {filepath!s:60}: {ex}")  # f!s is str(f)
            else:
                print(f"Uploaded {filepath!s:60}", end="\r")

    print(f"Uploaded {dirpath!s:60}")


# Alternatively, if you installed Google Cloud CLI (it's preinstalled on Colab), then you can use the following `gsutil` command:

#!gsutil -m cp -r my_mnist_model gs://{bucket_name}/


from google.cloud import aiplatform

server_image = "gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest"

aiplatform.init(project=project_id, location=location)
mnist_model = aiplatform.Model.upload(
    display_name="mnist",
    artifact_uri=f"gs://{bucket_name}/my_mnist_model/0001",
    serving_container_image_uri=server_image,
)


# **Warning**: this cell may take several minutes to run, as it waits for Vertex AI to provision the compute nodes:

endpoint = aiplatform.Endpoint.create(display_name="mnist-endpoint")

endpoint.deploy(
    mnist_model,
    min_replica_count=1,
    max_replica_count=5,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1
)


response = endpoint.predict(instances=X_new.tolist())


import numpy as np

np.round(response.predictions, 2)


endpoint.undeploy_all()  # undeploy all models from the endpoint
endpoint.delete()


# ## Running Batch Prediction Jobs on Vertex AI

batch_path = Path("my_mnist_batch")
batch_path.mkdir(exist_ok=True)
with open(batch_path / "my_mnist_batch.jsonl", "w") as jsonl_file:
    for image in X_test[:100].tolist():
        jsonl_file.write(json.dumps(image))
        jsonl_file.write("\n")

upload_directory(bucket, batch_path)


batch_prediction_job = mnist_model.batch_predict(
    job_display_name="my_batch_prediction_job",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=5,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1,
    gcs_source=[f"gs://{bucket_name}/{batch_path.name}/my_mnist_batch.jsonl"],
    gcs_destination_prefix=f"gs://{bucket_name}/my_mnist_predictions/",
    sync=True  # set to False if you don't want to wait for completion
)


batch_prediction_job.output_info  # extra code – shows the output directory


y_probas = []
for blob in batch_prediction_job.iter_outputs():
    print(blob.name)  # extra code
    if "prediction.results" in blob.name:
        for line in blob.download_as_text().splitlines():
            y_proba = json.loads(line)["prediction"]
            y_probas.append(y_proba)


y_pred = np.argmax(y_probas, axis=1)
accuracy = np.sum(y_pred == y_test[:100]) / 100


accuracy


mnist_model.delete()


# Let's delete all the directories we created on GCS (i.e., all the blobs with these prefixes):

for prefix in ["my_mnist_model/", "my_mnist_batch/", "my_mnist_predictions/"]:
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()

#bucket.delete()  # uncomment and run if you want to delete the bucket itself
batch_prediction_job.delete()


# # Deploying a Model to a Mobile or Embedded Device

converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
tflite_model = converter.convert()
with open("my_converted_savedmodel.tflite", "wb") as f:
    f.write(tflite_model)


# extra code – shows how to convert a Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)


converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()
with open("my_converted_keras_model.tflite", "wb") as f:
    f.write(tflite_model)


# # Running a Model in a Web Page

# Code examples for this section are hosted on glitch.com, a website that lets you create Web apps for free.
# 
# * https://homl.info/tfjscode: a simple TFJS Web app that loads a pretrained model and classifies an image.
# * https://homl.info/tfjswpa: the same Web app setup as a WPA. Try opening this link on various platforms, including mobile devices.
# ** https://homl.info/wpacode: this WPA's source code.
# * https://tensorflow.org/js: The TFJS library.
# ** https://www.tensorflow.org/js/demos: some fun demos.

# # Using GPUs to Speed Up Computations

# Let's check that TensorFlow can see the GPU:

physical_gpus = tf.config.list_physical_devices("GPU")
physical_gpus


# If you want your TensorFlow script to use only GPUs \#0 and \#1 (based on PCI order), then you can set the environment variables `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0,1` before starting your script, or in the script itself before using TensorFlow.

# ## Managing the GPU RAM

# To limit the amount of RAM to 2GB per GPU:

#for gpu in physical_gpus:
#    tf.config.set_logical_device_configuration(
#        gpu,
#        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
#    )


# To make TensorFlow grab memory as it needs it (only releasing it when the process shuts down):

#for gpu in physical_gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)


# Equivalently, you can set the `TF_FORCE_GPU_ALLOW_GROWTH` environment variable to `true` before using TensorFlow.

# To split a physical GPU into two logical GPUs:

#tf.config.set_logical_device_configuration(
#    physical_gpus[0],
#    [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
#     tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
#)


logical_gpus = tf.config.list_logical_devices("GPU")
logical_gpus


# ## Placing Operations and Variables on Devices

# To log every variable and operation placement (this must be run just after importing TensorFlow):

#tf.get_logger().setLevel("DEBUG")  # log level is INFO by default
#tf.debugging.set_log_device_placement(True)


a = tf.Variable([1., 2., 3.])  # float32 variable goes to the GPU
a.device


b = tf.Variable([1, 2, 3])  # int32 variable goes to the CPU
b.device


# You can place variables and operations manually on the desired device using a `tf.device()` context:

with tf.device("/cpu:0"):
    c = tf.Variable([1., 2., 3.])

c.device


# If you specify a device that does not exist, or for which there is no kernel, TensorFlow will silently fallback to the default placement:

# extra code

with tf.device("/gpu:1234"):
    d = tf.Variable([1., 2., 3.])

d.device


# If you want TensorFlow to throw an exception when you try to use a device that does not exist, instead of falling back to the default device:

tf.config.set_soft_device_placement(False)

# extra code
try:
    with tf.device("/gpu:1000"):
        d = tf.Variable([1., 2., 3.])
except tf.errors.InvalidArgumentError as ex:
    print(ex)

tf.config.set_soft_device_placement(True)  # extra code – back to soft placement


# ## Parallel Execution Across Multiple Devices

# If you want to set the number of inter-op or intra-op threads (this may be useful if you want to avoid saturating the CPU, or if you want to make TensorFlow single-threaded, to run a perfectly reproducible test case):

#tf.config.threading.set_inter_op_parallelism_threads(10)
#tf.config.threading.set_intra_op_parallelism_threads(10)


# # Training Models Across Multiple Devices

# ## Training at Scale Using the Distribution Strategies API

# extra code – creates a CNN model for MNIST using Keras
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28],
                                dtype=tf.uint8),
        tf.keras.layers.Rescaling(scale=1 / 255),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                               padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                               padding="same"), 
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                               padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation="softmax"),
    ])


tf.random.set_seed(42)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()  # create a Keras model normally
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
                  metrics=["accuracy"])  # compile the model normally

batch_size = 100  # preferably divisible by the number of replicas
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)


type(model.weights[0])


model.predict(X_new).round(2)  # extra code – the batch is split across all replicas


# extra code – shows that saving a model does not preserve its distribution
#              strategy
model.save("my_mirrored_model", save_format="tf")
model = tf.keras.models.load_model("my_mirrored_model")
type(model.weights[0])


with strategy.scope():
    model = tf.keras.models.load_model("my_mirrored_model")


type(model.weights[0])


# If you want to specify the list of GPUs to use:

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


# If you want to change the default all-reduce algorithm:

strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# If you want to use the `CentralStorageStrategy`:

strategy = tf.distribute.experimental.CentralStorageStrategy()


# To train on a TPU in Google Colab:
#if "google.colab" in sys.modules and "COLAB_TPU_ADDR" in os.environ:
#  tpu_address = "grpc://" + os.environ["COLAB_TPU_ADDR"]
#else:
#  tpu_address = ""
#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
#tf.config.experimental_connect_to_cluster(resolver)
#tf.tpu.experimental.initialize_tpu_system(resolver)
#strategy = tf.distribute.experimental.TPUStrategy(resolver)


# ## Training a Model on a TensorFlow Cluster

# A TensorFlow cluster is a group of TensorFlow processes running in parallel, usually on different machines, and talking to each other to complete some work, for example training or executing a neural network. Each TF process in the cluster is called a "task" (or a "TF server"). It has an IP address, a port, and a type (also called its role or its job). The type can be `"worker"`, `"chief"`, `"ps"` (parameter server) or `"evaluator"`:
# * Each **worker** performs computations, usually on a machine with one or more GPUs.
# * The **chief** performs computations as well, but it also handles extra work such as writing TensorBoard logs or saving checkpoints. There is a single chief in a cluster. If it is not defined, then it is worker #0.
# * A **parameter server** (ps) only keeps track of variable values, it is usually on a CPU-only machine.
# * The **evaluator** obviously takes care of evaluation. There is usually a single evaluator in a cluster.
# 
# The set of tasks that share the same type is often called a "job". For example, the "worker" job is the set of all workers.
# 
# To start a TensorFlow cluster, you must first define it. This means specifying all the tasks (IP address, TCP port, and type). For example, the following cluster specification defines a cluster with 3 tasks (2 workers and 1 parameter server). It's a dictionary with one key per job, and the values are lists of task addresses:

cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",     # /job:worker/task:0
        "machine-b.example.com:2222"      # /job:worker/task:1
    ],
    "ps": ["machine-a.example.com:2221"]  # /job:ps/task:0
}


# Every task in the cluster may communicate with every other task in the server, so make sure to configure your firewall to authorize all communications between these machines on these ports (it's usually simpler if you use the same port on every machine).
# 
# When a task is started, it needs to be told which one it is: its type and index (the task index is also called the task id). A common way to specify everything at once (both the cluster spec and the current task's type and id) is to set the `TF_CONFIG` environment variable before starting the program. It must be a JSON-encoded dictionary containing a cluster specification (under the `"cluster"` key), and the type and index of the task to start (under the `"task"` key). For example, the following `TF_CONFIG` environment variable defines the same cluster as above, with 2 workers and 1 parameter server, and specifies that the task to start is worker \#0:

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 0}
})


# Some platforms (e.g., Google Vertex AI) automatically set this environment variable for you.

# TensorFlow's `TFConfigClusterResolver` class reads the cluster configuration from this environment variable:

resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
resolver.cluster_spec()


resolver.task_type


resolver.task_id


# Now let's run a simpler cluster with just two worker tasks, both running on the local machine. We will use the `MultiWorkerMirroredStrategy` to train a model across these two tasks.
# 
# The first step is to write the training code. As this code will be used to run both workers, each in its own process, we write this code to a separate Python file, `my_mnist_multiworker_task.py`. The code is relatively straightforward, but there are a couple important things to note:
# * We create the `MultiWorkerMirroredStrategy` before doing anything else with TensorFlow.
# * Only one of the workers will take care of logging to TensorBoard. As mentioned earlier, this worker is called the *chief*. When it is not defined explicitly, then by convention it is worker #0.

get_ipython().run_cell_magic('writefile', 'my_mnist_multiworker_task.py', '\nimport tempfile\nimport tensorflow as tf\n\nstrategy = tf.distribute.MultiWorkerMirroredStrategy()  # at the start!\nresolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()\nprint(f"Starting task {resolver.task_type} #{resolver.task_id}")\n\n# extra code – Load and split the MNIST dataset\nmnist = tf.keras.datasets.mnist.load_data()\n(X_train_full, y_train_full), (X_test, y_test) = mnist\nX_valid, X_train = X_train_full[:5000], X_train_full[5000:]\ny_valid, y_train = y_train_full[:5000], y_train_full[5000:]\n\nwith strategy.scope():\n    model = tf.keras.Sequential([\n        tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28],\n                                dtype=tf.uint8),\n        tf.keras.layers.Rescaling(scale=1 / 255),\n        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",\n                               padding="same", input_shape=[28, 28, 1]),\n        tf.keras.layers.MaxPooling2D(pool_size=2),\n        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",\n                               padding="same"), \n        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",\n                               padding="same"),\n        tf.keras.layers.MaxPooling2D(pool_size=2),\n        tf.keras.layers.Flatten(),\n        tf.keras.layers.Dense(units=64, activation="relu"),\n        tf.keras.layers.Dropout(0.5),\n        tf.keras.layers.Dense(units=10, activation="softmax"),\n    ])\n    model.compile(loss="sparse_categorical_crossentropy",\n                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),\n                  metrics=["accuracy"])\n\nmodel.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)\n\nif resolver.task_id == 0:  # the chief saves the model to the right location\n    model.save("my_mnist_multiworker_model", save_format="tf")\nelse:\n    tmpdir = tempfile.mkdtemp()  # other workers save to a temporary directory\n    model.save(tmpdir, save_format="tf")\n    tf.io.gfile.rmtree(tmpdir)  # and we can delete this directory at the end!\n')


# In a real world application, there would typically be a single worker per machine, but in this example we're running both workers on the same machine, so they will both try to use all the available GPU RAM (if this machine has a GPU), and this will likely lead to an Out-Of-Memory (OOM) error. To avoid this, we could use the `CUDA_VISIBLE_DEVICES` environment variable to assign a different GPU to each worker. Alternatively, we can simply disable GPU support, by setting `CUDA_VISIBLE_DEVICES` to an empty string.

# We are now ready to start both workers, each in its own process. Notice that we change the task index:

get_ipython().run_cell_magic('bash', '--bg', '\nexport CUDA_VISIBLE_DEVICES=\'\'\nexport TF_CONFIG=\'{"cluster": {"worker": ["127.0.0.1:9901", "127.0.0.1:9902"]},\n                   "task": {"type": "worker", "index": 0}}\'\npython my_mnist_multiworker_task.py > my_worker_0.log 2>&1\n')


get_ipython().run_cell_magic('bash', '--bg', '\nexport CUDA_VISIBLE_DEVICES=\'\'\nexport TF_CONFIG=\'{"cluster": {"worker": ["127.0.0.1:9901", "127.0.0.1:9902"]},\n                   "task": {"type": "worker", "index": 1}}\'\npython my_mnist_multiworker_task.py > my_worker_1.log 2>&1\n')


# **Note**: if you get warnings about `AutoShardPolicy`, you can safely ignore them. See [TF issue #42146](https://github.com/tensorflow/tensorflow/issues/42146) for more details.

# That's it! Our TensorFlow cluster is now running, but we can't see it in this notebook because it's running in separate processes (but you can see the progress in `my_worker_*.log`).
# 
# Since the chief (worker #0) is writing to TensorBoard, we use TensorBoard to view the training progress. Run the following cell, then click on the settings button (i.e., the gear icon) in the TensorBoard interface and check the "Reload data" box to make TensorBoard automatically refresh every 30s. Once the first epoch of training is finished (which may take a few minutes), and once TensorBoard refreshes, the SCALARS tab will appear. Click on this tab to view the progress of the model's training and validation accuracy.

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=./my_mnist_multiworker_logs --port=6006')


# strategy = tf.distribute.MultiWorkerMirroredStrategy(
#     communication_options=tf.distribute.experimental.CommunicationOptions(
#         implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))


# ## Running Large Training Jobs on Vertex AI

# Let's copy the training script, but add `import os` and change the save path to be the GCS path that the `AIP_MODEL_DIR` environment variable will point to:

get_ipython().run_cell_magic('writefile', 'my_vertex_ai_training_task.py', '\nimport os\nfrom pathlib import Path\nimport tempfile\nimport tensorflow as tf\n\nstrategy = tf.distribute.MultiWorkerMirroredStrategy()  # at the start!\nresolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()\n\nif resolver.task_type == "chief":\n    model_dir = os.getenv("AIP_MODEL_DIR")  # paths provided by Vertex AI\n    tensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")\n    checkpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")\nelse:\n    tmp_dir = Path(tempfile.mkdtemp())  # other workers use a temporary dirs\n    model_dir = tmp_dir / "model"\n    tensorboard_log_dir = tmp_dir / "logs"\n    checkpoint_dir = tmp_dir / "ckpt"\n\ncallbacks = [tf.keras.callbacks.TensorBoard(tensorboard_log_dir),\n             tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)]\n\n# extra code – Load and prepare the MNIST dataset\nmnist = tf.keras.datasets.mnist.load_data()\n(X_train_full, y_train_full), (X_test, y_test) = mnist\nX_valid, X_train = X_train_full[:5000], X_train_full[5000:]\ny_valid, y_train = y_train_full[:5000], y_train_full[5000:]\n\n# extra code – build and compile the Keras model using the distribution strategy\nwith strategy.scope():\n    model = tf.keras.Sequential([\n        tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28],\n                                dtype=tf.uint8),\n        tf.keras.layers.Lambda(lambda X: X / 255),\n        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",\n                               padding="same", input_shape=[28, 28, 1]),\n        tf.keras.layers.MaxPooling2D(pool_size=2),\n        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",\n                               padding="same"), \n        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",\n                               padding="same"),\n        tf.keras.layers.MaxPooling2D(pool_size=2),\n        tf.keras.layers.Flatten(),\n        tf.keras.layers.Dense(units=64, activation="relu"),\n        tf.keras.layers.Dropout(0.5),\n        tf.keras.layers.Dense(units=10, activation="softmax"),\n    ])\n    model.compile(loss="sparse_categorical_crossentropy",\n                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),\n                  metrics=["accuracy"])\n\nmodel.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10,\n          callbacks=callbacks)\nmodel.save(model_dir, save_format="tf")\n')


custom_training_job = aiplatform.CustomTrainingJob(
    display_name="my_custom_training_job",
    script_path="my_vertex_ai_training_task.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    model_serving_container_image_uri=server_image,
    requirements=["gcsfs==2022.3.0"],  # not needed, this is just an example
    staging_bucket=f"gs://{bucket_name}/staging"
)


mnist_model2 = custom_training_job.run(
    machine_type="n1-standard-4",
    replica_count=2,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,
)


# Let's clean up:

mnist_model2.delete()
custom_training_job.delete()
blobs = bucket.list_blobs(prefix=f"gs://{bucket_name}/staging/")
for blob in blobs:
    blob.delete()


# # Hyperparameter Tuning on Vertex AI

get_ipython().run_cell_magic('writefile', 'my_vertex_ai_trial.py', '\nimport argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument("--n_hidden", type=int, default=2)\nparser.add_argument("--n_neurons", type=int, default=256)\nparser.add_argument("--learning_rate", type=float, default=1e-2)\nparser.add_argument("--optimizer", default="adam")\nargs = parser.parse_args()\n\nimport tensorflow as tf\n\ndef build_model(args):\n    with tf.distribute.MirroredStrategy().scope():\n        model = tf.keras.Sequential()\n        model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))\n        for _ in range(args.n_hidden):\n            model.add(tf.keras.layers.Dense(args.n_neurons, activation="relu"))\n        model.add(tf.keras.layers.Dense(10, activation="softmax"))\n        opt = tf.keras.optimizers.get(args.optimizer)\n        opt.learning_rate = args.learning_rate\n        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,\n                      metrics=["accuracy"])\n        return model\n\n# extra code – loads and splits the dataset\nmnist = tf.keras.datasets.mnist.load_data()\n(X_train_full, y_train_full), (X_test, y_test) = mnist\nX_valid, X_train = X_train_full[:5000], X_train_full[5000:]\ny_valid, y_train = y_train_full[:5000], y_train_full[5000:]\n\n# extra code – use the AIP_* environment variable and create the callbacks\nimport os\nmodel_dir = os.getenv("AIP_MODEL_DIR")\ntensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")\ncheckpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")\ntrial_id = os.getenv("CLOUD_ML_TRIAL_ID")\ntensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_log_dir)\nearly_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)\ncallbacks = [tensorboard_cb, early_stopping_cb]\n\nmodel = build_model(args)\nhistory = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),\n                    epochs=10, callbacks=callbacks)\nmodel.save(model_dir, save_format="tf")  # extra code\n\nimport hypertune\n\nhypertune = hypertune.HyperTune()\nhypertune.report_hyperparameter_tuning_metric(\n    hyperparameter_metric_tag="accuracy",  # name of the reported metric\n    metric_value=max(history.history["val_accuracy"]),  # max accuracy value\n    global_step=model.optimizer.iterations.numpy(),\n)\n')


trial_job = aiplatform.CustomJob.from_local_script(
    display_name="my_search_trial_job",
    script_path="my_vertex_ai_trial.py",  # path to your training script
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    staging_bucket=f"gs://{bucket_name}/staging",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,  # in this example, each trial will have 2 GPUs
)


from google.cloud.aiplatform import hyperparameter_tuning as hpt

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="my_hp_search_job",
    custom_job=trial_job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-3, max=10, scale="log"),
        "n_neurons": hpt.IntegerParameterSpec(min=1, max=300, scale="linear"),
        "n_hidden": hpt.IntegerParameterSpec(min=1, max=10, scale="linear"),
        "optimizer": hpt.CategoricalParameterSpec(["sgd", "adam"]),
    },
    max_trial_count=100,
    parallel_trial_count=20,
)
hp_job.run()


def get_final_metric(trial, metric_id):
    for metric in trial.final_measurement.metrics:
        if metric.metric_id == metric_id:
            return metric.value

trials = hp_job.trials
trial_accuracies = [get_final_metric(trial, "accuracy") for trial in trials]
best_trial = trials[np.argmax(trial_accuracies)]


max(trial_accuracies)


best_trial.id


best_trial.parameters


# # Extra Material – Distributed Keras Tuner on Vertex AI

# Instead of using Vertex AI's hyperparameter tuning service, you can use [Keras Tuner](https://keras.io/keras_tuner/) (introduced in Chapter 10) and run it on Vertex AI VMs. Keras Tuner provides a simple way to scale hyperparameter search by distributing it across multiple machines: it only requires setting three environment variables on each machine, then running your regular Keras Tuner code on each machine. You can use the exact same script on all machines. One of the machines acts as the chief, and the others act as workers. Each worker asks the chief which hyperparameter values to try—it acts as the oracle—then the worker trains the model using these hyperparameter values, and finally it reports the model's performance back to the chief, which can then decide which hyperparameter values the worker should try next.
# 
# The three environment variables you need to set on each machine are:
# 
# * `KERASTUNER_TUNER_ID`: equal to `"chief"` on the chief machine, or a unique identifier on each worker machine, such as `"worker0"`, `"worker1"`, etc.
# * `KERASTUNER_ORACLE_IP`: the IP address or hostname of the chief machine. The chief itself should generally use `"0.0.0.0"` to listen on every IP address on the machine.
# * `KERASTUNER_ORACLE_PORT`: the TCP port that the chief will be listening on.
# 
# You can use distributed Keras Tuner on any set of machines. If you want to run it on Vertex AI machines, then you can spawn a regular training job, and just modify the training script to set the three environment variables properly before using Keras Tuner.
# 
# For example, the script below starts by parsing the `TF_CONFIG` environment variable, which will be automatically set by Vertex AI, just like earlier. It finds the address of the task of type `"chief"`, and it extracts the IP address or hostname, and the TCP port. It then defines the tuner ID as the task type followed by the task index, for example `"worker0"`. If the tuner ID is `"chief0"`, it changes it to `"chief"`, and it sets the IP to `"0.0.0.0"`: this will make it listen on all IPv4 address on its machine. Then it defines the environment variables for Keras Tuner. Next, the script creates a tuner, just like in Chapter 10, the it runs the search, and finally it saves the best model to the location given by Vertex AI:

get_ipython().run_cell_magic('writefile', 'my_keras_tuner_search.py', '\nimport json\nimport os\n\ntf_config = json.loads(os.environ["TF_CONFIG"])\n\nchief_ip, chief_port = tf_config["cluster"]["chief"][0].rsplit(":", 1)\ntuner_id = f\'{tf_config["task"]["type"]}{tf_config["task"]["index"]}\'\nif tuner_id == "chief0":\n    tuner_id = "chief"\n    chief_ip = "0.0.0.0"\n    # extra code – since the chief doesn\'t work much, you can optimize compute\n    # resources by running a worker on the same machine. To do this, you can\n    # just make the chief start another process, after tweaking the TF_CONFIG\n    # environment variable to set the task type to "worker" and the task index\n    # to a unique value. Uncomment the next few lines to give this a try:\n    # import subprocess\n    # import sys\n    # tf_config["task"]["type"] = "workerX"  # the worker on the chief\'s machine\n    # os.environ["TF_CONFIG"] = json.dumps(tf_config)\n    # subprocess.Popen([sys.executable] + sys.argv,\n    #                  stdout=sys.stdout, stderr=sys.stderr)\n\nos.environ["KERASTUNER_TUNER_ID"] = tuner_id\nos.environ["KERASTUNER_ORACLE_IP"] = chief_ip\nos.environ["KERASTUNER_ORACLE_PORT"] = chief_port\n\nfrom pathlib import Path\nimport keras_tuner as kt\nimport tensorflow as tf\n\ngcs_path = "/gcs/my_bucket/my_hp_search"  # replace with your bucket\'s name\n\ndef build_model(hp):\n    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)\n    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)\n    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,\n                             sampling="log")\n    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])\n    if optimizer == "sgd":\n        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)\n    else:\n        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)\n\n    model = tf.keras.Sequential()\n    model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))\n    for _ in range(n_hidden):\n        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))\n    model.add(tf.keras.layers.Dense(10, activation="softmax"))\n    model.compile(loss="sparse_categorical_crossentropy",\n                  optimizer=optimizer,\n                  metrics=["accuracy"])\n    return model\n\nhyperband_tuner = kt.Hyperband(\n    build_model, objective="val_accuracy", seed=42,\n    max_epochs=10, factor=3, hyperband_iterations=2,\n    distribution_strategy=tf.distribute.MirroredStrategy(),\n    directory=gcs_path, project_name="mnist")\n\n# extra code – Load and split the MNIST dataset\nmnist = tf.keras.datasets.mnist.load_data()\n(X_train_full, y_train_full), (X_test, y_test) = mnist\nX_valid, X_train = X_train_full[:5000], X_train_full[5000:]\ny_valid, y_train = y_train_full[:5000], y_train_full[5000:]\n\ntensorboard_log_dir = os.environ["AIP_TENSORBOARD_LOG_DIR"] + "/" + tuner_id\ntensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_log_dir)\nearly_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)\nhyperband_tuner.search(X_train, y_train, epochs=10,\n                       validation_data=(X_valid, y_valid),\n                       callbacks=[tensorboard_cb, early_stopping_cb])\n\nif tuner_id == "chief":\n    best_hp = hyperband_tuner.get_best_hyperparameters()[0]\n    best_model = hyperband_tuner.hypermodel.build(best_hp)\n    best_model.save(os.getenv("AIP_MODEL_DIR"), save_format="tf")\n')


# Note that Vertex AI automatically mounts the `/gcs` directory to GCS, using the open source [GCS Fuse adapter](https://cloud.google.com/storage/docs/gcs-fuse). This gives us a shared directory across the workers and the chief, which is required by Keras Tuner. Also note that we set the distribution strategy to a `MirroredStrategy`. This will allow each worker to use all the GPUs on its machine, if there's more than one.
# 

# Replace `/gcs/my_bucket/` with <code>/gcs/<i>{bucket_name}</i>/</code>:

with open("my_keras_tuner_search.py") as f:
    script = f.read()

with open("my_keras_tuner_search.py", "w") as f:
    f.write(script.replace("/gcs/my_bucket/", f"/gcs/{bucket_name}/"))


# Now all we need to do is to start a custom training job based on this script, exactly like in the previous section. Don't forget to add `keras-tuner` to the list of `requirements`:

hp_search_job = aiplatform.CustomTrainingJob(
    display_name="my_hp_search_job",
    script_path="my_keras_tuner_search.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    model_serving_container_image_uri=server_image,
    requirements=["keras-tuner~=1.1.2"],
    staging_bucket=f"gs://{bucket_name}/staging",
)


mnist_model3 = hp_search_job.run(
    machine_type="n1-standard-4",
    replica_count=3,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,
)


# And we have a model!

# Let's clean up:

mnist_model3.delete()
hp_search_job.delete()
blobs = bucket.list_blobs(prefix=f"gs://{bucket_name}/staging/")
for blob in blobs:
    blob.delete()


# # Extra Material – Using AutoML to Train a Model

# Let's start by exporting the MNIST dataset to PNG images, and prepare an `import.csv` pointing to each image, and indicating the split (training, validation, or test) and the label:

import matplotlib.pyplot as plt

mnist_path = Path("datasets/mnist")
mnist_path.mkdir(parents=True, exist_ok=True)
idx = 0
with open(mnist_path / "import.csv", "w") as import_csv:
    for split, X, y in zip(("training", "validation", "test"),
                           (X_train, X_valid, X_test),
                           (y_train, y_valid, y_test)):
        for image, label in zip(X, y):
            print(f"\r{idx + 1}/70000", end="")
            filename = f"{idx:05d}.png"
            plt.imsave(mnist_path / filename, np.tile(image, 3))
            line = f"{split},gs://{bucket_name}/mnist/{filename},{label}\n"
            import_csv.write(line)
            idx += 1


# Let's upload this dataset to GCS:

upload_directory(bucket, mnist_path)


# Now let's create a managed image dataset on Vertex AI:

from aiplatform.schema.dataset.ioformat.image import single_label_classification

mnist_dataset = aiplatform.ImageDataset.create(
    display_name="mnist-dataset",
    gcs_source=[f"gs://{bucket_name}/mnist/import.csv"],
    project=project_id,
    import_schema_uri=single_label_classification,
    sync=True,
)


# Create an AutoML training job on this dataset:

# **TODO**

# # Exercise Solutions

# ## 1. to 8.

# 1. A SavedModel contains a TensorFlow model, including its architecture (a computation graph) and its weights. It is stored as a directory containing a _saved_model.pb_ file, which defines the computation graph (represented as a serialized protocol buffer), and a _variables_ subdirectory containing the variable values. For models containing a large number of weights, these variable values may be split across multiple files. A SavedModel also includes an _assets_ subdirectory that may contain additional data, such as vocabulary files, class names, or some example instances for this model. To be more accurate, a SavedModel can contain one or more _metagraphs_. A metagraph is a computation graph plus some function signature definitions (including their input and output names, types, and shapes). Each metagraph is identified by a set of tags. To inspect a SavedModel, you can use the command-line tool `saved_model_cli` or just load it using `tf.saved_model.load()` and inspect it in Python.
# 2. TF Serving allows you to deploy multiple TensorFlow models (or multiple versions of the same model) and make them accessible to all your applications easily via a REST API or a gRPC API. Using your models directly in your applications would make it harder to deploy a new version of a model across all applications. Implementing your own microservice to wrap a TF model would require extra work, and it would be hard to match TF Serving's features. TF Serving has many features: it can monitor a directory and autodeploy the models that are placed there, and you won't have to change or even restart any of your applications to benefit from the new model versions; it's fast, well tested, and scales very well; and it supports A/B testing of experimental models and deploying a new model version to just a subset of your users (in this case the model is called a _canary_). TF Serving is also capable of grouping individual requests into batches to run them jointly on the GPU. To deploy TF Serving, you can install it from source, but it is much simpler to install it using a Docker image. To deploy a cluster of TF Serving Docker images, you can use an orchestration tool such as Kubernetes, or use a fully hosted solution such as Google Vertex AI.
# 3. To deploy a model across multiple TF Serving instances, all you need to do is configure these TF Serving instances to monitor the same _models_ directory, and then export your new model as a SavedModel into a subdirectory.
# 4. The gRPC API is more efficient than the REST API. However, its client libraries are not as widely available, and if you activate compression when using the REST API, you can get almost the same performance. So, the gRPC API is most useful when you need the highest possible performance and the clients are not limited to the REST API.
# 5. To reduce a model's size so it can run on a mobile or embedded device, TFLite uses several techniques:
#     * It provides a converter which can optimize a SavedModel: it shrinks the model and reduces its latency. To do this, it prunes all the operations that are not needed to make predictions (such as training operations), and it optimizes and fuses operations whenever possible.
#     * The converter can also perform post-training quantization: this technique dramatically reduces the model’s size, so it’s much faster to download and store.
#     * It saves the optimized model using the FlatBuffer format, which can be loaded to RAM directly, without parsing. This reduces the loading time and memory footprint.
# 6. Quantization-aware training consists in adding fake quantization operations to the model during training. This allows the model to learn to ignore the quantization noise; the final weights will be more robust to quantization.
# 7. Model parallelism means chopping your model into multiple parts and running them in parallel across multiple devices, hopefully speeding up the model during training or inference. Data parallelism means creating multiple exact replicas of your model and deploying them across multiple devices. At each iteration during training, each replica is given a different batch of data, and it computes the gradients of the loss with regard to the model parameters. In synchronous data parallelism, the gradients from all replicas are then aggregated and the optimizer performs a Gradient Descent step. The parameters may be centralized (e.g., on parameter servers) or replicated across all replicas and kept in sync using AllReduce. In asynchronous data parallelism, the parameters are centralized and the replicas run independently from each other, each updating the central parameters directly at the end of each training iteration, without having to wait for the other replicas. To speed up training, data parallelism turns out to work better than model parallelism, in general. This is mostly because it requires less communication across devices. Moreover, it is much easier to implement, and it works the same way for any model, whereas model parallelism requires analyzing the model to determine the best way to chop it into pieces. That said, research in this domain is making quick progress (e.g., PipeDream or Pathways), so a mix of model parallelism and data parallelism is probably the way forward.
# 8. When training a model across multiple servers, you can use the following distribution strategies:
#     * The `MultiWorkerMirroredStrategy` performs mirrored data parallelism. The model is replicated across all available servers and devices, and each replica gets a different batch of data at each training iteration and computes its own gradients. The mean of the gradients is computed and shared across all replicas using a distributed AllReduce implementation (NCCL by default), and all replicas perform the same Gradient Descent step. This strategy is the simplest to use since all servers and devices are treated in exactly the same way, and it performs fairly well. In general, you should use this strategy. Its main limitation is that it requires the model to fit in RAM on every replica.
#     * The `ParameterServerStrategy` performs asynchronous data parallelism. The model is replicated across all devices on all workers, and the parameters are sharded across all parameter servers. Each worker has its own training loop, running asynchronously with the other workers; at each training iteration, each worker gets its own batch of data and fetches the latest version of the model parameters from the parameter servers, then it computes the gradients of the loss with regard to these parameters, and it sends them to the parameter servers. Lastly, the parameter servers perform a Gradient Descent step using these gradients. This strategy is generally slower than the previous strategy, and a bit harder to deploy, since it requires managing parameter servers. However, it can be useful in some situations, especially when you can take advantage of the asynchronous updates, for example to reduce I/O bottlenecks. This depends on many factors, including hardware, network topology, number of servers, model size, and more, so your mileage may vary.

# ## 9.
# _Exercise: Train a model (any model you like) and deploy it to TF Serving or Google Vertex AI. Write the client code to query it using the REST API or the gRPC API. Update the model and deploy the new version. Your client code will now query the new version. Roll back to the first version._

# Please follow the steps in the <a href="#Deploying-TensorFlow-models-to-TensorFlow-Serving-(TFS)">Deploying TensorFlow models to TensorFlow Serving</a> section above.

# # 10.
# _Exercise: Train any model across multiple GPUs on the same machine using the `MirroredStrategy` (if you do not have access to GPUs, you can use Colaboratory with a GPU Runtime and create two virtual GPUs). Train the model again using the `CentralStorageStrategy `and compare the training time._

# Please follow the steps in the [Distributed Training](#Distributed-Training) section above.

# # 11.
# _Exercise: Train a small model on Google Vertex AI, using TensorFlow Cloud Tuner for hyperparameter tuning._

# Please follow the instructions in the _Hyperparameter Tuning using TensorFlow Cloud Tuner_ section in the book.

# # Congratulations!

# You've reached the end of the book! I hope you found it useful. 😊
