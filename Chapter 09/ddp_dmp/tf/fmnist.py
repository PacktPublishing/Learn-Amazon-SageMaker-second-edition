import tensorflow as tf
import numpy as np
import argparse, os
from model import FMNISTModel

# SMDataParallel: initialization
import smdistributed.dataparallel.tensorflow as sdp
sdp.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')

print("TensorFlow version", tf.__version__)

# Process command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

args, _ = parser.parse_known_args()

epochs     = args.epochs
lr         = args.learning_rate*sdp.size()
batch_size = args.batch_size*sdp.size()

model_dir  = args.model_dir
training_dir   = args.training
validation_dir = args.validation

# Load data set
x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

# Add extra dimension for channel: (28,28) --> (28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_val   = x_val[..., tf.newaxis]

# Prepare training and validation iterators
#  - define batch size
#  - normalize pixel values to [0,1]
#  - one-hot encode labels
preprocess = lambda x, y: (tf.divide(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10)))

train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
steps = len(train)//batch_size
train = train.map(preprocess)
train = train.repeat()

val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
val = val.map(preprocess)
val = val.repeat()

# Build model
model = FMNISTModel()
loss = tf.losses.CategoricalCrossentropy()
opt = tf.optimizers.Adam(lr)

sdp.broadcast_variables(model.variables, root_rank=0)
sdp.broadcast_variables(opt.variables(), root_rank=0)

@tf.function
def training_step(images, labels):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = sdp.DistributedGradientTape(tape)
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    # SMDataParallel: average the loss across workers
    loss_value = sdp.oob_allreduce(loss_value)
    return loss_value

for e in range(epochs):    
    if sdp.rank() == 0:
        print("Start epoch %d" % (e))
    for batch, (images, labels) in enumerate(train.take(steps)):
        loss_value = training_step(images, labels)
        if batch % 10 == 0 and sdp.rank() == 0:
            print("Step #%d\tLoss: %.6f" % (batch, loss_value))

# SMDataParallel: save model only on GPU 0
if sdp.rank() == 0:
    model.save(os.path.join(model_dir, '1'))
   