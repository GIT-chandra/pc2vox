import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def Convo3d(layer,num_filters):
    return tf.layers.conv3d(layer,num_filters,[5,5,5],padding = 'same',activation = tf.nn.relu)

def Mpool(layer):
    return tf.layers.max_pooling3d(layer,[2,2,2],2)

def model_fn(features,labels,mode):
    '''
    Input will be sent as 512 x 512 x (512 + 256 + ... + 4) = 512 x 512 x 1020
    This will then be split to recover the individual grids
    '''
    input512 = tf.reshape(features["x"], [-1, 64, 64, 64, 1])
    c1b = Convo3d(input512,8)
    p1 = Mpool(c1b)

    input256 = tf.reshape(features["x2"], [-1, 32, 32, 32, 1])
    ccat1 = tf.concat([input256,p1],4)
    c2b = Convo3d(ccat1,16)

    # c2b = Convo3d(p1,16)

    p2 = Mpool(c2b)
    c3b = Convo3d(p2,32)
    c3bflat = tf.reshape(c3b,[-1,16*16*16*32])
    logits = tf.layers.dense(c3bflat,units = 10)

    pred = {
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)


def sample_gen():

    # count = 0
    # count += batch_size
    # if count>l.shape[0]:
    #     count = 0
    #     yield f1[0:batch_size], f2[0:batch_size], l[0:batch_size]
    # else:
    #     yield f1[count - batch_size: count], f2[count - batch_size: count], l[count - batch_size: count]

    # images = np.random.random((64,64,64,1)).astype(np.float32)
    # images2 = np.random.random((32,32,32,1)).astype(np.float32)
    # labels = np.random.randint(10)
    # while(1):
    #     yield images, images2, labels

    images = np.random.random((10,64,64,64,1)).astype(np.float32)
    images2 = np.random.random((10,32,32,32,1)).astype(np.float32)
    labels = np.random.randint(10,size = 10)
    count = -1
    #if count<10:
    while(1):
        count += 1
        yield images[count%10], images2[count%10], labels[count%10]

def get_train_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            s = 1000
            images = np.random.random((s,64,64,64,1)).astype(np.float32)
            images2 = np.random.random((s,32,32,32,1)).astype(np.float32)
            labels = np.random.randint(10, size = s)

            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            images2_placeholder = tf.placeholder(
                images2.dtype, images2.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator

            # dataset = tf.contrib.data.Dataset.from_tensor_slices(
            #     (images_placeholder,images2_placeholder, labels_placeholder))
            # dataset = dataset.repeat(None)  # Infinite iterations
            # dataset = dataset.shuffle(buffer_size=1000)
            # dataset = dataset.batch(batch_size)

            dataset = tf.data.Dataset.from_generator(
            generator = sample_gen,
            output_types = (tf.float32, tf.float32, tf.int64),
            output_shapes= ([64,64,64,1],[32,32,32,1],[]))
            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            # iterator = dataset.make_one_shot_iterator()
            next_example, next_example2, next_label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'x':next_example,'x2':next_example2}, next_label
    return train_inputs, iterator_initializer_hook

def main(unused_argv):
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./tmp_model")

    # s = 1000
    # train_data = np.random.random((s,64,64,64,1)).astype(np.float32)
    # train_data2 = np.random.random((s,32,32,32,1)).astype(np.float32)
    # train_labels = np.random.randint(10, size = s)

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'x': train_data,'x2':train_data2},
    #     y=train_labels,
    #     batch_size=10,
    #     num_epochs=None,
    #     shuffle=True)

    train_input_fn, train_input_hook = get_train_inputs(batch_size=100)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[train_input_hook])

if __name__ == '__main__':
    tf.app.run()
