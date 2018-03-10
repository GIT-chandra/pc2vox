import tensorflow as tf
import numpy as np

TRAIN_FILE_LIST = 'trainFiles.txt'
TEST_FILE_LIST = 'testFiles.txt'
DIM = 128

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
    input128 = tf.reshape(features['128'], [-1, 128, 128, 128, 1])
    logits = Convo3d(input128,1)
    # p1 = Mpool(c1b)
    #
    # input256 = tf.reshape(features['32'], [-1, 32, 32, 32, 1])
    # ccat1 = tf.concat([input256,p1],4)
    # c2b = Convo3d(ccat1,16)
    #
    # # c2b = Convo3d(p1,16)
    #
    # p2 = Mpool(c2b)
    # c3b = Convo3d(p2,32)
    # c3bflat = tf.reshape(c3b,[-1,16*16*16*32])
    # logits = tf.layers.dense(c3bflat,units = 10)


    pred = {
    "classes": logits,
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    loss = tf.losses.mean_squared_error(labels,logits)

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
    with open(TRAIN_FILE_LIST,'r') as f:
        trainFiles = np.array(f.read().split('\n')[:-1])
    num_files = trainFiles.shape[0]
    trainFiles = trainFiles[np.random.permutation(num_files)]

    count = -1
    while(1):
        count += 1
        if count >= num_files:
            count = 0

        # preparing the lable file
        with open(trainFiles[count],'r') as f:
            '''
            first 5 values are not voxel indices:
            refer to https://github.com/topskychen/voxelizer
            + 1 because voxelization done at dimension - 2; to get padding of 1 all around
            '''
            gt_vox = np.array(f.read().replace('\n',' ').split(' ')[5:-1]).reshape((-1,3)).astype(int) + 1
        labels = np.zeros((DIM,DIM,DIM))
        labels[gt_vox[:,0],gt_vox[:,1],gt_vox[:,2]] = 1

        # Input file
        inputFull = np.load(trainFiles[count][:-3] + 'npy', encoding='latin1')

        yield inputFull[0].astype(np.float32), \
        inputFull[1].astype(np.float32), \
        inputFull[2].astype(np.float32), \
        inputFull[3].astype(np.float32), \
        inputFull[4].astype(np.float32), \
        inputFull[5].astype(np.float32), \
        labels.reshape((DIM,DIM,DIM,1)).astype(np.int64)

def get_train_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):

            dataset = tf.data.Dataset.from_generator(
            generator = sample_gen,
            output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64),
            output_shapes= ([128,128,128], [64,64,64], [32,32,32], [16,16,16], [8,8,8], [4,4,4], [128,128,128,1]))
            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            # iterator = dataset.make_one_shot_iterator()
            inp128, inp64, inp32, inp16, inp8, inp4, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'128':inp128, '64':inp64, '32':inp32, '16':inp16, '8':inp8, '4':inp4}, label
    return train_inputs, iterator_initializer_hook

def main(unused_argv):
    mycfg = tf.estimator.RunConfig(model_dir=None,
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=100,
    save_checkpoints_secs=None,
    session_config=None,
    keep_checkpoint_max=5,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=10)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./tmp_model", config = mycfg)

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

    train_input_fn, train_input_hook = get_train_inputs(batch_size=4)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[train_input_hook])

if __name__ == '__main__':
    tf.app.run()
