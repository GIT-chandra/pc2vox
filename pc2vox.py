import tensorflow as tf
import numpy as np

TRAIN_FILE_LIST = 'trainFiles.txt'
TEST_FILE_LIST = 'testFiles.txt'

tf.logging.set_verbosity(tf.logging.INFO)

def Convo3d(layer,num_filters):
    cnv =  tf.layers.conv3d(layer,num_filters,[5,5,5],padding = 'same',activation = tf.nn.relu)
    return tf.nn.dropout(cnv,0.5)

def upConvo3d(layer,num_filters):
    cnv =  tf.layers.conv3d_transpose(layer,num_filters,[2,2,2],strides = (2,2,2),activation = tf.nn.relu, use_bias = False)
    return tf.nn.dropout(cnv,0.5)

def Mpool(layer):
    return tf.layers.max_pooling3d(layer,[2,2,2],2)

def model_fn(features,labels,mode):
    input128 = tf.reshape(features['128'], [-1, 128, 128, 128, 1])
    input64 = tf.reshape(features['64'], [-1, 64, 64, 64, 1])
    input32 = tf.reshape(features['32'], [-1, 32, 32, 32, 1])
    input16 = tf.reshape(features['16'], [-1, 16, 16, 16, 1])
    input8 = tf.reshape(features['8'], [-1, 8, 8, 8, 1])
    input4 = tf.reshape(features['4'], [-1, 4, 4, 4, 1])

    c1a = Convo3d(input128,32)
    c1b = Convo3d(c1a,16)
    p1 = Mpool(c1b) # [-1,64,64,64,f]

    c2a = Convo3d(input64,32)
    ccat1 = tf.concat([c2a,p1],4)
    c2b = Convo3d(ccat1,16)
    p2 = Mpool(c2b) # [-1,32,32,32,f]

    c3a = Convo3d(input32,32)
    ccat2 = tf.concat([c3a,p2],4)
    c3b = Convo3d(ccat2,16)
    p3 = Mpool(c3b) # [-1,16,16,16,f]

    c4a = Convo3d(input16,16)
    ccat3 = tf.concat([c4a,p3],4)
    c4b = Convo3d(ccat3,32)
    p4 = Mpool(c4b) # [-1,8,8,8,f]

    c5a = Convo3d(input8,8)
    ccat4 = tf.concat([c5a,p4],4)
    c5b = Convo3d(ccat4,32)
    p5 = Mpool(c5b) # [-1,4,4,4,f]

    c6a = Convo3d(input4,4)
    ccat5 = tf.concat([c6a,p5],4)
    c6b = Convo3d(ccat5,64)

    # deconvolutions
    d1a = upConvo3d(c6b,16) # # [-1,8,8,8,f]
    # d1a = tf.layers.conv3d_transpose(inputs = c6b,filters = 32,kernel_size = [2,2,2],strides = (2,2,2),activation = tf.nn.relu, )
    dccat1 = tf.concat([c5b,d1a],4)
    d1b = Convo3d(dccat1,32)

    d2a = upConvo3d(d1b,16) # [-1,16,16,16,f]
    dccat2 = tf.concat([c4b,d2a],4)
    d2b = Convo3d(dccat2,32)

    d3a = upConvo3d(d2b,8) # [-1,32,32,32,f]
    dccat3 = tf.concat([c3b,d3a],4)
    d3b = Convo3d(dccat3,16)

    d4a = upConvo3d(d3b,8) # [-1,64,64,64,f]
    dccat4 = tf.concat([c2b,d4a],4)
    d4b = Convo3d(dccat4,16)

    d5a = upConvo3d(d4b,8) # [-1,128,128,128,f]
    dccat5 = tf.concat([c1b,d5a],4)
    d5b = Convo3d(dccat5,16)

    logits = tf.layers.conv3d(d5b,2,[5,5,5],padding = 'same')

    # pred = {
    # "predictions": tf.argmax(input = logits,axis = 4)
    # }
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
        labels = np.zeros((128,128,128))
        labels[gt_vox[:,0],gt_vox[:,1],gt_vox[:,2]] = 1

        # Input file
        inputFull = np.load(trainFiles[count][:-3] + 'npy', encoding='latin1')

        yield inputFull[0].astype(np.float32), \
        inputFull[1].astype(np.float32), \
        inputFull[2].astype(np.float32), \
        inputFull[3].astype(np.float32), \
        inputFull[4].astype(np.float32), \
        inputFull[5].astype(np.float32), \
        labels.reshape((128,128,128,1)).astype(np.int64)

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
    log_step_count_steps=5)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./tmp_model", config = mycfg)

    train_input_fn, train_input_hook = get_train_inputs(batch_size=2)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[train_input_hook])

if __name__ == '__main__':
    tf.app.run()
