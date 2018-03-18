import tensorflow as tf
import numpy as np

TRAIN_FILE_LIST = 'trainFiles.txt'
EVAL_FILE_LIST = 'evalFiles.txt'
NUM_CLASSES = 10

CAT_DICT = {'bathtub':0,
            'bed':1,
            'chair':2,
            'desk':3,
            'dresser':4,
            'monitor':5,
            'night_stand':6,
            'sofa':7,
            'table':8,
            'toilet':9}

CATEGORIES = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

tf.logging.set_verbosity(tf.logging.INFO)

def Convo3d(layer,num_filters):
    cnv =  tf.layers.conv3d(layer,num_filters,[5,5,5],padding = 'same',activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    return tf.nn.dropout(tf.contrib.layers.batch_norm(cnv),0.5)
    # return tf.nn.dropout(cnv,0.5)
    # return cnv

def upConvo3d(layer,num_filters):
    cnv =  tf.layers.conv3d_transpose(layer,num_filters,[2,2,2],strides = (2,2,2),activation = tf.nn.relu, use_bias = False)
    return tf.nn.dropout(tf.contrib.layers.batch_norm(cnv),0.5)
    # return tf.nn.dropout(cnv,0.5)
    # return cnv

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

    c6a = Convo3d(p5,64)
    c6b = Convo3d(c6a,64)

    dense1 = tf.layers.dense(tf.layers.Flatten()(c6b),100,activation=tf.nn.relu)
    logits = tf.layers.dense(dense1,10)

    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=1e-2,beta2=1e-2)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    else:
        pred = {"predictions": tf.argmax(input = logits,axis = 1)}
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=pred["predictions"])}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred["predictions"])

class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def train_sample_gen():
    with open(TRAIN_FILE_LIST,'r') as f:
        trainFiles = np.array(f.read().split('\n')[:-1])
    num_files = trainFiles.shape[0]
    trainFiles = trainFiles[np.random.permutation(num_files)]
    trainFiles = trainFiles[np.random.permutation(num_files)]
    trainFiles = trainFiles[np.random.permutation(num_files)]

    count = -1
    while(1):
        count += 1
        if count >= num_files:
            count = 0
            trainFiles = trainFiles[np.random.permutation(num_files)]

        # preparing the lable file
        cat = trainFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        inputFull = np.load(trainFiles[count][:-3] + 'npy', encoding='latin1')

        yield inputFull[0].astype(np.float32), \
        inputFull[1].astype(np.float32), \
        inputFull[2].astype(np.float32), \
        inputFull[3].astype(np.float32), \
        inputFull[4].astype(np.float32), \
        inputFull[5].astype(np.float32), \
        labels

def get_train_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = train_sample_gen,
            output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64),
            output_shapes= ([128,128,128], [64,64,64], [32,32,32], [16,16,16], [8,8,8], [4,4,4], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            inp128, inp64, inp32, inp16, inp8, inp4, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'128':inp128, '64':inp64, '32':inp32, '16':inp16, '8':inp8, '4':inp4}, label
    return train_inputs, iterator_initializer_hook

def eval_sample_gen():
    with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
    num_files = evalFiles.shape[0]
    evalFiles = evalFiles[np.random.permutation(num_files)]

    count = -1
    while(1):
        count += 1
        if count >= num_files:
            count = 0
            evalFiles = evalFiles[np.random.permutation(num_files)]

        # preparing the lable file
        cat = evalFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        inputFull = np.load(evalFiles[count][:-3] + 'npy', encoding='latin1')

        yield inputFull[0].astype(np.float32), \
        inputFull[1].astype(np.float32), \
        inputFull[2].astype(np.float32), \
        inputFull[3].astype(np.float32), \
        inputFull[4].astype(np.float32), \
        inputFull[5].astype(np.float32), \
        labels

def get_eval_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def eval_inputs():
        with tf.name_scope('Evaluation_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = eval_sample_gen,
            output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64),
            output_shapes= ([128,128,128], [64,64,64], [32,32,32], [16,16,16], [8,8,8], [4,4,4], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            inp128, inp64, inp32, inp16, inp8, inp4, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'128':inp128, '64':inp64, '32':inp32, '16':inp16, '8':inp8, '4':inp4}, label
    return eval_inputs, iterator_initializer_hook

def pred_inp_fn():
    inputFull = np.load('/home/raman/DDP_17-18/pc2vox/ModelNet10/chair/train/chair_0001.npy', encoding='latin1')
    return {'128':inputFull[0].astype(np.float32), \
        '64':inputFull[1].astype(np.float32), \
        '32':inputFull[2].astype(np.float32), \
        '16':inputFull[3].astype(np.float32), \
        '8':inputFull[4].astype(np.float32), \
        '4':inputFull[5].astype(np.float32)}, None

def main(unused_argv):
    mycfg = tf.estimator.RunConfig(model_dir=None,
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=100,
    save_checkpoints_secs=None,
    session_config=None,
    keep_checkpoint_max=5,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=100)

    est = tf.estimator.Estimator(model_fn=model_fn, model_dir="./model_dir", config = mycfg)

    train_input_fn, train_input_hook = get_train_inputs(batch_size = 3)
    eval_input_fn, eval_input_hook = get_eval_inputs(batch_size = 3)

    train_spec = tf.estimator.TrainSpec(
    input_fn = train_input_fn,
    hooks=[train_input_hook],
    max_steps=30000)

    eval_spec = tf.estimator.EvalSpec(
    input_fn = eval_input_fn,
    hooks=[eval_input_hook],
    throttle_secs=600,
    start_delay_secs=600
    )

    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

    # inputFull = np.load('ModelNet10/bed/test/bed_0516.npy', encoding='latin1')
    # pred_input_fn = tf.estimator.inputs.numpy_input_fn(\
    # x = {'128':inputFull[0].astype(np.float32), \
    #     '64':inputFull[1].astype(np.float32), \
    #     '32':inputFull[2].astype(np.float32), \
    #     '16':inputFull[3].astype(np.float32), \
    #     '8':inputFull[4].astype(np.float32), \
    #     '4':inputFull[5].astype(np.float32)},
    # y = None,
    # batch_size=1,
    # num_epochs=1,
    # shuffle=False)


    # res = est.predict(input_fn = pred_inp_fn)
    # for r in res:
    #     voxx = r['predictions']
    #     break
    # np.save('pred.npy',voxx)
    #
    # voxx = np.load('pred.npy').reshape(128,128,128)
    #
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # idxs = np.where(voxx > 0.5)
    # ax.scatter(idxs[0],idxs[1],idxs[2],c='red')
    # plt.show()

if __name__ == '__main__':
    tf.app.run()
