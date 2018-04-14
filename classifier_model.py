import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from classifier_misc import *

tf.logging.set_verbosity(tf.logging.INFO)

def Convo2d(layer,num_filters, isTrain):
    cnv =  tf.layers.conv2d(layer,num_filters,[5,5],padding = 'same',\
    activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    return cnv
    
def Mpool(layer):
    return tf.layers.max_pooling2d(layer,[2,2],2)

def model_fn(features,labels,mode):
    isTrain = (mode == tf.estimator.ModeKeys.TRAIN)
    inpx = tf.reshape(features['x'], [-1, IMG_RES, IMG_RES, 5])

    c2a = Convo2d(inpx,8,isTrain)
    c2b = Convo2d(c2a,8,isTrain)
    p2 = Mpool(c2b) # [-1,32,32,32,f]

    c3a = Convo2d(p2,8,isTrain)
    c3b = Convo2d(c3a,8,isTrain)
    p3 = Mpool(c3b) # [-1,16,16,16,f]

    c4a = Convo2d(p3,8,isTrain)
    c4b = Convo2d(c4a,8,isTrain)
    p4 = Mpool(c4b) # [-1,8,8,8,f]

    c5a = Convo2d(p4,8,isTrain)
    c5b = Convo2d(c5a,8,isTrain)

    dense1 = tf.layers.dense(tf.layers.Flatten()(c5b),64,activation=tf.nn.relu)
    logits = tf.layers.dense(dense1,NUM_CLASSES)

    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=B1, beta2=B2)
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

    count = -1
    while(1):
        count += 1
        if count >= num_files:
            count = 0
            trainFiles = trainFiles[np.random.permutation(num_files)]

        # preparing the label file
        cat = trainFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        mesh = trimesh.load_mesh(trainFiles[count])

        input_x = get_images(mesh, axis.x)
        input_y = get_images(mesh, axis.y)
        input_z = get_images(mesh, axis.z)
        yield input_x.astype(np.float32), input_y.astype(np.float32), input_z.astype(np.float32), labels

def get_train_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = train_sample_gen,
            output_types = (tf.float32, tf.float32, tf.float32, tf.int64),
            output_shapes= ([IMG_RES,IMG_RES,5], [IMG_RES,IMG_RES,5], [IMG_RES,IMG_RES,5], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)
            # dataset = dataset.shuffle(buffer_size=3500)

            iterator = dataset.make_initializable_iterator()
            inpx, inpy, inpz, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'x':inpx, 'y':inpy, 'z':inpz}, label
    return train_inputs, iterator_initializer_hook

def eval_sample_gen():
    with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
    num_files = evalFiles.shape[0]

    count = -1
    while(1):
        count += 1
        if count >= num_files:
            count = 0

        # preparing the label file
        cat = evalFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        mesh = trimesh.load_mesh(evalFiles[count])

        input_x = get_images(mesh, axis.x, use_seed=True)
        input_y = get_images(mesh, axis.y, use_seed=True)
        input_z = get_images(mesh, axis.z, use_seed=True)
        yield input_x.astype(np.float32), input_y.astype(np.float32), input_z.astype(np.float32), labels

def get_eval_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def eval_inputs():
        with tf.name_scope('Evaluation_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = eval_sample_gen,
            output_types = (tf.float32, tf.float32, tf.float32, tf.int64),
            output_shapes= ([IMG_RES,IMG_RES,5], [IMG_RES,IMG_RES,5], [IMG_RES,IMG_RES,5], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            inpx, inpy, inpz, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'x':inpx, 'y':inpy, 'z':inpz}, label
    return eval_inputs, iterator_initializer_hook

def pred_sample_gen():
    with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
    num_files = evalFiles.shape[0]

    count = 0
    while(count < num_files):
        
        # preparing the lable file
        cat = evalFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        # inputFull = np.load(evalFiles[count][:-3] + 'npy', encoding='latin1')
        mesh = trimesh.load_mesh(evalFiles[count])
        inputFull = createGrid(mesh, use_seed=True)

        count += 1

        yield inputFull[0].astype(np.float32), \
        inputFull[1].astype(np.float32), \
        inputFull[2].astype(np.float32), \
        inputFull[3].astype(np.float32), \
        inputFull[4].astype(np.float32), \
        inputFull[5].astype(np.float32), \
        labels

def get_pred_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def pred_inputs():
        with tf.name_scope('Evaluation_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = pred_sample_gen,
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
    return pred_inputs, iterator_initializer_hook

if __name__ == '__main__':
    # FNAME = 'ModelNet10/sofa/test/sofa_0681.off'
    # mesh = trimesh.load_mesh(FNAME)
    # imgs = get_images(mesh, axis.z,n = 2**14, use_seed = True)
    # for i in range(imgs.shape[2]):
    #     plt.imshow(imgs[:,:,i], cmap='gray', origin='lower')
    #     plt.show()
    # exit()

    mycfg = tf.estimator.RunConfig(model_dir=None,
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=100,
    save_checkpoints_secs=None,
    session_config=None,
    keep_checkpoint_max=800,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=100)

    est = tf.estimator.Estimator(model_fn=model_fn, model_dir="./model_dir", config = mycfg)

    train_input_fn, train_input_hook = get_train_inputs(batch_size = BATCH_TRAIN)
    eval_input_fn, eval_input_hook = get_eval_inputs(batch_size = BATCH_EVAL)

    train_spec = tf.estimator.TrainSpec(
    input_fn = train_input_fn,
    hooks=[train_input_hook],
    max_steps=NUM_ITERS)

    eval_spec = tf.estimator.EvalSpec(
    input_fn = eval_input_fn,
    steps = int(NUM_FILES_EVAL/BATCH_EVAL),
    hooks=[eval_input_hook],
    throttle_secs=eval_delay,
    start_delay_secs=eval_start_delay
    )

    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

    # pred_input_fn, pred_input_hook = get_pred_inputs(batch_size = BATCH_EVAL)
    # res = est.predict(input_fn = pred_input_fn, hooks=[pred_input_hook])
    
    # labels = np.zeros(NUM_FILES_EVAL,dtype=int)
    # with open(EVAL_FILE_LIST,'r') as f:
    #     evalFiles = np.array(f.read().split('\n')[:-1])    
    # count = 0
    # print('Preparing labels')
    # while(count < NUM_FILES_EVAL):       
    #     cat = evalFiles[count].split('/')[1]
    #     labels[count] = int(CAT_DICT[cat])
    #     count += 1
    # print('Done labels')

    # confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES))
    # count = 0
    # for r in res:
    #     pred = int(r)
    #     print('Sample',count,'prediction : ',pred,'GT : ',labels[count])

    #     confusion_matrix[labels[count],pred] += 1

    #     count += 1
    #     if count >= NUM_FILES_EVAL:
    #         break

    # tp = 0
    # for i in range(NUM_CLASSES):
    #     tp += confusion_matrix[i,i]
    # print('Accuracy',tp/NUM_FILES_EVAL)


    # plt.imshow(confusion_matrix, cmap='gray')
    # plt.colorbar()
    # plt.show()