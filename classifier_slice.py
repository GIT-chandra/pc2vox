import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class axis(Enum):
    x = 0
    y = 1
    z = 2

import trimesh

EXTENT = 100.0
CLEARANCE = 0.1
L2PARAM = 0.02
LEARNING_RATE = 5e-6
B1 = 0.9
B2 = 0.9

BATCH_TRAIN = 16
BATCH_EVAL = 24

eval_start_delay = 600
eval_delay = 600

REG_PARAM = 0.05

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def get_image(mesh, img_axis, img_res=256, random_num_pts = False, n = 2**12, use_seed = False):
    if use_seed:
        np.random.seed(2)
    faces = mesh.faces
    v1s,v2s,v3s = mesh.vertices[faces[:,0],:],mesh.vertices[faces[:,1],:],mesh.vertices[faces[:,2],:]

    areas = triangle_area(v1s,v2s,v3s)

    probs = areas/areas.sum()
    # number of points to sample
    if random_num_pts == True:
        n = np.random.randint(2**10,2**14)
    weighted_rand_inds = np.random.choice(range(len(areas)),size = n, p = probs)

    sel_v1s = v1s[weighted_rand_inds]
    sel_v2s = v2s[weighted_rand_inds]
    sel_v3s = v3s[weighted_rand_inds]

    # barycentric co-ords
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)

    invalids = u + v >1 # u+v+w =1

    u[invalids] = 1 - u[invalids]
    v[invalids] = 1 - v[invalids]
    w = 1-(u+v)

    pt_cld = (sel_v1s * u) + (sel_v2s * v) + (sel_v3s * w)   

     # centering the model
    delta = (np.max(pt_cld,axis=0) + np.min(pt_cld,axis=0))/2.0
    if sum(np.abs(delta)) != 0.0:
        pt_cld -= delta

    # scaling the model to have largest extent of 100
    span = np.max(np.max(pt_cld,axis=0) - np.min(pt_cld,axis=0))
    pt_cld /= span
    pt_cld *= (EXTENT - 2*CLEARANCE)
    pt_cld += EXTENT/2.0    # to get indices correctly on dividing by unit length 

    # getting the image 
    unit = EXTENT/float(img_res)
    
    inds_xyz = (pt_cld/unit).astype(int)
    inds_img = inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2))[0]]
    for i in range(20):
        inds_img = np.concatenate([inds_img, inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2 - i))[0]] ],axis = 0)
        inds_img = np.concatenate([inds_img, inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2 + i))[0]] ],axis = 0)
    img = np.zeros((img_res,img_res))

    img_axes_map = {0:[2,1],1:[2,0],2:[0,1]}
    for pix in inds_img:
        img[pix[img_axes_map[img_axis.value][0]], pix[img_axes_map[img_axis.value][1]]] = 1
    # max_intensity = np.max(img)
    # if  max_intensity != 0:
    #     img /= max_intensity
    
    return img




TRAIN_FILE_LIST = 'trainFilesShuffled.txt'
EVAL_FILE_LIST = 'evalFiles.txt'
NUM_CLASSES = 10

with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
NUM_FILES_EVAL = evalFiles.shape[0]


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

def Convo3d(layer,num_filters, isTrain):
    cnv =  tf.layers.conv3d(layer,num_filters,[5,5,5],padding = 'same',\
    activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    return cnv

def upConvo3d(layer,num_filters, isTrain):
    cnv =  tf.layers.conv3d_transpose(layer,num_filters,[2,2,2],strides = (2,2,2),\
    activation = tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer(),\
     use_bias = False)
    return cnv

def Mpool(layer):
    return tf.layers.max_pooling3d(layer,[2,2,2],2)

def model_fn(features,labels,mode):
    isTrain = (mode == tf.estimator.ModeKeys.TRAIN)
    input64 = tf.reshape(features['64'], [-1, 64, 64, 64, 1])

    c2a = Convo3d(input64,8,isTrain)
    c2b = Convo3d(c2a,8,isTrain)
    p2 = Mpool(c2b) # [-1,32,32,32,f]

    c3a = Convo3d(p2,8,isTrain)
    c3b = Convo3d(c3a,8,isTrain)
    p3 = Mpool(c3b) # [-1,16,16,16,f]

    c4a = Convo3d(p3,8,isTrain)
    c4b = Convo3d(c4a,8,isTrain)
    p4 = Mpool(c4b) # [-1,8,8,8,f]

    c5a = Convo3d(p4,8,isTrain)
    c5b = Convo3d(c5a,8,isTrain)

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
        mesh = trimesh.load_mesh(trainFiles[count])
      
        # angle_rad = np.random.rand()*2*np.pi    
        # M = trimesh.transformations.rotation_matrix(angle_rad,[0,0,1])
        # mesh.apply_transform(M)

        inputFull = createGrid(mesh)
        yield inputFull.astype(np.float32), labels

def get_train_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = train_sample_gen,
            output_types = (tf.float32, tf.int64),
            output_shapes= ([64,64,64], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)
            # dataset = dataset.shuffle(buffer_size=3500)

            iterator = dataset.make_initializable_iterator()
            inp64, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'64':inp64}, label
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

        # preparing the lable file
        cat = evalFiles[count].split('/')[1]
        labels = CAT_DICT[cat]

        # Input file
        mesh = trimesh.load_mesh(evalFiles[count])
        inputFull = createGrid(mesh, use_seed=True)

        yield inputFull.astype(np.float32),labels

def get_eval_inputs(batch_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def eval_inputs():
        with tf.name_scope('Evaluation_data'):
            dataset = tf.data.Dataset.from_generator(
            generator = eval_sample_gen,
            output_types = (tf.float32, tf.int64),
            output_shapes= ([64,64,64], []))

            dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            inp64, label = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer)
            return {'64':inp64}, label
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
    FNAME = 'ModelNet10/sofa/test/sofa_0681.off'
    mesh = trimesh.load_mesh(FNAME)
    img = get_image(mesh, axis.y)
    plt.imshow(img, cmap='gray', origin='lower')
    plt.show()
    exit()

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
    max_steps=40000)

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