import tensorflow as tf
import numpy as np
import os
import ConfigParser
import image_preprocessing
import CNN_structure

cf = ConfigParser.ConfigParser()
cf.read('CNN.conf')

def new_train():
    ######### train reference #################

    PIC_DIR                   = cf.get      ( 'CNN_train', 'PIC_DIR'                   )
    VALIDATION_PERCENTAGE     = cf.getfloat ( 'CNN_train', 'VALIDATION_PERCENTAGE'     )
    TESTING_PERCENTAGE        = cf.getfloat ( 'CNN_train', 'TESTING_PERCENTAGE'        )
    IMAGE_WIDTH               = cf.getint   ( 'CNN_train', 'IMAGE_WIDTH'               )
    IMAGE_HEIGHT              = cf.getint   ( 'CNN_train', 'IMAGE_HEIGHT'              )
    IMAGE_DEPTH               = cf.getint   ( 'CNN_train', 'IMAGE_DEPTH'               )
    SHUFFLE_DATA              = cf.getint   ( 'CNN_train', 'SHUFFLE_DATA'              )

    BATCH_SIZE                = cf.getint   ( 'CNN_train', 'BATCH_SIZE'                )
    STEPS                     = cf.getint   ( 'CNN_train', 'STEPS'                     )
    REGULARIZATION_RATE       = cf.getfloat ( 'CNN_train', 'REGULARIZATION_RATE'       )

    OPTIMIZER                 = cf.get      ( 'CNN_train', 'OPTIMIZER'                 )

    CNN_MODEL                 = cf.get      ( 'CNN_train', 'CNN_MODEL'                 )

    USE_DECAYED_LEARNING_RATE = cf.get      ( 'CNN_train', 'USE_DECAYED_LEARNING_RATE' )
    LEARNING_RATE             = cf.getfloat ( 'CNN_train', 'LEARNING_RATE'             )
    DECAY_STEPS               = cf.getint   ( 'CNN_train', 'DECAY_STEPS'               )
    DECAY_RATE                = cf.getfloat ( 'CNN_train', 'DECAY_RATE'                )


    DROP_OUT                  = cf.getfloat ( 'CNN_train', 'DROP_OUT'                  )

    MODEL_DIR                 = cf.get      ( 'CNN_train', 'MODEL_DIR'                 )
    MODEL_FILE                = cf.get      ( 'CNN_train', 'MODEL_FILE'                )

    LOG_DIR                   = cf.get      ( 'CNN_train', 'LOG_DIR'                   )

    ######Data and Calc########################
    #X,Y         = mnist.train.next_batch(dataset_size)
    #vX,vY       = mnist.validation.next_batch(VALIDATION_dataset_size)
    #tX,tY       = mnist.test.next_batch(TEST_dataset_size)

    category_number, dataset_size, vdataset_size, tdataset_size, X, Y, vX, vY, tX, tY  = image_preprocessing.subdir_and_image_to_array(
        PIC_DIR,
        VALIDATION_PERCENTAGE,
        TESTING_PERCENTAGE,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DEPTH,
        SHUFFLE_DATA
    )

    print('training dataset   : %d' %dataset_size)
    print('validation dataset : %d' %vdataset_size)
    print('testing dataset    : %d' %tdataset_size)

    ######INPUT_NODE and OUTPUT_NODE depented on image_preprocessing.py###################
    INPUT_NODE  = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
    OUTPUT_NODE = category_number

    x           = tf.placeholder(tf.float32, shape = (None,INPUT_NODE),  name = 'x-input')
    y_          = tf.placeholder(tf.float32, shape = (None,OUTPUT_NODE), name = 'y-input')
    keep_prob   = tf.placeholder(tf.float32,                             name = 'keep_prob')
    global_step = tf.Variable(0, name = 'global_step')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    if CNN_MODEL == 'VGG16_slim':
        y                           = CNN_structure.VGG16_slim (x, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, category_number)
    if CNN_MODEL == 'LeNet3':
        y                           = CNN_structure.LeNet3     (x, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, category_number)
    if CNN_MODEL == 'LeNet3_slim':
        y                           = CNN_structure.LeNet3_slim(x, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, category_number)
    if CNN_MODEL == 'LeNet5_slim':
        y                           = CNN_structure.LeNet5_slim(x, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, category_number)

    ######Loss###############################
    #loss              = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
    #loss              = tf.reduce_mean(tf.square(y-y_))
    cross_entropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(labels       = tf.argmax(y_,1),logits = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    aaa                = tf.get_collection('losses')
    regulation_loss    = tf.add_n(tf.get_collection('losses'))
    loss               = cross_entropy_mean + regulation_loss

    ######Accuracy###########################
    corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy           = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

    ######Learning rate#######################
    if USE_DECAYED_LEARNING_RATE == 'YES':
        learning_rate  = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
    else:
        learning_rate  = LEARNING_RATE

    ######Train###############################
    if OPTIMIZER == 'GradientDescent':
        train_step         = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    if OPTIMIZER == 'Adam':
        train_step         = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ######Save###############################
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % dataset_size
            end   = min(start+BATCH_SIZE,dataset_size)
            sess.run(train_step, feed_dict = {x:X[start:end], y_:Y[start:end], keep_prob:DROP_OUT})
            if i % 10 == 0:
                print('---------------- After %d training steps ----------------' % (sess.run(global_step) - 1))
                print('training dataset accuracy is   : %f'  % (sess.run(accuracy , feed_dict = {x:X[start:end] , y_:Y[start:end] , keep_prob:1})))
                print('validation dataset accuracy is : %f'  % (sess.run(accuracy , feed_dict = {x:vX           , y_:vY           , keep_prob:1})))
                print('loss is                        : ' + str(sess.run(aaa      , feed_dict = {x:X[start:end] , y_:Y[start:end] , keep_prob:1})))
                print('')
        accuracy_sum = 0
        for i in range(len(tX)):
            accuracy_sum = accuracy_sum + sess.run(accuracy,feed_dict={x:np.reshape(tX[i],[-1,INPUT_NODE]),y_:np.reshape(tY[i],[-1,OUTPUT_NODE]), keep_prob:1})
        print('testing dataset accuracy is    : %f'  % (accuracy_sum/len(tX)))

        write = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
        saver.save(sess, os.path.join(MODEL_DIR, MODEL_FILE))
        saver.export_meta_graph(os.path.join(MODEL_DIR, MODEL_FILE + '.meta.json'), as_text=True)
        graph_def = tf.get_default_graph().as_graph_def()

        ######output_graph_def need check output layers name#####
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,['layer-output/Reshape'])
        with tf.gfile.GFile(os.path.join(MODEL_DIR, MODEL_FILE + '.pb'), 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def continue_train():
    PIC_DIR                   = cf.get      ( 'CNN_train', 'PIC_DIR'                   )
    VALIDATION_PERCENTAGE     = cf.getfloat ( 'CNN_train', 'VALIDATION_PERCENTAGE'     )
    TESTING_PERCENTAGE        = cf.getfloat ( 'CNN_train', 'TESTING_PERCENTAGE'        )
    IMAGE_WIDTH               = cf.getint   ( 'CNN_train', 'IMAGE_WIDTH'               )
    IMAGE_HEIGHT              = cf.getint   ( 'CNN_train', 'IMAGE_HEIGHT'              )
    IMAGE_DEPTH               = cf.getint   ( 'CNN_train', 'IMAGE_DEPTH'               )
    SHUFFLE_DATA              = cf.getint   ( 'CNN_train', 'SHUFFLE_DATA'              )

    BATCH_SIZE                = cf.getint   ( 'CNN_train', 'BATCH_SIZE'                )
    STEPS                     = cf.getint   ( 'CNN_train', 'STEPS'                     )
    REGULARIZATION_RATE       = cf.getfloat ( 'CNN_train', 'REGULARIZATION_RATE'       )

    OPTIMIZER                 = cf.get      ( 'CNN_train', 'OPTIMIZER'                 )

    USE_DECAYED_LEARNING_RATE = cf.get      ( 'CNN_train', 'USE_DECAYED_LEARNING_RATE' )
    LEARNING_RATE             = cf.getfloat ( 'CNN_train', 'LEARNING_RATE'             )
    DECAY_STEPS               = cf.getint   ( 'CNN_train', 'DECAY_STEPS'               )
    DECAY_RATE                = cf.getfloat ( 'CNN_train', 'DECAY_RATE'                )


    DROP_OUT                  = cf.getfloat ( 'CNN_train', 'DROP_OUT'                  )

    MODEL_DIR                 = cf.get      ( 'CNN_train', 'MODEL_DIR'                 )
    MODEL_FILE                = cf.get      ( 'CNN_train', 'MODEL_FILE'                )

    LOG_DIR                   = cf.get      ( 'CNN_train', 'LOG_DIR'                   )

    category_number, dataset_size, vdataset_size, tdataset_size, X, Y, vX, vY, tX, tY  = image_preprocessing.subdir_and_image_to_array(
        PIC_DIR,
        VALIDATION_PERCENTAGE,
        TESTING_PERCENTAGE,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DEPTH,
        SHUFFLE_DATA
    )

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(MODEL_DIR, MODEL_FILE + '.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        INPUT_NODE  = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
        OUTPUT_NODE = category_number

        graph       = tf.get_default_graph()
        x           = graph.get_tensor_by_name('x-input:0')
        y_          = graph.get_tensor_by_name('y-input:0')
        y           = graph.get_tensor_by_name('layer-output/Reshape:0')
        keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        global_step = graph.get_tensor_by_name('global_step:0')
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

        ######Loss###############################
        #loss              = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
        #loss              = tf.reduce_mean(tf.square(y-y_))
        cross_entropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(labels       = tf.argmax(y_,1),logits = y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        aaa                = tf.get_collection('losses')
        regulation_loss    = tf.add_n(tf.get_collection('losses'))
        loss               = cross_entropy_mean + regulation_loss

        ######Accuracy###########################
        corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
        accuracy           = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

        ######Learning rate#######################
        if USE_DECAYED_LEARNING_RATE == 'YES':
            learning_rate  = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
        else:
            learning_rate  = LEARNING_RATE

        ######Train###############################
        if OPTIMIZER == 'GradientDescent':
            train_step         = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        if OPTIMIZER == 'Adam':
            train_step         = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % dataset_size
            end   = min(start+BATCH_SIZE,dataset_size)
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end], keep_prob:DROP_OUT})
            if i % 10 == 0:
                print('---------------- After %d training steps ----------------' % (sess.run(global_step) - 1))
                print('training dataset accuracy is   : %f'  % (sess.run(accuracy , feed_dict = {x:X[start:end] , y_:Y[start:end] , keep_prob:1})))
                print('validation dataset accuracy is : %f'  % (sess.run(accuracy , feed_dict = {x:vX           , y_:vY           , keep_prob:1})))
                print('loss is                        : ' + str(sess.run(aaa      , feed_dict = {x:X[start:end] , y_:Y[start:end] , keep_prob:1})))
                print('')
        accuracy_sum = 0
        for i in range(len(tX)):
            accuracy_sum = accuracy_sum + sess.run(accuracy,feed_dict={x:np.reshape(tX[i],[-1,INPUT_NODE]),y_:np.reshape(tY[i],[-1,OUTPUT_NODE]), keep_prob:1})
        print('testing dataset accuracy is    : %f'  % (accuracy_sum/len(tX)))

        write = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
        saver.save(sess, os.path.join(MODEL_DIR, MODEL_FILE))
        saver.export_meta_graph(os.path.join(MODEL_DIR, MODEL_FILE + '.meta.json'), as_text=True)
        graph_def = tf.get_default_graph().as_graph_def()

        ######output_graph_def need check output layers name#####
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,['layer-output/Reshape'])
        with tf.gfile.GFile(os.path.join(MODEL_DIR, MODEL_FILE + '.pb'), 'wb') as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    TRAIN_STATE = cf.get ( 'CNN_train', 'TRAIN_STATE')
    if TRAIN_STATE == 'new_train':
        new_train()
    if TRAIN_STATE == 'continue_train':
        continue_train()

