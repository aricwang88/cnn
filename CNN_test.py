import tensorflow as tf
import numpy as np
import os
import ConfigParser
import image_preprocessing

cf = ConfigParser.ConfigParser()
cf.read('CNN.conf')
np.set_printoptions(threshold='nan')

def test_use_pb():
    PIC_DIR               = cf.get     ('CNN_test', 'PIC_DIR'               )
    TEST_OUTPUT_DIR       = cf.get     ('CNN_test', 'TEST_OUTPUT_DIR'       )
    VALIDATION_PERCENTAGE = cf.getfloat('CNN_test', 'VALIDATION_PERCENTAGE' )
    TESTING_PERCENTAGE    = cf.getfloat('CNN_test', 'TESTING_PERCENTAGE'    )
    IMAGE_WIDTH           = cf.getint  ('CNN_test', 'IMAGE_WIDTH'           )
    IMAGE_HEIGHT          = cf.getint  ('CNN_test', 'IMAGE_HEIGHT'          )
    IMAGE_DEPTH           = cf.getint  ('CNN_test', 'IMAGE_DEPTH'           )
    SHUFFLE_DATA          = cf.getint  ('CNN_test', 'SHUFFLE_DATA'          )

    BATCH_SIZE            = cf.getint  ('CNN_test', 'BATCH_SIZE'            )

    DROP_OUT              = cf.getfloat('CNN_test', 'DROP_OUT'              )

    MODEL_DIR             = cf.get     ('CNN_test', 'MODEL_DIR'             )
    MODEL_FILE            = cf.get     ('CNN_test', 'MODEL_FILE'            )

    category_number, dataset_size, vdataset_size, tdataset_size, X, Y, vX, vY, tX, tY  = image_preprocessing.subdir_and_image_to_array(
        PIC_DIR,
        VALIDATION_PERCENTAGE,
        TESTING_PERCENTAGE,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DEPTH,
        SHUFFLE_DATA
    )
    #print(tX.shape)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE + '.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        graph            = tf.get_default_graph()
        tensor_input     = graph.get_tensor_by_name('x-input:0')
        tensor_output    = graph.get_tensor_by_name('layer-output/Reshape:0')
        keep_prob        = graph.get_tensor_by_name('keep_prob:0')
        steps            = tdataset_size / BATCH_SIZE
        remaining        = tdataset_size % BATCH_SIZE
        if remaining != 0:
            steps = steps + 1
        for i in range(steps):
            start  = (i * BATCH_SIZE) % tdataset_size
            end    = min(start + BATCH_SIZE, tdataset_size)
            y_out  = sess.run(tensor_output, {tensor_input:tX[start:end], keep_prob:DROP_OUT})
            tY_sub = sess.run(tf.argmax(y_out,1))
            if i == 0:
                tY = tY_sub
            else:
                tY = np.append(tY,tY_sub)
        print(tY)
        print(tX.shape,tY.shape)
        image_preprocessing.array_to_subdir_and_image(TEST_OUTPUT_DIR,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_DEPTH,
                                                      tX,
                                                      tY)


def test_use_meta():
    PIC_DIR               = cf.get     ('CNN_test', 'PIC_DIR'               )
    TEST_OUTPUT_DIR       = cf.get     ('CNN_test', 'TEST_OUTPUT_DIR'       )
    VALIDATION_PERCENTAGE = cf.getfloat('CNN_test', 'VALIDATION_PERCENTAGE' )
    TESTING_PERCENTAGE    = cf.getfloat('CNN_test', 'TESTING_PERCENTAGE'    )
    IMAGE_WIDTH           = cf.getint  ('CNN_test', 'IMAGE_WIDTH'           )
    IMAGE_HEIGHT          = cf.getint  ('CNN_test', 'IMAGE_HEIGHT'          )
    IMAGE_DEPTH           = cf.getint  ('CNN_test', 'IMAGE_DEPTH'           )
    SHUFFLE_DATA          = cf.getint  ('CNN_test', 'SHUFFLE_DATA'          )

    BATCH_SIZE            = cf.getint  ('CNN_test', 'BATCH_SIZE'            )

    DROP_OUT              = cf.getfloat('CNN_test', 'DROP_OUT'              )

    MODEL_DIR             = cf.get     ('CNN_test', 'MODEL_DIR'             )
    MODEL_FILE            = cf.get     ('CNN_test', 'MODEL_FILE'            )

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

        graph            = tf.get_default_graph()
        tensor_input     = graph.get_tensor_by_name('x-input:0')
        tensor_output    = graph.get_tensor_by_name('layer-output/Reshape:0')
        keep_prob        = graph.get_tensor_by_name('keep_prob:0')
        steps            = tdataset_size / BATCH_SIZE
        remaining        = tdataset_size % BATCH_SIZE
        if remaining != 0:
            steps = steps + 1
        for i in range(steps):
            start = (i * BATCH_SIZE) % tdataset_size
            end   = min(start + BATCH_SIZE, tdataset_size)
            y_out = sess.run(tensor_output, {tensor_input:tX[start:end], keep_prob:DROP_OUT})
            tY_sub = sess.run(tf.argmax(y_out,1))
            if i == 0:
                tY = tY_sub
            else:
                tY = np.append(tY,tY_sub)
        print(tY)
        print(tX.shape,tY.shape)
        image_preprocessing.array_to_subdir_and_image(TEST_OUTPUT_DIR,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_DEPTH,
                                                      tX,
                                                      tY)

if __name__ == '__main__':
    test_use_pb()
