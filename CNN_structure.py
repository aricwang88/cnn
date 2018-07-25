import tensorflow as tf


def VGG16_slim(input_tensor, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, CATEGORY_NUMBER):

    ######### CNN structure reference #########
    CONV_SIZE      = 3
    CONV1_DEEP     = 64
    CONV2_DEEP     = 128
    CONV3_DEEP     = 256
    CONV4_DEEP     = 512
    CONV5_DEEP     = 512

    FC_SIZE1       = 512
    FC_SIZE2       = 512
    ############################################

    slim = tf.contrib.slim
    input_tensor = tf.reshape(input_tensor,[-1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])

    with tf.variable_scope('layer1-conv1'):
        relu11 = slim.conv2d(input_tensor,CONV1_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope('layer1-conv2'):
        relu12 = slim.conv2d(relu11,CONV1_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.name_scope("layer1-pool1"):
        pool11 = tf.nn.max_pool(relu12, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope('layer2-conv1'):
        relu21 = slim.conv2d(pool11,CONV2_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer2-conv2"):
        relu22 = slim.conv2d(relu21,CONV2_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.name_scope("layer2-pool1"):
        pool21 = tf.nn.max_pool(relu22, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope('layer3-conv1'):
        relu31 = slim.conv2d(pool21,CONV3_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer3-conv2"):
        relu32 = slim.conv2d(relu31,CONV3_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer3-conv3"):
        relu33 = slim.conv2d(relu32,CONV3_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.name_scope("layer3-pool1"):
        pool31 = tf.nn.max_pool(relu33, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope('layer4-conv1'):
        relu41 = slim.conv2d(pool31,CONV4_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer4-conv2"):
        relu42 = slim.conv2d(relu41,CONV4_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer4-conv3"):
        relu43 = slim.conv2d(relu42,CONV4_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.name_scope("layer4-pool1"):
        pool41 = tf.nn.max_pool(relu43, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope('layer5-conv1'):
        relu51 = slim.conv2d(pool41,CONV5_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer5-conv2"):
        relu52 = slim.conv2d(relu51,CONV5_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.variable_scope("layer5-conv3"):
        relu53 = slim.conv2d(relu52,CONV5_DEEP,[CONV_SIZE,CONV_SIZE])

    with tf.name_scope("layer5-pool1"):
        pool51 = tf.nn.max_pool(relu53, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

        pool_shape = pool51.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
        reshaped = tf.reshape(pool51, [-1, nodes])

    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE1], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE1, CATEGORY_NUMBER],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [CATEGORY_NUMBER], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.variable_scope('layer-output'):
        logit = tf.reshape(logit,[-1,CATEGORY_NUMBER])

    writer = tf.summary.FileWriter('log', tf.get_default_graph())
    writer.close()
    return logit

def LeNet3(input_tensor, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, CATEGORY_NUMBER):

    ######### CNN structure reference #########
    CONV1_DEEP              = 32
    CONV1_SIZE              = 3

    CONV2_DEEP              = 64
    CONV2_SIZE              = 3

    FC_SIZE                 = 512
    ###########################################

    input_tensor = tf.reshape(input_tensor,[-1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, IMAGE_DEPTH, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, CATEGORY_NUMBER],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [CATEGORY_NUMBER], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.variable_scope('layer-output'):
        logit = tf.reshape(logit,[-1,CATEGORY_NUMBER])

    writer = tf.summary.FileWriter('log', tf.get_default_graph())
    writer.close()
    return logit

def LeNet3_slim(input_tensor, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, CATEGORY_NUMBER):

    ######### CNN structure reference #########
    CONV1_DEEP              = 32
    CONV1_SIZE              = 3

    CONV2_DEEP              = 64
    CONV2_SIZE              = 3

    FC_SIZE                 = 512
    ###########################################

    slim = tf.contrib.slim
    input_tensor = tf.reshape(input_tensor,[-1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])

    with tf.variable_scope('layer1-conv1'):
        relu1 = slim.conv2d(input_tensor,CONV1_DEEP,[CONV1_SIZE,1])

    with tf.variable_scope('layer2-conv2'):
        relu2 = slim.conv2d(relu1,CONV1_DEEP,[1,CONV1_SIZE])

    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('layer4-conv3'):
        relu3 = slim.conv2d(pool1,CONV1_DEEP,[CONV2_SIZE,1])

    with tf.variable_scope("layer5-conv4"):
        relu4 = slim.conv2d(relu3,CONV2_DEEP,[1,CONV2_SIZE])

    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, CATEGORY_NUMBER],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [CATEGORY_NUMBER], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.variable_scope('layer-output'):
        logit = tf.reshape(logit,[-1,CATEGORY_NUMBER])

    writer = tf.summary.FileWriter('log', tf.get_default_graph())
    writer.close()
    return logit

def LeNet5_slim(input_tensor, keep_prob, regularizer, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, CATEGORY_NUMBER):

    ######### CNN structure reference #########
    CONV1_DEEP              = 64
    CONV1_SIZE              = 5

    CONV2_DEEP              = 128
    CONV2_SIZE              = 5

    FC_SIZE                 = 512
    ###########################################

    slim = tf.contrib.slim
    input_tensor = tf.reshape(input_tensor,[-1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])

    with tf.variable_scope('layer1-conv1'):
        relu1 = slim.conv2d(input_tensor,CONV1_DEEP,[CONV1_SIZE,1])

    with tf.variable_scope('layer2-conv2'):
        relu2 = slim.conv2d(relu1,CONV1_DEEP,[1,CONV1_SIZE])

    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('layer4-conv3'):
        relu3 = slim.conv2d(pool1,CONV1_DEEP,[CONV2_SIZE,1])

    with tf.variable_scope("layer5-conv4"):
        relu4 = slim.conv2d(relu3,CONV2_DEEP,[1,CONV2_SIZE])

    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, CATEGORY_NUMBER],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [CATEGORY_NUMBER], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.variable_scope('layer-output'):
        logit = tf.reshape(logit,[-1,CATEGORY_NUMBER])

    writer = tf.summary.FileWriter('log', tf.get_default_graph())
    writer.close()
    return logit
