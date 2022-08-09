# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import shutil
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

real_num = 8
TRAINING_STEPS = 120*real_num
BATCH_SIZE = 200
TRAINING_EPOCH = 100
train_num = 24000*real_num
valid_num = 6000*real_num
CLASSES = 12
input_shape = 64
output_shape2 = CLASSES

initial_learning_rate = 0.01
decay_steps = TRAINING_STEPS
decay_rate = 0.96
staircase = True

log_dir = './tensorboard_log'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
IS_SAVING_GRAPH = False
train_path = r'E:\Datas\DATA11\GitHub\8types\Train.tfrecord'
valid_path = r'E:\Datas\DATA11\GitHub\8types\Valid.tfrecord'
save_model_path = r'E:\Datas\DATA11\GitHub\8types'


def _parse_record(example_proto):
    features = {
        'Data': tf.FixedLenFeature((), tf.string),
        'imf': tf.FixedLenFeature((), tf.string),
        'IMF_sum': tf.FixedLenFeature((), tf.string),
        'Feat1_M': tf.FixedLenFeature((), tf.string),
        'Feat2_M': tf.FixedLenFeature((), tf.string),
        'Label': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features=features)
    return parsed_features


def get_processed_data(data_batch):

    IData_String = data_batch['Data']
    IData_Merge = []
    for idata in IData_String:
        IData = np.fromstring(idata, dtype=np.float64)
        IData_Merge.append(IData)
    IData_Merge = np.array(IData_Merge)

    imf_String = data_batch['imf']
    imf_Merge = []
    for item in imf_String:
        imf = np.fromstring(item, dtype=np.float64)
        imf_Merge.append(imf)
    imf_Merge = np.array(imf_Merge)

    IMF_sum_String = data_batch['IMF_sum']
    IMF_sum_Merge = []
    for item in IMF_sum_String:
        IMF_sum = np.fromstring(item, dtype=np.float64)
        IMF_sum = np.reshape(IMF_sum, (-1))
        IMF_sum_Merge.append(IMF_sum)
    IMF_sum_Merge = np.array(IMF_sum_Merge)

    Feat1_M_String = data_batch['Feat1_M']
    Feat1_M_Merge = []
    for item in Feat1_M_String:
        Feat1_M = np.fromstring(item, dtype=np.float64)
        Feat1_M_Merge.append(Feat1_M)
    Feat1_M_Merge = np.array(Feat1_M_Merge)

    Feat2_M_String = data_batch['Feat2_M']
    Feat2_M_Merge = []
    for item in Feat2_M_String:
        Feat2_M = np.fromstring(item, dtype=np.float64)
        Feat2_M_Merge.append(Feat2_M)
    Feat2_M_Merge = np.array(Feat2_M_Merge)

    Label_String = data_batch['Label']
    Label_Merge = []
    for labeldata in Label_String:
        LabelData = np.fromstring(labeldata, dtype=np.float64)
        Label_Merge.append(LabelData)
    Label_Merge = np.array(Label_Merge, dtype=np.float64)

    IData_Merge = np.reshape(IData_Merge, [BATCH_SIZE, -1, 1])
    imf_Merge = np.reshape(imf_Merge, [BATCH_SIZE, -1])
    IMF_sum_Merge = np.reshape(IMF_sum_Merge, [BATCH_SIZE, -1])
    Feat1_M_Merge = np.reshape(Feat1_M_Merge, [BATCH_SIZE, -1])
    Feat2_M_Merge = np.reshape(Feat2_M_Merge, [BATCH_SIZE, -1])
    Label_Merge = np.reshape(Label_Merge, [BATCH_SIZE, -1])

    return IData_Merge, imf_Merge, IMF_sum_Merge, Feat1_M_Merge, Feat2_M_Merge, Label_Merge


def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('MFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000.0, params.total_parameters))


def train():
    Data, imf, IMF_sum, Feat1_M, Feat2_M, y_, keep_prob, is_training = inputs()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, decay_steps,
        decay_rate, staircase, name='learning_rate')

    logits_imf, logits_IMF_sum, logits_Feat1_M, logits_Feat2_M, logits, CURRENT_MODEL_NAME = \
        RFFsNet_SEI(Data, is_training)

    loss1 = tf.reduce_mean(tf.square(imf-logits_imf)) + tf.reduce_mean(tf.square(IMF_sum-logits_IMF_sum))
    loss2 = tf.reduce_mean(tf.square(Feat1_M-logits_Feat1_M)) + tf.reduce_mean(tf.square(Feat2_M-logits_Feat2_M))
    loss3 = tf.reduce_mean(tf.square(logits-y_))
    loss = loss1 + loss2 + loss3

    tf.add_to_collection("loss", loss)
    tf.add_to_collection("loss1", loss1)
    tf.add_to_collection("loss2", loss2)
    tf.add_to_collection("loss3", loss3)

    train_op1, train_op2, train_op3, train_op = train_optimizer(learning_rate,
                                                                loss1, loss2, loss3, loss, global_step=global_step)
    # Train
    dataset_train = tf.data.TFRecordDataset(train_path)
    dataset_train = dataset_train.map(_parse_record)
    dataset_train = dataset_train.shuffle(buffer_size=train_num).batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_train = dataset_train.make_one_shot_iterator()
    data_train = iterator_train.get_next()

    # Valid
    dataset_valid = tf.data.TFRecordDataset(valid_path)
    dataset_valid = dataset_valid.map(_parse_record)
    dataset_valid = dataset_valid.shuffle(buffer_size=valid_num).batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_valid = dataset_valid.make_one_shot_iterator()
    data_valid = iterator_valid.get_next()

    # region Create the corresponding folder
    Train_Valid_Dir = r'E:\Datas\DATA11\GitHub\8types\Compare\\'
    if os.path.exists(Train_Valid_Dir):
        shutil.rmtree(Train_Valid_Dir)
    os.makedirs(Train_Valid_Dir)
    Test_Dir = r'E:\Datas\DATA11\GitHub\8types\CompareTest\\'
    if os.path.exists(Test_Dir):
        shutil.rmtree(Test_Dir)
    os.makedirs(Test_Dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/gpu:0'):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            count_flops(sess.graph)
            for epoch in range(TRAINING_EPOCH):
                time_start = time.time()
                for i in range(TRAINING_STEPS):
                    data_train_batch = sess.run(data_train)
                    # The training data is converted into parameters that can be input into the training network
                    IData_Merge, imf_Merge, IMF_sum_Merge, Feat1_M_Merge, Feat2_M_Merge, Label_Merge = get_processed_data(data_train_batch)
                    t_op, loss_value,  step = sess.run([train_op, loss, global_step],
                                                        feed_dict={Data: IData_Merge,
                                                                   imf: imf_Merge,
                                                                   IMF_sum: IMF_sum_Merge,
                                                                   Feat1_M: Feat1_M_Merge,
                                                                   Feat2_M: Feat2_M_Merge,
                                                                   y_: Label_Merge, keep_prob: 0.9, is_training: True})

                    pass
                    if i % 20 == 19:
                        data_valid_batch = sess.run(data_valid)
                        # The valid data is converted into parameters that can be used to valid the network
                        IData_Merge, imf_Merge, IMF_sum_Merge, Feat1_M_Merge, Feat2_M_Merge, Label_Merge = \
                            get_processed_data(data_valid_batch)

                        # Calculate the loss of the validation set
                        valid_loss_value, valid_y_output = sess.run([loss, logits],
                                                                    feed_dict={Data: IData_Merge,
                                                                               imf: imf_Merge,
                                                                               IMF_sum: IMF_sum_Merge,
                                                                               Feat1_M: Feat1_M_Merge,
                                                                               Feat2_M: Feat2_M_Merge,
                                                                               y_: Label_Merge, keep_prob: 1.0,
                                                                               is_training: False})

                        fileName = Train_Valid_Dir + CURRENT_MODEL_NAME + '_' + str(step) + '_true.csv'
                        np.savetxt(fileName, Label_Merge, delimiter=",", fmt='%.8f')
                        fileName = Train_Valid_Dir + CURRENT_MODEL_NAME + '_' + str(step) + '_valid.csv'
                        np.savetxt(fileName, valid_y_output, delimiter=",")

                        print("%d,%g,%g"
                              % (step, loss_value, valid_loss_value))
                        del IData_Merge, imf_Merge, IMF_sum_Merge, Feat1_M_Merge, Feat2_M_Merge, Label_Merge
                time_end = time.time()
                print("the %dth epoch use time=%f" % (epoch+1, time_end - time_start))
                is_save = False
                if epoch <= 50:
                    if epoch % 8 == 1:
                        is_save = True
                if epoch > 50:
                    if epoch % 5 == 1:
                        is_save = True
                if is_save:
                    floderName = CURRENT_MODEL_NAME + '_'
                    floderName += datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    floderName += "_Epoch_" + str(epoch+1)
                    filePath = os.path.join(save_model_path, floderName)
                    builder = tf.saved_model.builder.SavedModelBuilder(filePath)
                    inputs_params = {'input_x': tf.saved_model.utils.build_tensor_info(Data),
                                     'input_y': tf.saved_model.utils.build_tensor_info(y_),
                                     'is_training': tf.saved_model.utils.build_tensor_info(is_training),
                                     'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)}
                    outputs = {'logits_imf': tf.saved_model.utils.build_tensor_info(logits_imf),
                               'logits_IMF_sum': tf.saved_model.utils.build_tensor_info(logits_IMF_sum),
                               'logits_Feat1_M': tf.saved_model.utils.build_tensor_info(logits_Feat1_M),
                               'logits_Feat2_M': tf.saved_model.utils.build_tensor_info(logits_Feat2_M),
                               'logits': tf.saved_model.utils.build_tensor_info(logits),
                               }

                    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs_params, outputs,
                                                                                       'test_sig_name')
                    builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature': signature})
                    builder.save()
                    del builder
            pass
        pass
    pass


def CBL(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True, is_sigmoid = False):
    """
    Implementation of the CBL block as defined in Figure 10

    Arguments:
    input_data -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters_shape -- integer, specifying the shape of the middle CONV's window for the main path
    trainable -- Bool,specifying the status of training
    name -- current name of the block
    downsample -- Bool,specifying the status of downsample
    activate -- Bool,specifying the status of activation
    bn -- Bool,specifying the status of Batch Norm
    is_sigmoid -- Bool,specifying the status of Sigmoid

    Returns:
    conv -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    with tf.variable_scope(name):
        if downsample:
            strides = (1, 2, 1, 1)
            padding = 'SAME'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate:
            if not is_sigmoid:
                conv = tf.nn.leaky_relu(conv, alpha=0.1)
            else:
                conv = tf.nn.sigmoid(conv)

    return conv


def ResUnit(X_input, kernel_size, filters, stage, block, stride=2, training=True):
    """
    Implementation of the ResUnit block as defined in Figure 10

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used
    training -- Bool,specifying the status of training

    Returns:
    add_result -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2 = filters

        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(kernel_size, 1),
                                 strides=(stride, stride),
                                 name=conv_name_base+'a', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'a', training=training)
        x = tf.nn.leaky_relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, 1), name=conv_name_base + 'b', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + 'b', training=training)
        x = tf.nn.leaky_relu(x)

        # SHORTCUT PATH
        # Adopt the method in resnetv2
        X_shortcut = tf.layers.conv2d(X_shortcut, filter2, (1, 1),
                                      strides=(stride, stride), name=conv_name_base + 'c')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + 'c', training=training)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x, name=bn_name_base + 'd')
        add_result = tf.nn.leaky_relu(X_add_shortcut)

    return add_result


def RFFsNet_SEI(input, is_training):
    """
    Implementation of the RFFsNet_SEI as defined in Figure 9

    Arguments:
    input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    is_training -- Bool,specifying the status of training

    Returns:
    logits_imf -- output of imf with shape (BATCH_SIZE, -1)
    logits_IMF_sum -- output of IMF_sum(including time modes and spectral modes) with shape (BATCH_SIZE, -1)
    logits_Feat1_M -- output of Feat1(including time features) with shape (BATCH_SIZE, -1)
    logits_Feat2_M -- output of Feat2(including spectral features) with shape (BATCH_SIZE, -1)
    logits2 -- output of classification with shape (BATCH_SIZE, -1)
    CURRENT_MODEL_NAME -- name of the current function
    """
    CURRENT_MODEL_NAME = sys._getframe().f_code.co_name
    x = tf.expand_dims(input, -1)

    basic_filter = 16

    # The beginning of ResBlock1
    # stage 2
    filters = basic_filter
    x = ResUnit(x, kernel_size=3, filters=[filters, filters], stage=2, block='a', stride=1, training=is_training)

    # stage 3
    filters = basic_filter
    x = ResUnit(x, kernel_size=3, filters=[filters, filters], stage=3, block='a', stride=1, training=is_training)

    # output the imf and IMF_sum
    # stage 4
    filters = basic_filter
    x0 = ResUnit(x, kernel_size=3, filters=[filters, filters], stage=4, block='a', stride=1, training=is_training)
    # The end of ResBlock1

    # The Detail of CBF
    x = CBL(x0, (1, 1,  filters, 2*filters), is_training, name='conv_CBF1_1', activate=True, bn=True)
    x = CBL(x, (3, 1,  2*filters, filters), is_training, name='conv_CBF1_2', activate=True, bn=True)
    x1 = CBL(x, (1, 1,  filters, 4), is_training, name='conv_CBF1_3', activate=True, bn=True)

    logits_imf = x1[:, :, :, 0:4]
    logits_imf = tf.reshape(logits_imf, (BATCH_SIZE, -1))

    # The Detail of CBF
    x = CBL(x0, (1, 1,  filters, 2*filters), is_training, name='conv_CBF2_1', activate=True, bn=True)
    x = CBL(x, (3, 1,  2*filters, filters), is_training, name='conv_CBF2_2', activate=True, bn=True)
    x2 = CBL(x, (1, 1,  filters, 1), is_training, name='conv_CBF2_3', activate=True, bn=True)

    logits_IMF_sum = x2[:, :, :, 0]
    logits_IMF_sum = tf.reshape(logits_IMF_sum, (BATCH_SIZE, -1))

    # Add the characteristics of auxiliary branches, and input the mixed network for classification
    x = CBL(x0, (1, 1,  filters, 2*filters), is_training, name='conv_CBF3_1', activate=True, bn=True)
    x = CBL(x, (3, 1,  2*filters, filters), is_training, name='conv_CBF3_2', activate=True, bn=True)
    xc = CBL(x, (1, 1,  filters, 3), is_training, name='conv_CBF3_3', activate=True, bn=True)

    x0_ = tf.concat([x1, x2, xc], axis=3)
    x0_ = CBL(x0_, (1, 1,  8, 8), is_training, name='conv_extra_merge', activate=True, bn=True)

    # The beginning of ResBlock2
    # stage 5
    filters = basic_filter
    x = ResUnit(x0_, kernel_size=3, filters=[filters, filters], stage=5, block='a', stride=2, training=is_training)

    # stage 6
    filters = basic_filter
    x = ResUnit(x, kernel_size=3, filters=[filters, filters], stage=6, block='a', stride=2, training=is_training)

    # stage 7
    filters = basic_filter
    x0_ = ResUnit(x, kernel_size=3, filters=[filters, filters], stage=7, block='a', stride=2, training=is_training)

    x = CBL(x0_, (1, 1,  filters, 2*filters), is_training, name='conv_CBF4_1', downsample=True, activate=True, bn=True)
    x = CBL(x, (3, 1,  2*filters, filters), is_training, name='conv_CBF4_2', downsample=True, activate=True, bn=True)
    x3 = CBL(x, (1, 1,  filters, 3), is_training, name='conv_CBF4_3', downsample=True, activate=True, bn=True, is_sigmoid=True)
    logits_Feat1_M = x3[:, :, :, 0:3]
    logits_Feat1_M = tf.reshape(logits_Feat1_M, (BATCH_SIZE, -1))

    x = CBL(x0_, (1, 1,  filters, 2*filters), is_training, name='conv_CBF5_1', downsample=True, activate=True, bn=True)
    x = CBL(x, (3, 1,  2*filters, filters), is_training, name='conv_CBF5_2', downsample=True, activate=True, bn=True)
    x4 = CBL(x, (1, 1,  filters, 3), is_training, name='conv_CBF5_3', downsample=True, activate=True, bn=True, is_sigmoid=True)
    logits_Feat2_M = x4[:, :, :, 0:3]
    logits_Feat2_M = tf.reshape(logits_Feat2_M, (BATCH_SIZE, -1))

    x = CBL(x0_, (1, 1,  filters, 2*filters), is_training, name='conv_CBF6_1', downsample=True, activate=True, bn=True)
    x = CBL(x, (1, 1,  2*filters, filters), is_training, name='conv_CBF6_2', downsample=True, activate=True, bn=True)
    xc_ = CBL(x, (1, 1,  filters, 2), is_training, name='conv_CBF6_3', downsample=True, activate=True, bn=True)
    x = tf.concat([x3, x4, xc_], axis=3)
    x = CBL(x, (1, 1,  8, 8), is_training, name='conv_extra_merge2', activate=True, bn=True)

    x = tf.reshape(x, (BATCH_SIZE, 8, 1, 1))
    x = CBL(x, (1, 1,  1, filters), is_training, name='conv_CBF7_1', downsample=True, activate=True, bn=True)
    x = CBL(x, (3, 1,  filters, 2*filters), is_training, name='conv_CBF7_2', downsample=True, activate=True, bn=True)
    x = CBL(x, (1, 1,  2*filters, filters), is_training, name='conv_CBF7_3', downsample=True, activate=True, bn=True)
    flatten2 = tf.layers.flatten(x, name='flatten')
    logits2 = tf.layers.dense(flatten2, name='logits', units=output_shape2, activation=tf.nn.sigmoid)

    return logits_imf, logits_IMF_sum, logits_Feat1_M, logits_Feat2_M, logits2, CURRENT_MODEL_NAME


def inputs():
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    Data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, input_shape, 1], name="Data")
    imf = tf.placeholder(tf.float32, shape=[BATCH_SIZE, input_shape*4], name="imf")
    IMF_sum = tf.placeholder(tf.float32, shape=[BATCH_SIZE, input_shape], name="IMF_sum")
    Feat1_M = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3], name="Feat1_M")
    Feat2_M = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3], name="Feat2_M")
    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, output_shape2], name="labels")
    is_training = tf.placeholder(dtype=tf.bool, name='is_trainning')
    return Data, imf, IMF_sum, Feat1_M, Feat2_M, y_, keep_prob, is_training


def train_optimizer(learning_rate,loss1, loss2, loss3, loss, global_step=None):
    """
    Implementation of the train_optimizer

    Arguments:
    learning_rate -- the learning rate
    loss1 -- mode decomposition loss
    loss2 -- feature extraction loss
    loss3 -- classification loss
    loss -- the total loss of the proposed method
    global_step -- the global step of the training procedure
    Returns:
    train_op1 -- the optimizer for loss1 during the training
    train_op2  -- the optimizer for loss2 during the training
    train_op3 -- the optimizer for loss3 during the training
    train_op -- the optimizer for loss during the training
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    update_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    update_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    update_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1, global_step=global_step,
                                                                   var_list=update_list1)
        train_op2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2, global_step=global_step,
                                                                   var_list=update_list2)
        train_op3 = tf.train.AdamOptimizer(learning_rate).minimize(loss3, global_step=global_step,
                                                                   var_list=update_list3)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op1, train_op2, train_op3, train_op


if __name__ == "__main__":
    time_start = time.time()
    train()
    time_end = time.time()
    print("Use time=%f" % (time_end - time_start))


