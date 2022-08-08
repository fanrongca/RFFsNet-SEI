import numpy as np
import pandas as pd
import os
import math
import multiprocessing
import warnings
import tensorflow as tf
import time
import sys
from RFFsNet_SEI_train import _parse_record
from RFFsNet_SEI_train import get_processed_data

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

saved_model_dir = r'E:\Datas\DATA11\GitHub\8types\\'
test_path = valid_path = r'E:\Datas\DATA11\GitHub\8types\Valid.tfrecord'
real_num = 8

valid_num = 800*real_num
BATCH_SIZE = 200
TRAINING_EPOCH = 1
TRAINING_STEPS = int(valid_num/BATCH_SIZE)

Model_Name = 'RFFsNetSEI'
is_drop = False


def sei_predict():
    dataset_test = tf.data.TFRecordDataset(test_path)
    dataset_test = dataset_test.map(_parse_record)
    dataset_test = dataset_test.batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_test = dataset_test.make_one_shot_iterator()
    data_test = iterator_test.get_next()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            signature_key = 'test_signature'
            input_key_x = 'input_x'
            input_key_y = 'input_y'
            output_key = 'logits'
            is_training_key = 'is_training'
            keep_prob_key = 'keep_prob'
            meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], saved_model_dir)
            signature = meta_graph_def.signature_def
            x_tensor_name = signature[signature_key].inputs[input_key_x].name
            y_tensor_name = signature[signature_key].inputs[input_key_y].name
            keep_prob_name = signature[signature_key].inputs[keep_prob_key].name
            is_training_name = signature[signature_key].inputs[is_training_key].name
            output_tensor_name = signature[signature_key].outputs[output_key].name
            x = sess.graph.get_tensor_by_name(x_tensor_name)
            y = sess.graph.get_tensor_by_name(y_tensor_name)
            is_training = sess.graph.get_tensor_by_name(is_training_name)
            if is_drop:
                keep_prob = sess.graph.get_tensor_by_name(keep_prob_name)
            logits = sess.graph.get_tensor_by_name(output_tensor_name)

            Test_Dir = saved_model_dir + r'\\Predict\\'
            if (False == os.path.exists(Test_Dir)):
                os.makedirs(Test_Dir)
            total_time = 0.0
            for epoch in range(TRAINING_EPOCH):
                epoch_time = 0.0
                for i in range(TRAINING_STEPS):
                    data_test_batch = sess.run(data_test)
                    xs_test, imf_Merge, IMF_sum_Merge, Feat1_M_Merge, Feat2_M_Merge, ys_test = get_processed_data(
                        data_test_batch)
                    time_start = time.time()
                    if is_drop:
                        y_output = sess.run(logits, feed_dict={x: xs_test, is_training: False, keep_prob: 1.0})
                    else:
                        y_output = sess.run(logits, feed_dict={x: xs_test, is_training: False})
                    time_end = time.time()
                    infer_time = time_end - time_start
                    total_time += infer_time
                    epoch_time += infer_time
                    fileName1 = Test_Dir + Model_Name + '_' + str(i) + '_true.csv'
                    np.savetxt(fileName1, ys_test, delimiter=",", fmt='%s')
                    fileName2 = Test_Dir + Model_Name + '_' + str(i) + '_valid.csv'
                    np.savetxt(fileName2, y_output, delimiter=",", fmt='%.6f')

                    percent = 100 * (i + 1) * 1.0/TRAINING_STEPS
                    sys.stdout.write('\r'+ "[%.1f%%]" % percent)
                    sys.stdout.flush()
                    del xs_test, ys_test
                print("epoch %d, %f" % (epoch+1, epoch_time))
                print("Total Infer time=%f" % total_time)
                average_infer = total_time/((epoch+1)*1.0)
                FPS = (BATCH_SIZE*TRAINING_STEPS)/average_infer
                print("InferTime,=%.2f,FPS,%.1f " % (average_infer, FPS))
    pass


def get_topk_index(array, topk):
    shape = array.shape
    error_index =[]
    merge_index =[]
    for i in range(shape[0]):
        temp = array[i]
        if((temp==0).all()):
            error_index.append(i)
        c_topk = temp.argsort()[::-1][0:topk]
        sort = np.sort(c_topk)
        merge_index.append(sort)
        pass
    merge_index = np.array(merge_index)
    merge_index.reshape([shape[0], topk])
    return merge_index, error_index
    pass


def accuracy_test(y_real, y_pred):

    temp = y_real - y_pred
    acc = len(temp) - np.count_nonzero(temp)
    return acc/len(temp)


def analyse_predict_file(lableFile, predictFile):
    topk = 1
    y_true = pd.read_csv(lableFile, header=None).values
    y_predict = pd.read_csv(predictFile, header=None).values

    y_pred_topk, erro_index = get_topk_index(y_predict[:, 0:12], topk)
    y_true_topk, _ = get_topk_index(y_true[:, 0:12], topk)

    y_pred_topk.reshape(-1, 1)
    y_true_topk.reshape(-1, 1)

    reslut = np.zeros(y_true.shape[0]*4).reshape(y_true.shape[0], -1)
    Id = np.arange(1, y_true.shape[0]+1)
    Id = Id[:,np.newaxis]
    Id.reshape(-1, 1)
    reslut[:, 0] = Id[:, 0]
    reslut[:, 1] = y_true_topk[:, 0] - y_pred_topk[:, 0]
    reslut[:, 2] = y_true_topk[:, 0]
    reslut[:, 3] = y_pred_topk[:, 0]
    test_acc_topk = accuracy_test(y_true_topk, y_pred_topk)
    temp = os.path.basename(lableFile)
    fileDir = os.path.abspath(os.path.join(os.path.dirname(lableFile), "..")) + "\\Analyse"
    # if not os.path.exists(fileDir):
    #     os.makedirs(fileDir)
    input_path = temp.replace("true", "TopkAcc%.3f" % test_acc_topk)
    input_path = os.path.join(fileDir, input_path)
    np.savetxt(input_path, reslut, delimiter=",", fmt='%s')
    step = temp.split('_')[1]

    # step = int(step)
    step = int(step)
    print("%d,%.3f" % (step, test_acc_topk))
    return step, test_acc_topk


def analyse_sei(files, root):
    handleList = []
    steps = []
    test_acc_topks = []
    for item in files:
        exchange = ""
        temp = fileName = item
        dirname = root
        if "valid" in fileName:
            exchange = fileName.replace("valid", "true")
            fileName = fileName.replace("valid", "")
        if "true" in fileName:
            exchange = fileName.replace("true", "valid")
            fileName = fileName.replace("true", "")
        if fileName in handleList:
            pass
        else:
            handleList.append(fileName)
            item_valid = os.path.join(dirname, exchange)
            item = os.path.join(dirname, item)
            if(os.path.isfile(item_valid) and os.path.isfile(item)):
                if "valid" in temp:
                    step, test_acc_topk = analyse_predict_file(item_valid, item)
                    steps.append(step)
                    test_acc_topks.append(test_acc_topk)
                else:
                    step,test_acc_topk= analyse_predict_file(item, item_valid)
                    steps.append(step)
                    test_acc_topks.append(test_acc_topk)
                    pass
                pass
            pass
    steps = np.array(steps)
    test_acc_topks = np.array(test_acc_topks)
    reslut = np.zeros(steps.shape[0] * 2).reshape(steps.shape[0], -1)
    reslut[:, 0] = steps
    reslut[:, 1] = test_acc_topks
    return reslut


if __name__ == '__main__':

    # get the predicted files
    sei_predict()

    csv_dir = saved_model_dir
    m = 1
    Source_Dir = csv_dir + r'\\Predict\\'
    filePath = csv_dir + r'RMSE.csv'
    fileDir = csv_dir + "\Analyse"
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    resluts = []
    file_list = []
    for root, dirs, files in os.walk(Source_Dir):
        for file in files:
            file_name = os.path.join(root, file)
            file_list.append(file_name)
    n = int(math.ceil(len(file_list) / float(m)))
    pool = multiprocessing.Pool(processes=m)
    for i in range(0, len(file_list), n):
        resluts.append(pool.apply_async(analyse_sei, args=(file_list[i: i + n], csv_dir)))
    pool.close()
    pool.join()
    result = []
    for item in resluts:
        result.append(item.get())
    result = np.array(result)
    result = np.reshape(result, (-1, 2))
    np.savetxt(filePath, result, delimiter=",", fmt='%s')
    print("Acc %.3f" % (np.mean(result[:, 1])))
    pass


