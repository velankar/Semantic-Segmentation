import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from os import listdir
from os.path import isfile, join
train_image_files = [f for f in listdir("image/train") if isfile(join("image/train", f))]
train_label_files = [f for f in listdir("label/train") if isfile(join("label/train", f))]
#print(len(train_image_files))
#print(train_label_files)


def get_train_data():
    data = np.empty([244,352,1216,3])
    label = np.empty([244,352,1216])
    for i in range(len(train_image_files)):
        data[i] = cv2.imread("image/train/"+train_image_files[i])
        with open("label/train/"+train_label_files[i],"rb") as fo:
        	label[i] = pickle.load(fo, encoding='bytes')
    return data,label

train_data,train_label = get_train_data()
train_data = train_data.astype(int)
train_label = train_label.astype(int)
train_label*=(train_label>0)
traindata = train_data[:199]
dataval = train_data[199:244]
trainlabel = train_label[:199]
vallabel = train_label[199:244]
num_examples = traindata.shape[0]
epochs_completed =0
index_in_epoch =0

def next_batch(batch_size):

    global traindata
    global trainlabel
    global epochs_completed
    global index_in_epoch
    index_in_epoch +=batch_size
    index = np.random.randint(low=0, high=traindata.shape[0], size=batch_size)
    x_train = traindata[index]
    y_train = trainlabel[index]
    if index_in_epoch>num_examples:
        index_in_epoch =0
        epochs_completed+=1

    return x_train,y_train

x = tf.placeholder(tf.float32,shape=[None,352,1216,3])
y_true = tf.placeholder(tf.float32,shape=[None,352,1216])
def init_weights(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
def init_bias(shape):
    initializer=tf.zeros_initializer()
    return tf.Variable(initializer(shape))
def conv2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')
def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def convolution_layer(input_x, shape):
    W =init_weights(shape)
    b =init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)



convo1_1 = convolution_layer(x,[3,3,3,64])
convo1_2 = convolution_layer(convo1_1,[3,3,64,64])

pool1 = maxpool(convo1_2)

convo2_1 = convolution_layer(pool1,[3,3,64,128])
convo2_2 = convolution_layer(convo2_1,[3,3,128,128])

pool2 = maxpool(convo2_2)

convo3_1 = convolution_layer(pool2,[3,3,128,256])
convo3_2 = convolution_layer(convo3_1,[3,3,256,256])
convo3_3 = convolution_layer(convo3_2,[3,3,256,256])

pool3 = maxpool(convo3_3)

convo4_1 = convolution_layer(pool3,[3,3,256,512])
convo4_2 = convolution_layer(convo4_1,[3,3,512,512])
convo4_3 = convolution_layer(convo4_2,[3,3,512,512])

pool4 = maxpool(convo4_3)

convo5_1 = convolution_layer(pool4,[3,3,512,512])
convo5_2 = convolution_layer(convo5_1,[3,3,512,512])
convo5_3 = convolution_layer(convo5_2,[3,3,512,512])
pool5 = maxpool(convo5_3)
print("after pool",pool5.shape)
# converting FC
convo6 = convolution_layer(pool5,[7, 7, 512, 4096])
convo7 = convolution_layer(convo6,[1,1,4096,4096])


convo8 = convolution_layer(convo7,[1,1,4096,1])
deconv = tf.layers.conv2d_transpose(convo8,1,[64,64],32,padding='same')
print("deconv shape is ",deconv.shape)
# unflatening
# pred = convo8
 
#pred_reshape = tf.reshape(deconv, [-1])
#gt_reshape = tf.reshape(y_true, [-1])
mask = (y_true >= 0)
print(mask)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.boolean_mask(y_true,mask),logits=tf.boolean_mask(deconv[:,:,:,0],mask)))
train = tf.train.MomentumOptimizer(0.001,0.99).minimize(loss)
pred_probab = tf.nn.sigmoid(deconv)

#loss tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_reshape,logits=pred_reshape),tf.float32))
loss_list_train=[]
loss_list_validation=[]
epochs =[]
error_train =0.0
error_validation=0.0
totaltrainerr=0.0
with tf.Session() as sess:
    i=0
    j=0
    global epochs_completed
    sess.run(tf.global_variables_initializer())
    while i<1990*2:
        batch = next_batch(1)
        #print(batch[1].shape)
        _,error_train = sess.run([train,loss],feed_dict={x: batch[0], y_true: batch[1]})
        error_validation += sess.run(loss,feed_dict = {x:dataval[:2],y_true:vallabel[:2]})
        totaltrainerr+=error_train
        if i%99 == 0:
            #error_validation = sess.run(loss,feed_dict = {x:dataval[:5],y_true:vallabel[:5]})
            #loss_list_train.append(error_train/98)
            #loss_list_validation.append(error_validation/98)
            print("Train loss :", (totaltrainerr/99))
            totaltrainerr=0.0
            print("Validation loss :", (error_validation/99))
            error_validation=0.0
        i+=1
    error_validation = sess.run(loss,feed_dict = {x:dataval[:2],y_true:vallabel[:2]})
    print("After complete training",error_validation)
    pred_label = sess.run(pred_probab, feed_dict={x: dataval[:2], y_true: vallabel[:2]})
    print(pred_label[0].shape)
    pred_label[pred_label > 0.5] = 1
    pred_label[pred_label<=0.5] =0
    print (pred_label[:,:,:,0])
    TP = np.sum(np.logical_and(pred_label[:,:,:,0] == 1, vallabel[:2] == 1))
    TN = np.sum(np.logical_and(pred_label[:,:,:,0] == 0, vallabel[:2] == 0))
    FP = np.sum(np.logical_and(pred_label[:,:,:,0] == 1, vallabel[:2] == 0))
    FN = np.sum(np.logical_and(pred_label[:,:,:,0] == 0, vallabel[:2] == 1))
    IOU = float(TP)/(TP+FP+FN)
    print("IOU: ",IOU)
