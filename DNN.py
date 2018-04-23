import tensorflow as tf
from config import *
from dataFeed import *
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

print(stopWin)
print(stopLost)

print(dt.datetime.utcnow())
dataSet = loadData(normalize=True,length=300000,offset=30000)
print(dt.datetime.utcnow())

dataSet.test.data.shape


ACTIVATION = tf.nn.sigmoid
N_LAYERS = 3
N_HIDDEN_UNITS = 1000
OUTPUT_LAYER = 2


def fix_seed(seed=1):
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)


def built_net(xs, ys, norm,dropOut):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False, dropOut=False):
        # 添加层功能
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.random_normal([1, out_size]) + 0.1)
        
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        Wx_plus_b_ = tf.cond(tf.equal(dropOut, tf.constant(True))
                              , lambda: tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
                              , lambda: Wx_plus_b)            
        if norm:    # 判断书否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0],   # 想要 normalize 的维度, [0] 代表 batch 维度
                        # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            #mean, var = mean_var_with_update()      # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
            mean, var = tf.nn.moments(xs, axes=[0, 1], keep_dims=False)
            # 将修改后的 mean / var 放入下面的公式
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    fix_seed(1)
    if norm:
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        xs = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)

    layers_inputs = [xs]    # 记录每层的 input

    # loop 建立所有层
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
        )
        layers_inputs.append(output)    # 把 output 加入记录

    # 建立 output layer
    prediction = add_layer(layers_inputs[-1], N_HIDDEN_UNITS, OUTPUT_LAYER, activation_function=None,dropOut=dropOut)

    #cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    #train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    train_op = tf.train.AdadeltaOptimizer(0.02).minimize(cost)
    return [train_op, cost, layers_inputs, prediction]

x = tf.placeholder(tf.float32,[None, historyLength*column])
y = tf.placeholder(tf.float32,[None,OUTPUT_LAYER])
drop = tf.placeholder(tf.bool)


#train_op, cost, layers_inputs, y_predict = built_net(x_flatten, y, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm, y_predict_norm = built_net(x, y, norm=True, dropOut=drop) # with BN

#correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

correct_prediction_norm = tf.equal(tf.argmax(y_predict_norm, 1), tf.argmax(y, 1))
accuracy_norm = tf.reduce_mean(tf.cast(correct_prediction_norm, tf.float32))


def calculateProfit(tradeList):
    close = dataSet.test.rawData[:,3]
    currentPosition = 0
    entryPrice = 0
    totalTrade = 0
    tradeHistory = np.zeros([1, tradeList.shape[1]])
    for trade in range(1, tradeList.shape[1] - 1):
        if(currentPosition == 0):
            if(tradeList[0, trade] == 1 and tradeList[0, trade-1] == 1 and tradeList[0, trade-2] == 1 ):
                currentPosition = 1
                entryPrice = close[trade]
        if(currentPosition == 1):
            #if(tradeList[0, trade] == 0 and tradeList[0, trade-1] == 0 and tradeList[0, trade-2] == 0 and tradeList[0, trade-3] == 0):
            #    currentPosition = 0
            #    tradeHistory[0,trade] = close[trade] - entryPrice
            #    entryPrice = 0
            #    totalTrade = totalTrade+1
            if(close[trade] - entryPrice > stopWin):
                tradeHistory[0,trade] = stopWin
            elif(close[trade] - entryPrice < -stopLost):
                tradeHistory[0,trade] = -stopLost
            else:
                continue
            currentPosition = 0
            entryPrice = 0
            totalTrade = totalTrade + 1
                
                
    return tradeHistory, totalTrade

def getTradeList(y_predict_norm):
    y_oneHot = tf.argmin(y_predict_norm, 1)
    y_oneHot = tf.reshape(y_oneHot,[1,-1])
    return y_oneHot
tradeList = getTradeList(y_predict_norm)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(500000):
        batch_xs, batch_ys = dataSet.train.next_batch(100)        
        sess.run(train_op_norm, feed_dict={x:batch_xs,y:batch_ys})
        if step % 1000 == 0:
            accuracy_norm_ = sess.run(accuracy_norm, feed_dict={x: batch_xs, y: batch_ys, drop:False})
            print('step {}: accuracy train is {}'.format(step, accuracy_norm_))            
            accuracy_norm_ = sess.run(accuracy_norm, feed_dict={x: dataSet.test.data, y: dataSet.test.label, drop:False})
            print('step {}: accuracy test is {}'.format(step, accuracy_norm_))            
                
            tradeList_ = sess.run(tradeList, feed_dict={x:dataSet.test.data,drop:False})            
            profit_, totalTrade_ = calculateProfit(tradeList_)            
            profit_ = profit_.cumsum()
            print("Profit: {}, Trade#: {}".format(profit_[-1],totalTrade_))
            plt.plot(profit_)
            plt.show()
            
            