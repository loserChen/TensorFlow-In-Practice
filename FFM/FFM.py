import tensorflow as tf
import numpy as np
import os


input_x_size = 20
field_size = 2

vector_dimension = 3

# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 1

all_data_size = 1000

alpha = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"


def createTwoDimensionWeight(input_x_size,field_size,vector_dimension):          #初始化w2
    weights = tf.truncated_normal([input_x_size,field_size,vector_dimension])   #默认生成均值为0，标准差为1的[input_x_size,field_size,vector_dimension]维度的张量

    tf_weights = tf.Variable(weights)

    return tf_weights

def computation(input_x,input_x_field,TwoWeights):
    thirdValue=tf.Variable(0.0,dtype=tf.float32)
    input_shape=input_x_size
    for i in range(input_shape-1):
        featureIndex1 = i             #对应每个x1的特征序号
        fieldIndex1 = int(input_x_field[i])         #对应特征x1的field序号
        for j in range(i + 1, input_shape):
            featureIndex2 = j         #对应每个x2的特征序号
            fieldIndex2 = int(input_x_field[j])        #对应特征x2的field序号
            vectorLeft = tf.convert_to_tensor([[featureIndex1, fieldIndex2, i] for i in range(vector_dimension)])     #转换成张量
            weightLeft = tf.gather_nd(TwoWeights, vectorLeft)         #取对应位置的值，只不过是在张量上
            weightLeftAfterCut = tf.squeeze(weightLeft)         #消除维度为1的shape

            vectorRight = tf.convert_to_tensor([[featureIndex2, fieldIndex1, i] for i in range(vector_dimension)])
            weightRight = tf.gather_nd(TwoWeights, vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut, weightRightAfterCut))

            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(xi, xj))

            secondItemVal = tf.multiply(tempValue, product)

            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

        return thirdValue

def gen_data():
    labels = [-1,1]
    y = [np.random.choice(labels,1)[0]for _ in range(all_data_size)]        #表示在【-1，1】中选择一个数，后面加【0】是为了取值，而不是保持array类型
    x_field = [i // 10 for i in range(input_x_size)]
    x = np.random.randint(0,2,size=(all_data_size,input_x_size))

    return x,y,x_field

if __name__=='__main__':
    global_step = tf.Variable(0, trainable=False)
    trainx, trainy, trainx_field = gen_data()

    input_x = tf.placeholder(tf.float32, [input_x_size])
    input_y = tf.placeholder(tf.float32)

    lambda_v = tf.constant(0.001, name='lambda_v')

    weight = createTwoDimensionWeight(input_x_size,  # 创建二次项的权重变量
                                           field_size,
                                           vector_dimension)  # n * f * k

    y_ = computation(input_x, trainx_field, weight)

    l2_norm = tf.reduce_sum(tf.multiply(lambda_v, tf.pow(weight, 2)))

    loss = tf.log(1 + tf.exp(-input_y * y_)) + l2_norm

    train_step = tf.train.AdagradOptimizer(learning_rate=alpha,initial_accumulator_value=1).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)        #只保留最后的一个模型
    max_acc=100000
    is_train=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(10):
            if is_train:
                for t in range(all_data_size):
                    input_x_batch = trainx[t]
                    input_y_batch = trainy[t]
                    predict_loss, _, steps = sess.run([loss, train_step,global_step],
                                                        feed_dict={input_x: input_x_batch, input_y: input_y_batch})
                    print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                            .format(step=steps, predict_loss=predict_loss))
                    global_step+=1
                    if predict_loss<max_acc:
                        max_acc=predict_loss
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
            else:
                model_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH+'/')
                saver.restore(sess, model_file)
                for t in range(all_data_size):
                    val_loss, yhat = sess.run([loss, y_], feed_dict={input_x: trainx[t],input_y: trainy[t]})
                    print("loss on training batch is {predict_loss} ,prediction is {yhat},real y is {y}"
                          .format(predict_loss=val_loss,yhat=yhat,y=trainy[t]))