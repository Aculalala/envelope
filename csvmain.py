import tensorflow as tf
import numpy as np
def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx
const_d=5
const_lambda=1
const_gamma=0.4
X = np.genfromtxt('x.csv', delimiter=',')
Y = np.genfromtxt('y.csv', delimiter=',')
const_p=X.shape[1]
const_r=Y.shape[1]
split_control=int(0.9*X.shape[0])
trainX=X[:split_control]
trainY=Y[:split_control]
testX=X[split_control:]
testY=Y[split_control:]
T_X=tf.placeholder(name="X",shape=(None,const_p),dtype=tf.float32)
T_Y=tf.placeholder(name="Y",shape=(None,const_r),dtype=tf.float32)

T_A=tf.get_variable("A",shape=(const_p-const_d,const_d),dtype=tf.float32,trainable=True)
T_GA=tf.concat([tf.eye(const_d),T_A],axis=0)
#T_W=tf.ones(shape=const_p-const_d,dtype=tf.float32)
T_W=tf.get_variable("W",shape=const_p-const_d,dtype=tf.float32,trainable=False,initializer=tf.initializers.ones)
T_XY=tf.concat((T_X,T_Y),axis=1)
T_Total_Cov=tf_cov(T_XY)

T_SX=T_Total_Cov[:const_p,:const_p]
T_SX_I=tf.linalg.inv(T_SX)
T_SY=T_Total_Cov[const_p:const_p+const_r,const_p:const_p+const_r]
T_SXY=T_Total_Cov[:const_p,const_p:const_p+const_r]
T_SX_Y=T_SX-tf.matmul(T_SXY,tf.matmul(tf.linalg.inv(T_SY),tf.transpose(T_SXY)))
#tf.linalg.logdet
loss1=-2*tf.log(tf.linalg.det(tf.matmul(T_GA,T_GA,transpose_a=True)))
loss2=tf.matmul(T_GA,T_SX_Y,transpose_a=True)
loss2=tf.matmul(loss2,T_GA)
loss2=tf.log(tf.linalg.det(loss2))
loss3=tf.matmul(T_GA,T_SX_I,transpose_a=True)
loss3=tf.matmul(loss3,T_GA)
loss3=tf.log(tf.linalg.det(loss3))
lossr=const_lambda*tf.reduce_sum(tf.multiply(T_W,tf.sqrt(tf.reduce_sum(tf.square(T_A),axis=1))))
loss=loss1+loss2+loss3+lossr
train=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(loss)
max_reweight=10
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    Bestloss=float('inf')
    BestA=None
    while(True):
        try:
            a,l,_=sess.run([T_A,loss,train],feed_dict={T_X:trainX,T_Y:trainY})
        except:
            print(sess.run(T_GA))
            print(sess.run(loss))
            raise
        if(l<Bestloss):
            BestA=a
            Bestloss=l
            tl=sess.run(loss,feed_dict={T_X:testX,T_Y:testY})
            print("New best loss:%.4f,%.4f"%(l,tl))
            
        else:
            if(max_reweight>0):
                max_reweight=max_reweight-1
                print("%d reweight left."%max_reweight)
                newweight=sess.run(tf.pow(tf.sqrt(tf.reduce_sum(tf.square(T_A),axis=1)),-const_gamma))
                sess.run(tf.assign(T_W,newweight))
                sess.run(tf.assign(T_A,BestA))
                Bestloss=float('inf')
            else:
                break
    print(sess.run(T_A))
