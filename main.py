import tensorflow as tf
import numpy as np
const_r=100
const_p=80
const_d=50
const_n=2000
const_lambda=1
const_gamma=0.4
truth_mu=np.random.randn(const_r)*0.0
#truth_beta=np.random.randn(const_p,const_r)
truth_beta=np.random.choice([0.0,1.0,-3.0],size=(const_p,const_r),p=[0.8,0.15,0.05])
X=np.random.randn(const_n,const_p)
noise=np.random.randn(const_n,const_r)*5

X_mean=np.mean(X,axis=0)
X_cov=np.cov(X)

step=X-X_mean
nstep=np.zeros((const_n,const_r))
for i in range(const_n):
    nstep[i]=np.dot(truth_beta.T,step[i])
Y=nstep+noise+truth_mu

T_X=tf.convert_to_tensor(X,dtype=tf.float32)
T_Y=tf.convert_to_tensor(Y,dtype=tf.float32)

T_A=tf.get_variable("A",shape=(const_p-const_d,const_d),dtype=tf.float32,trainable=True)
T_GA=tf.concat([tf.eye(const_d),T_A],axis=0)
#T_W=tf.ones(shape=const_p-const_d,dtype=tf.float32)
T_W=tf.get_variable("W",shape=const_p-const_d,dtype=tf.float32,trainable=False,initializer=tf.initializers.ones)
XY=np.hstack((X,Y))
Total_Cov=np.cov(XY)

SX=Total_Cov[:const_p,:const_p]
SX_I=np.linalg.inv(SX)
SY=Total_Cov[const_p:const_p+const_r,const_p:const_p+const_r]
SXY=Total_Cov[:const_p,const_p:const_p+const_r]
SX_Y=SX-np.dot(SXY,np.dot(np.linalg.inv(SY),SXY.T))

T_SX_Y=tf.convert_to_tensor(SX_Y,dtype=tf.float32)
T_SX_I=tf.convert_to_tensor(SX_I,dtype=tf.float32)
loss1=-2*tf.linalg.logdet(tf.matmul(T_GA,T_GA,transpose_a=True))
loss2=tf.matmul(T_GA,T_SX_Y,transpose_a=True)
loss2=tf.matmul(loss2,T_GA)
loss2=tf.linalg.logdet(loss2)
loss3=tf.matmul(T_GA,T_SX_I,transpose_a=True)
loss3=tf.matmul(loss3,T_GA)
loss3=tf.linalg.logdet(loss3)
lossr=const_lambda*tf.reduce_sum(tf.multiply(T_W,tf.sqrt(tf.reduce_sum(tf.square(T_A),axis=1))))
loss=loss1+loss2+loss3+lossr
train=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(loss)
max_reweight=3
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    Bestloss=float('inf')
    BestA=None
    while(True):
        try:
            a,l,_=sess.run([T_A,loss,train])
        except:
            print(sess.run(T_GA))
            print(sess.run(loss))
            raise
        if(l<Bestloss):
            BestA=a
            Bestloss=l
            print("New best loss:%.4f"%l)
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
