import numpy as np
import warnings
warnings.filterwarnings("ignore")
import math
xtrain=np.load('mnist_train_images.npy')
ytrain=np.load('mnist_train_labels.npy')
xval=np.load('mnist_validation_images.npy')
yval=np.load('mnist_validation_labels.npy')
xtest=np.load('mnist_test_images.npy')
ytest=np.load('mnist_test_labels.npy')
cost=np.inf

def create_mini_batch(x,y,start_index,end_index,mini_batch_size):#transpose x before passing as parameter
    x_minibatches=[]
    y_minibatches=[]
    num_minibatches=math.ceil(x.shape[0]/mini_batch_size)
    for minibatch in range(num_minibatches):
        x_mini=x[start_index:end_index]
        y_mini=y[start_index:end_index]
        start_index=end_index
        end_index+=mini_batch_size
        x_minibatches.append(x_mini)
        y_minibatches.append(y_mini)
    return x_minibatches,y_minibatches,num_minibatches

def initialize_w_and_b(nodes_every_layer,x_rows):
    w_dict={}
    b_dict={}
    for i in range(len(nodes_every_layer)):
        if i!=len(nodes_every_layer)-1:
            c=(nodes_every_layer[i+1])**-0.5
        else:
            c=10**-0.5
        if i==0:
            w_dict['w1']=np.random.uniform(-c/2,c/2,(nodes_every_layer[i],x_rows))
            b_dict['b1']=np.ones((nodes_every_layer[i],1))*0.01
            
        else:
            w_dict['w'+str(i+1)]=np.random.uniform(-c/2,c/2,(nodes_every_layer[i],nodes_every_layer[i-1]))
            b_dict['b'+str(i+1)]=np.ones((nodes_every_layer[i],1))*0.01
        #if(i==2):
            #print(w_dict['w3'])
    return w_dict,b_dict

def compute_zi(wi,bi,hi):
    #print(wi.shape,bi.shape,x.shape)
    zi=wi.dot(hi)+bi
    #print('\n',zi.shape,wi.shape,hi.shape,bi.shape)
    return zi

def compute_hi(zi):#compute relu
    #print(zi)
    hi=np.maximum(0,zi)
    return hi

def compute_softmax(zi):
    c=zi.T
    for i in range(c.shape[0]):#for every row
        numerator=np.exp(c[i])#e raised to all elements in that row
        #print(numerator)
        denominator=numerator.sum()#sum of all elements in the row
        #print(denominator)
        if(denominator<1e-07):
            denominator=1e-03
        c[i]=(numerator / denominator)#softmax
        #print(i)
        #print(i.sum())
    yhat=c
    return yhat

def forward_prop(x,nodes_every_layer,z_dict,h_dict,yhat_dict,w_dict,b_dict):
    for i in range(len(nodes_every_layer)):
        #print(each_mini_batch.shape)
        if i==0:
            #print("x",x.shape)
            #print("w",w_dict["w1"].shape)
            #print("b1",b_dict["b1"].shape)
            z_dict['z1']=compute_zi(w_dict['w1'],b_dict['b1'],x)
            h_dict['h1']=compute_hi(z_dict['z1'])
            #print(z_and_h['h1'].shape)
        elif (i!=0) and (i!=len(nodes_every_layer)-1):
            z_dict['z'+str(i+1)]=compute_zi(w_dict['w'+str(i+1)],b_dict['b'+str(i+1)],h_dict['h'+str(i)])
            h_dict['h'+str(i+1)]=compute_hi(z_dict['z'+str(i+1)])
            #print('h',i+1,'is',h_dict['h'+str(i+1)])
        else:
            #print("bL",b_dict["b2"].shape)
            z_dict['z'+str(i+1)]=compute_zi(w_dict['w'+str(i+1)],b_dict['b'+str(i+1)],h_dict['h'+str(i)])
            yhat_dict['yhat']=compute_softmax(z_dict['z'+str(i+1)])
    return z_dict,h_dict,yhat_dict
    
def compute_cost(yhat,y):
    m = yhat.shape[0]
    #print(m)
    cost = (y.dot(np.log(yhat)))/(-m)
    cost = np.squeeze(cost)    
    return np.sum(cost)

def relu_prime(zi):
    zi[zi <= 0] = 0
    #zi[zi>0]=1
    return zi

def update_parameters(w_dict,b_dict,gradients,epsilon):
    for i in range(len(list(w_dict.keys()))):
        #print(i)
        #print(w_dict['w'+str(i+1)].shape)
        #print(gradients['dw'+str(i+1)].shape)
        #print(b_dict['b'+str(i+1)].shape)
        #print('gradients',gradients['db'+str(i+1)].shape)
        w_dict['w'+str(i+1)]=w_dict['w'+str(i+1)]-epsilon*gradients['dw'+str(i+1)]
        b_dict['b'+str(i+1)]=b_dict['b'+str(i+1)]-epsilon*gradients['db'+str(i+1)]
        #b_dict['b'+str(i+1)]=b_dict['b'+str(i+1)].reshape(b_dict['b'+str(i+1)].shape[0],1)
        
    #b_dict['b4']=b_dict['b4'].T
    #gradients['dw4']=gradients['dw4'].T
    
    gradients['dw1']=gradients['dw1'].T
    return w_dict,b_dict

def back_prop(y,yhat,h_dict,w_dict,b_dict,nodes_every_layer):
    for i in range(len(nodes_every_layer)):
        #print(i)
        if i==0:
            #print('yhat',yhat.shape)
            #print('y',y.shape)
            g=(y-yhat)
            #print(g.shape)
            gradients['db'+str(len(nodes_every_layer))]=(-1/5000)*np.sum(g,axis=1)
            gradients['dw'+str(len(nodes_every_layer))]=(-1/5000)*(g.T.dot(h_dict['h'+str(len(nodes_every_layer)-1)].T))
            #g=g.T.dot(w_dict['w'+str(len(nodes_every_layer)-i)])
        else:
            g=((g.dot(w_dict['w'+str(len(nodes_every_layer)-i+1)]))*(relu_prime(z_dict['z'+str(len(nodes_every_layer)-i)])).T)
            gradients['db'+str(len(nodes_every_layer)-i)]=(-1/10000)*np.sum(g,axis=1)
            gradients['dw'+str(len(nodes_every_layer)-i)]=(-1/10000)*(g.T.dot(h_dict['h'+str(len(nodes_every_layer)-i-1)].T)).T
            #g=g.dot(w_dict['w'+str(len(nodes_every_layer)-i)])
            
    #b_dict['b4']=b_dict['b4'].T
    #gradients['dw4']=gradients['dw4'].T
    gradients['dw1']=gradients['dw1'].T
    return gradients

def findBestHyperparameters(x,y,batch_sizes,epochs,alphas,epsilons,nodes_every_layer,z_dict,h_dict,yhat_dict,w_dict,b_dict):
    for batch_size in batch_sizes:
        for epoch in epochs:
            for alpha in alphas:
                for epsilon in epsilons:
                    x_minibatches,y_minibatches,num_minibatches=create_mini_batch(x,y,0,batch_size,batch_size)
                    for xval,yval in zip(x_minibatches,y_minibatches):
                        #x_rows=xval.shape[1]
                        for i in range(epoch):
                            z_dict,h_dict,yhat_dict=forward_prop(xval.T,nodes_every_layer,z_dict,h_dict,yhat_dict,w_dict,b_dict)
                            h_dict['h0']=xval.T
                            y=yval
                            yhat=yhat_dict['yhat']
                            cost=compute_cost(yhat,y.T)
                            #print('epoch',i+1)
                            #print('yhat',yhat.shape)
                            #print('y',y.shape)
                            gradients=back_prop(y,yhat,h_dict,w_dict,b_dict,nodes_every_layer)
                            w_dict,b_dict=update_parameters(w_dict,b_dict,gradients,0.2)
                            max_prob_pred = np.argmax(yhat,axis = 1)
                            max_prob_test=np.argmax(y,axis=1)
                            acc=100*np.sum(max_prob_pred==max_prob_test)/size
                            #print('Validation accuracy:',acc)
                            #print(cost)
                            best_hyperparameters=[batch_size,epochs,alphas,epsilons,cost,acc]
                            return best_hyperparameters



nodes_every_layer=[40,10]
x_rows=xval.shape[1]
w_dict,b_dict = initialize_w_and_b(nodes_every_layer,x_rows)
z_dict={}
h_dict={}
yhat_dict={}
gradients={} 
batch_sizes=[5000]
epochs=[100,300]
alphas=[0,0.01]
epsilons=[0.22,0.1]
best_hyperparameters=findBestHyperparameters(xval,yval,batch_sizes,epochs,alphas,epsilons,nodes_every_layer,z_dict,h_dict,yhat_dict,w_dict,b_dict)



x_rows=xtrain.shape[1]
nodes_every_layer=[40,10]
size=10000
w_dict,b_dict = initialize_w_and_b(nodes_every_layer,x_rows)
z_dict={}
h_dict={}
yhat_dict={}
gradients={} 
epsilon=0.22
alpha=0.01  
epochs=300
x_minibatches,y_minibatches,num_minibatches=create_mini_batch(xtrain,ytrain,0,size,size)
x_minibatches.pop()
y_minibatches.pop()
for xtrain,ytrain in zip(x_minibatches,y_minibatches):
    #k=0
    #k+=1
    #if(k==5):
        #break
    x_rows=xtrain.shape[1]
    nodes_every_layer=[40,10]
    
    for i in range(epochs):
        z_dict,h_dict,yhat_dict=forward_prop(xtrain.T,nodes_every_layer,z_dict,h_dict,yhat_dict,w_dict,b_dict)
        #print(z_dict)
        h_dict['h0']=xtrain.T
        y=ytrain
        yhat=yhat_dict['yhat']
        prev_cost=cost
        #print('yhat',yhat.shape)
        #print('y',y.shape)
        cost=compute_cost(yhat,y.T)
        print('epoch',i+1)
        #print('yhat',yhat.shape)
        #print('y',y.shape)
        gradients=back_prop(y,yhat,h_dict,w_dict,b_dict,nodes_every_layer)
        w_dict,b_dict=update_parameters(w_dict,b_dict,gradients,alpha)
        max_prob_pred = np.argmax(yhat,axis = 1)
        max_prob_test=np.argmax(y,axis=1)
        acc=100*np.sum(max_prob_pred==max_prob_test)/size
        print('Validation accuracy:',acc)
print('\nBest Hyperparameters are\n batch size=',size,'\nepsilon=',epsilon,'\nalpha=',alpha,'\nepochs=',epochs,'\nlayers=',len(nodes_every_layer))

#TEST
z_dict={}
h_dict={}
yhat_dict={}
x_rows=xtest.shape[1]
wbest=w_dict
bbest=b_dict
a=[]
#print(w_dict,b_dict)
x_minibatches,y_minibatches,num_minibatches=create_mini_batch(xtest,ytest,0,size,size)
for xtest,ytest in zip(x_minibatches,y_minibatches):
#bbest['b1']=bbest['b1'][:10000]
    z_dict,h_dict,yhat_dict=forward_prop(xtest.T,nodes_every_layer,z_dict,h_dict,yhat_dict,wbest,bbest)
    #print(ytest.shape)
    test=ytest
    hat=yhat_dict['yhat']
    max_prob_pred = np.argmax(hat,axis=1)
    max_prob_test=np.argmax(test,axis=1)
    acc=(np.sum(max_prob_pred==max_prob_test))
    a.append(acc)
print('\nTesting accuracy:',sum(a)/100,'\n')