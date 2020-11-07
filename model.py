import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

#the default is relu function
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)
    #return tf.maximum(0.0, x)
    #return tf.nn.tanh(x)
    #return tf.nn.elu(x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] ,tf.shape(y)[3]])], 3)

class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            #fc = tcl.batch_norm(fc)
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = tcl.batch_norm(fc)
                #fc = leaky_relu(fc)
                fc = tf.nn.tanh(fc)
            
            output = tcl.fully_connected(
                fc, 1, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, concat_every_fcl=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            y = z[:,self.input_dim:]
            fc = tcl.fully_connected(
                z, self.nb_units,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
            fc = leaky_relu(fc)
            #fc = tf.nn.dropout(fc,0.1)
            if self.concat_every_fcl:
                fc = tf.concat([fc, y], 1)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
                
                fc = leaky_relu(fc)
                if self.concat_every_fcl:
                    fc = tf.concat([fc, y], 1)
            
            output = tcl.fully_connected(
                fc, self.output_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                #activation_fn=tf.sigmoid
                activation_fn=tf.identity
                )
            #output = tc.layers.batch_norm(output,decay=0.9,scale=True,updates_collections=None,is_training = True)
            #output = tf.nn.relu(output)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#representation and infer cluster label
class Encoder_cluster(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            #return output[:, 0:self.feat_dim], y, logits
            return output, y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            #logits = output[:, self.feat_dim:]
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Attention(object):
    def __init__(self, name, tau=1.0, output_dim=2, nb_layers=3, nb_units=256):
        self.name = name
        self.tau = tau
        self.output_dim = output_dim
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            fc = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            logits = tf.nn.sigmoid(fc)/self.tau
            #logits = fc
            output = tf.nn.softmax(logits)
            return tf.reduce_mean(output,axis=0)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Cluster(object):
    def __init__(self, K, name, nb_layers=2, nb_units=256):
        self.K = K
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            logits = tcl.fully_connected(
                fc, self.K, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            output = tf.nn.softmax(logits)
            return output
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]

            if self.dataset=="mnist":
                z = tf.reshape(z, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                z = tf.reshape(z, [bs, 32, 32, 3])
            conv = tcl.convolution2d(z, 64, [4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #(bs, 14, 14, 32)
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, 128, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 7, 7, 32)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(
                fc, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',is_training=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            y = z[:,-10:]
            #yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = tf.concat([fc, y], 1)

            if self.dataset=='mnist':
                fc = tcl.fully_connected(
                    fc, 7*7*128,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))
            elif self.dataset=='cifar10':
                fc = tcl.fully_connected(
                    fc, 8*8*128,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = conv_cond_concat(fc,yb)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #(bs,14,14,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = tf.nn.relu(conv)
            if self.dataset=='mnist':
                output = tcl.convolution2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            elif self.dataset=='cifar10':
                output = tcl.convolution2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#encoder for images, H()
class Encoder_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',cond=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset=="mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tcl.convolution2d(x,64,[4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, self.nb_units, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, 
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity)
            
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, self.output_dim, 
                activation_fn=tf.identity
                )        
            logits = output[:, -self.nb_classes:]
            y = tf.nn.softmax(logits)
            return output[:, :-self.nb_classes], y, logits     

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


def get_kernel(mat,bd=1.0,kernel_type='Gaussian'):
    
    l2_norm = tf.reduce_sum(mat**2,axis=1)
    size = tf.shape(l2_norm)[0]
    l2_norm = tf.reshape(l2_norm,[size,1])
    A = tf.tile(l2_norm,[1, size])
    A_t = tf.transpose(A)
    diff_mat = tf.sqrt(A+A_t-2*tf.matmul(mat,tf.transpose(mat)))
    if kernel_type == 'Gaussian':
        return tf.math.exp(-diff_mat/(2*bd))
    else:
        return -diff_mat

def get_paired_kernel(mat1, mat2, bd=1.0, kernel_type='Gaussian'):
    #paired similarity matrix between rows of mat1 and mat2
    l2_norm1 = tf.reduce_sum(mat1**2,axis=1)
    l2_norm2 = tf.reduce_sum(mat2**2,axis=0)
    size1 = tf.shape(l2_norm1)[0]
    size2 = tf.shape(l2_norm2)[0]
    l2_norm1 = tf.reshape(l2_norm1,[size1,1])
    l2_norm2 = tf.reshape(l2_norm2,[1,size2])
    A = tf.tile(l2_norm1,[1, size2])
    B = tf.tile(l2_norm2,[size1,1])
    diff_mat = tf.sqrt(A+B-2*tf.matmul(mat1,tf.transpose(mat2)))
    if kernel_type == 'Gaussian':
        return tf.math.exp(-diff_mat/(2*bd))
    else:
        return -diff_mat


def get_corr(y__, gene_idx, re_idx_list, rna_reducer, atac_reducer,y_dim=30):
    #components : (n_comp, org_feat), mean_ : (org_feat_)
    #np.dot(y, self.y_sampler.rna_reducer.components_) + self.y_sampler.rna_reducer.mean_
    #gene_idx, re_idx_list
    rna_mean = rna_reducer.mean_.astype('float32')
    rna_comp = rna_reducer.components_.astype('float32')
    atac_mean = atac_reducer.mean_.astype('float32')
    atac_comp = atac_reducer.components_.astype('float32')
    rna_rec = y__[:,:int(y_dim/2)]
    atac_rec = y__[:,int(y_dim/2):]

    rna_rec_org = tf.matmul(rna_rec,rna_comp)+rna_mean
    atac_rec_org = tf.matmul(atac_rec,atac_comp)+atac_mean
    rna_select_rec_org = tf.transpose(tf.gather(tf.transpose(rna_rec_org),gene_idx))

    atac_select_rec_org = tf.transpose(tf.stack([tf.reduce_sum(tf.transpose(tf.gather(tf.transpose(atac_rec_org),item)),axis=1) \
            for item in re_idx_list]))
    sim = util.get_cosine_similarity(rna_select_rec_org,atac_select_rec_org)
    return tf.reduce_mean(sim)
    

if __name__ == "__main__":
    import sys
    import numpy as np
    import util 
    import hickle as hkl 
    ys = util.scAI_Sampler(n_components=15,mode=3)
    y__ = tf.convert_to_tensor(np.random.normal(size=(32,30)),dtype=tf.float32)
    gene_idx, re_idx_list = hkl.load('index.hkl')
    #gene_idx, re_idx_list = util.find_RE_index(ys.genes, ys.peaks, thred = 100000)
    #hkl.dump([gene_idx, re_idx_list],'index.hkl')
    
    sim = get_corr(y__, gene_idx, re_idx_list, ys.rna_reducer, ys.atac_reducer,y_dim=30)
    print(sim)

    batch_size = 3
    embeds1 = tf.constant([[1, 2],
                            [0, 0],
                            [2, 2]], dtype=tf.float32)
    embeds2 = tf.constant([[1, 1],
                            [0, 1],
                            [1, 1]], dtype=tf.float32)                            
    embeds_combine = tf.concat([embeds1,embeds2],axis=1,name='embeds_combine')
    embeds_stack = tf.stack([embeds1,embeds2],axis=0)
    att_net = Attention(name='att_net', tau=10.0, output_dim=2, nb_layers=3, nb_units=256)
    w = att_net(embeds_combine,reuse=False)
    print(w)
    w1 = tf.constant([0.2,0.8])
    w1 = tf.reshape(w1,[2,1,1])
    print(w1)
    fused_feats = tf.reduce_sum(w1*embeds_stack  ,axis=0)
    assignments = tf.constant([[0.5, 0.5],
                            [0.1, 0.9],
                            [0.2, 0.8]], dtype=tf.float32)
    kernel1 = get_kernel(embeds1)
    kernel2 = get_kernel(embeds2)
    print(kernel1,kernel2)
    kernel_fused = get_kernel(fused_feats)
    kernel_stack = tf.stack([kernel1, kernel2],axis=0)
    kernel_c = tf.reduce_sum(w1 * kernel_stack,axis=0)

    run_config = tf.ConfigProto()
    sess = tf.Session(config=run_config)

    numerator = tf.matmul(tf.matmul(assignments,kernel_fused,transpose_a=True), assignments)
    print('numberator',numerator)
    diag = tf.diag_part(numerator)
    print('diag',diag)
    diag = tf.reshape(diag,[1, tf.shape(diag)[0]])
    denominator = tf.sqrt(tf.matmul(diag,diag,transpose_a=True))
    print('denominator',denominator)
    mat_sc = tf.math.divide(numerator,denominator)
    loss_sc = (tf.reduce_sum(mat_sc)-tf.reduce_sum(tf.diag_part(mat_sc)))/2.0
    print(loss_sc)

    print(diag)

    reg_mat = tf.transpose(get_paired_kernel(tf.transpose(assignments),tf.eye(batch_size))) #(bs, K)

    numerator = tf.matmul(tf.matmul(reg_mat,kernel_fused,transpose_a=True), reg_mat)
    diag = tf.diag_part(numerator)
    diag = tf.reshape(diag,[1, tf.shape(diag)[0]])
    denominator = tf.sqrt(tf.matmul(diag,diag,transpose_a=True))
    mat_sc = tf.math.divide(numerator,denominator)
    loss_reg = (tf.reduce_sum(mat_sc)-tf.reduce_sum(tf.diag_part(mat_sc)))/2.0
    print(diag)
    
    loss_ort = (tf.reduce_sum(tf.matmul(assignments,assignments,transpose_a=True)) - \
        tf.reduce_sum(tf.diag_part(tf.matmul(assignments,assignments,transpose_a=True))))/2.0
    a = tf.constant([[0.2,0.8,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8,9]])
    v = tf.constant([[1.0,2],[2,3]])
    m = tf.constant([1,2])
    b=[0,2]
    c=[5]
    d=[tf.reduce_sum(tf.gather(a,b)),tf.reduce_sum(tf.gather(a,c))]
    print(d)
    aa = tf.gather(tf.transpose(a),b)
    print(sess.run(tf.transpose(aa)))
    d = tf.stack(d)
    print(d)
    print(sess.run(tf.stack([tf.reduce_sum(a,axis=1),tf.reduce_sum(a,axis=1)])))
    #sim = cosine_similarity(embeds1,embeds2)
    print(sess.run(tf.transpose(sim)))
    #print(sess.run(tf.matmul(v,np.array([[1,0],[0,2]],dtype='float32'))))
    #print(sess.run(tf.matmul(assignments,assignments,transpose_a=True)))
    #print(sess.run(loss_ort))




