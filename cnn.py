import keras
from keras.layers import *
from keras.models import Sequential,Model,load_model
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer

import tensorflow as tf
import numpy as np
from scipy import stats
import subprocess

def get_neff(fasta):
  # gather information about fasta
  ref_seq = ""
  ref_cut = ""
  neff = 0.0
  gremlin_neff = subprocess.run(["./gremlin","-i",fasta,"-only_neff"],stdout=subprocess.PIPE)
  for line in str(gremlin_neff.stdout).split("\\n"):
      var = line.split(" ")
      if len(var) > 1:
          if var[1] == "SEQ": ref_seq = var[2]
          if var[1] == "CUT": ref_cut = var[2]
          if var[1] == "NEFF": neff = float(var[2])
  return(ref_seq,ref_cut,neff)

a2n = {'-':0,'X':0,'B':0,'O':0,'U':0,'Z':0,'J':0,
       'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,
       'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,
       'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}
def aa2num(seq):
    return np.array([a2n[a] for a in seq])

def ex0(x):
  return np.expand_dims(x,0)

def normalize(v):
  v = stats.boxcox(v - np.amin(v) + 1.0)[0]
  v_mean = np.mean(v)
  v_std = np.std(v)
  return((v-v_mean)/v_std)

def pp_2D(aa_data,pp_idx_data,pp_data):
  ln = aa_data.shape[0]
  pp = np.zeros((ln,ln))

  pp[pp_idx_data] = normalize(pp_data)
  pp += pp.T
  return np.expand_dims(pp,-1)

def mask_2D(aa_data,pp_ref_data):
  ln = aa_data.shape[0]
  mask = np.zeros((ln,ln))
  mask[pp_ref_data[None,:],pp_ref_data[:,None]] = 1
  return np.expand_dims(mask,-1)

def nf_ln(aa_data,pp_ref_data,neff):
  ln = aa_data.shape[0]
  nf_ln = (np.log(neff),np.log(pp_ref_data.shape[0]))
  return np.reshape(np.tile(nf_ln,ln*ln),(ln,ln,2))

def get_data(aa_data,pp_data,pp_ref_data,pp_idx_data,neff):
  mtx = pp_2D(aa_data,pp_idx_data,pp_data)
  mask = np.ones_like(mtx)
  yield [ex0(aa_data),ex0(mtx),ex0(mask),ex0(nf_ln(aa_data,pp_ref_data,neff))]

#######################################################################################

class Lookup(Layer):
    def __init__(self, cat, filters,
                 kernel_regularizer=None,
                 symm_cat=None,
                 **kwargs):
      self.cat = cat
      self.fil = filters
      self.symm_cat = symm_cat
      super(Lookup, self).__init__(**kwargs)
      self.kernel_regularizer = regularizers.get(kernel_regularizer)
    def build(self, input_shape):
      self.W = self.add_weight(name='W',
                       shape=(self.cat,self.fil),
                       initializer='uniform',
                       trainable=True,
                       regularizer=self.kernel_regularizer)
      if(self.symm_cat != None):
        self.W = tf.reshape(self.W,(self.symm_cat,self.symm_cat,self.fil))
        self.W = 0.5 * (self.W + tf.transpose(self.W,[1,0,2]))
        self.W = tf.reshape(self.W,(self.cat,self.fil))
      super(Lookup, self).build(input_shape)

    def call(self, x):
      return tf.gather(self.W,tf.cast(x,tf.int32))
    def compute_output_shape(self, input_shape):
      return input_shape + (self.fil,)


def tp_concat_se(x):
  a = tf.transpose(x,[0,3,1,2])
  b = tf.transpose(x,[0,3,2,1])
  A = tf.transpose(tf.matrix_band_part(a,-1,0) + tf.matrix_band_part(b,0,-1) - tf.matrix_band_part(a,0,0),[0,2,3,1])
  B = tf.transpose(tf.matrix_band_part(a,0,-1) + tf.matrix_band_part(b,-1,0) - tf.matrix_band_part(a,0,0),[0,2,3,1])
  return tf.concat((A,B),-1)

def pairwise_idx_se(x):
  ln = tf.shape(x)[1]
  x = tf.squeeze(x,-1)
  a = tf.tile(x[:,:,None],(1,1,ln))
  b = tf.tile(x[:,None,:],(1,ln,1))
  A = tf.matrix_band_part(a,-1,0) + tf.matrix_band_part(b,0,-1) - tf.matrix_band_part(a,0,0)
  B = tf.matrix_band_part(a,0,-1) + tf.matrix_band_part(b,-1,0) - tf.matrix_band_part(a,0,0)
  return tf.stack((A,B),-1)

def seqsep_se(x):
  r = tf.map_fn(lambda y: tf.cast(tf.range(0,tf.shape(y)[0]),tf.float32),x)
  val = (r[:,:,None]-r[:,None,:])
  return tf.expand_dims(val,-1)

########################################################################################
K.clear_session()

win = 9
lays_cc = []
for l in range(4):
	lays_cc.append(
	Convolution2D(
		20,(win,win),activation='elu',
		kernel_initializer='lecun_normal',
		kernel_regularizer=regularizers.l2(1e-6),
		name='lay_cc'+str(l+1)))

ti = Dense(10,activation='elu',kernel_initializer='lecun_normal',name="t_input")
to = Dense(1,activation="sigmoid",kernel_initializer='lecun_normal',name="t_output")

mod_ll = Dense(1,activation='tanh',use_bias=False,name="mod_ll")
mod_nf = Dense(1,activation='linear',name="mod_nf")
mod_ln = Dense(1,activation='linear',name="mod_ln")

en_lookup = Lookup(cat=21,filters=2,kernel_regularizer=regularizers.l2(0.0001))
en_lookup_pair = Lookup(cat=441,symm_cat=21,filters=1,kernel_regularizer=regularizers.l2(0.0001))

########################################################################################
def mk_model(lays=4):
  aa = Input(shape=(None,1), name="aa")
  idx = Lambda(pairwise_idx_se)(aa)

  aa_i = en_lookup(Lambda(lambda x: x[...,0])(idx))
  aa_j = en_lookup(Lambda(lambda x: x[...,1])(idx))
  aa_ij = en_lookup_pair(Lambda(lambda x: 21*x[...,0]+x[...,1])(idx))
  aa_en = concatenate([aa_i,aa_j,aa_ij])

  pp = Input(shape=(None,None,1),name="pp")
  nf_ln = Input(shape=(None,None,2),name="nf_ln")
  nf = mod_nf(Lambda(lambda x: x[...,0,None])(nf_ln))
  ln = mod_ln(Lambda(lambda x: x[...,1,None])(nf_ln))
  mask = Input(shape=(None,None,1),name="mask")

  A = concatenate([aa_en,pp,nf,ln])
  A = ti(A)

  #############
  # PADDING TIME
  pad = int((lays)*(win-1)/2)
  pad_shape = [[0,0],[pad,pad],[pad,pad],[0,0]]
  A = Lambda(lambda x: tf.pad(x,pad_shape,constant_values=0.0))(A)
  #############

  for i in range(lays):
    A = concatenate([A,mod_ll(Lambda(seqsep_se)(A))])
    A = lays_cc[i](A)

  A = Lambda(tp_concat_se)(A)
  A = to(A)
  cc_mask = Multiply(name="cc")([A,mask])

  return Model(inputs=[aa,pp,mask,nf_ln],outputs=[cc_mask])

########################################################################################
model = mk_model()
model.load_weights('model_D_16Nov2018_n1.weights')
model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'], loss_weights=[1.0])
########################################################################################
