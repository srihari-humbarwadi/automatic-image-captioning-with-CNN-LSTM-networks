
# coding: utf-8

# In[1]:


from tensorflow.python.keras.layers import Dense, Input, LSTM, Bidirectional, TimeDistributed, concatenate, Lambda, RepeatVector, Embedding, Dropout, add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.xception import Xception, preprocess_input


# In[2]:


def xception_feature_extractor():
    inp = Input(shape=(None, None, 3), name='image_input')
    return Xception(include_top=False, input_tensor=inp, pooling='avg', weights='imagenet')


# In[15]:


def caption_model(max_len=33, vocab_size=10431, train=False, feature_extractor_model=None):
    print(f'Vocab size = {vocab_size}')
    print(f'Max sequence lenth = {max_len}\n')
    if train:
        print('Creating training network. . .\nInput shape=(None,2048)')
        inp_1 = Input(shape=(2048,), name='image_embedding')
    else:
        print('Creating inference network. . .\nInput shape=(None, 299, 299, 3)')
        inp_1 = feature_extractor_model.output
    y = Dense(units=300, activation='relu', name='image_embedding_dense')(inp_1)
    y = RepeatVector(max_len, name='repeat_layer')(y)
    
    inp_2 = Input(shape=(max_len,), name='partial_captions')
    x = Embedding(input_dim=vocab_size+1, output_dim=300, input_length=max_len)(inp_2)
    x = LSTM(units=256, return_sequences=True)(x)
    x = TimeDistributed(Dense(units=300, activation='linear'))(x)
    
    merge_layer = add([y, x], name=f'add_{train}')
    z = Bidirectional(LSTM(units=256, return_sequences=False), name='Bidirectional-LSTM1')(merge_layer)
    out = Dense(units=vocab_size+1, activation='softmax', name='word_output')(z)
    print(f'Successfully created training network. . .\nOutput shape=(None,{vocab_size+1})\n')

    if train:
        return Model(inputs=[inp_1, inp_2], outputs=out, name='Caption-Model')
    return Model(inputs=[feature_extractor_model.input, inp_2], outputs=out, name='c-Model')

