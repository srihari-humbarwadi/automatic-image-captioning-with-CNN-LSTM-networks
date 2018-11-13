
# coding: utf-8

# In[1]:


from models import xception_feature_extractor, caption_model
from tensorflow.python.keras.applications.xception import preprocess_input
from tensorflow.python.keras.utils import multi_gpu_model, Sequence, to_categorical
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import os


# In[2]:


img_height, img_width = 299, 299


# In[3]:


image_dir = 'images/train2014'
def get_image(path:str):
    image = load_img(path, target_size=(img_height, img_width, 3))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image


# In[4]:


with open('wordtoidx.pkl', 'rb') as f:
    word_idx = pickle.load(f)
idx_word = {v:k for k,v in word_idx.items()}
vocab_size = len(idx_word)

def wordtoidx(word:str):
    idx = word_idx[word]
    return idx

def idxtoword(idx:int):
    if idx == 0:
        return ''
    word = idx_word[idx]
    return word


# In[5]:


# fe = xception_feature_extractor()
# image_embeddings = {}
# for image in tqdm(os.listdir('images/train2014')):
#     img = get_image(f'images/train2014/{image}')
#     embedding = fe.predict(np.expand_dims(img, axis=0))
#     image_embeddings[image] = embedding
# with open('image_embeddings.pkl', 'wb') as f:
#     pickle.dump(image_embeddings, f)


# In[6]:


with open('image_embeddings.pkl', 'rb') as f:
    image_embeddings = pickle.load( f)


# In[7]:


try:
    image_names = np.load('image_names.npy')
    encoded_partial_captions = np.load('encoded_partial_captions.npy')
    next_words = np.load('next_words.npy')
#     next_words = to_categorical(next_words, num_classes=21306)
    print('\nloaded training data')
except:
    print('\nError in loading traning data')


# In[8]:


class Data_generator(Sequence):
    def __init__(self, image_embedding_dict, image_names, partial_caps, next_words, batch_size):
        assert len(partial_caps) == len(next_words)
        self.image_embeddings = image_embedding_dict
        self.images = image_names
        self.partial_caps = partial_caps
        self.next_words = next_words
        self.batch_size = batch_size
        self.samples = len(partial_caps)
        print(f'Found {self.samples} datapoints\n')

    def __len__(self):
        return int(np.ceil(len(self.partial_caps) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.samples, self.batch_size)
        batch_ximg, batch_xcap, batch_y = [], [], []
        for i in idx:
            image = self.image_embeddings[self.images[i]][0]
            partial_caption = self.partial_caps[i]
            next_word = self.next_words[i]
            batch_ximg.append(image)
            batch_xcap.append(partial_caption)
            batch_y.append(next_word)
        return [np.array(batch_ximg), np.array(batch_xcap)], np.array(batch_y)


# In[9]:


train_generator = Data_generator(image_embeddings, image_names, encoded_partial_captions, next_words, 512)
steps = train_generator.__len__()


# In[10]:

print(f'Image height = {img_height}')
print(f'Image width = {img_width}\n')
c_model = caption_model(max_len=50, vocab_size=vocab_size, train=True)


# In[11]:


with open('embedding_layer_weights', 'rb') as f:
    embedding_layer_weights = pickle.load(f)
layer = c_model.get_layer('embedding')
try:
    print('Loading embedding layer weights')
    layer.set_weights([embedding_layer_weights])
except:
    print('!!!!! failed to load weights proceede with caution !!!!!')

c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
print(f'Parameters in model = {c_model.count_params()}\n\n')


# In[12]:


tb = TensorBoard(log_dir='logs', write_graph=True)
chkpt = ModelCheckpoint('models/top_weights_p.h5', monitor='acc', save_best_only=True, save_weights_only=True, verbose=1)
callbacks = [tb, chkpt]


# In[ ]:


history = c_model.fit_generator(train_generator, steps_per_epoch=steps, epochs=30, callbacks=callbacks)
c_model.save('models/model.h5')

