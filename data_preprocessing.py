
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import shutil
from sklearn.model_selection import train_test_split


##################### Reading the data ########################

data = pd.read_csv('FullIJCNN2013/gt.txt', sep=";", header=None)
data.columns = ["img", "x1", "y1", "x2", "y2", "id"]

tmp = data.copy()


print "Data has image files with traffic signs numbers:", len(data['img'].unique())
print "Data has traffic signs class numbers:", len(data['id'].unique())
print "Data has traffic signs instance numbers:", data['id'].count()


##################### Changing all the files to JPEG ######################

## Uncomment this part if you don't already have the JPEG images in the folder

# for i in range(data.shape[0]):
#     data['img'][i] = data['img'][i].replace('ppm','jpg')


# train_img_dir = 'FullIJCNN2013'
# png_img_dir = 'jpg_FullIJCNN2013'

# if os.path.exists(png_img_dir):
#     shutil.rmtree(png_img_dir)
# os.makedirs(png_img_dir) 

# for img_name in os.listdir(train_img_dir):
# #for img_name in ["00000.ppm","00001.ppm","00002.ppm"]:
#     if img_name[-3:] == "ppm":
#         img_path = os.path.join(train_img_dir, img_name)
#         img = Image.open(img_path)
#         png_path = os.path.join(png_img_dir, img_name[:-3]+'jpg')
#         print (png_path)
#         img.save(png_path)
#         #img.show()



################### Selecting the positive and negative classes ######################

## class 1 refers to red_round_labels
## class 0 refers to negative samples
red_round_labels = [0,1,2,3,4,5,7,8,9,10,15,16]   ## prohibitory classes
data['id'] = np.where(tmp['id'].isin(red_round_labels), 1, 0)
labels = data['id'].values
full_data = data.drop('id',1)


################## Preparing the train and test data #####################

x_train, x_test, y_train, y_test = train_test_split(full_data, labels, test_size=0.2, random_state=42)

train_data = []
test_data = []

## here we extract the traffic sign, resize it to 48 x 48 and perform constrast enhancement

data_dir = 'jpg_FullIJCNN2013'

for i in range(x_train.shape[0]): 
    curr_im = os.path.join(data_dir, x_train['img'].iloc[i])
    img = Image.open(curr_im)
    cropped_rect = (x_train['x1'].iloc[i], x_train['y1'].iloc[i], x_train['x2'].iloc[i], x_train['y2'].iloc[i])
    crop_im = img.crop(cropped_rect)
    ## resize here to 48 x 48
    crop_im = crop_im.resize((48, 48), Image.ANTIALIAS)
    crop_im = ImageOps.autocontrast(crop_im)
    train_data.append(np.array(crop_im))
    

for i in range(x_test.shape[0]): 
    curr_im = os.path.join(data_dir, x_test['img'].iloc[i])
    img = Image.open(curr_im)
    cropped_rect = (x_test['x1'].iloc[i], x_test['y1'].iloc[i], x_test['x2'].iloc[i], x_test['y2'].iloc[i])
    crop_im = img.crop(cropped_rect)
    ## resize here to 48 x 48
    crop_im = crop_im.resize((48, 48), Image.ANTIALIAS)
    crop_im = ImageOps.autocontrast(crop_im)
    test_data.append(np.array(crop_im))
    
train_data = np.array(train_data)
test_data = np.array(test_data)


##################### One hot encoding of labels ####################

num_train = train_data.shape[0]
num_test = test_data.shape[0]
num_classes = 2

train_label = np.zeros((num_train, num_classes), dtype=np.int8)
test_label = np.zeros((num_test, num_classes), dtype=np.int8)

for i in range(len(y_train)):
    if y_train[i] == 1:
        train_label[i][1] = 1
    else:
        train_label[i][0] = 1
        
for i in range(len(y_test)):
    if y_test[i] == 1:
        test_label[i][1] = 1
    else:
        test_label[i][0] = 1



count = 0
for i in range(len(y_train)):
    if y_train[i] == 1:
        count += 1

#################### Class distribution in training data ###################

print "Number of red round signs in training data: ", count
print "Number of negatives in training data: ", num_train - count


## saving the training and testing data
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('train_labels.npy', train_label)
np.save('test_labels.npy', test_label)


