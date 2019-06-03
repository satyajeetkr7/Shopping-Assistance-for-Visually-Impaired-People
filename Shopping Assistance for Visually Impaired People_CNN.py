from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os.path

labels = pd.read_csv("D:\\index2\\UPC_index.txt",sep="\t")
print(labels.shape)
print(labels.head(3))
#
tag=[]
train_image = []
for i in (range(1,120)):
    j=1
    while(j!=0):
        file_path='D:\\inVitro\\'+str(i)+'\\web\\PNG\\'+'web'+str(j)+'.png'
        if os.path.isfile(file_path):
            img = image.load_img(file_path, target_size=(28,28,1), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            train_image.append(img)
            tag.append(labels.iloc[i]['product_name'])
            #print(i,":",labels.iloc[i]['product_name'])
            j=j+1
        else:
            j=0

print("tag length:",len(tag))
print("Total InVitro Images: ",len(train_image))
list_of_tuples=list(zip(train_image,tag))
df = pd.DataFrame(list_of_tuples, columns = ['img', 'label'])
print(df.shape)

X = np.array(train_image)

y=pd.get_dummies(df['label']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print("Train Images:",X_train.shape[0])
print("Test Images:",X_test.shape[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(119, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

tag1=[]
test_image = []
for i in (range(1,120)):
    j=1
    while(j!=0):
        file_path='D:\\inSitu\\'+str(i)+'\\video\\video'+str(j)+'.png'
        if os.path.isfile(file_path):
            img = image.load_img(file_path, target_size=(28,28,1), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            test_image.append(img)
            tag1.append(labels.iloc[i]['product_name'])
            j=j+1
        else:
            j=0

#print("tag length:",len(tag))
print("Total InSitu Images: ",len(train_image))

list_of_tuples1=list(zip(test_image,tag1))
df1 = pd.DataFrame(list_of_tuples1, columns = ['img', 'label'])
print(df1.shape)
test = np.array(test_image)
ytest=pd.get_dummies(df1['label']).values

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





