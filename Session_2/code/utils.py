import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
DataPath = './SegNet-Tutorial/CamVid/'

# global params
image_height = 360
image_width = 480
data_shape = image_height * image_width

#Helper function to get image from file
def get_image(mode, index=None):
  with open(DataPath + mode +'.txt') as f:
    txt_list = f.readlines()
    txt_list = [line.strip().split(' ') for line in txt_list]
    file_count = len(txt_list)
    if(index == None):
      index = randint(0, file_count-1)
    if(index > file_count):
      raise Exception("Index can not be greater than population size!")
    img_path = DataPath + txt_list[index][0][15:]
    return cv2.imread(img_path)

#Helper function to get a random image from file
def get_random_image(mode):
  return get_image(mode)
  
  
def get_label(mode, index):
  with open(DataPath + mode +'.txt') as f:
    txt_list = f.readlines()
    txt_list = [line.strip().split(' ') for line in txt_list]
    file_count = len(txt_list)
    if(index > file_count):
      raise Exception("Index can not be greater than population size!")
    img_path = DataPath + txt_list[index][1][15:]
    return Visualizer().create_vis_mat(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

#-------------------------------------------------------------------------------------------------------
# Helper functions to print the layer data
def print_last_layer_info(model):
  '''
    takes a model as input and prints the info
    of the last layer
    
  '''
  print ("last layer information")
  print ("name: " + model.layers[-1].name)
  print ("input shape:" + str(model.layers[-1].input_shape))
  print ("output shape:" + str(model.layers[-1].output_shape))
#-------------------------------------------------------------------------------------------------------

  
#Class for visualization of the neural network output

class Visualizer:
  '''
    Helper class used for visualization of model output on test images
  '''
  def __init__(self):
    # RGB color coding for the respective classes
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]

    self.label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                              Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    class_colours = np.divide(self.label_colours, 255.0)
    # for drawing the labels
    classes = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist", "Unlabelled"]
    self.patches = [ mpatches.Patch(color=class_colours[i], label=classes[i]) for i in range(len(classes)) ]

  def create_vis_mat(self, mat):
    '''
    Function to create visualization matrix given the output
    '''
    
    r = mat.copy()
    g = mat.copy()
    b = mat.copy()
    
    for l in range(0,11):
        r[mat==l] = self.label_colours[l,0]
        g[mat==l] = self.label_colours[l,1]
        b[mat==l] = self.label_colours[l,2]

    rgb = np.zeros((mat.shape[0], mat.shape[1], 3))
    rgb[:,:,0] = (r/255.0)
    rgb[:,:,1] = (g/255.0)
    rgb[:,:,2] = (b/255.0)

    return rgb

  def image_to_data(self, image):
    '''
    Takes image as input and outputs numpy array required by the model
    '''

    img = np.rollaxis(normalized(image),2)
    return np.array([img])
  
  def visualize(self, image, model):
    '''
    Function to visualize output of model on given image
    '''
    gen_seg = self.image_to_data(image)
    output = model.predict(gen_seg)
    label_mat = np.argmax(output[0],axis=1).reshape((image_height, image_width))
    pred = self.create_vis_mat(label_mat)

    plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.imshow(pred)
    plt.figure(2)
    plt.imshow(image)
    
  def visualize_ground_truth(self, image, label):
    '''
      Function to visualize the ground truth annotations for a given
      image and its class label map
    '''
    #label_img = self.create_vis_mat(np.argmax(label_mat),axis=1)
    plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.imshow(label)
    plt.figure(2)
    plt.imshow(image)
    

#-------------------------------------------------------------------------------------------------------

#Utility function for plotting history of training loss and validation loss

def plot_history(training_loss, val_loss, training_acc, val_acc):
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    
    plt.figure(2)
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, val_acc, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show();
    
def normalized(rgb):
  
  '''
    <input>: opencv image object
    <output>: normalized opencv image object
    
    This method runs a histogram equalization of the
    input image to nullify the effect of uneven
    lighting conditions.
    
  '''
  # initialize with a zero matrix
  norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

  # convert RGB to opencv BGR format
  b = rgb[:,:,0]
  g = rgb[:,:,1]
  r = rgb[:,:,2]

  # run a histogram equalization on the image
  # filter out the effects due to uneven lighting conditions
  norm[:,:,0]=cv2.equalizeHist(b)
  norm[:,:,1]=cv2.equalizeHist(g)
  norm[:,:,2]=cv2.equalizeHist(r)

  # return the normalized image
  return norm
    