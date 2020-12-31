import os
import pandas as pd
import cv2
import numpy as np

class TaskSpecificDataset():
    
    def __init__(self, root: str):
        self.root = root
        self.data = []
        self.features = []
        self.images = []
        self.__traverse_folder__(self.root)
        self.__filter_dataset__()
    
    def get_features(self):
        return self.features
    
    def get_images(self):
        return self.images
    
    def __traverse_folder__(self, path: str):
        for entry in os.scandir(path):
            if entry.is_file():
                self.data.append(entry.name)
    
    def __filter_dataset__(self):
        self.features = []
        self.images = []
        
        for item in self.data:
            if item.__contains__(".csv"):
                self.features.append(item)
            elif item.__contains__(".jpg"):
                self.images.append(item)
    
class PandasDataset():
    
    def __init__(self, dataset, root):
        self.root = root 
        self.csv_dataset = dataset
        self.dataset = pd.DataFrame(data={})
        self.__parse_csv_to_dataframe__()
        
    def get_data_frame(self):
        return self.dataset
        
    def __parse_csv_to_dataframe__(self): 
        for item in self.csv_dataset: 
            new_df = pd.read_csv(os.path.join(self.root, item))
            self.dataset = self.dataset.append(new_df)
            
class ImageDataset():
    
    def __init__(self, root, data):
        self.root = root
        self.data = data
    
    def __load_and_resize__(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        max_width = 224
        max_height = 224
        
        scale_factor_width = max_width / img.shape[1]
        scale_factor_height = max_height / img.shape[0]

#         scale_percent = 60 # percent of original size
#         width = int(img.shape[1] * scale_percent / 100)
#         height = int(img.shape[0] * scale_percent / 100)
#         dim = (width, height)
    
        dim = (int(img.shape[1] * scale_factor_width), int(img.shape[0] * scale_factor_height))
        
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        return resized
    
    def __load_and_scale_to_default__(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        width = 480
        height = 720
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        return resized

    # TODO consider to add padding to remain proportions
    def get_images(self):
        result = []
        for item in self.data:
            # TODO resize to the smallest dimension in list or/and padding image to the median dimensions
            # convert to bytes, mat.shape is height, width, channel
            mat = self.__load_and_resize__(os.path.join(self.root, item + ".jpg"))
            result.append(mat.reshape(-1))
                    
        result = np.asarray(result)
        
        return result
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        