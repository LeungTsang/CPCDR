import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import glob
import networkx as nx
from scipy import interpolate
import os 
import time
import threading


from skimage import filters
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76, deltaE_ciede2000
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import mark_boundaries
from skimage.future import graph

from iterative_infomap_clean import hierarchy_im_downtop, upsample

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))



K = np.array([[0.58, 0, 0.5, 0],
              [0, 1.92, 0.5, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32) 


paths = sorted(glob.glob("data_semantics_inst/training/rgb/*.png"))
#paths = sorted(glob.glob("data_semantics/training/image_2/*.png"))
cmap = plt.get_cmap("prism")
random_color = cmap(np.random.rand(20000))[:,0:3]


def sp_graph(paths):
    time_start = time.time() 
    for i,img_path in enumerate(paths):
        #load data
        #i = 0
        print(id,i, time.time()-time_start)
        #depth_path = "data_semantics_inst/training/disp/"+paths[i][-13:]#[:-4]+"_disp.png"
        #img_path = paths[i]
        save_path = "inst_out/"#+paths[i]
        img_name = img_path.split("/")[-1][:-4]
        img = np.array(pil.open(img_path))#.resize((1024,512),pil.ANTIALIAS))
        img = img.astype(np.float32)/255
        disp = np.array(pil.open(img_path.replace("rgb","disp"))).astype(np.float32)/65535
        #plt.imshow(disp)
        #plt.show()
        disp[disp<0.005] = 0.005
        z = 1/disp
        print(id,i, time.time()-time_start,"rd")
        
        h,w,_ = img.shape
        K_i=K.copy()
        K_i[0, :] *= w
        K_i[1, :] *= h
        inv_K = np.linalg.pinv(K_i)
        y = np.linspace(h-1,0,h,dtype = np.int32)
        x = np.linspace(0,w-1,w,dtype = np.int32)
        x, y = np.meshgrid(x,y)
        
        #get 3d coordinate
        coords = np.stack([x,y,np.ones_like(x)], axis=0).astype(np.float32)
        coords = coords.reshape(3,coords.shape[1]*coords.shape[2])
        coords_3d = (z.reshape(1,h*w)*np.matmul(inv_K[ :3, :3],coords)).reshape(3,h,w)
        coords_2d= coords[:2]
        #coords_3d[1] = coords_3d[1]+coords_3d.min()
        #get superpixel
        
        #print(id,i, time.time()-time_start,"slic")
        sp = slic(img, n_segments = 10000,compactness=10, sigma = 1)
        sp_graph = graph.RAG(sp)
        
        time1 =  time.time()
        num_sp = sp.max()+1
        #attributes
        coords_3d = coords_3d/coords_3d.std(axis=(1,2),keepdims = True)
        grad = np.gradient(coords_3d,axis=(1,2))/coords_3d[2]
    
        #superpixel features
        sp_center_3d = np.zeros((num_sp,3),dtype = np.float32)
        sp_center_2d = np.zeros((num_sp,2),dtype = np.float32)
        sp_color = np.zeros((num_sp,3),dtype = np.float32)
        sp_grad = np.zeros((num_sp,2,3),dtype = np.float32)
        sp_normal = np.zeros((num_sp,3),dtype = np.float32)
 
        sp_flatten = sp.reshape(h*w)
        coords_3d_flatten = coords_3d.reshape(3,h*w).transpose((1,0))
        coords_2d_flatten = coords_2d.reshape(2,h*w).transpose((1,0))
        count = np.bincount(sp_flatten)
        np.add.at(sp_center_3d, sp_flatten, coords_3d_flatten)
        np.add.at(sp_center_2d, sp_flatten, coords_2d_flatten)
        sp_center_3d = sp_center_3d/count.reshape(-1,1)
        sp_center_2d = sp_center_2d/count.reshape(-1,1)

        color_flatten = img.reshape(h*w,3)
        np.add.at(sp_color, sp_flatten, color_flatten)
        sp_color_rgb = sp_color/count.reshape(-1,1)
        sp_color = rgb2lab(sp_color_rgb)

        grad_flatten = grad.reshape(2,3,h*w).transpose((2,0,1))
        np.add.at(sp_grad,sp_flatten,grad_flatten)
        sp_grad = sp_grad/count.reshape(-1,1,1)
        print(sp_grad.shape)
        sp_normal = np.cross(sp_grad[:,1],sp_grad[:,0])
        sp_normal = sp_normal/np.linalg.norm(sp_normal,axis=1,keepdims= True)
    
        #set edge weight
    
        edges = np.array(sp_graph.edges)
        num_edges = edges.shape[0]
        edge_weight = np.zeros((num_edges),dtype=np.float32)

        #color_w,center_3d_w,sup_w,bias = 0.006,36, 2.5, -4
        color_w,center_3d_w,sup_w,bias = 0, 48.0, 00.0, -4.0
        center_3d = sp_center_3d[edges,:]
        color = sp_color[edges,:]
        y = sp_center_3d[edges,1]
        depth = sp_center_3d[edges,2]
        normal_y = sp_normal[edges,1]
    
        #support plane attributes
        
        height = y.mean(axis = -1)
        y_min_id = np.argmin(y,axis = 1)
        y_max_id = 1-y_min_id
        id_order = np.arange(y.shape[0])
        sup = normal_y[id_order, y_min_id]
        sup[sup<0] = 0
        normal_y_diff = (normal_y[id_order,y_min_id]-normal_y[id_order,y_max_id])
        normal_y_diff[normal_y_diff<0] = 0 
        d_sup = sup*normal_y_diff*(height<0)*(height)*(height)
    
        #color distance
        d_color = np.linalg.norm(color[:,1]-color[:,0],axis=1)*(height>0)*(height)
    
        #3d distance
        d_center_3d = np.linalg.norm(center_3d[:,1]-center_3d[:,0],axis=1)/(depth.max(axis = 1)+depth.min(axis = 1))
        weight = (color_w*d_color+center_3d_w*d_center_3d+sup_w*d_sup+bias)
        
        edge_weight = sigmoid(weight)

        communities = hierarchy_im_downtop(g = sp_graph, hierarchy = 0,edge_weight = edge_weight, separate_w = 0.20, img = img, seg = sp, pos = sp_center_2d, color = sp_color_rgb)

        seg = communities[sp]
        seg_img = random_color[seg]
        seg_img = (0.25*seg_img + 0.75*img)
        seg_img = mark_boundaries(seg_img,seg,color=(0,0.75,0))
        plt.imshow(seg_img)
        plt.show()
    
        seg = pil.fromarray(seg.astype(np.uint16))
        seg_im = pil.fromarray((seg_img*255).astype(np.uint8))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        seg.save(save_path+paths[i][-13:])
        seg_im.save(save_path[:-4]+paths[i][-13:]+"_seg.jpg")
        print(id,i, time.time()-time1,"seg")

sp_graph(paths)
