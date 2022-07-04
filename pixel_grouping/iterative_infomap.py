import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import infomap
from skimage.segmentation import mark_boundaries
import time

cmap = plt.get_cmap('prism')
random_color = cmap(np.random.rand(20000))[:,0:3]

def upsample(img, scale = 1):
    img = np.repeat(img, scale, axis = 0)
    img = np.repeat(img, scale, axis = 1)
    return img

def get_seg_img(img, seg, node_communities):
    h, w = seg.shape
    seg_img = np.zeros((h,w,3),np.float32)
    for i in range(len(node_communities)):
        seg_img[seg==i] = random_color[node_communities[i]]

    seg_img = 0.25*seg_img + 0.75*img
    seg_img[seg==-1] = 0.25*img[seg==-1]
    #seg_img = mark_boundaries(seg_img, seg, color=(1,1,0))
    seg_img = seg_img
    return seg_img

def merge_nodes(g, edge_weight, node_communities, separate_w):
    
    community_edges = node_communities[np.array(list(g.edges))]
    #print(community_edges)
    out_edges = community_edges[:,0]!=community_edges[:,1]
    #community_edges = community_edges[mask]
    #edge_weight = edge_weight[mask]
    community_edges = np.stack((community_edges.min(axis = 1), community_edges.max(axis = 1)),axis = 1)
    print(edge_weight)
    out_small_edges = np.logical_and(edge_weight < separate_w, out_edges)
    node_communities_unique,idx_ = np.unique(community_edges, return_inverse = True)

    connected_node_communities = np.zeros(node_communities_unique.shape[0],dtype=np.bool)
    out_small_edges = np.repeat(out_small_edges, 2, axis = 0)
    np.bitwise_or.at(connected_node_communities, idx_, out_small_edges)

    community_edges = community_edges[out_edges]
    edge_weight = edge_weight[out_edges]
    connected_edges = connected_node_communities[community_edges].all(axis=1)
    connected_community_edges = community_edges[connected_edges]
    edge_weight = edge_weight[connected_edges]
    
    connected_edges_unique, idx = np.unique(connected_community_edges, return_inverse = True, axis = 0)
    next_edge_weight = np.zeros(connected_edges_unique.shape[0])
    count = np.bincount(idx)
    np.add.at(next_edge_weight, idx, edge_weight)
    next_edge_weight = next_edge_weight/count
    
    connected_node_communities_unique = node_communities_unique[connected_node_communities]
    next_node_idx = np.arange(0,connected_node_communities_unique.shape[0],dtype=np.int32)
    next_edge_idx = -np.ones(node_communities_unique.shape[0],dtype=np.int32)
    next_edge_idx[connected_node_communities] = next_node_idx
    next_connected_edges_unique = next_edge_idx[connected_edges_unique]
    #print(next_connected_edges_unique)

    next_g = nx.Graph()
    next_g.add_nodes_from(next_node_idx)
    next_nodes = list(next_g.nodes)
    next_g.add_edges_from(next_connected_edges_unique)
    nx.set_edge_attributes(next_g, dict(zip(list(map(tuple,connected_edges_unique)),next_edge_weight)), "weight")

    return next_g, next_edge_weight, connected_node_communities



def hierarchy_im_downtop(g, hierarchy, edge_weight, separate_w, img, seg, pos, color):
    num_nodes = len(g.nodes)
    num_edges = len(g.edges)
    nodes = list(g.nodes)
    edges = list(g.edges)
    seg_img = get_seg_img(img, seg, nodes)

    #print(num_nodes, num_edges)

    nx.set_edge_attributes(g, dict(zip(edges,1/edge_weight)), "weight")

    #run infomap
    im = infomap.Infomap(silent =True, preferred_number_of_modules = 10, two_level = True, teleportation_probability=0.0)
    im.add_networkx_graph(g)
    im.run()
    communities_n2c = im.get_modules()

    node_communities = np.array(list(communities_n2c.values()))-1
    

    #nx.draw(next_g, next_pos, node_color = next_g_color, edge_color = next_edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 20)
    #nx.draw(g, pos, node_color = g_color, edge_color = edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 20)
    
    #plt.show()

    #communities_unique, idx = np.unique(node_communities, return_inverse = True)

    #get community graph
    community_g, community_edge_weight, connected_communities = merge_nodes(g, edge_weight, node_communities, separate_w)
    
    next_seg = -np.ones_like(seg)
    #next_seg = 
    for i in range(node_communities.shape[0]):
        next_seg[seg==i] = node_communities[i]
    
    print("==="+str(hierarchy)+"===")
    print("community seg and graph")
    #plt.figure(figsize=(20,10))
    #plt.tight_layout()
    #plt.axis("off")
    #plt.imshow(mark_boundaries(np.ones_like(img), upsample(next_seg), color=(0.2,0.2,0.2)))
    #plt.show()
    
    plt.figure(figsize=(20,12))
    plt.subplot(2,2,1)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(seg_img)
    plt.subplot(2,2,3)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(mark_boundaries(seg_img, upsample(next_seg), color=(0,0.75,0)))
    
    
    #sp_flatten = sp.reshape(h*w)
    #coords_3d_flatten = coords_3d.reshape(3,h*w).transpose((1,0))
    #coords_2d_flatten = coords_2d.reshape(2,h*w).transpose((1,0))
    next_pos = np.zeros((connected_communities.shape[0],2),dtype = np.float32)
    next_color = np.zeros((connected_communities.shape[0],3),dtype = np.float32)
    count = np.bincount(node_communities)
    np.add.at(next_pos, node_communities, pos)
    next_pos = next_pos/count.reshape(-1,1)
    next_pos = next_pos[connected_communities]
    np.add.at(next_color, node_communities, color)
    next_color = next_color/count.reshape(-1,1)
    next_color = next_color[connected_communities]
    #plt.figure(figsize=(20,10))
    #plt.tight_layout()
    #plt.axis("off")
    #nx.draw(g, pos, node_color = color, edge_color = edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 8)
    #plt.show()
    
    plt.subplot(2,2,2)
    plt.tight_layout()
    plt.axis("off")
    nx.draw(g, pos, node_color = color, edge_color = edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 10)
    plt.subplot(2,2,4)
    plt.tight_layout()
    plt.axis("off")
    nx.draw(community_g, next_pos, node_color = next_color, edge_color = community_edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 10)
    #nx.draw(g, pos, node_color = g_color, edge_color = edge_weight, edge_cmap = plt.get_cmap('Reds'), node_size = 20)
    plt.show()
    
    
    
    
    disconnected_communities = np.logical_not(connected_communities)
    #communities_idx = np.arange(0,disconnected_communities.sum())
    
    #print(node_communities)
    disconnected_nodes = disconnected_communities[node_communities]
    connected_nodes = np.logical_not(disconnected_nodes)
    disconnected_node_communities = node_communities[disconnected_nodes]
    disconnected_node_communities_unique, idx = np.unique(disconnected_node_communities, return_inverse = True)
    communities_idx = np.arange(0,disconnected_node_communities_unique.shape[0],dtype=np.int32)
    node_communities[disconnected_nodes] = communities_idx[idx]
    next_seg[disconnected_communities[next_seg]] = -1
    print(next_seg.shape)
    next_seg_unique, next_seg = np.unique(next_seg, return_inverse = True)
    print(next_seg.shape)
    next_seg = next_seg.reshape(seg.shape)-1


    if disconnected_node_communities.shape[0] == node_communities.shape[0]:
        return node_communities
    else:
        next_communities = hierarchy_im_downtop(community_g, hierarchy+1, community_edge_weight, separate_w, img, next_seg, next_pos, next_color)
        connected_node_communities = node_communities[connected_nodes]
        connected_node_communities_unique, idx = np.unique(connected_node_communities, return_inverse = True)
        node_communities[connected_nodes] = next_communities[idx]+disconnected_node_communities_unique.shape[0]
        #node_communities[np.logical_not(disconnected_nodes)] = next_communities+disconnected_node_communities_unique.shape[0]


    #print(node_communities)
    return node_communities


    

        
