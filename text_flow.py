#!/usr/local/bin/python
import h5py
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.misc import imread
import networkx as nx
import cv2
import numpy as np
import random
def construct_reach_graph(bbox,**kwargs):

    if 't_v' not in kwargs:
        kwargs['t_v'] = 0.6
    if 't_s' not in kwargs:
        kwargs['t_s'] = 0.2

    def meet_t_v(box1, box2):
        h1=box1[3]-box1[1]+1
        h2=box2[3]-box2[1]+1
        overlaps=min(box2[3], box1[3])-max(box1[1], box2[1])
        overlaps=overlaps if overlaps>0 else 0
        if float(overlaps)/min(h1, h2)>kwargs['t_v']:
            return True
        return False

    def meet_t_s(box1, box2):
        h1=box1[3]-box1[1]+1
        h2=box2[3]-box2[1]+1
        if float(abs(h1-h2))/min(h1, h2)<kwargs['t_s']:
            return True
        else:
            return False

    G = nx.DiGraph()
    bbox = sorted(bbox,key=lambda x: x[0])
    sorted_bbox = np.array(bbox, dtype=np.float64)
    length = len(sorted_bbox)
    for i in range(length):
        G.add_node(i)
    iter_list = [(i,j) for i in range(length) for j in range(length) if i<j ]
    for i,j in iter_list:
        if meet_t_s(sorted_bbox[i],sorted_bbox[j]) and meet_t_v(sorted_bbox[i],sorted_bbox[j]):
            G.add_edge(i,j)

    return G,sorted_bbox

def constrcut_flow_graph(reach_graph,sorted_bbox,flow_graph=None,scores=None):
    def to_left_node(index):
        return 2*index+1

    def to_right_node(index):
        return 2*index+2

    def scale_to_int(score):
        MAX_VALUE = 10000
        return int(score*MAX_VALUE)

    def compute_box_center(box):
        center_y = (box[3]+box[1])/2
        center_x = (box[2]+box[0])/2
        return center_x,center_y

    def get_data_cost(score):
        return scale_to_int(score)

    def get_smoothness_cost(box1, box2):
        alpha=0.0
        h1=box1[3]-box1[1]+1
        h2=box2[3]-box2[1]+1
        x1, y1=compute_box_center(box1)
        x2, y2=compute_box_center(box2)
        d=((x1-x2)**2+(y1-y2)**2)**0.5/min(h1,h2)
        s=float(abs(h1-h2))/min(h1, h2)
        return scale_to_int(alpha*d+(1-alpha)*s)

    def get_entry_cost(index,reach_graph,scores):
        reach_node_costs = [scores[left] for left, right in reach_graph.edges() if right == index]
        sorted_costs = sorted(reach_node_costs)
        if len(sorted_costs) > 0:
            cost = scale_to_int(-sorted_costs[-1])
        else:
            cost = 0
        return cost

    def get_exit_cost(index,reach_graph,scores):
        reach_node_costs = [scores[right] for left, right in reach_graph.edges() if left == index]
        sorted_costs = sorted(reach_node_costs)
        if len(sorted_costs) > 0:
            cost = scale_to_int(-sorted_costs[-1])
        else:
            cost = 0
        return cost

    def overlaps(box1,box2):
        h1=box1[3]-box1[1]+1
        h2=box2[3]-box2[1]+1
        overlaps=min(box2[3], box1[3])-max(box1[1], box2[1])
        overlaps=overlaps if overlaps>0 else 0
        return float(overlaps)/h2

    length = len(sorted_bbox)
    scores = [-1 for i in range(length)]

    if flow_graph == None:
        #box
        print 'construct graph'
        flow_graph = nx.DiGraph()
        ENTRY_NODE = -1
        flow_graph.add_node(ENTRY_NODE,demand = -1)
        EXIT_NODE = -2
        flow_graph.add_node(EXIT_NODE,demand = 1)
        # data cost
        for index in range(length):
            left_node = to_left_node(index)
            right_node = to_right_node(index)
            flow_graph.add_node(left_node)
            flow_graph.add_node(right_node)
            data_cost = get_data_cost(scores[index])
            flow_graph.add_edge(left_node,right_node,weight=data_cost,capacity = 1)
        # smoothness cost
        for left,right in reach_graph.edges():
            left_node = to_right_node(left)
            right_node = to_left_node(right)
            smoothness_cost = get_smoothness_cost(sorted_bbox[left],sorted_bbox[right])
            flow_graph.add_edge(left_node,right_node,weight=smoothness_cost,capacity = 1)
        # entry cost
        for index in range(length):
            entry_cost = get_entry_cost(index,reach_graph,scores)
            left_node = to_left_node(index)
            flow_graph.add_edge(ENTRY_NODE,left_node,weight=entry_cost,capacity = 1)
        # exit cost
        for index in range(length):
            exit_cost = get_exit_cost(index,reach_graph,scores)
            right_node = to_right_node(index)
            flow_graph.add_edge(right_node,EXIT_NODE,weight=exit_cost,capacity = 1)
    box_inds=[]
    flowDict = nx.min_cost_flow(flow_graph)
    flowCost = nx.min_cost_flow_cost(flow_graph)
    for index in range(length):
        left_node=to_left_node(index)
        right_node=to_right_node(index)
        try:
            find = [node for node, value in flowDict[left_node].iteritems() if value == 1]
        except:
            find = []
        if len(find)>0:
            box_inds.append(index)
    # find node overlaps over 0.5
    remove_inds = box_inds
    '''
    for index in box_inds:
        print index
        node_neighbor_right = [right for left, right in reach_graph.edges() if left == index]
        node_neighbor_left = [left for left, right in reach_graph.edges() if right == index]
        for node_index in node_neighbor_right:
            if overlaps(sorted_bbox[node_index],sorted_bbox[index]) > 0.5:
                remove_inds.append(node_index)
        for node_index in node_neighbor_left:
            if overlaps(sorted_bbox[node_index],sorted_bbox[index]) > 0.5:
                remove_inds.append(node_index)
    '''
    # remove node for next iteration
    for index in remove_inds:
        left_node = to_right_node(index)
        right_node = to_left_node(index)
        if left_node > 0:
            flow_graph.remove_node(left_node)
        if right_node > 0:
            flow_graph.remove_node(right_node)
    return box_inds,flowCost,flow_graph



def text_flow(bbox):
    flow_list = []
    reach_graph,sorted_bbox = construct_reach_graph(box_array)
    result_index,flowCost,flow_graph = constrcut_flow_graph(reach_graph,sorted_bbox,flow_graph = None,scores=None)
    flow_list.append(result_index)
    count = 0
    while flowCost<0 and len(flow_graph.nodes()) != 2:
        result_index,flowCost,flow_graph = constrcut_flow_graph(reach_graph,sorted_bbox,flow_graph,scores=None)
        flow_list.append(result_index)
        count = count + 1
    flow_graph = None
    return flow_list,sorted_bbox

import time
for i in range(233):

    image = imread('test_original/img_%d.jpg'%(i+1))
    with open("result/res_img_{}.txt".format(i+1),"r")as file:
            coord = file.read().splitlines()
    height,width = image.shape[:2]
    image = imresize(image,(500,500))
    box_list = []
    for item in coord:
        xmin,ymin,xmax,ymax = item.split(',')
        xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
        cv2.rectangle(image, (int(xmin/width*500.) ,int(ymin/height*500.)), (int(xmax/width*500.),int(ymax/height*500.)), (0,255,0),2)
        box_list.append((xmin,ymin,xmax,ymax))
    box_array = np.array(box_list)
    start = time.clock()
    flow_list, sorted_bbox= text_flow(box_array)
    print (time.clock() - start)*1000
    r = lambda: random.randint(0,255)
    for flow in flow_list:
        color = (r(),r(),r())
        for index in flow:
            xmin,ymin,xmax,ymax = sorted_bbox[index]
            cv2.rectangle(image, (int(xmin/width*500.) ,int(ymin/height*500.)), (int(xmax/width*500.),int(ymax/height*500.)), color,2)
    plt.imshow(image)
    plt.show()
