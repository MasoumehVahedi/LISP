import numpy as np
import pickle
import sys

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon



def get_list_of_coord_polygons(geom):
    coords = list(geom.exterior.coords)
    return coords


def polygon_area(xs, ys):
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def polygon_centroid(xs, ys):
    xy = np.array([xs, ys])
    center_point = np.dot(xy + np.roll(xy, 1, axis=1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys) / (6 * polygon_area(xs, ys))
    return center_point


def get_mbb(sorted_cluster):
    # polygons is a list of the 100 polygons in the cluster
    polygons = [Polygon(coords) for coords in sorted_cluster]
    # Convert the list of polygons to a MultiPolygon object
    cluster = MultiPolygon(polygons)
    # Calculate the minimum bounding box of the cluster
    minx, miny, maxx, maxy = cluster.bounds
    mbb = [(minx, miny), (maxx, maxy)]
    # Create a rectangle object from the bounding box coordinates
    #bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    return mbb


def calculate_bounding_box(rectangles):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for polygon, rectangle in rectangles:
        x_min = min(x_min, rectangle[0])
        y_min = min(y_min, rectangle[1])
        x_max = max(x_max, rectangle[2])
        y_max = max(y_max, rectangle[3])
    mbb = [(x_min, y_min), (x_max, y_max)]
    return mbb



def get_node_size(node):
    # Calculate the size of the node in bytes
    size = sys.getsizeof(node)
    # Adjust the attributes according to your specific implementation
    size += sys.getsizeof(node.z_ranges)
    size += sys.getsizeof(node.clusters)
    size += sys.getsizeof(node.labels)
    size += sys.getsizeof(node.left_child)
    size += sys.getsizeof(node.right_child)
    size += sys.getsizeof(node.internal_model)
    size += sys.getsizeof(node.leaf_model)
    size += sys.getsizeof(node.cdfs)
    return size


def get_model_index_size(root):
    if root is None:
        return 0
    # Calculate the size of each node recursively
    size = get_node_size(root)
    # Add the size of child nodes
    size += get_model_index_size(root.left_child)
    size += get_model_index_size(root.right_child)
    return size


def get_rtree_index_size(idx):
    # Calculate the size of the index in bytes
    size = sys.getsizeof(idx)
    # Calculate the size of entries within the index
    num_entries = sum(1 for _ in idx.intersection(idx.bounds, objects=True))
    size += num_entries * sys.getsizeof(idx.bounds)
    return size


def save_clusters(clusters, filename="clusters.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(clusters, f)

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def save_hashTable(model, filename="hashTable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def get_size(d):
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        if isinstance(value, dict):
            size += get_size(value)
        else:
            size += sys.getsizeof(value)
    return size


