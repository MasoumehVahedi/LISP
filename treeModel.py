import numpy as np
from sklearn.linear_model import LinearRegression
from pympler.asizeof import asizeof


class Node:
    def __init__(self, z_ranges, clusters):
        self.z_ranges = z_ranges
        self.clusters = clusters
        self.labels = list(set(clusters))  # store unique cluster labels
        self.left_child = None
        self.right_child = None
        self.internal_model = None
        self.leaf_model = None
        self.cdfs = None  # initialize cdfs attribute to None


class TreeBuilder:
    # We have no constructor
    def build_tree(self, z_ranges, clusters, max_depth):
        if len(clusters) == 0:
            return None
        # Build leaf node
        if max_depth == 0 or len(z_ranges) == 1:
            node = Node(z_ranges, clusters)
            cdfs = [(i + 1) / len(clusters) for i in range(len(clusters))]
            node.cdfs = cdfs
            X = np.array([[z_range[0], z_range[1]] for z_range in z_ranges])
            y = np.array(clusters)
            node.leaf_model = LinearRegression().fit(X, y)
            return node
        else:
            # Initialize Root node
            node = Node(z_ranges, clusters)
            midpoint = (z_ranges[0][0] + z_ranges[-1][1]) / 2.0
            left_clusters = []
            left_Zranges = []
            right_clusters = []
            right_Zranges = []
            overlapping_clusters = []
            overlapping_Zranges = []
            for i in range(len(clusters)):
                if z_ranges[i][1] <= midpoint:
                    left_clusters.append(clusters[i])
                    left_Zranges.append(z_ranges[i])
                elif z_ranges[i][0] > midpoint:
                    right_clusters.append(clusters[i])
                    right_Zranges.append(z_ranges[i])
                else:
                    # This cluster overlaps with both subtrees
                    overlapping_clusters.append(clusters[i])
                    overlapping_Zranges.append(z_ranges[i])

            # Compute CDF and fit regression model for left subtree
            if len(left_clusters) > 0:
                cdfs = [(i + 1) / len(left_clusters) for i in range(len(left_clusters))]
                node.left_child = self.build_tree(left_Zranges, left_clusters, max_depth - 1)
                node.left_child.cdfs = cdfs
                if node.left_child.internal_model is None:
                    X = np.array([[z_range[0], z_range[1]] for z_range in left_Zranges])
                    y = np.array(left_clusters)
                    node.left_child.internal_model = LinearRegression().fit(X, y)
                if node.left_child.leaf_model is None:
                    X = np.array([[z_range[0], z_range[1]] for z_range in left_Zranges])
                    y = np.array(left_clusters)
                    node.left_child.leaf_model = LinearRegression().fit(X, y)

            # Compute CDF and fit regression model for right subtree
            if len(right_clusters) > 0:
                cdfs = [(i + 1) / len(right_clusters) for i in range(len(right_clusters))]
                node.right_child = self.build_tree(right_Zranges, right_clusters, max_depth - 1)
                node.right_child.cdfs = cdfs
                if node.right_child.internal_model is None:
                    X = np.array([[z_range[0], z_range[1]] for z_range in right_Zranges])
                    y = np.array(right_clusters)
                    node.right_child.internal_model = LinearRegression().fit(X, y)
                if node.right_child.leaf_model is None:
                    X = np.array([[z_range[0], z_range[1]] for z_range in right_Zranges])
                    y = np.array(right_clusters)
                    node.right_child.leaf_model = LinearRegression().fit(X, y)

            # Fit regression model for overlapping clusters
            if len(overlapping_clusters) > 0:
                X = np.array([[z_range[0], z_range[1]] for z_range in overlapping_Zranges])
                y = np.array(overlapping_clusters)
                if node.internal_model is None:
                    node.internal_model = LinearRegression().fit(X, y)
                if node.leaf_model is None:
                    node.leaf_model = LinearRegression().fit(X, y)
            return node

    def count_nodes(self, node):
        if node is None:  # base case: if node is None, return 0
            return 0
        else:  # recursive case: return the count of nodes in the left subtree + the count in the right subtree + 1 (for the current node)
            return self.count_nodes(node.left_child) + self.count_nodes(node.right_child) + 1

    def get_tree_size(self, node):
        if node is None:
            return 0
        else:
            return self.get_tree_size(node.left_child) + self.get_tree_size(node.right_child) + get_node_size(node)




def get_node_size(node):
    if node is None:
        return 0
    # size of node object itself
    size = asizeof(node) - asizeof(node.left_child) - asizeof(node.right_child)
    # add the sizes of node's attributes (excluding child nodes)
    attributes = ['z_ranges', 'clusters', 'labels', 'internal_model', 'leaf_model', 'cdfs']
    for attr in attributes:
        if hasattr(node, attr):
            size += asizeof(getattr(node, attr))
    return size