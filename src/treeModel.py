import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




class Node:
    def __init__(self, z_ranges, clusters):
        self.z_ranges = z_ranges
        self.clusters = clusters
        self.labels = list(set(clusters))  # store unique cluster labels
        self.left_child = None
        self.right_child = None
        self.internal_model = None
        self.leaf_model = None
        self.cdfs = None


class TreeBuilder:
    def __init__(self, global_percentage, capacity_node):
        self.global_percentage = global_percentage
        self.capacity_node = capacity_node


    def are_all_z_ranges_identical(self, z_ranges):
        """Check if all z_ranges are identical."""
        first_z_range = z_ranges[0]
        return all(z_range == first_z_range for z_range in z_ranges)


    def buildTreeModel(self, z_ranges, clusters):
        if len(clusters) == 0:
            return None

        if self.are_all_z_ranges_identical(z_ranges) and len(z_ranges) > self.capacity_node:
            return self.handleIdenticalZRanges(z_ranges, clusters)
        elif len(z_ranges) <= self.capacity_node:
            return self.buildLeafNode(z_ranges, clusters)
        else:
            return self.buildInternalNode(z_ranges, clusters)


    def buildLeafNode(self, z_ranges, clusters):
        node = Node(z_ranges, clusters)
        cdfs = [(i + 1) / len(clusters) for i in range(len(clusters))]
        node.cdfs = cdfs
        X = np.array([[z_range[0], z_range[1]] for z_range in z_ranges])
        y = np.array(clusters)
        node.leaf_model = LinearRegression().fit(X, y)
        self.evaluateModel(node.leaf_model, X, y, z_ranges)
        return node


    def buildInternalNode(self, z_ranges, clusters):
        node = Node(z_ranges, clusters)
        midpoint = (z_ranges[0][0] + z_ranges[-1][1]) / 2.0
        left_clusters, left_Zranges, right_clusters, right_Zranges, overlapping_clusters, overlapping_Zranges = self.splitClusters(z_ranges, clusters, midpoint)

        # Compute CDF and fit regression model for left subtree
        if len(left_clusters) > 0:
            cdfs = [(i + 1) / len(left_clusters) for i in range(len(left_clusters))]
            node.left_child = self.buildTreeModel(left_Zranges, left_clusters)
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
            node.right_child = self.buildTreeModel(right_Zranges, right_clusters)
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

    def handleIdenticalZRanges(self, z_ranges, clusters):
        # Since all z_ranges are identical and we can't split them based on value,
        # split them into two groups arbitrarily to reduce the problem size.
        half_index = len(clusters) // 2
        left_clusters = clusters[:half_index]
        right_clusters = clusters[half_index:]
        left_Zranges = z_ranges[:half_index]
        right_Zranges = z_ranges[half_index:]

        if len(left_Zranges) > 0:
            self.buildLeafNode(left_Zranges, left_clusters)
        if len(right_Zranges) > 0:
            self.buildLeafNode(right_Zranges, right_clusters)


    def splitClusters(self, z_ranges, clusters, midpoint):
        left_clusters, left_Zranges, right_clusters, right_Zranges, overlapping_clusters, overlapping_Zranges = [], [], [], [], [], []
        for i, z_range in enumerate(z_ranges):
            if z_range[1] <= midpoint:
                left_clusters.append(clusters[i])
                left_Zranges.append(z_range)
            elif z_range[0] > midpoint:
                right_clusters.append(clusters[i])
                right_Zranges.append(z_range)
            else:
                # This cluster overlaps with both subtrees
                overlapping_clusters.append(clusters[i])
                overlapping_Zranges.append(z_range)

        print(f"Original size: {len(z_ranges)}, Left size: {len(left_Zranges)}, Right size: {len(right_Zranges)}")
        return left_clusters, left_Zranges, right_clusters, right_Zranges, overlapping_clusters, overlapping_Zranges


    def evaluateModel(self, model, X, y, z_ranges):
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        error_bound = self.calErrorBound(z_ranges)
        #print(f"Model MSE: {mse: .4f}, Error Bound: {error_bound:.4f}, Within Bound: {mse <= error_bound}")


    def calErrorBound(self, z_ranges):
        global_min_z = min(z_ranges, key=lambda x: x[0])[0]
        global_max_z = max(z_ranges, key=lambda x: x[1])[1]
        global_range = global_max_z - global_min_z
        return global_range * self.global_percentage
