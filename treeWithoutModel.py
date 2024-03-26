
# Node class as provided
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


# TreeBuilder class as provided
class TreeBuilder:
    def build_tree(self, z_ranges, clusters, max_depth):
        # Simplified build_tree function to focus on the search part
        if len(clusters) == 0:
            return None
        if max_depth == 0 or len(z_ranges) == 1:
            node = Node(z_ranges, clusters)
            # Omitting model fitting for simplicity
            return node

        else:
            # Simplified logic for splitting the tree
            midpoint = len(z_ranges) // 2
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
            # Initialize Root node
            node = Node(z_ranges, clusters)
            node.left_child = self.build_tree(left_Zranges, left_clusters, max_depth - 1)
            node.right_child = self.build_tree(right_Zranges, right_clusters, max_depth - 1)
            return node


def search(node, Z):
    if node is None or Z[1] < node.z_ranges[0][0] or Z[0] > node.z_ranges[-1][1]:
        return set()
    if node.left_child is None and node.right_child is None:
        return {cluster_id for z_range, cluster_id in zip(node.z_ranges, node.clusters) if z_range[1] >= Z[0] and z_range[0] <= Z[1]}
    result = set()
    if node.left_child is not None:
        result.update(search(node.left_child, Z))
    if node.right_child is not None and Z[1] >= node.right_child.z_ranges[0][0] and Z[0] <= node.right_child.z_ranges[-1][1]:
        result.update(search(node.right_child, Z))
    return result

