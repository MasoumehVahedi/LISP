import pickle
import gzip
import gc
import shelve
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.cluster import Birch
from sklearn.linear_model import LinearRegression

from ConfigParam import Config
from ZAdress import MortonCode
from utils import *


class LISP:
    def __init__(self, polygons, page_size):
        self.polygons = polygons
        self.page_size = page_size
        self.config = Config()
        self.leaf_count = 0
        self.internal_count = 0

        self.X = np.array([self.compute_features(polygon) for polygon in self.polygons], dtype=np.float32)
        self.clusters, self.cluster_labels = self.get_clusters()

    def extract_features(self, polygon):
        return [len(polygon.exterior.coords)]

    def get_byte_size(self, polygon):
        return len(polygon[0].exterior.coords) * 16

    def regression_model(self, X, page_numbers):
        y = page_numbers.reshape(-1, 1)
        regressor = LinearRegression().fit(X, y)
        y_pred = regressor.predict(X).reshape(-1)
        return np.maximum(y_pred, 0)

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj)

    @staticmethod
    def loads(serialized_data):
        return pickle.loads(serialized_data)

    def save_pages_to_disk(self, filename="pages.bin"):
        page_map = []
        with open(filename, "wb") as f:
            for page in self.pages:
                start_pos = f.tell()
                pickle.dump(page, f)
                end_pos = f.tell()
                page_map.append((start_pos, end_pos))
        with open("page_map.pkl", "wb") as f:
            pickle.dump(page_map, f)

    def get_disk_pages(self):
        byte_sizes_gen = (self.get_byte_size(polygon) for cluster in self.sorted_clusters for polygon in cluster)
        X = np.array([[byte_size] for byte_size in byte_sizes_gen])
        byte_locations = np.cumsum(X)
        page_numbers = byte_locations // self.page_size
        y_pred = self.regression_model(X, page_numbers)

        # 1. Collect Data
        vertex_counts = [len(polygon) for cluster in self.sorted_clusters for polygon in cluster]
        byte_sizes = [self.get_byte_size(polygon) for cluster in self.sorted_clusters for polygon in cluster]

        # 2. Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(vertex_counts, byte_sizes, color='blue', alpha=0.5)

        # Fitting a linear regression line
        m, b = np.polyfit(vertex_counts, byte_sizes, 1)
        plt.plot(vertex_counts, m * np.array(vertex_counts) + b, color='red')

        plt.title("Relationship between Vertex Counts and Byte Sizes")
        plt.xlabel("Vertex Count")
        plt.ylabel("Byte Size")
        plt.grid(True)
        plt.show()

        # Calculate R-squared value
        correlation_matrix = np.corrcoef(vertex_counts, byte_sizes)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        print(f"R-squared value: {r_squared:.2f}")

        self.pages = []
        self.cluster_hash_tables = defaultdict(dict)

        i = 0
        for j, cluster in enumerate(self.sorted_clusters):
            polygon_page_nums_cluster = {}
            for k, (original_polygon, bounding_box_polygon) in enumerate(cluster):
                bbp_tuple = tuple(bounding_box_polygon)
                page_index = max(int(y_pred[i]), 0)
                while len(self.pages) <= page_index:
                    self.pages.append([])
                self.pages[page_index].append((i, self.dumps(original_polygon)))
                #simplified_polygon = original_polygon.simplify(tolerance=0.5)
                # polygon_page_nums_cluster[bbp_tuple] = (simplified_polygon.wkb, page_index)
                polygon_page_nums_cluster[bbp_tuple] = (self.dumps(original_polygon.simplify(tolerance=0.5)), page_index)
                #polygon_page_nums_cluster[bbp_tuple] = (k, page_index)
                i += 1
            self.cluster_hash_tables[j] = polygon_page_nums_cluster
        self.save_pages_to_disk()

        #serialized_data = pickle.dumps(self.cluster_hash_tables)
        #with gzip.open("cluster_hash_tables.pkl.gz", 'wb') as f:
        #    f.write(serialized_data)
        yield self.cluster_hash_tables

        del self.pages
        del self.cluster_hash_tables
        gc.collect()  # Force a garbage collection to free up memory


    def compute_features(self, polygon):
        return np.array(polygon.bounds)


    def get_clusters(self):
        birch = Birch(branching_factor=self.config.bf, n_clusters=self.config.n_clusters, threshold=self.config.threshold).fit(self.X)
        self.cluster_labels = birch.labels_
        self.num_clusters = len(set(self.cluster_labels))
        #print('Number of clusters:', self.num_clusters)
        # Using list comprehension and generator to reduce memory
        combined_clusters = [
            [(polygon, rectangle) for polygon, rectangle in
             zip(self.polygons[self.cluster_labels == n], self.X[self.cluster_labels == n])]
            for n in range(self.num_clusters)
        ]
        self.clusters = combined_clusters
        return self.clusters, self.cluster_labels


    def sort_clusters_Zaddress(self, clusters):
        MBR_clusters = []
        for i, cluster in enumerate(clusters):
            mbb = calculate_bounding_box(cluster)
            if np.all(np.isfinite(mbb)):  # Check if all values are finite
                MBR_clusters.append(mbb)
                #print(f"MBB for cluster {i+1}: {mbb}")

        self.all_z_addresses = []
        for mbr in MBR_clusters:
            z_addresses = [(MortonCode().interleave_latlng(mbr[0][1], mbr[0][0])),
                           (MortonCode().interleave_latlng(mbr[1][1], mbr[1][0]))]
            self.all_z_addresses.append(z_addresses)

        self.z_ranges_sorted = sorted(self.all_z_addresses, key=lambda x: x[0])
        self.sorted_indices = [self.all_z_addresses.index(c) for c in self.z_ranges_sorted]
        self.sorted_clusters = [self.clusters[i] for i in self.sorted_indices]
        self.sorted_clusters_IDs = [i for i, _ in enumerate(self.sorted_clusters)]
        return self.z_ranges_sorted, self.sorted_clusters_IDs

    def pred_cluster_ids(self, node, z_range):
        if node.leaf_model is not None:
            self.leaf_count += 1
            for (z_min, z_max), cluster_id in zip(node.z_ranges, node.clusters):
                if z_min <= z_range[1] and z_max >= z_range[0]:
                    yield (cluster_id, 1 / len(node.labels))
        else:
            self.internal_count += 1
            if node.internal_model is not None and node.z_ranges[0][0] <= z_range[0] and node.z_ranges[-1][1] >= \
                    z_range[1]:
                X = np.array([[z_range[0], z_range[1]]])
                probs = node.internal_model.predict(X).flatten()
                left_cluster_probs = list(self.pred_cluster_ids(node.left_child, z_range))
                right_cluster_probs = list(self.pred_cluster_ids(node.right_child, z_range))

                for cluster_id in node.labels:
                    left_prob = sum(p for c_id, p in left_cluster_probs if c_id == cluster_id)
                    right_prob = sum(p for c_id, p in right_cluster_probs if c_id == cluster_id)
                    prob = probs[cluster_id] + left_prob + right_prob
                    yield (cluster_id, prob)
            else:
                if node.left_child and node.left_child.z_ranges[-1][1] >= z_range[0]:
                    yield from self.pred_cluster_ids(node.left_child, z_range)
                if node.right_child and node.right_child.z_ranges[0][0] <= z_range[1]:
                    yield from self.pred_cluster_ids(node.right_child, z_range)


    def pred_cluster_ids_for_point_query(self, node, point_query):
        if node.leaf_model is not None:
            # leaf node: return the first cluster ID where point_query falls within the z_range
            return [(cluster_id, 1 / len(node.labels)) for (z_min, z_max), cluster_id in zip(node.z_ranges, node.clusters)
                    if z_min <= point_query <= z_max]
        else:
            # internal node
            if node.z_ranges[0][0] <= point_query <= node.z_ranges[-1][1]:
                if node.internal_model is not None:
                    # the point_query falls within the z_range of the internal node
                    X = np.array([[point_query]])
                    probs = node.internal_model.predict(X).flatten()
                    cluster_probs = []
                    # search left child
                    if node.left_child and node.left_child.z_ranges[0][0] <= point_query <= \
                            node.left_child.z_ranges[-1][1]:
                        left_cluster_probs = self.pred_cluster_ids_for_point_query(node.left_child, point_query)
                        cluster_probs.extend(left_cluster_probs)
                    # search right child
                    if node.right_child and node.right_child.z_ranges[0][0] <= point_query <= \
                            node.right_child.z_ranges[-1][1]:
                        right_cluster_probs = self.pred_cluster_ids_for_point_query(node.right_child, point_query)
                        cluster_probs.extend(right_cluster_probs)
                    if cluster_probs:
                        return cluster_probs
                    # the point_query does not fall within the z_range of any child nodes
                    for cluster_id in node.labels:
                        prob = probs[cluster_id]
                        return (cluster_id, prob)


    def get_predict_clusters(self, model, z_range):
        return self.pred_cluster_ids(model, z_range)


    def get_range_query_result(self, query_rect, hash_tables):
        xim, xmax, ymin, ymax = query_rect
        for cluster_polygons in hash_tables:
            for polygon_mbb, value in cluster_polygons.items():
                if polygon_mbb[2] > xim and polygon_mbb[0] < xmax and polygon_mbb[3] > ymin and polygon_mbb[1] < ymax:
                    yield value


    def get_predict_point_clusters(self, model, z_range):
        return self.pred_cluster_ids_for_point_query(model, z_range)


    def get_point_query_result(self, query_point, hash_tables):
        # Given a rectangle with points (x1,y1) and (x2,y2) and assuming x1 < x2 and y1 < y2, a point (x,y) is within that rectangle if x1 < x < x2 and y1 < y < y2.
        for pred_clusters in hash_tables:
            for polygon_mbb, value in pred_clusters.items():
                if polygon_mbb[0] <= query_point[0] <= polygon_mbb[2] and polygon_mbb[1] <= query_point[1] <= polygon_mbb[3]:
                    yield value


