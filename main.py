import sys
import os

sys.path.append('../')

import time
import pickle

from shapely.geometry import box, Point
from pympler import asizeof
from ConfigParam import Config
from treeModel import TreeBuilder
from SPLindex import *
from ZAdress import *
import utils




def load_data(data_dir):
    # Read data
    polygons_path = os.path.join(data_dir, Config().land_polygon_name)
    polygons = np.load(polygons_path, allow_pickle=True)
    return polygons


def index_construction():
    ########### Load data ###########
    data_dir = "./"
    polygons = load_data(data_dir)

    ########### Build SPLIndex ###########
    print("-------- SPLindex building---------")
    spli = SPLindex(polygons, page_size=Config().page_size)
    clusters, cluster_labels = spli.clusters, spli.cluster_labels

    z_ranges_sorted, sorted_clusters_IDs = spli.sortClustersZaddress(clusters)
    hash_tables_generator = spli.getDiskPages()

    hash_tables = defaultdict(dict)
    for new_hash_tables in hash_tables_generator:
        hash_tables.update(new_hash_tables)

    # Build the tree model
    tree = TreeBuilder(global_percentage=0.05, capacity_node=10)  # Assuming a 5% error bound for illustration
    tree_model = tree.buildTreeModel(z_ranges_sorted, sorted_clusters_IDs)

    # Save clusters and models
    utils.save_clusters(clusters, "clusters.pkl")
    utils.save_model(tree_model, "tree_model.pkl")
    utils.save_hashTable(hash_tables, "hashTable.pkl")
    return spli, tree_model, hash_tables


def splindexRangeQuery(spli, tree_model, query_ranges, hash_tables):
    ########### Range Query ###########
    print("-------- Range Query ---------")
    for query_rect in query_ranges:
        xim, xmax, ymin, ymax = query_rect
        query_rect_poly = Polygon([(xim, ymin), (xmax, ymin), (xmax, ymax), (xim, ymax)])
        query_results, hash_pred_clusters = spli.getRangeQueryWithModel(tree_model, query_rect, hash_tables)

        result = []
        for i in query_results:
            geom = pickle.loads(i[0])
            if query_rect_poly.intersects(geom):
                result.append(geom)
        print(f"Range query result = {len(result)}")


def splindexPointQuery(spli, tree_model, point_queries, hash_tables):
    ########### Point Query ###########
    print("-------- Point Query ---------")
    for query_point in point_queries:
        query_point_box = box(query_point[0], query_point[1], query_point[0], query_point[1])
        query_results, hash_pred_clusters = spli.pointQuery(tree_model, query_point, hash_tables)

        result = []
        for i in query_results:
            geom = pickle.loads(i[0])
            if query_point_box.intersects(geom):
                result.append(geom)
        print(f"Point query result = {result}")


def main():
    range_query_path = "./"
    #query_path = os.path.join(range_query_path, Config().land_query_range_path)
    query_path = "landuse_query_ranges_1%.npy"
    query_ranges = np.load(query_path, allow_pickle=True)
    spli, tree_model, hash_tables = index_construction()

    ######## Range Query ##########
    splindexRangeQuery(spli, tree_model, query_ranges, hash_tables)

    ######## Point Query ##########
    #splindexPointQuery(spli, tree_model, query_ranges, hash_tables)


if __name__ == "__main__":
    main()