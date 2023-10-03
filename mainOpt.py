import sys
import os

sys.path.append('../')

import time


from shapely.geometry import box, Point
from pympler import asizeof
from ConfigParam import Config
from treeModel import TreeBuilder
from optLISP import *
from ZAdress import *
import utils



def get_size(d):
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        if isinstance(value, dict):
            size += get_size(value)
        else:
            size += sys.getsizeof(value)
    return size


def range_query(lisp, model, query_rect, hash_tables):
    lisp.leaf_count = 0
    lisp.internal_count = 0
    z_min = MortonCode().interleave_latlng(query_rect[2], query_rect[0])
    z_max = MortonCode().interleave_latlng(query_rect[3], query_rect[1])
    z_range = [z_min, z_max]
    ############## Step 1- Filtering step to predict cluster IDs ###################
    t1 = time.time()
    predicted_labels = lisp.get_predict_clusters(model, z_range)
    t2 = time.time()
    filter_step_1 = t2 - t1
    #print("Time for Filter step 1 (predicting cluster IDs) = ", filter_step_1)
    leaf_count, internal_count = lisp.leaf_count, lisp.internal_count
    io_filtering_step_1 = leaf_count + internal_count
    #print("IO cost for Filter step (internal_leaf_nodes) =", io_filtering_step_1)

    ############## Step 2- Intermediate Filtering step ###################
    t_1 = time.time()
    hash_pred_clusters = []
    for label in predicted_labels:
        hash_pred_clusters.append(hash_tables.get(label[0]))
    query_results = lisp.get_range_query_result(query_rect, hash_pred_clusters)
    t_2 = time.time()
    filter_step_2 = t_2 - t_1
    #print("Time for Filter step 2 (Intermediate Filter) = ", filter_step_2)
    #print("query_results =", len(query_results))
    return query_results, hash_pred_clusters, filter_step_1, filter_step_2, io_filtering_step_1


def pointsQuery(lisp, model, query_point, hash_tables):
    point_query = Point(query_point)
    # Encode the latitude and longitude using Morten encoding
    z_min = MortonCode().interleave_latlng(point_query.y, point_query.x)
    ############## Step 1- Filtering step to predict cluster IDs ###################
    t1 = time.time()
    predicted_labels = lisp.get_predict_point_clusters(model, z_min)
    t2 = time.time()
    filter_step_1 = t2 - t1
    # print("Time for Filter step 1 (predicting cluster IDs) = ", filter_step_1)
    leaf_count, internal_count = lisp.leaf_count, lisp.internal_count
    io_filtering_step_1 = leaf_count + internal_count
    # print("IO cost for Filter step (internal_leaf_nodes) =", io_filtering_step_1)

    ############## Step 2- Intermediate Filtering step ###################
    t_1 = time.time()
    hash_pred_clusters = []
    for label in predicted_labels:
        hash_pred_clusters.append(hash_tables.get(label[0]))
    query_results = lisp.get_point_query_result(query_point, hash_pred_clusters)
    t_2 = time.time()
    filter_step_2 = t_2 - t_1
    # print("Time for Filter step 2 (Intermediate Filter) = ", filter_step_2)
    #print("query_results =", len(query_results))
    return query_results, hash_pred_clusters, filter_step_1, filter_step_2, io_filtering_step_1


def load_data(data_dir):
    # Read data
    #polygons_path = os.path.join(data_dir, Config().land_polygon_name)
    polygons_path = "/Users/vahedi/Library/CloudStorage/OneDrive-RoskildeUniversitet/Python Projects/FinalLISP/data/LandPolygons.npy"
    polygons = np.load(polygons_path, allow_pickle=True)
    return polygons


def index_construction():
    ########### Load data ###########
    data_dir = "./"
    polygons = load_data(data_dir)

    ########### Build LISP ###########
    print("-------- LISP Index building---------")
    lisp = LISP(polygons, page_size=Config().page_size)
    start_initialize_time_index = time.time()
    clusters, cluster_labels = lisp.get_clusters()

    z_ranges_sorted, sorted_clusters_IDs = lisp.sort_clusters_Zaddress(clusters)
    end_initialize_time_index = time.time()
    initialize_time_index = end_initialize_time_index - start_initialize_time_index

    hash_tables_generator = lisp.get_disk_pages()

    hash_tables = defaultdict(dict)
    for new_hash_tables in hash_tables_generator:
        hash_tables.update(new_hash_tables)


    start_time_model = time.time()
    tree = TreeBuilder()
    tree_model = tree.build_tree(z_ranges_sorted, sorted_clusters_IDs, Config().max_depth)
    end_time_model = time.time()

    time_lisp = end_time_model - start_time_model

    total_nodes = tree.count_nodes(tree_model)
    print(f"Total number of nodes in the tree: {total_nodes}")

    print("Total Time required for indexing:", (initialize_time_index + time_lisp))
    return lisp, tree_model, hash_tables


def lisp_range_query(lisp, tree_model, query_ranges, hash_tables):
    ########### Range Query ###########
    print("-------- Range Query ---------")
    total_filter_step_1_time = 0
    total_filter_step_2_time = 0
    refinement_time_query = 0
    filtering_io_cost = 0
    intermediate_filtering_io_cost = 0
    refinement_io_cost = 0
    memory_usage = 0
    total_size_hash_tables = 0
    ### For 100 range query ###
    #mem_usage_start = memory_usage(-1, interval=0.1, timeout=None)[0]
    for query_rect in query_ranges:
        xim, xmax, ymin, ymax = query_rect
        query_rect_poly = Polygon([(xim, ymin), (xmax, ymin), (xmax, ymax), (xim, ymax)])

        ################# 1- Filter Step ##################
        query_results, hash_pred_clusters, filter_time_1, filter_time_2, io_filtering_step_1 = range_query(lisp,
                                                                                                           tree_model,
                                                                                                           query_rect,
                                                                                                           hash_tables)
        total_filter_step_1_time += filter_time_1
        total_filter_step_2_time += filter_time_2

        query_results_list = list(query_results)
        # print("query_results =", query_results_list)
        print("query_results =", len(query_results_list))

        ############ IO cost for Filtering step = counting the number of internal and leaf nodes #############
        filtering_io_cost += io_filtering_step_1

        ############ IO cost for Intermediate Filtering step = counting the number of hash tables need to consider #############
        io_filterning = len(hash_pred_clusters)
        intermediate_filtering_io_cost += io_filterning

        ############ IO cost for Refinement step= the total number of disk pages loaded in main memory ###########
        number_pages = list(set(value[1] for value in query_results))
        #print(number_pages)
        Io_refinement = len(number_pages)
        #print("Io_refinement = ", Io_refinement)
        refinement_io_cost += Io_refinement

        result = []
        # Now, you can run your processing code on the `pages` data:
        t1 = time.time()
        for i in query_results:
            geom = pickle.loads(i[0])
            if query_rect_poly.intersects(geom):
                result.append(geom)
        # Step 1: Utilize a generator to create query_values_set
        t2 = time.time()
        time_Q_IO = t2 - t1
        refinement_time_query += time_Q_IO
        #print(len(result))

        # Memory consumption for IO cost = total number of pages times page size
        memory_consumption_range_query = (Io_refinement * Config().page_size) / (1024 * 1024)
        memory_usage += memory_consumption_range_query
        hash_table_size = asizeof.asizeof(hash_pred_clusters)
        total_size_hash_tables += hash_table_size

    print("Total time for Filter step 1 across all queries:", total_filter_step_1_time)
    print("Total time for Filter step 2 across all queries:", total_filter_step_2_time)
    print(f"Refinement time for 100 range query: {(refinement_time_query):.2f} s")

    print("IO cost for Filter step 1 (internal and leaf nodes) =", filtering_io_cost)
    print("IO cost for Intermediate Filter step 2 (hash tables) =", intermediate_filtering_io_cost)
    print("IO cost for Refinement step (Total number of pages) = ", refinement_io_cost)


def lisp_point_query(lisp, tree_model, point_queries, hash_tables):
    ########### Point Query ###########
    print("-------- Point Query ---------")
    total_filter_step_1_time = 0
    total_filter_step_2_time = 0
    refinement_time_query = 0
    filtering_io_cost = 0
    intermediate_filtering_io_cost = 0
    refinement_io_cost = 0
    memory_usage = 0
    total_size_hash_tables = 0
    ### For 100 range query ###
    for query_point in point_queries:
        point_query = Point(query_point)
        query_point_box  = box(query_point[0], query_point[1], query_point[0], query_point[1])
        ################# 1- Filter Step ##################
        query_results, hash_pred_clusters, filter_time_1, filter_time_2, io_filtering_step_1 = pointsQuery(lisp,
                                                                                                           tree_model,
                                                                                                           query_point,
                                                                                                           hash_tables)
        total_filter_step_1_time += filter_time_1
        total_filter_step_2_time += filter_time_2

        ############ IO cost for Filtering step = counting the number of internal and leaf nodes #############
        filtering_io_cost += io_filtering_step_1

        ############ IO cost for Intermediate Filtering step = counting the number of hash tables need to consider #############
        io_filterning = len(hash_pred_clusters)
        intermediate_filtering_io_cost += io_filterning

        ############ IO cost for Refinement step= the total number of disk pages loaded in main memory ###########
        number_pages = list(set(value[1] for value in query_results))
        Io_refinement = len(number_pages)
        refinement_io_cost += Io_refinement

        result = []
        # Now, you can run your processing code on the `pages` data:
        t1 = time.time()
        for i in query_results:
            geom = pickle.loads(i[0])
            if query_point_box.intersects(geom):
                result.append(geom)
        # Step 1: Utilize a generator to create query_values_set
        t2 = time.time()
        time_Q_IO = t2 - t1
        refinement_time_query += time_Q_IO
        print("result = ", len(result))

        # Memory consumption for IO cost = total number of pages times page size
        memory_consumption_range_query = (Io_refinement * Config().page_size) / (1024 * 1024)
        memory_usage += memory_consumption_range_query
        hash_table_size = asizeof.asizeof(hash_pred_clusters)
        total_size_hash_tables += hash_table_size

    print("Total time for Filter step 1 across all queries:", total_filter_step_1_time)
    print("Total time for Filter step 2 across all queries:", total_filter_step_2_time)
    print(f"Refinement time for 100 point query: {(refinement_time_query):.4f} s")

    print("IO cost for Filter step 1 (internal and leaf nodes) =", filtering_io_cost)
    print("IO cost for Intermediate Filter step 2 (hash tables) =", intermediate_filtering_io_cost)
    print("IO cost for Refinement step (Total number of pages) = ", refinement_io_cost)


def main():
    range_query_path = "./"
    query_path = os.path.join(range_query_path, Config().land_query_range_path)
    query_ranges = np.load(query_path, allow_pickle=True)
    lisp, tree_model, hash_tables = index_construction()

    ######## Range Query ##########
    lisp_range_query(lisp, tree_model, query_ranges, hash_tables)

    """ ######## Point Query ##########
    lisp_point_query(lisp, tree_model, query_ranges, hash_tables)"""


if __name__ == "__main__":
    main()