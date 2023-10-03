import os
import sys
sys.path.append("../")


class Config(object):
    """Configuration class."""

    class __Singleton(object):
        """Singleton design pattern."""

        def __init__(self):
            # params for clustering algorithm
            #self.bf = 15  # The branching factor for 5M
            self.bf = 10  # The branching factor for land and water
            self.n_clusters = None
            self.threshold = 11.5
            #self.threshold = 0.5
            self.max_depth = 300  # number of tree depth
            self.page_size = 4096

            # Data path
            filename = "data/"
            self.land_query_range_path = os.path.join(filename, "land_query_ranges_2%.npy")
            self.water_query_range_path = os.path.join(filename, "water_query_ranges_1%.npy")
            self.corr_query_range_path = os.path.join(filename, "landuse_query_ranges_0.05%.npy")
            self.zipf_query_range_path = os.path.join(filename, "polyZipf_query_ranges_0.05%.npy")
            self.uniform_polygon_name = filename + "data/SynPolyUniform_5M.npy"  # PolyUniform data path
            self.land_polygon_name = filename + "data/LandPolygons.npy"  # LandPolygon data path
            self.water_polygon_name = filename + "data/water_poly.npy"  # WaterPolygon data path
            self.corr_polygon_name = filename + "data/PolyCorrelated5M.npy"  # PolyCorrelated data path
            self.zipf_polygon_name = filename + "data/Zipf/PolyZipf5M.npy"  # PolyZipf data path

            print('---------Config is initilized----------')
    instance = None

    def __new__(cls):
        """Return singleton instance."""
        if not Config.instance:
            Config.instance = Config.__Singleton()
        return Config.instance

    def __getattr__(self, name):
        """Get singleton instance's attribute."""
        return getattr(self.instance, name)

    def __setattr__(self, name):
        """Set singleton instance's attribute."""
        return setattr(self.instance, name)
