# SPLindex

## Introduntion

This paper proposes SPLindex, a Spatial Polygon Learned Index, designed for disk-based systems. To achieve this, SPLindex is based on four main ideas: 1) dividing polygons into clusters, 2) mapping clusters to one-dimensional ordering using Z-address, 3) a hierarchical model that predicts cluster IDs for a given spatial queries, and 4) locating polygons on disk layout and accessing via a table for optimal disk access.

## Highlights
<ul>
  <li>Spatial object clustering</li>
  <li>Ranking clusters by Z-addresses</li>
  <li>The Hierarchical Z-Interval Tree (HZIT)</li>
</ul>


## Dependencies
<ul>
   <li>Python 3.9</li>
   <li>Numpy</li>
   <li>scikit-learn</li>
</ul>

## Code Structure

To run the [`main.py`](https://github.com/MasoumehVahedi/SPLindex/blob/master/main.py) file, you can follow these steps:

1. Install the necessary packages.

2. Clone the repository under a directory `data`.

3. Download datasets and put them in the directory specified in [`ConfigParam.py`](https://github.com/MasoumehVahedi/SPLindex/blob/master/ConfigParam.py). 

4. Navigate to the directory and run [`main.py`](https://github.com/MasoumehVahedi/SPLindex/blob/master/main.py).

## Example Commands

1. To install the necessary packages, you can use:
    ```sh
    pip install -r requirements.txt
    ```

2. To clone the repository:
    ```sh
    git clone https://github.com/MasoumehVahedi/SPLindex/tree/master
    ```

3. Make sure your [`ConfigParam.py`](https://github.com/MasoumehVahedi/SPLindex/blob/master/ConfigParam.py) file points to the correct dataset directory:
    ```python
    polygons_path = "path/to/data"
    ```

4. To run the [`main.py`](https://github.com/MasoumehVahedi/SPLindex/blob/master/main.py) file, use:
    ```sh
    cd data
    python main.py
    ```

## Datasets
Link to access datasets:

https://drive.google.com/drive/folders/1MGTFAlZ_N7WAg8xzIzKXZmMYlqBlAPJ_?usp=sharing

