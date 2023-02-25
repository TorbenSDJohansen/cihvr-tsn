import itertools
import numpy as np
from pointcloud import PointCloud


def lists_to_tuples(lst): return [list(map(tuple, l)) for l in lst] 


def flatten_list(lst): return list(itertools.chain(*lst))


def tuples_to_lists(tpls): return list(map(list, tpls))
    

#def distribute_keypoints(lines, keypoints_per_line):
#
#    def _sort_by_y(array): return array[array[:, 1].argsort()]
#    
#    def _get_evenly_spaced_numbers(stop, n):
#        return np.floor(np.linspace(0, stop, n)).astype(int)
#
#    keypoints = []
#    for line in lines:
#        distributed = _get_evenly_spaced_numbers(
#                stop=len(line)-1,
#                n=min(keypoints_per_line, len(line)-1)
#        )
#        squeezed = line.squeeze()
#        sorted_by_y = _sort_by_y(squeezed)
#        keypoints.append(sorted_by_y[distributed])
#    array = np.array(keypoints).squeeze()
#    to_shape = (array.shape[0] * array.shape[1], 2)
#    array = array.reshape(to_shape)
#    return PointCloud(array).sort()




