import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

hico_path = osp.join(this_dir, 'eval_hico')
add_path(hico_path)

vcoco_path = osp.join(this_dir, 'eval_vcoco')
add_path(vcoco_path)
