import os
import pickle
import numpy as np


class VidOR:

    def __init__(self, DS_ROOT):
        obj_label_path = os.path.join(DS_ROOT, 'object_labels.txt')
        pre_label_path = os.path.join(DS_ROOT, 'predicate_labels.txt')

        self.obj_vecs = self._load_object_vectors(DS_ROOT)
        self.obj_name2ind = {}  # 0 base
        self.pre_name2ind = {}  # 0 base
        self.spatial_set = set()

        with open(obj_label_path) as f:
            self.obj_ind2name = f.readlines()
        for i in range(len(self.obj_ind2name)):
            self.obj_ind2name[i] = self.obj_ind2name[i].strip()
            self.obj_name2ind[self.obj_ind2name[i]] = i

        with open(pre_label_path) as f:
            self.pre_ind2name = f.readlines()
        for i in range(len(self.pre_ind2name)):
            self.pre_ind2name[i] = self.pre_ind2name[i].strip()
            self.pre_name2ind[self.pre_ind2name[i]] = i

        self.spatial_set.add('towards')
        self.spatial_set.add('next_to')
        self.spatial_set.add('inside')
        self.spatial_set.add('in_front_of')
        self.spatial_set.add('beneath')
        self.spatial_set.add('behind')
        self.spatial_set.add('away')
        self.spatial_set.add('above')

    def is_spatial(self, predicate):
        return predicate in self.spatial_set

    def is_subject(self, object):
        return True

    def _load_object_vectors(self, DS_ROOT):
        # load object word2vec
        o2v_path = os.path.join(DS_ROOT, 'object_vectors.mat')
        w2v_path = os.path.join(DS_ROOT, 'GoogleNews-vectors-negative300.bin')
        if not os.path.exists(o2v_path):
            self._prepare_w2v(w2v_path, o2v_path)
        with open(o2v_path, 'r') as f:
            obj_vecs = pickle.load(f)
        return obj_vecs

    def _prepare_w2v(self, w2v_path, o2v_path):
        import gensim
        # load pre-trained word2vec model
        print('Loading pretrained word vectors ...')
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        obj_vecs = self._extract_object_vectors(w2v_model)

        # save object label vectors
        with open(o2v_path, 'w') as f:
            pickle.dump(obj_vecs, f)
        print('VidOR object word vectors saved at: %s' % o2v_path)

    def _extract_object_vectors(self, w2v_model, vec_len=300, debug=False):
        # object labels to vectors
        print('Extracting word vectors for VidOR object labels ...')
        obj_vecs = np.zeros((len(self.obj_ind2name), vec_len))
        for i in range(len(self.obj_ind2name)):
            obj_label = self.obj_ind2name[i]
            obj_label = obj_label.split('/')[0]

            if obj_label == 'traffic_light':
                obj_label = 'signal'
            if obj_label == 'stop_sign':
                obj_label = 'sign'
            if obj_label == 'baby_seat':
                obj_label = 'stroller'
            if obj_label == 'electric_fan':
                obj_label = 'fan'
            if obj_label == 'baby_walker':
                obj_label = 'walker'

            vec = w2v_model[obj_label]
            if debug and vec is None or len(vec) == 0 or np.sum(vec) == 0:
                print('[WARNING] %s' % obj_label)
            obj_vecs[i] = vec
        return obj_vecs