import scipy.io as sio

similarity_data = sio.loadmat('human_semantic_similarity.mat')
sim_matrix = similarity_data['similarity']

with open('objectInfo150.txt') as f:
    lines = f.readlines()[1:]
    obj_infos = [line.split('\t') for line in lines]

obj_names = [info[-1].strip() for info in obj_infos]
obj2ind = dict(zip(obj_names, range(len(obj_names))))
obj_ratios = zip([info[-1].strip() for info in obj_infos], [float(info[1]) for info in obj_infos])
obj_ratios_sorted = sorted(obj_ratios, key=lambda item: item[1], reverse=True)
obj2ratio = dict(obj_ratios)

obj2group = {obj_name: None for obj_name in obj_names}
group_list = []
for obj, ratio in obj_ratios_sorted:
    sims = sim_matrix[obj2ind[obj]]
    if obj2group[obj] is None:
        group = []
        group_list.append(group)
    else:
        group = obj2group[obj]
    for sim_obj_ind in range(len(sims)):
        if sims[sim_obj_ind] > 0.2 and \
                obj2group[obj_names[sim_obj_ind]] is None:
            group.append(obj_names[sim_obj_ind])
            obj2group[obj_names[sim_obj_ind]] = group

group_ratios = [0] * len(group_list)
for i in range(len(group_list)):
    for obj in group_list[i]:
        group_ratios[i] += obj2ratio[obj]

for i in range(len(group_list)):
    print('%s: %.4f' % (group_list[i][0], group_ratios[i]))

print('==========')
print('group num: %d' % len(group_list))
print('class num: %d' % sum([len(group) for group in group_list]))



