import matplotlib.pyplot as plt
import numpy as np


def convert_list_to_dict(list):
    item = [(index+1, item) for index, item in enumerate(list)]
    item = dict(item)
    return item


def dictToList(dic):
    list1 = []
    list1.append(0)
    for i in dic.keys():
        list1.append(dic[i]*100)
    return list1


#Tomcat Project
tomcat_rVSM = {1: 0.349, 2: 0.454, 3: 0.516, 4: 0.552, 5: 0.585, 6: 0.622, 7: 0.651, 8: 0.663, 9: 0.678, 10: 0.692, 11: 0.703, 12: 0.712, 13: 0.725, 14: 0.734, 15: 0.739, 16: 0.743, 17: 0.749, 18: 0.753, 19: 0.757, 20: 0.764}
tomcat_svm_semantic_knowledge = {1: 0.313, 2: 0.3888, 3: 0.4541, 4: 0.4891, 5: 0.5139, 6: 0.5397, 7: 0.5531, 8: 0.5670999999999999, 9: 0.5813, 10: 0.5954, 11: 0.6144000000000001, 12: 0.6248, 13: 0.6286, 14: 0.6361, 15: 0.6399, 16: 0.6504, 17: 0.6553, 18: 0.6581, 19: 0.6646, 20: 0.6734}
tomcat_svm_rVSM_knowledge = {1: 0.3844, 2: 0.4857, 3: 0.5369, 4: 0.5851, 5: 0.6212, 6: 0.646, 7: 0.6658000000000001, 8: 0.6772, 9: 0.6886, 10: 0.6962, 11: 0.7075, 12: 0.7172, 13: 0.7286, 14: 0.7361, 15: 0.7454999999999999, 16: 0.7538, 17: 0.7614, 18: 0.7624, 19: 0.7662, 20: 0.77}
tomcat_svm_rVSM_semantic_knowledge = {1: 0.3844, 2: 0.4857, 3: 0.5378, 4: 0.586, 5: 0.6212, 6: 0.6413, 7: 0.6638000000000001, 8: 0.6762, 9: 0.6876, 10: 0.6962, 11: 0.7065, 12: 0.7172, 13: 0.7286, 14: 0.7361, 15: 0.7454999999999999, 16: 0.7538, 17: 0.7614, 18: 0.7624, 19: 0.7662, 20: 0.77}


tomcat_rVSM = dictToList(tomcat_rVSM)
tomcat_svm_semantic_knowledge = dictToList(tomcat_svm_semantic_knowledge)
tomcat_svm_rVSM_knowledge = dictToList(tomcat_svm_rVSM_knowledge)
tomcat_svm_rVSM_semantic_knowledge = dictToList(tomcat_svm_rVSM_semantic_knowledge)

fig, (ax2) = plt.subplots(ncols=1,nrows=1)

ax2.plot(tomcat_svm_semantic_knowledge,"r-")
ax2.plot(tomcat_rVSM,"g-")
ax2.plot(tomcat_svm_rVSM_knowledge,"m-")
ax2.plot(tomcat_svm_rVSM_semantic_knowledge,"b-")

ax2.set(title='Tomcat Accuracy', xlabel='K', ylabel='Accuracy(%)', xlim=([1, 20]), ylim=(0, 100))

plt.legend(['rVSM', 'smv+semantic', 'svm+rVSM', 'svm+rVSM+semantic'], loc="lower right")
fig.tight_layout()
plt.show()
fig.savefig('data_output/output10.png',dpi=100)
