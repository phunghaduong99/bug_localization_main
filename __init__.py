import matplotlib.pyplot as plt
import numpy as np

def dictToList(dic):
    list1 = []
    list1.append(0)
    for i in dic.keys():
        list1.append(dic[i]*100)
    return list1

#Ajpect Project
aspectj_dnn = {1: 0.073, 2: 0.148, 3: 0.196, 4: 0.217, 5: 0.254, 6: 0.279, 7: 0.302, 8: 0.34, 9: 0.367, 10: 0.379, 11: 0.404, 12: 0.418, 13: 0.441, 14: 0.468, 15: 0.493, 16: 0.518, 17: 0.539, 18: 0.559, 19: 0.575, 20: 0.6}
aspectj_dnn_meta = {1: 0.544, 2: 0.68, 3: 0.747, 4: 0.794, 5: 0.824, 6: 0.843, 7: 0.858, 8: 0.873, 9: 0.883, 10: 0.888, 11: 0.899, 12: 0.902, 13: 0.911, 14: 0.916, 15: 0.92, 16: 0.922, 17: 0.923, 18: 0.928, 19: 0.931, 20: 0.933}
aspectj_rVSM = {1: 0.584, 2: 0.692, 3: 0.778, 4: 0.801, 5: 0.831, 6: 0.854, 7: 0.877, 8: 0.891, 9: 0.899, 10: 0.904, 11: 0.909, 12: 0.918, 13: 0.92, 14: 0.927, 15: 0.931, 16: 0.932, 17: 0.934, 18: 0.934, 19: 0.938, 20: 0.941}
aspectj_dnn_rVSM = {1: 0.587, 2: 0.698, 3: 0.781, 4: 0.803, 5: 0.835, 6: 0.86, 7: 0.877, 8: 0.891, 9: 0.899, 10: 0.906, 11: 0.913, 12: 0.916, 13: 0.922, 14: 0.927, 15: 0.929, 16: 0.932, 17: 0.935, 18: 0.94, 19: 0.945, 20: 0.947}
aspectj_dnnLoc = {1: 0.698, 2: 0.844, 3: 0.892, 4: 0.914, 5: 0.923, 6: 0.933, 7: 0.94, 8: 0.948, 9: 0.949, 10: 0.954, 11: 0.955, 12: 0.957, 13: 0.957, 14: 0.962, 15: 0.963, 16: 0.967, 17: 0.967, 18: 0.967, 19: 0.967, 20: 0.968}

#Tomcat Project
tomcat_dnn = {1: 0.031, 2: 0.063, 3: 0.09, 4: 0.124, 5: 0.165, 6: 0.197, 7: 0.233, 8: 0.266, 9: 0.301, 10: 0.326, 11: 0.355, 12: 0.384, 13: 0.41, 14: 0.434, 15: 0.462, 16: 0.484, 17: 0.507, 18: 0.528, 19: 0.544, 20: 0.572}
tomcat_dnn_meta = {1: 0.487, 2: 0.609, 3: 0.682, 4: 0.719, 5: 0.746, 6: 0.775, 7: 0.798, 8: 0.818, 9: 0.835, 10: 0.851, 11: 0.864, 12: 0.876, 13: 0.883, 14: 0.892, 15: 0.9, 16: 0.907, 17: 0.913, 18: 0.918, 19: 0.923, 20: 0.926}
tomcat_rVSM = {1: 0.676, 2: 0.787, 3: 0.842, 4: 0.867, 5: 0.881, 6: 0.896, 7: 0.901, 8: 0.911, 9: 0.918, 10: 0.925, 11: 0.93, 12: 0.933, 13: 0.934, 14: 0.938, 15: 0.94, 16: 0.944, 17: 0.945, 18: 0.947, 19: 0.95, 20: 0.951}
tomcat_dnn_rVSM = {1: 0.675, 2: 0.786, 3: 0.841, 4: 0.866, 5: 0.881, 6: 0.896, 7: 0.901, 8: 0.912, 9: 0.92, 10: 0.925, 11: 0.929, 12: 0.934, 13: 0.938, 14: 0.939, 15: 0.943, 16: 0.944, 17: 0.945, 18: 0.947, 19: 0.95, 20: 0.952}
tomcat_dnnLoc = {1: 0.724, 2: 0.823, 3: 0.859, 4: 0.883, 5: 0.902, 6: 0.912, 7: 0.919, 8: 0.927, 9: 0.935, 10: 0.941, 11: 0.945, 12: 0.947, 13: 0.949, 14: 0.95, 15: 0.951, 16: 0.951, 17: 0.951, 18: 0.951, 19: 0.953, 20: 0.954}

#Eclipse Project
eclipse_dnn =  {1: 0.065, 2: 0.11, 3: 0.148, 4: 0.185, 5: 0.217, 6: 0.248, 7: 0.281, 8: 0.311, 9: 0.34, 10: 0.367, 11: 0.395, 12: 0.419, 13: 0.443, 14: 0.464, 15: 0.491, 16: 0.512, 17: 0.533, 18: 0.557, 19: 0.583, 20: 0.603}
eclipse_dnn_meta = {1: 0.62, 2: 0.738, 3: 0.788, 4: 0.818, 5: 0.84, 6: 0.858, 7: 0.874, 8: 0.885, 9: 0.893, 10: 0.902, 11: 0.909, 12: 0.915, 13: 0.922, 14: 0.927, 15: 0.931, 16: 0.934, 17: 0.937, 18: 0.94, 19: 0.943, 20: 0.945}
eclipse_rVSM = {1: 0.631, 2: 0.761, 3: 0.821, 4: 0.86, 5: 0.884, 6: 0.899, 7: 0.909, 8: 0.919, 9: 0.927, 10: 0.934, 11: 0.939, 12: 0.946, 13: 0.95, 14: 0.953, 15: 0.957, 16: 0.959, 17: 0.96, 18: 0.961, 19: 0.964, 20: 0.966}
eclipse_dnn_rVSM = {1: 0.632, 2: 0.764, 3: 0.822, 4: 0.859, 5: 0.884, 6: 0.901, 7: 0.912, 8: 0.921, 9: 0.929, 10: 0.936, 11: 0.941, 12: 0.947, 13: 0.95, 14: 0.954, 15: 0.958, 16: 0.96, 17: 0.962, 18: 0.963, 19: 0.965, 20: 0.967}
eclipse_dnnLoc = {1: 0.747, 2: 0.854, 3: 0.893, 4: 0.914, 5: 0.927, 6: 0.936, 7: 0.943, 8: 0.947, 9: 0.95, 10: 0.954, 11: 0.957, 12: 0.96, 13: 0.962, 14: 0.964, 15: 0.966, 16: 0.967, 17: 0.968, 18: 0.969, 19: 0.97, 20: 0.971}

#SWT Project
swt_dnn = {1: 0.179, 2: 0.299, 3: 0.391, 4: 0.464, 5: 0.523, 6: 0.572, 7: 0.612, 8: 0.651, 9: 0.685, 10: 0.716, 11: 0.739, 12: 0.763, 13: 0.784, 14: 0.799, 15: 0.813, 16: 0.827, 17: 0.84, 18: 0.853, 19: 0.863, 20: 0.873}
swt_dnn_meta = {1: 0.468, 2: 0.646, 3: 0.739, 4: 0.807, 5: 0.849, 6: 0.874, 7: 0.898, 8: 0.912, 9: 0.923, 10: 0.935, 11: 0.94, 12: 0.946, 13: 0.952, 14: 0.958, 15: 0.962, 16: 0.965, 17: 0.97, 18: 0.971, 19: 0.973, 20: 0.976}
swt_rVSM = {1: 0.436, 2: 0.612, 3: 0.705, 4: 0.769, 5: 0.81, 6: 0.84, 7: 0.864, 8: 0.879, 9: 0.896, 10: 0.91, 11: 0.921, 12: 0.93, 13: 0.937, 14: 0.943, 15: 0.949, 16: 0.954, 17: 0.958, 18: 0.961, 19: 0.963, 20: 0.967}
swt_dnn_rVSM = {1: 0.468, 2: 0.646, 3: 0.739, 4: 0.807, 5: 0.849, 6: 0.874, 7: 0.898, 8: 0.912, 9: 0.923, 10: 0.935, 11: 0.94, 12: 0.946, 13: 0.952, 14: 0.958, 15: 0.962, 16: 0.965, 17: 0.97, 18: 0.971, 19: 0.973, 20: 0.976}
swt_dnnLoc = {1: 0.618, 2: 0.783, 3: 0.854, 4: 0.892, 5: 0.912, 6: 0.928, 7: 0.937, 8: 0.944, 9: 0.949, 10: 0.954, 11: 0.959, 12: 0.962, 13: 0.963, 14: 0.965, 15: 0.967, 16: 0.968, 17: 0.97, 18: 0.971, 19: 0.972, 20: 0.972}

aspectj_dnn = dictToList(aspectj_dnn)
aspectj_dnn_meta = dictToList(aspectj_dnn_meta)
aspectj_rVSM = dictToList(aspectj_rVSM)
aspectj_dnn_rVSM = dictToList(aspectj_dnn_rVSM)
aspectj_dnnLoc = dictToList(aspectj_dnnLoc)

tomcat_dnn = dictToList(tomcat_dnn)
tomcat_dnn_meta = dictToList(tomcat_dnn_meta)
tomcat_rVSM = dictToList(tomcat_rVSM)
tomcat_dnn_rVSM = dictToList(tomcat_dnn_rVSM)
tomcat_dnnLoc = dictToList(tomcat_dnnLoc)

eclipse_dnn = dictToList(eclipse_dnn)
eclipse_dnn_meta = dictToList(eclipse_dnn_meta)
eclipse_rVSM = dictToList(eclipse_rVSM)
eclipse_dnn_rVSM = dictToList(eclipse_dnn_rVSM)
eclipse_dnnLoc = dictToList(eclipse_dnnLoc)

swt_dnn = dictToList(swt_dnn)
swt_dnn_meta = dictToList(swt_dnn_meta)
swt_rVSM = dictToList(swt_rVSM)
swt_dnn_rVSM = dictToList(swt_dnn_rVSM)
swt_dnnLoc = dictToList(swt_dnnLoc)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2,nrows=2)

ax1.plot(aspectj_dnn,"r-")
ax1.plot(aspectj_dnn_meta,"g-")
ax1.plot(aspectj_rVSM,"m-")
ax1.plot(aspectj_dnn_rVSM,"b-")
ax1.plot(aspectj_dnnLoc,"y-")
ax1.set(title='Aspectj Accuracy', xlabel='K', ylabel='Accuracy(%)', xlim=([1,20]), ylim=(0,100))

ax2.plot(tomcat_dnn,"r-")
ax2.plot(tomcat_dnn_meta,"g-")
ax2.plot(tomcat_rVSM,"m-")
ax2.plot(tomcat_dnn_rVSM,"b-")
ax2.plot(tomcat_dnnLoc,"y-")
ax2.set(title='Tomcat Accuracy', xlabel='K', ylabel='Accuracy(%)', xlim=([1,20]), ylim=(0,100))

ax3.plot(eclipse_dnn,"r-")
ax3.plot(eclipse_dnn_meta,"g-")
ax3.plot(eclipse_rVSM,"m-")
ax3.plot(eclipse_dnn_rVSM,"b-")
ax3.plot(eclipse_dnnLoc,"y-")
ax3.set(title='Eclipse Accuracy', xlabel='K', ylabel='Accuracy(%)', xlim=([1,20]), ylim=(0,100))

ax4.plot(swt_dnn,"r-")
ax4.plot(swt_dnn_meta,"g-")
ax4.plot(swt_rVSM,"m-")
ax4.plot(swt_dnn_rVSM,"b-")
ax4.plot(swt_dnnLoc,"y-")
ax4.set(title='SWT Accuracy', xlabel='K', ylabel='Accuracy(%)', xlim=([1,20]), ylim=(0,100))

plt.legend(['DNN','DNN+meta','rVSM','DNN+rVSM','DNNLoc'],loc="lower right")
fig.tight_layout()
plt.show()
fig.savefig('../data/output10.png',dpi=100)
