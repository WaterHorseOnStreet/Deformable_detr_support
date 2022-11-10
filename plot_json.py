import json
import numpy as np
import matplotlib.pyplot as plt
# Opening JSON file
f = open('base.txt', 'r')
s = open('atten.txt', 'r')
m = open('small_atten.txt', 'r')
# returns JSON object as 
# a dictionary
atten_results = []
for jsonObj in s:
    atten_result = json.loads(jsonObj)
    atten_results.append(atten_result['test_coco_eval_bbox'])
atten_results = np.array(atten_results)

samll_atten_results = []
for jsonObj in m:
    samll_atten_result = json.loads(jsonObj)
    samll_atten_results.append(samll_atten_result['test_coco_eval_bbox'])
samll_atten_results = np.array(samll_atten_results)

# Iterating through the json
# list
results = []
for jsonObj in f:
    result = json.loads(jsonObj)
    results.append(result['test_coco_eval_bbox'])

    
results = np.array(results)
x = list(range(0,50))

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(x, results[:,0],label='base')
axs[0, 0].plot(x, atten_results[:,0],label='atten')
axs[0, 0].plot(x, samll_atten_results[:,0], label='small_atten')
axs[0, 0].set_title('AP')
axs[0, 0].legend(loc="lower right")

axs[0, 1].plot(x, results[:,1],label='base')
axs[0, 1].plot(x, atten_results[:,1],label='atten')
axs[0, 1].plot(x, samll_atten_results[:,1],label='small_atten')
axs[0, 1].set_title('IOU50')
axs[0, 1].legend(loc="lower right")

axs[0, 2].plot(x,  results[:,2],label='base')
axs[0, 2].plot(x,  atten_results[:,2],label='atten')
axs[0, 2].plot(x,  samll_atten_results[:,2],label='small_atten')
axs[0, 2].set_title('IOU75')
axs[0, 2].legend(loc="lower right")

axs[1, 0].plot(x, results[:,3],label='base')
axs[1, 0].plot(x, atten_results[:,3],label='atten')
axs[1, 0].plot(x, samll_atten_results[:,3],label='small_atten')

axs[1, 0].set_title('APsmall')
axs[1, 0].legend(loc="lower right")

axs[1, 1].plot(x, results[:,4],label='base')
axs[1, 1].plot(x, atten_results[:,4],label='atten')
axs[1, 1].plot(x, samll_atten_results[:,4],label='small_atten')

axs[1, 1].set_title('APmiddle')
axs[1, 1].legend(loc="lower right")

axs[1, 2].plot(x, results[:,5],label='base')
axs[1, 2].plot(x, atten_results[:,5],label='atten')
axs[1, 2].plot(x, samll_atten_results[:,5],label='small_atten')

axs[1, 2].set_title('APlarge')
axs[1, 1].legend(loc="lower right")

fig.tight_layout()
plt.savefig('result.png')