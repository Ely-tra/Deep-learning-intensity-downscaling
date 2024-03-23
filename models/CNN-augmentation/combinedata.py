root='/N/slate/kmluong/Training_data/'
import numpy as np
name=['NA','WP','EP']
for mode in ['13']:
  for xy in ['features','labels']:
    if xy=='features':
      end='fixed'
    else:
      end=''
    a=np.load(root+'CNN'+xy+mode+name[0]+end+'.npy')
    for n in name[1:]:
      b=np.load(root+'CNN'+xy+mode+n+end+'.npy')
      a=np.append(a,b,axis=0)
    print(a.shape)
    print(root+'CNN'+xy+mode+'AL'+end+'.npy')
    np.save(root+'CNN'+xy+mode+'AL'+end+'.npy', a)


