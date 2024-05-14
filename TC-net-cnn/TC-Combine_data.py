# If regionize is set to False in TC-extract_data.py, use this script to combine the datasets together
# Output a combined dataset with suffix AL at the end
root='/N/slate/kmluong/Training_data/'        # Data root
import numpy as np
name=['NA','WP','EP']                         # Choosing basins to combine
for mode in ['13']:                           # Number of layers of data
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
