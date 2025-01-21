import numpy as np

def bone(kpts3d):
    dict={}
    result=kpts3d[11]-kpts3d[13]
    result=result/np.linalg.norm(result)
    print(result)
    dict["leftUpperArm"]=result.tolist()
    return dict


