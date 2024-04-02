from Tensor import Tensor
import numpy as np

class Parameter(Tensor):  #基类就是个Tensor
    def __init__(self,data,training=True,depends_on=[],name="input"):
        if type(data)== np.ndarray:
            super().__init__(data,training,depends_on,name)
        elif type(data)==Tensor:
            super().__init__(data.data,training,depends_on,name)
