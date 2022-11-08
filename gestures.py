from enum import IntEnum



class Gestures(IntEnum):                                      
                                                              
    # see https://github.com/aljazfrancic/myo-readings-dataset
                                                              
    NEUTRAL = 0                                               
    FLEXION = 1                                               
    EXTENSION = 2                                             
    FIST = 7 


GESTURES_DICT = {g.name: g.value for g in Gestures}
