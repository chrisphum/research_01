import copy
from math import ceil 
import random

class globalModel():

    def __init__(self):
        self.resultList = []
        self.globalModelDict = {}
        
    def serveGlobalModel(self):
        return copy.deepcopy( self.globalModelDict )
    
    # Recieving model as a state dict
    def recieveClientModel(self, updateModel):
        self.resultList.append( updateModel )

    def clipAndNoise(self, instanceDict):
        # return instanceDict
        sensitivity = 0.5
        for key in instanceDict:
            # Difference between globelModel and Wk
            # NOT A COMPLETE OR ACCURATE FUNCTION
            # Page 3 of https://openreview.net/pdf?id=SkVRTj0cYQ
            differential = instanceDict[key] - self.globalModelDict[key]
            differential = differential * sensitivity
            instanceDict[key] = differential + self.globalModelDict[key]
        return instanceDict

    def updateGlobalModel(self, Mpercent):
        # M is the number of clients to select
        M = ceil( Mpercent * len( self.resultList ))

        # Select random indexes from client list to use
        randomlist = random.sample(range(0, len( self.resultList )), M)

        # Average M Number of Models Together
        globalSD  = self.resultList[randomlist[0]]
        
        # For first update only
        if not self.globalModelDict:
            self.globalModelDict = globalSD

        for k in randomlist[1:]:
            r = self.clipAndNoise( self.resultList[k] )
            for key in r:
                globalSD[key] = (globalSD[key] + r[key] )

        for key in globalSD:
            globalSD[key] = globalSD[key] / M

        self.globalModelDict = globalSD
        self.resultList = [] 