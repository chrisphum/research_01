from apiFuncs import globalModel
from client import Client

B = 100               # Batch
E = 5                 # Epochs
Mpercent = 0.25       # Percent Clients to Update From
seed = 1              # Seed to initialize clients with
numClientsDataReturned = 0

clientList = []
gm = globalModel()


# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE
# NOT COMPLETE - JUST AN EXAMPLE



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/getmodel')
def getmodel():
    # add some sort of polling system to reduce communication between client server 
    # so that updates only happen when needed
    currentGlobalModel = gm.serveGlobalModel()        
    return jsonify( {'model': currentGlobalModel, 'seed': seed, 'batch': B, 'epoch': E } )
                   
@app.route('/uploadmodel')
def uploadmodel():
    data = request.json 
    gm.recieveClientModel(data)
    numClientsDataReturned += 1
    if numClientsDataReturned > 100:
        gm.updateGlobalModel(Mpercent)
        numClientsDataReturned = 0

app.run()

