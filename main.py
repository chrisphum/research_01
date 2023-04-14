from apiFuncs import globalModel
from client import Client

B = 100               # Batch
E = 5                 # Epochs
Mpercent = 0.25       # Percent Clients to Update From
seed = 1              # Seed to initialize clients with
numClientsCreate = 10
interations = 10

clientList = []
gm = globalModel()

# Initialize Clients With Set Seed
for j in range(numClientsCreate):
  currentGlobalModel = gm.serveGlobalModel()
  clientInstance = Client( currentGlobalModel, 1 )
  clientList.append( clientInstance )

for q in range(interations):
  # Run an iteration on the client
  for clientInstance in clientList:
    clientInstance.run(E,B)

  # Get Results from the client
  for clientInstance in clientList:
    result = clientInstance.getModel()
    gm.recieveClientModel(result)

  # Update Global Model
  gm.updateGlobalModel(Mpercent)

  # Update Client Models
  for clientInstance in clientList:
    currentGlobalModel = gm.serveGlobalModel()
    clientInstance = clientInstance.updateModel( currentGlobalModel )