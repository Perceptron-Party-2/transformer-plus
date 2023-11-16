import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from encoder import Encoder
import dataset
import constants
import tokenizer
from decoder import Transformer
import wandb



wand = False

##### cuda magic ####
def getDevice():
  is_cuda = torch.cuda.is_available()
  return "cuda:0" if is_cuda else "cpu"


#initiate wanb #
if wand == True:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Transformer",
        
        # track hyperparameters and run metadata
        config= {
        "learning_rate": constants.LEARNING_RATE,
        "dimensions": constants.EMBEDDING_DIM,
        "vocab_size": constants.VOCAB_SIZE,
        "epochs": constants.EPOCHS,
        "num_heads" : constants.NUM_HEADS,
        "num_layers" : constants.NUM_LAYERS,
        "ff_dim" : constants.FF_DIM,
        "dropout" : constants.DROPOUT,
        "batch_size" : constants.BATCH_SIZE
        }
    )


device = getDevice()

#load dataset
ds = dataset.Dataset()  
dl = torch.utils.data.DataLoader(ds, batch_size= constants.BATCH_SIZE, shuffle=True, collate_fn = ds.collate_fn)

#instantiate transformer and optimiser
transformer = Transformer(constants.VOCAB_SIZE,constants.EMBEDDING_DIM,constants.FF_DIM,constants.NUM_HEADS,constants.NUM_LAYERS,constants.DROPOUT, with_encoder =True).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

transformer.train()


##### Training Loop ####
for epoch in range(0, constants.EPOCHS):

    #iterate over data batches
    for idx,batch in enumerate(dl):
        d_input = batch['d_input_train'].to(device)
        d_labels = batch['d_target_train'].to(device)
        e_input = batch['e_text_train'].to(device)


        output = transformer(e_input,d_input) # generate outputs from transformer model

        #reshape output and labels for loss calc
        #output = output.view(-1, output.size(-1))
        #d_labels = d_labels.view(-1)


        loss = torch.nn.functional.cross_entropy(output,d_labels) # calculate loss from model 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}/{constants.EPOCHS}, Loss: {loss}")
    torch.save(transformer.state_dict(), f"./transformer_epoch_{epoch+1}.pt")  # save model at current epoch 

    if wand == False:
        wandb.log({"loss":loss})

if wand == False:   
    wandb.finish()




