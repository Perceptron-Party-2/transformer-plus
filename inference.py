import torch
from model import Transformer
from tokenizer import Tokenizer

import constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(
    vocab_size=constants.VOCAB_SIZE,
    embedding_dim=constants.EMBEDDING_DIM,
    ff_dim=constants.FF_DIM,
    num_heads=constants.NUM_HEADS,
    num_layers=constants.NUM_LAYERS,
    dropout=constants.DROPOUT,
    with_encoder=constants.WITH_ENCODER,
)

model = model.to(device)

text_tokenizer = Tokenizer('questions')
sql_tokenizer = Tokenizer('answers')
# model.load_weights('model.pt')

def inference(text):
    input = torch.tensor([text_tokenizer.encode(text)]).to(device)
    output = model(input)
    return sql_tokenizer(output[0].argmax(dim=-1).tolist())

if __name__ == '__main__':
    question = 'What is the name of the team with the longest name?'

    print(inference(question))