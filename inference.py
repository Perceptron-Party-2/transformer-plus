import torch
from decoder import Transformer
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
    tokenised_text = text_tokenizer.encode(text)
    encoder_input = torch.tensor([tokenised_text]).to(device)
    max_sequence_length = 40

    iterable_sql = sql_tokenizer.sp.bos_id()
    break_count = 0

    while (iterable_sql != sql_tokenizer.sp.eos_id()) & (break_count <= max_sequence_length):
        iterable_decoder_tensor = torch.tensor([iterable_sql]).view(1,-1)
        probability_matrix = model(encoder_input=encoder_input, decoder_input=iterable_decoder_tensor)
        #print(probability_matrix.size())
        probability_vector = probability_matrix[0, -1, :]
        #print(probability_vector.size())
        next_token_id = (torch.argmax(probability_vector))
        #print(next_token_id)
        iterable_sql = [iterable_sql] + [next_token_id.item()]
        #print(tk.decode(iterable_text))
        break_count += 1

if __name__ == '__main__':
    question = 'What is the name of the team with the longest name?'

    print(inference(question))








