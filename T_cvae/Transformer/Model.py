import  torch
from    torch import nn
from    torch.nn import functional as F
from    Transformer.EncoderDecoder import EncoderCell, DecoderCell, PositionWiseFeedForwardNetworks
from    Transformer.Embedding import Embedding
from    Transformer.Attention import MultiHeadAttention

def pad_mask(inputs, PAD):
    return (inputs==PAD).unsqueeze(-1).unsqueeze(1)

def triu_mask(batch, length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0).unsqueeze(1).bool()

class Model(nn.Module):

    def __init__(self, d_model, d_vocab, num_layer_encoder, num_layer_decoder, 
                Encoder_Embed, Decoder_Embed, Attention, FFN, dropout_sublayer=0.):
        super().__init__()
        self.encoder_embed = Encoder_Embed
        self.decoder_embed = Decoder_Embed
        self.Encoder = nn.ModuleList([EncoderCell(d_model, Attention, FFN, dropout_sublayer) for _ in range(num_layer_encoder)])
        self.Decoder = nn.ModuleList([DecoderCell(d_model, Attention, FFN, dropout_sublayer) for _ in range(num_layer_decoder)])
        self.project = nn.Linear(d_model, d_vocab)
        
        nn.init.xavier_uniform_(self.project.weight)
    
    def teacher_forcing(self, inputs, encoder_outputs, PAD):
        PadMask = pad_mask(inputs, PAD).to(inputs.device)
        SeqMask = triu_mask(inputs.size(0), inputs.size(1)).to(inputs.device) + PadMask
        outputs = self.decoder_embed(inputs)
        for DecoderCell in self.Decoder:
            outputs = DecoderCell(outputs, encoder_outputs, PadMask, SeqMask)
        return outputs

    def greedy_search(self, encoder_outputs, maxLen, BOS, EOS):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        sentence = torch.LongTensor([BOS]).repeat(batch_size).reshape(-1, 1).to(device)
        outputs = None
        EOS_index = torch.Tensor(batch_size).fill_(False).to(device)

        for i in range(maxLen):

            embed = self.decoder_embed(sentence)
            SeqMask = triu_mask(batch_size, i+1).to(device)
            for DecoderCell in self.Decoder:
                embed = DecoderCell(embed, encoder_outputs, None, SeqMask)
            prob = F.softmax(self.project(embed[:, -1, :]), dim=-1)
            word = prob.max(dim=-1)[1].unsqueeze(-1).long()
            sentence = torch.cat((sentence, word), dim=1)
            
            EOS_index += (word==EOS).reshape(-1)
            if (EOS_index==False).sum() == 0:   break

        sentence = sentence.tolist()
        for i in range(batch_size):
            if EOS in sentence[i]:
                index = sentence[i].index(EOS)
                sentence[i] = sentence[i][:index]
            sentence[i] = sentence[i][1:]
        return sentence

    def beam_search(self):
        pass

    def forward(self, inputs, targets, BOS, EOS, PAD):
        SrcPadMask = pad_mask(inputs, PAD).to(inputs.device)
        encoder_outputs = self.encoder_embed(inputs)
        for EncoderCell in self.Encoder:
            encoder_outputs = EncoderCell(encoder_outputs, SrcPadMask)
        outputs = self.teacher_forcing(targets, encoder_outputs, PAD)
        return self.project(outputs)

    def predict(self, inputs, maxLen, PAD, BOS, EOS):
        PadMask = pad_mask(inputs, PAD).to(inputs.device)
        encoder_outputs = self.encoder_embed(inputs)
        for EncoderCell in self.Encoder:
            encoder_outputs = EncoderCell(encoder_outputs, PadMask)
        return self.greedy_search(encoder_outputs, maxLen, BOS, EOS)

def Transformer(encoder_vocab_size, decoder_vocab_size, embedding_dim=512, d_ff=2048, num_head=8,
              num_layer_encoder=6, num_layer_decoder=6, dropout_embed=0.2, dropout_sublayer=0.1, dropout_ffn=0.):
    if embedding_dim % num_head != 0:
        raise ValueError("Parameter Error, require embedding_dim % num head == 0.")
    d_qk = d_v = int(embedding_dim / num_head)
    Encode_Embedding = Embedding(encoder_vocab_size, embedding_dim)
    Decode_Embedding = Embedding(decoder_vocab_size, embedding_dim)
    Attention = MultiHeadAttention(embedding_dim, d_qk, d_v, num_head)
    FFN = PositionWiseFeedForwardNetworks(embedding_dim, embedding_dim, d_ff, dropout_ffn)
    return Model(embedding_dim, decoder_vocab_size, num_layer_encoder, num_layer_decoder, 
                 Encode_Embedding, Decode_Embedding, Attention, FFN, dropout_sublayer)
