import  torch
from    torch import nn
import  torchvision
from    torch.nn.utils.rnn import pack_padded_sequence
import  torch.nn.functional as F
from    Transformer.EncoderDecoder import EncoderCell, DecoderCell, PositionWiseFeedForwardNetworks
from    Transformer.Embedding import Embedding
from    Transformer.Attention import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def triu_mask(batch, length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0).unsqueeze(1).bool()

class ImgEncoder(nn.Module):
    """
    Image Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(ImgEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        batch_size = images.size(0)
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out_mean = out.view(batch_size, -1, 2048).mean(dim=1)  # (batch_size, 2048)
        return out, out_mean

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class CapEncoder(nn.Module):
    """
    Caption Auto-encoder.
    """

    def __init__(self, embed, embed_dim, encoder_dim):
        super(CapEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.embedding = embed
        self.encoder = nn.ModuleList([EncoderCell(embed_dim, MultiHeadAttention(embed_dim, int(embed_dim/8), int(embed_dim/8), num_head=8),
                                      PositionWiseFeedForwardNetworks(embed_dim, embed_dim, 2048, dropout=0.1), dropout=0.1) for _ in range(6)])
        # self.encoder = nn.LSTM(embed_dim, encoder_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)


    def forward(self, encoded_captions, caption_lengths):
        embeddings = self.embedding(encoded_captions)
        PadMask = torch.ones(encoded_captions.size()).bool().to(embeddings.device)
        for i in range(PadMask.size(0)):
            PadMask[i,:caption_lengths[i]] = False

        # packed_input = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)

        # _1, (hidden, _2) = self.encoder(packed_input)  # hidden: (2, batch_size, encoder_dim)
        # hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        for encodercell in self.encoder:
            embeddings = encodercell(embeddings, PadMask.unsqueeze(1).unsqueeze(-1))
        return F.relu(self.fc(embeddings.sum(dim=1)))

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class AttnDecoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed, attention_dim, embed_dim, img_size, decoder_dim, latent_size, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param latent_size: size of latent variable
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(AttnDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = embed
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.dropout = nn.Dropout(p=self.dropout)
        # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.decoder = nn.ModuleList([DecoderCell(embed_dim, MultiHeadAttention(embed_dim, int(embed_dim/8), int(embed_dim/8), num_head=8),
                                      PositionWiseFeedForwardNetworks(embed_dim, embed_dim, 2048, dropout=0.1), dropout=0.1) for _ in range(6)])

        # self.decode_step = nn.ModuleList([DeocderCell(embed_dim,  Attention(embed_dim, embed_dim/8, embed_dim/8, num_head=8),
        #                                   PositionWiseFeedForwardNetworks(embed_dim, embed_dim, 2048, dropout=0.1)) for _ in range(6)])
        self.init_h = nn.Linear(latent_size, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(latent_size, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc_attn = nn.Linear(encoder_dim, embed_dim)
        self.fc_embed = nn.Linear(embed_dim * 2 + img_size, embed_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

        nn.init.xavier_uniform_(self.fc_attn.weight)
        nn.init.xavier_uniform_(self.fc_embed.weight)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        # self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, z):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param z: latent variable
        :return: hidden state, cell state
        """
        h = self.init_h(z)  # (batch_size, decoder_dim)
        c = self.init_c(z)
        return h, c

    def forward(self, encoded_imgs, encoded_captions, caption_lengths, z):
        """
        Forward propagation.

        :param encoded_imgs: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :param z: latent variable
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoded_imgs.size(0)
        img_dim = encoded_imgs.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoded_imgs = encoded_imgs.view(batch_size, -1, img_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoded_imgs.size(1)

        # Already sorted
        # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # encoded_imgs = encoded_imgs[sort_ind]
        # encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        # h, c = self.init_hidden_state(z)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # print(max(decode_lengths), embeddings.size())
        # Create tensors to hold word predicion scores and alphas
        # predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        # alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        alphas = None

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        encoded_imgs = F.relu(self.fc_attn(encoded_imgs))
        PadMask = torch.ones(encoded_captions.size()).bool().to(embeddings.device)
        for i in range(PadMask.size(0)):
            PadMask[i,:caption_lengths[i]] = False
        PadMask = PadMask.unsqueeze(1).unsqueeze(-1)
        SeqMask = triu_mask(embeddings.size(0), embeddings.size(1)).to(embeddings.device) + PadMask
        for decodercell in self.decoder:
            embeddings, weight = decodercell(embeddings, encoded_imgs, PadMask, SeqMask)
            # print(alphas.size(), weight.size())
            if alphas is None:
                alphas = weight.mean(dim=1)
            else:
                alphas = alphas + weight.mean(dim=1)
            
        # for t in range(max(decode_lengths)):
        #     batch_size_t = sum([l > t for l in decode_lengths])
        #     attention_weighted_encoding, alpha = self.attention(encoded_imgs[:batch_size_t],
        #                                                         h[:batch_size_t])
        #     gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
        #     attention_weighted_encoding = gate * attention_weighted_encoding
        #     h, c = self.decode_step(
        #         torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
        #         (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
        #     preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        #     predictions[:batch_size_t, t, :] = preds
        #     alphas[:batch_size_t, t, :] = alpha
        # print(z.size(), embeddings.size())
        
        embeddings = F.relu(self.fc_embed(torch.cat((embeddings, z.unsqueeze(1).repeat(1, embeddings.size(1), 1)), dim=-1)))
        predictions = self.fc(embeddings)

        return predictions, alphas / 6, decode_lengths


class CvaeEncoder(nn.Module):

    def __init__(self, standard_gaussian, hidden_size, img_size, latent_size):
        super(CvaeEncoder, self).__init__()
        self.standard_gaussian = standard_gaussian
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.latent_size = latent_size

        self.fc1 = nn.Linear(hidden_size + img_size, hidden_size)
        self.hidden2mu1 = nn.Linear(hidden_size, latent_size)
        self.hidden2logv1 = nn.Linear(hidden_size, latent_size)
        self.rec = nn.Linear(latent_size, hidden_size + img_size)
        self.fc2 = nn.Linear(img_size, hidden_size)
        self.hidden2mu2 = nn.Linear(hidden_size, latent_size)
        self.hidden2logv2 = nn.Linear(hidden_size, latent_size)

    def forward(self, imgs_mean, caps, val):
        batch_size = imgs_mean.size(0)
        img_text = torch.cat([imgs_mean, caps], dim=-1)
        h1 = torch.relu(self.fc1(img_text))
        mu1 = self.hidden2mu1(h1)
        logv1 = self.hidden2logv1(h1)
        h2 = torch.relu(self.fc2(imgs_mean))
        mu2 = self.hidden2mu2(h2)
        logv2 = self.hidden2logv2(h2)

        if not val:
            std = torch.exp(0.5 * logv1)
            z = torch.randn([batch_size, self.latent_size])
            if torch.cuda.is_available():
                z = z.cuda()
            z = z * std + mu1
        else:
            if self.standard_gaussian:
                z = torch.randn([batch_size, self.latent_size])
                if torch.cuda.is_available():
                    z = z.cuda()
            else:
                std = torch.exp(0.5 * logv2)
                z = torch.randn([batch_size, self.latent_size])
                if torch.cuda.is_available():
                    z = z.cuda()
                z = z * std + mu2
        img_text_rec = self.rec(z)

        return mu1, logv1, mu2, logv2, img_text.detach(), img_text_rec

class ImgCapCVAE(nn.Module):
    """
    Image Caption CVAE: ImgEncoder + CapEncoder + AttnDecoder + mappings
    """

    def __init__(self, standard_gaussian, attn_dim, embed_dim, hidden_size, latent_size, vocab_size, img_size=2048):
        super(ImgCapCVAE, self).__init__()

        self.standard_gaussian = standard_gaussian
        self.attn_dim = attn_dim
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.img_size = img_size

        self.embedding = Embedding(vocab_size, embed_dim, dropout=0.2)  # embedding layer
        self.imgEncoder = ImgEncoder()
        self.capEncoder = CapEncoder(self.embedding, embed_dim, hidden_size)
        self.attnDecoder = AttnDecoder(self.embedding, attn_dim, embed_dim, img_size, hidden_size, latent_size, vocab_size)
        self.cvaeEncoder = CvaeEncoder(standard_gaussian, hidden_size, img_size, latent_size)

        # # mappings
        # self.fc1 = nn.Linear(hidden_size + img_size, hidden_size)
        # self.hidden2mu1 = nn.Linear(hidden_size, latent_size)
        # self.hidden2logv1 = nn.Linear(hidden_size, latent_size)
        # self.fc2 = nn.Linear(img_size, hidden_size)
        # self.hidden2mu2 = nn.Linear(hidden_size, latent_size)
        # self.hidden2logv2 = nn.Linear(hidden_size, latent_size)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, imgs, encoded_captions, caption_lengths, val=False):
        batch_size = encoded_captions.size(0)
        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        imgs = imgs[sort_ind]

        imgs, imgs_mean = self.imgEncoder(imgs)
        caps = self.capEncoder(encoded_captions, caption_lengths)

        mu1, logv1, mu2, logv2, img_text, img_text_rec = self.cvaeEncoder(imgs_mean, caps, val)

        preds, alphas, decode_lengths = self.attnDecoder(imgs, encoded_captions, caption_lengths, img_text_rec)
        return preds, alphas, mu1, logv1, mu2, logv2, encoded_captions, decode_lengths, sort_ind

    def generate(self, beam, imgs, BOS, EOS):
        batch_size = imgs.size(0)
        imgs, imgs_mean = self.imgEncoder(imgs)
        caps = torch.randn(imgs_mean.size(0), self.hidden_size).to(imgs.device)
        mu1, logv1, mu2, logv2, img_text, img_text_rec = self.cvaeEncoder(imgs_mean, caps, val=True)
        # caps = self.capEncoder(encoded_captions, caption_lengths)

        # print(imgs_mean.size())

        # r = torch.randn(imgs_mean.size(0), self.hidden_size).to(imgs.device)
        # h1 = torch.relu(self.fc1(torch.cat([imgs_mean, r], dim=-1)))
        # mu1 = self.hidden2mu1(h1)
        # logv1 = self.hidden2logv1(h1)
        # h2 = torch.relu(self.fc2(imgs_mean))
        # mu2 = self.hidden2mu2(h2)
        # logv2 = self.hidden2logv2(h2)

        # if self.standard_gaussian:
        #     z = torch.randn([batch_size, self.latent_size])
        #     if torch.cuda.is_available():
        #         z = z.cuda()
        # else:
        #     std = torch.exp(0.5 * logv2)
        #     z = torch.randn([batch_size, self.latent_size])
        #     if torch.cuda.is_available():
        #         z = z.cuda()
        #     z = z * std + mu2

        return self.beam_search(beam, imgs, img_text_rec, BOS, EOS)

    def beam_search(self, beam, encoded_imgs, z, BOS, EOS):
        batch_size = encoded_imgs.size(0)
        img_dim = encoded_imgs.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoded_imgs = encoded_imgs.view(batch_size, -1, img_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoded_imgs.size(1)

        encoded_imgs = F.relu(self.attnDecoder.fc_attn(encoded_imgs))
        
        device = encoded_imgs.device
        srcLen = encoded_imgs.size(1)
        # print(encoded_imgs.size(), self.embed_dim)
        encoded_imgs = encoded_imgs.unsqueeze(1).repeat(1, beam, 1, 1).view(-1, srcLen, self.embed_dim)
        sentence = torch.LongTensor(batch_size*beam, 1).fill_(BOS).to(device)
        EOS_index = torch.BoolTensor(batch_size,  beam).fill_(False).to(device)

        totalProb = torch.zeros(batch_size, beam).to(device)
        for i in range(50):
            embeddings = self.embedding(sentence)    # [Batch*beam, 1, hidden]
        
            SeqMask = triu_mask(batch_size  * beam, i+1).to(device)
            
            for DecoderCell in self.attnDecoder.decoder:
                embeddings, _ = DecoderCell(embeddings, encoded_imgs, None, SeqMask)
            embeddings = F.relu(self.attnDecoder.fc_embed(torch.cat((embeddings, z.unsqueeze(1).repeat(beam, embeddings.size(1), 1)), dim=-1)))
            prob = F.softmax(self.attnDecoder.fc(embeddings[:, -1, :]), dim=-1)
            
            mask = EOS_index.view(batch_size*beam, 1)
            prob.masked_fill_(mask.repeat(1, self.vocab_size), 1)
            prob = torch.log(prob+1e-15) + totalProb.view(-1, 1)

            totalProb, index = prob.view(batch_size, -1).topk(beam, dim=-1, largest=True)
            # prob, index: [batch_size, beam]
            
            word = index % self.vocab_size
            index /= self.vocab_size
            EOS_index = EOS_index.gather(dim=-1, index=index)
            EOS_index |= (word==EOS)
            if EOS_index.sum() == batch_size * beam:
                break
            
            sentence = sentence.view(batch_size, beam, -1).gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i+1))
            sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1).view(batch_size*beam, -1)

        index = totalProb.max(1)[1]
        sentence = sentence.view(batch_size, beam, -1)
        outputs = []
        for i in range(batch_size):
            sent = sentence[i, index[i], :].tolist()
            if EOS in sent:
                sent = sent[:sent.index(EOS)]
            outputs.append(sent[1:])

        return outputs


