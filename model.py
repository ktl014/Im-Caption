import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, architecture='resnet50',
                 attention=False, encoded_image_size=11):
        """Load the pretrained ResNet-50 and replace top fc layer.
        :param attention:
        """
        super(EncoderCNN, self).__init__()
        self.architecture = architecture
        self.attention = attention
        if self.architecture == 'resnet50':
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features

            if not self.attention:
                modules = list(model.children())[:-1] #####MARKER
            else:
                modules = list(model.children())[:-2]

            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (encoded_image_size, encoded_image_size))

            self.model = nn.Sequential(*modules)

        elif self.architecture == 'alexnet':
            model = models.alexnet(pretrained=True)
            modules = list(model.children())[:-1]
            in_features = 4096
            self.model = nn.Sequential(*modules)
            self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(9216, in_features),
                                        nn.BatchNorm1d(in_features, momentum=0.01))
            
        elif self.architecture == 'vgg16':
            model = models.vgg16(pretrained=True)
            modules = list(model.children())[:-1]
            in_features = model.classifier[0].in_features
            self.model = nn.Sequential(*modules)

        elif self.architecture == 'densenet161':
            model = models.densenet161(pretrained=True)
            in_features = model.classifier.in_features
            modules = list(model.children())[:-1] 
            self.model = nn.Sequential(*modules, nn.AvgPool2d(kernel_size=7, stride=1, padding=0))

        self.embed = nn.Linear(in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.model(images)
        print(features.shape)

        #####MARKER
        if not self.attention:
            features = features.view(features.size(0), -1)
            if self.architecture == "alexnet":
                features = self.classifier(features)
            features = self.embed(features)
            print(features.shape)
            features = self.bn(features)
            print(features.shape)
        else:
            features = self.adaptive_pool(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted 
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted 
        sentences. This is the beam search approach.
        """
        # Top word idx sequences and their corresponding inputs and states
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point 
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]


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

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, vocab_size, attention_dim=512, embed_dim=512,
                 decoder_dim=512, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

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

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return:
            scores: scores for vocabulary (batch_size, 15, 9490)
            caps_sorted: sorted encoded captions (batch_size, 52)
            decode_lengths: (list) decode lengths
            alphas: weights (batch_size, 15, 196)
            sort_ind: sorted indices (batch_size)
        """
        encoder_out = encoder_out.permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0,
        #                                                             descending=True)
        '''
        caption_lengths = caption_lengths+2
        caption_lengths, sort_ind = caption_lengths.sort(dim=0,
                                                         descending=True)
        assert isinstance(caption_lengths[0].item(), int) # change squeeze dim if receiving errors


        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        '''
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        #decode_lengths = (caption_lengths - 1).tolist()
        decode_lengths = [encoded_captions.shape[1] - 1]

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths),
                                  vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths),
                             num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            '''
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
            '''

            attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                                h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        return predictions, alphas



if __name__ == '__main__':
    import json
    import os
    import pickle

    # Data parameters
    data_folder = 'data'  # folder with data files saved by create_input_files.py
    data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    f = open(os.path.join(data_folder, 'sample-input_curr-model.pkl'), 'rb')
    sample_input = pickle.load(f)
    img, caps, vocab_size, capslens = sample_input[0], sample_input[1], \
                               sample_input[2], sample_input[3]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    encoder = EncoderCNN(attention=True)
    decoder = DecoderRNN(vocab_size)
    decoder_attention = DecoderWithAttention(vocab_size=vocab_size)

    encoder.to(device)
    decoder.to(device)
    decoder_attention.to(device)

    img = encoder(img)

    # scores = decoder(img, caps)
    # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_attention(
    #     img, caps, capslens)
    scores, alphas = decoder_attention(
             img, caps)

    criterion = nn.CrossEntropyLoss().to(device)

    # Calculate the batch loss
    # scores --> (32, 8, 6335) enc: (batch_size, decode_length, vocab_size)
    scores = scores.view(-1, vocab_size)
    # scores --> XXX (256, 6335) ERROR HERE

    # targets --> (32 x 11)
    targets = caps[:,1:].flatten()
    # targets --> (352)

    loss = criterion(scores, targets)
    print(loss)
