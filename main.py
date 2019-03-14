"""Main Script"""

# Standard dist imports
import math
import os
import time

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Project level imports
from utils import train, validate, save_epoch, early_stopping, set_cuda, \
    clean_sentence, get_prediction, load_checkpoint
from data_loader import get_loader, transform
from model import EncoderCNN, DecoderRNN

# Module level constants
MODE = 'train'
CNN_ARCH = 'resnet50'
MODEL_DIR = './models/test-model'
RESUME = None

# Check if mode is specified correctly
assert MODE in ['train', 'eval', 'deploy']
# Initialize model directory to save checkpoints
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    # Set values for the training variables
    batch_size = 32  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 10  # number of training epochs
    lr = 0.001 # learning rate
    estop_threshold = 3 # early stop threshold
    start_epoch = 0 # starting epoch
    start_loss = 0 # starting loss

    # Build data loader, applying the transforms
    if MODE == 'train':
        train_loader = get_loader(transform=transform['train'],
                                  mode='train',
                                  batch_size=batch_size,
                                  vocab_threshold=vocab_threshold,
                                  vocab_from_file=vocab_from_file)
        val_loader = get_loader(transform=transform['val'],
                                mode='val',
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                vocab_from_file=vocab_from_file)

        # The size of the vocabulary
        vocab_size = len(train_loader.dataset.vocab)

        # Set the total number of training and validation steps per epoch
        total_train_step = math.ceil(len(
            train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
        total_val_step = math.ceil(len(
            val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
        print("Number of training steps:", total_train_step)
        print("Number of validation steps:", total_val_step)
    else:
        # Create the data loader
        test_loader = get_loader(transform=transform['test'],
                                 mode='test')
        total_test_step = math.ceil(len(test_loader.dataset.caption_lengths)
                                    / test_loader.batch_sampler.batch_size)
        print("Number of test steps:", total_test_step)

        # Get the vocabulary and its size
        vocab = test_loader.dataset.vocab
        vocab_size = len(vocab)

    # Initialize the encoder and decoder
    encoder = EncoderCNN(embed_size, architecture=CNN_ARCH)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available
    computing_device = set_cuda()
    encoder.to(computing_device)
    decoder.to(computing_device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(computing_device) \
        if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model
    params = list(decoder.parameters()) + list(
        encoder.embed.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=lr)

    # Resume from a checkpoint
    if RESUME or MODE != 'train':
        fn = RESUME if RESUME else MODEL_DIR + '/best-model.pkl'
        encoder, decoder, optimizer,\
        start_loss, start_epoch = load_checkpoint(encoder, decoder,
                                                  optimizer, MODE, fn)

    #========================== Train Network ================================#
    # Keep track of train and validation losses and validation Bleu-4 scores by epoch
    if MODE == 'train':
        train_losses = []
        val_losses = []
        val_bleus = []
        # Keep track of the current best validation Bleu score
        best_val_bleu = float("-INF")

        start_time = time.time()
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss = train(train_loader, encoder, decoder, criterion,
                               optimizer, vocab_size, epoch,
                               total_train_step, start_loss=start_loss)
            train_losses.append(train_loss)
            val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion,
                                          train_loader.dataset.vocab, epoch,
                                          total_val_step)
            val_losses.append(val_loss)
            val_bleus.append(val_bleu)
            if val_bleu > best_val_bleu:
                print(
                    "Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
                    format(best_val_bleu, val_bleu))
                best_val_bleu = val_bleu
                filename = os.path.join(MODEL_DIR, "best-model.pkl")
                save_epoch(filename, encoder, decoder, optimizer, train_losses,
                           val_losses,
                           val_bleu, val_bleus, epoch)
            else:
                print(
                    "Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(
                        epoch))
            # Save the entire model anyway, regardless of being the best model so far or not
            filename = os.path.join(MODEL_DIR, "model-{}.pkl".format(epoch))
            save_epoch(filename, encoder, decoder, optimizer, train_losses,
                       val_losses,
                       val_bleu, val_bleus, epoch)
            print("Epoch [%d/%d] took %ds" % (
            epoch, num_epochs, time.time() - start_time))
            if epoch > 5:
                # Stop if the validation Bleu doesn't improve for 3 epochs
                if early_stopping(val_bleus, estop_threshold):
                    break
            start_time = time.time()

    # ======================== Evaluate Network ==============================#
    elif MODE == 'eval':
        test_loss, test_bleu = validate(test_loader, encoder, decoder,
                                        criterion, optimizer,
                                        test_loader.dataset.vocab, total_test_step)
        print('FINAL RESULTS')
        print('Test Loss: {} | BLEU: {}'.format(test_loss, test_bleu))


    # ========================== Deploy Network ==============================#
    elif MODE == 'deploy':
        # Obtain sample image before and after pre-processing
        orig_image, image = next(iter(test_loader))
        # Convert image from torch.FloatTensor to numpy ndarray
        transformed_image = image.numpy()
        # Remove the first dimension which is batch_size euqal to 1
        transformed_image = np.squeeze(transformed_image)
        transformed_image = transformed_image.transpose((1, 2, 0))

        # Obtain the embedded image features.
        features = encoder(image.cuda()).unsqueeze(1)

        # Pass the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features)
        print('example output:', output)

        sentence = clean_sentence(output, vocab)
        print('example sentence:', sentence)

        get_prediction(test_loader, encoder, decoder, vocab)



if __name__ == '__main__':
    main()
