'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, RNN, Embedding}

        module_types = {'lstm', 'rnn'}
        models = {'lstm': torch.nn.LSTM, 'rnn': torch.nn.RNN}

        # Create embedding to convert discrete one-hot encoding to continuous fdrvp

        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        # Create the rnn
        self.rnn = models[self.cell_type](input_size=self.word_embedding_size,
                                          hidden_size=self.hidden_state_size,
                                          num_layers=self.num_hidden_layers,
                                          dropout=self.dropout,
                                          bidirectional=True)

        return

    def forward_pass(
            self,
            source_x: torch.LongTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        embedding = self.get_all_rnn_inputs(source_x)

        return self.get_all_hidden_states(embedding, source_x_lens, h_pad)

    def get_all_rnn_inputs(self, source_x: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   x (output) is shape (S, B, I)

        # Create the embedding
        return self.embedding(source_x)

    def get_all_hidden_states(
            self,
            x: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, B, I)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #   h (output) is of shape (S, B, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        packed = torch.nn.tutils.rnn.pack_padded_sequence(input=x, lengths=source_x_lens, enforce_sorted=False)

        # Get the output
        h_packed, _ = self.rnn(packed)

        # Convert it back
        h_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(h_packed, padding_value=h_pad)

        return h_padded


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell}

        models = {'lstm': torch.nn.LSTM, 'rnn': torch.nn.RNN}

        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.rnn = models[self.cell_type](input_size=self.word_embedding_size,
                                          hidden_size=self.hidden_state_size,
                                          num_layers=self.num_hidden_layers,
                                          dropout=self.dropout,
                                          bidirectional=True)

        self.output_layer = torch.nn.Linear(in_features=self.hidden_state_size,
                                            out_features=self.target_vocab_size)

        return

    def forward_pass(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> Tuple[
        torch.FloatTensor, Union[
            torch.FloatTensor,
            Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   target_y_tm1 is of shape (B,)
        #   htilde_tm1 is of shape (B, 2 * H)
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   logits_t (output) is of shape (B, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.

        xtilde_t = self.get_current_rnn_input(target_y_tm1, htilde_tm1, h, source_x_lens)
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   htilde_tm1 (output) is of shape (B, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        forward = h[source_x_lens - 1, [_ for _ in range(source_x_lens.shape[0])], :self.hidden_state_size // 2]
        backward = h[0, :, self.hidden_state_size // 2:]
        return torch.cat((forward, backward), dim=1).to(h.device)

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   target_y_tm1 is of shape (B,)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   xtilde_t (output) is of shape (B, Itilde)
        return self.embedding(htilde_tm1)

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (B, Itilde)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1
        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (B, V)

        if self.cell_type == "lstm":
            return self.output_layer(htilde_t[0])
        else:
            return self.output_layer(htilde_t)


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        cells = {"lstm": torch.nn.LSTMCell, "rnn": torch.nn.RNNCell}

        # Now since we have to work with attention, input size has to address context vectors
        self.cell = cells[self.cell_type](input_size=self.word_embedding_size + self.hidden_state_size,
                                          hidden_size=self.hidden_state_size)

        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.output_layer = torch.nn.Linear(in_features=self.hidden_state_size,
                                            out_features=self.target_vocab_size)

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.

        return torch.zeros_like(h[0]).to(h.device)

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t

        embedded = self.embedding(target_y_tm1)
        c_t = self.attend(htilde_tm1, h, source_x_lens)

        return torch.cat((embedded, c_t), dim=1).to(h.device)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, B, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, b, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``b``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[source_x_lens[b]:, b]``
            should all be ignored.
        source_x_lens : torch.LongTensor
            An integer tensor of shape ``(B,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(B, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        alpha_t = self.get_attention_weights(htilde_t, h, source_x_lens)

        alpha_trans = alpha_t.unsqueeze(dim=1).transpose(0, 2)

        h_trans = h.transpose(0, 1)

        return torch.matmul(alpha_trans, h_trans).squeeze(1)

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, B)
        a_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= source_x_lens.to(h.device)  # (S, B)
        a_t = a_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(a_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H)
        #   h is of shape (S, B, 2 * H)
        #   a_t (output) is of shape (S, B)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        if self.cell_type == "lstm":
            return torch.nn.CosineSimilarity(dim=-1)(htilde_t[0], h)
        else:
            return torch.nn.CosineSimilarity(dim=-1)(htilde_t, h)


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.output_layer, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        self.W = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Q = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)


def attend(
        self,
        htilde_t: Union[
            torch.FloatTensor,
            Tuple[torch.FloatTensor, torch.FloatTensor]],
        h: torch.FloatTensor,
        source_x_lens: torch.LongTensor) -> torch.FloatTensor:
    # Hints:
    # 1. You can use super().attend to call for the regular attention
    #   function.
    # 2. Relevant pytorch function:
    #   tensor().view, tensor().repeat_interleave
    # 3. Fun fact:
    #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
    #   tensor([1,2,3,4]).repeat_interleave(2) will output
    #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
    # 4. You *WILL* need self.heads at this point

    if self.cell_type == "lstm":
        htilde_t_n = (self.Wtilde(htilde_t[0]).view(h.shape[1] * self.heads, h.shape[2] // self.heads),
                      htilde_t[1].view(h.shape[1] * self.heads, h.shape[2] // self.heads))
    else:
        htilde_t_n = self.Wtilde(htilde_t).view(h.shape[1] * self.heads, h.shape[2] // self.heads)

        # Transformation: h_n = S * MK * H/K, after applying linear layer
    h_n = self.W(h).view(h.shape[0], h.shape[1] * self.heads, h.shape[2] // self.heads)

    lens_n = source_x_lens.repeat_interleave(self.heads)

    # Call attend (for single-head) and transform from MK * H/K to M * H context vector
    c_t_n = super().attend(htilde_t_n, h_n, lens_n).view(h.shape[1], h.shape[2])

    # Apply a final linear layer
    return self.Q(c_t_n)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it.

        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)

        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type,
                                     heads=self.heads)

    def translate(self, input_sentence):
        # This method translates the input sentence from the model's source
        # language to the target language.
        # 1. Tokenize the input sentence.
        # 2. Compute the length of the input sentence.
        # 3. Feed the tokenized sentence into the model.
        # 4. Decode the output of the sentence into a string.

        # Hints:
        # 1. You will need the following methods/attributs from the dataset.
        # Consult :class:`HansardEmptyDataset` for a description of parameters
        # and attributes.
        #   self.dataset.tokenize(input_sentence)
        #       This function tokenizes the input sentence.  For example:
        #       >>> self.dataset.tokenize('This is a sentence.')
        #       ['this', 'is', 'a', 'sentence']
        #   self.dataset.source_word2id
        #       A dictionary that maps tokens to ids for the source language.
        #       For example: `self.dataset.source_word2id['francophone'] -> 5127`
        #   self.dataset.source_unk
        #       The speical token for unknown input tokens.  Any token in the
        #       input sentence that isn't present in the source vocabulary should
        #       be converted to this special token.
        #   self.dataset.target_id2word
        #       A dictionary that maps ids to tokens for the target language.
        #       For example: `self.dataset.source_word2id[6123] -> 'anglophone'`
        # 
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave

        tokens = self.dataset.tokenize(input_sentence)
        source_ids = [self.dataset.source_word2id.get(token, self.dataset.source_unk) for token in tokens]

        # Convert ids to tensor
        source_tensor = torch.tensor(source_ids).view(-1, 1)

        # Compute length of input sentence
        source_len = torch.tensor([len(source_ids)])

        # Feed the tokenized sentence into the model
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(source_tensor, source_len)

            # Initialize the decoder input with the <sos> token
            decoder_input = torch.tensor([[self.dataset.target_sos]])

            # Initialize the decoder hidden state with the encoder hidden state
            decoder_hidden = hidden

            # Initialize a list to hold the decoded tokens
            decoded_tokens = []

            # Decode the output of the sentence into a string
            for _ in range(self.dataset.max_target_len):
                # Compute the scores for the next token
                scores, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

                # Get the index of the token with the highest score
                top_index = scores.argmax(1)

                if top_index.item() == self.dataset.target_eos:
                    # If end, break.
                    break

                # Append the decoded token to the list
                decoded_tokens.append(self.dataset.target_id2word[top_index.item()])

                # Update the decoder input with the top token
                decoder_input = top_index.view(-1, 1)

            # Convert the decoded tokens to a string
            output_sentence = ' '.join(decoded_tokens)

        return output_sentence

        return

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            target_y: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   target_y is of shape (T, B)
        #   logits (output) is of shape (T - 1, B, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than target_y (why?)

        htilde_t = None

        # Logits size
        logits = torch.zeros(target_y.shape[0] - 1, h.shape[1], self.target_vocab_size).to(h.device)

        # Iterate through timestep
        for t in range(1, target_y.shape[0]):
            htilde_tm1 = htilde_t

            # Inputs are Etm1 and htilde_tm1
            logits_t, htilde_t = self.decoder.forward(target_y[t - 1, :], htilde_tm1, h, source_x_lens)
            logits[t - 1, :, :] = logits_t

        logits = logits.to(h.device)
        return logits

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (B, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (B, K)
        #   b_tm1_1 is of shape (t, B, K)

        ## Output order:
        #   logpb_t (first output) is of shape (B, K)
        #   b_t_0 (second output) is of shape (B, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (third output) is of shape (t + 1, B, K)
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (X, Y),
        #   then the element z[a, b] maps to z'[a*Y + b]

        logpb_tm1_repeat = logpb_tm1.unsqueeze(-1).repeat(1, 1, self.target_vocab_size)

        extensions_t = (logpb_tm1_repeat + logpy_t).reshape(logpy_t.shape[0],
                                                            logpy_t.shape[1] * logpy_t.shape[2])

        logpb_t, indices = extensions_t.topk(logpb_tm1.shape[1], dim=1)

        valid_prefixes = torch.div(indices, self.target_vocab_size).unsqueeze(0)
        next_words = torch.remainder(indices, self.target_vocab_size).unsqueeze(0)

        b_tm1_1 = b_tm1_1.gather(2, valid_prefixes.expand_as(b_tm1_1).type(torch.int64))

        b_t_1 = torch.cat([b_tm1_1, next_words], dim=0)

        valid_prefixes = valid_prefixes.reshape(valid_prefixes.shape[1], valid_prefixes.shape[2],
                                                valid_prefixes.shape[0])

        if self.cell_type == 'lstm':
            b_t_0 = (htilde_t[0].gather(1, valid_prefixes.expand_as(htilde_t[0]).type(torch.int64)),
                     htilde_t[1].gather(1, valid_prefixes.expand_as(htilde_t[1]).type(torch.int64)))
        else:
            b_t_0 = htilde_t.gather(1, valid_prefixes.expand_as(htilde_t).type(torch.int64))

        return logpb_t, b_t_0, b_t_1
