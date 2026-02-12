
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder, 
    FairseqEncoderDecoderModel,
    register_model, 
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerModel
)
from fairseq.modules import (
    PositionalEmbedding, 
    RelPositionalEncoding,
    FairseqDropout
)
from fairseq.modules.conformer_layer import ConformerEncoderLayer

logger = logging.getLogger(__name__)


class ConformerUtutEncoder(FairseqEncoder):
    """
    Conformer Encoder for Unit-to-Unit translation.
    Adapts S2T Conformer to accept discrete unit inputs.
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        self.embed_tokens = embed_tokens
        self.padding_idx = dictionary.pad()
        self.max_source_positions = args.max_source_positions

        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0

        self.pos_enc_type = args.pos_enc_type
        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.pos_enc_type = "abs"
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.conformer_layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                    attn_type=args.attn_type,
                    pos_enc_type=self.pos_enc_type,
                    use_fp16=args.fp16,
                )
                for _ in range(args.encoder_layers)
            ]
        )
        if args.encoder_normalize_before:
            self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forwarded(self, src_tokens, src_lengths=None, return_all_hiddens=False):
        # Embed tokens
        x = self.embed_scale * self.embed_tokens(src_tokens)
        
        # Calculate padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Add positional encodings
        if self.pos_enc_type == "rel_pos":
            positions = self.embed_positions(x)
        elif self.pos_enc_type == "rope":
            positions = None
        else:
            if self.embed_positions is not None:
                positions = self.embed_positions(src_tokens)
                x += positions
            positions = None

        x = self.dropout_module(x)
        
        # Conformer layers expect [T, B, C]
        x = x.transpose(0, 1)

        encoder_states = []

        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask, positions)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False):
        return self.forwarded(src_tokens, src_lengths, return_all_hiddens)

    def reorder_encoder_out(self, encoder_out, new_order):
        return TransformerModel.reorder_encoder_out(self, encoder_out, new_order)


@register_model("conformer_utut")
class ConformerUtutModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--depthwise-conv-kernel-size",
            type=int,
            metavar="N",
            help="kernel size of depthwise convolution layers",
        )
        parser.add_argument(
            "--attn-type",
            type=str,
            metavar="STR",
            help="If not specified uses fairseq MHA. Other valid option is espnet",
        )
        parser.add_argument(
            "--pos-enc-type",
            type=str,
            metavar="STR",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        
        # make sure all arguments are present in older models
        conformer_utut_architecture(args)

        if getattr(args, "encoder_layers_to_keep", None):
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if getattr(args, "decoder_layers_to_keep", None):
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if task.source_dictionary is None:
            raise ValueError("Task must provide a source dictionary")
        if task.target_dictionary is None:
            raise ValueError("Task must provide a target dictionary")

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim
        )
        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim
        )
        
        encoder = ConformerUtutEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            decoder_embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        return cls(encoder, decoder)


@register_model_architecture("conformer_utut", "conformer_utut")
def conformer_utut_architecture(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    
    # Defaults for Transformer Decoder
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.max_source_positions = getattr(args, "max_source_positions", 4096)
    args.fp16 = getattr(args, "fp16", False)
