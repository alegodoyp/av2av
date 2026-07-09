from fairseq.models import register_model_architecture
from fairseq.models.bart.model import bart_large_architecture


@register_model_architecture("bart", "utut_large")
def utut_large_architecture(args):
    # Matches utut_sts_ft.pt exactly (confirmed via diagnose_utut_ckpt.py):
    # sinusoidal positional embeddings, no layernorm_embedding, scale embeddings.
    # Must be set BEFORE calling bart_large_architecture, which uses getattr(args, ..., True)
    # and would otherwise override to True if these are not already present in args.
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)
