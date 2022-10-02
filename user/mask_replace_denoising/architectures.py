from fairseq.models import register_model_architecture
from fairseq.models import transformer
from fairseq.models.bart import bart_base_architecture


@register_model_architecture('transformer', 'mbart_base_sinusoidal')
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)

    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    bart_base_architecture(args)


@register_model_architecture("transformer", "transformer_marss_base")
def transformer_marss_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.dropout = getattr(args, "dropout", 0.3)
    transformer.base_architecture(args)


@register_model_architecture("transformer", "transformer_baseline")
def transformer_marss_base(args):
    heads = 8
    layers = 6
    emb_dim = 1024
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", ffn_dim)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads",
                                           heads)
    args.encoder_layers = getattr(args, "encoder_layers", layers)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", ffn_dim)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads",
                                           heads)
    args.decoder_layers = getattr(args, "decoder_layers", layers)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.dropout = getattr(args, "dropout", 0.3)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            True)

    transformer.base_architecture(args)


@register_model_architecture("transformer", "transformer_baseline_iwslt")
def transformer_marss_base(args):
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            True)
    transformer.transformer_iwslt_de_en(args)
