import logging
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingConfig, MultilingualDenoisingTask

logger = logging.getLogger(__name__)

@register_task("utut_pretraining", dataclass=MultilingualDenoisingConfig)
class UTUTPretrainingTask(MultilingualDenoisingTask):
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        lang_list = self.cfg.langs.split(",")

        lang_token_ids = set()
        for lang in lang_list:
            token = "[{}]".format(lang)
            idx = self.dictionary.index(token)
            if idx != self.dictionary.unk():
                lang_token_ids.add(idx)

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        eos_token = "[{}]".format(self.target_language)
        eos_idx = self.dictionary.index(eos_token)
        if eos_idx != self.dictionary.unk():
            extra_gen_cls_kwargs["eos"] = eos_idx

        tokens_to_suppress = []
        for lang in lang_list:
            if lang != self.target_language:
                if self.dictionary.index("[{}]".format(lang)) != self.dictionary.unk():
                    tokens_to_suppress.append("[{}]".format(lang))
        
        # Add mask token
        if self.mask_idx in self.dictionary:
            tokens_to_suppress.append(self.dictionary[self.mask_idx])
            
        extra_gen_cls_kwargs["tokens_to_suppress"] = tokens_to_suppress

        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )
