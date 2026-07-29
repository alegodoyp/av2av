# Bloco E — Reconciliação documento × código

**Nota de execução importante**: este relatório foi gerado a partir de um checkout
Windows local do repositório (`c:\Users\lelex\source\repos\Mestrado\av2av`), **não**
na VM (`/mnt/disk/home/alexandregodoy/av2av`). Não foi possível ativar
`av2av_env`, carregar `checkpoints/full_model/checkpoint_best.pt` (esse arquivo nem
existe neste checkout — `checkpoints/` está vazio localmente, é ignorado pelo git) nem
rodar `nvidia-smi`/`torch` diretamente. Tudo abaixo foi verificado por uma destas três
fontes, sempre citada:
1. Código-fonte do repositório (mirror local via git, idêntico ao do GitHub).
2. `logs_treino_full.txt` (rastreado no git) — log real de uma execução de
   `fairseq-train` em 2026-07-09.
3. `mlflow.db` (rastreado no git) — consultado diretamente via `sqlite3`/Python.

Qualquer item que dependesse de ler o checkpoint ou o ambiente da VM diretamente está
marcado **NÃO CONFIRMADO**, com o comando exato para você rodar lá.

---

## E1 — Existe CTC em algum lugar?

**Resposta categórica: NÃO. Não há termo CTC na perda otimizada em nenhuma das duas
execuções de treino encontradas neste repositório.**

- `--criterion` efetivamente usado: `label_smoothed_cross_entropy`, em **ambas** as
  execuções reais encontradas:
  - Execução de 2026-07-09 (arch `utut_large`): `logs_treino_full.txt:140`,
    campo `'criterion': {'_name': 'label_smoothed_cross_entropy', 'label_smoothing': 0.2, ...}`.
  - Execução de 2026-02-13 (arch `conformer_utut`, ver E8): `mlflow.db`, params
    `criterion._name = label_smoothed_cross_entropy`, `criterion.label_smoothing = 0.1`.
- Busca em todo o repositório (`grep -rni "ctc" --include=*.py --include=*.yaml --include=*.json .`,
  excluindo `fairseq/tests/`): **todos** os hits de conteúdo real ("ctc" de verdade, não o
  falso-positivo `DictConfig`→"ct"+"C") caem em dois grupos, nenhum deles no caminho
  executado:
  - `fairseq/examples/**` (145 ocorrências) — exemplos genéricos do fairseq para outras
    tarefas (ASR com wav2vec/HuBERT, alinhamento MMS, Tacotron2, FastSpeech2). Nenhum
    desses módulos é importado por `scripts/`, `av2unit/`, `unit2unit/`, `unit2av/`,
    `inference.py` ou `util.py` (confirmei: zero imports de `fairseq.examples.*` nesses
    arquivos).
  - `fairseq/fairseq/criterions/ctc.py`, `label_smoothed_cross_entropy_with_ctc.py`,
    `fastspeech2_loss.py`, `tacotron2_loss.py`, `speech_to_speech_criterion.py`,
    `fairseq/fairseq/models/hubert/hubert_asr.py` (`hubert_ctc`), `wav2vec2_asr.py`
    (`wav2vec_ctc`), `s2s_conformer*.py`, `speech_to_text/*.py` — todos são
    **criterions/modelos alternativos da biblioteca fairseq, registrados sob outros
    nomes** (`ctc`, `hubert_ctc`, `wav2vec_ctc`, `label_smoothed_cross_entropy_with_ctc`
    etc.), nunca selecionados por `--criterion`/`--arch`/`--task` em nenhum script deste
    projeto.
- Também verifiquei o código-fonte de `unit2unit/models/conformer_utut.py` (a
  arquitetura Conformer da execução de 2026-02-13, ver E7/E8) diretamente: usa
  `TransformerDecoder` padrão do fairseq (linha 251), sem nenhuma referência a CTC no
  arquivo inteiro (`grep -n "ctc\|CTC"` retornou vazio).

**Conclusão para reescrita**: a afirmação de uma perda híbrida `L = λ·L_CTC + (1−λ)·L_Att`
com λ=0,3 não corresponde a nenhuma execução real registrada neste repositório, em
nenhuma das duas arquiteturas encontradas. É `label_smoothed_cross_entropy` puro nos
dois casos.

---

## E2 — Otimizador, scheduler e hiperparâmetros efetivos

**NÃO CONFIRMADO diretamente do `cfg` do checkpoint** (arquivo não acessível deste
checkout). Em vez disso, uso o `cfg`/Namespace **logado ao vivo por duas execuções reais
distintas de `fairseq-train`** — isso é uma cópia funcionalmente equivalente ao que
fairseq grava dentro do checkpoint (`ck['cfg']` é serializado a partir do mesmo objeto
de configuração impresso no log), mas não é bit-a-bit o mesmo arquivo. Para certeza
absoluta, rode na VM:
```bash
cd /mnt/disk/home/alexandregodoy/av2av && conda activate av2av_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -c "
import torch, json
ck = torch.load('checkpoints/full_model/checkpoint_best.pt', map_location='cpu')
print(json.dumps(ck['cfg'], default=str, indent=2))
"
```
(nota: confirme antes que `checkpoint_best.pt` existe — `ls checkpoints/full_model/`;
o log só mostra `checkpoint_last.pt` sendo salvo).

Há **duas execuções reais** registradas neste repositório, com configurações
**diferentes entre si** — isso é abordado com detalhe em E8. A tabela abaixo cobre a
execução mais recente (2026-07-09, arch `utut_large`, `logs_treino_full.txt:140`), que
é a que gerou o `checkpoint_last.pt` mais recente em `checkpoints/full_model/`:

| Parâmetro | Valor real (log 2026-07-09) | Valor alegado na dissertação |
|---|---|---|
| Arquitetura | `utut_large` (fairseq `bart`, ver E7) | Conformer encoder + Transformer decoder |
| Criterion | `label_smoothed_cross_entropy` | Híbrida CTC/Attention, λ=0,3 |
| Optimizer | `adam` (`_name: 'adam'`, `weight_decay: 0.0`) | AdamW |
| LR scheduler | `polynomial_decay` (`power=1.0`, ou seja, decaimento **linear**) | Warmup Cosine Decay |
| lr | `1e-4` (`[0.0001]`) | não informado na sua lista, mas note o scheduler diverge |
| warmup_updates | `3000` | — |
| total_num_update | `50000` | — |
| clip_norm | `0.1` | — |
| label_smoothing | `0.2` | — |
| dropout | `0.1` | `0,3` nas camadas densas |
| attention_dropout | `0.1` | `0,1` na atenção (**este bate**) |
| max_epoch | `100` (config), mas só chegou a epoch 7 seguindo o log disponível | — |
| batch_size / max_tokens | `16` / `200000` | — |
| seed | `1` (default do fairseq, nunca sobrescrito pelo script) | — |

**Divergências confirmadas**: optimizer (`adam`, não `AdamW`), scheduler
(`polynomial_decay` linear, não cosine), dropout das camadas densas (0,1, não 0,3).
`attention_dropout=0.1` é a única coincidência exata com o que está na dissertação.

A execução de 2026-02-13 (arch `conformer_utut`) tem valores ainda mais distantes:
`lr_scheduler=inverse_sqrt`, `lr=[0.0005]`, `warmup_updates=4000`, `clip_norm=0.0`,
`label_smoothing=0.1`, `dropout=0.1`, `finetune_from_model=None` (treinado do zero, não
fine-tuned), `disable_validation=True`. Fonte: `mlflow.db`, tabela `params`, run
`a27a9395759041689d2a58d7b8ce9c90`.

---

## E3 — Qual mecanismo produz a isocronia, de fato?

- **Taxa de unidades vs. frame rate**: razão **fixa e hardcoded**, não aprendida e não
  relacionada a CTC: `units_per_second = 50`, `frames_per_second = 25`,
  `code_frame_ratio = 50 // 25 = 2` — [unit2av/model.py:36-39](unit2av/model.py#L36-L39).
  Ou seja, **2 unidades discretas por frame de vídeo**, sempre.
- **`unit2unit` (UTUT) NÃO preserva comprimento de sequência.** É geração seq2seq por
  beam search comum (`task.build_generator`,
  [unit2unit/inference.py:79-81](unit2unit/inference.py#L79-L81)), com comprimento de
  saída controlado por `cfg.generation.max_len_a = 1.2` e `max_len_b = 100`
  ([unit2unit/inference.py:68-69](unit2unit/inference.py#L68-L69)) — ou seja, até
  aproximadamente `1.2×len(entrada) + 100` tokens, um teto flexível, não uma restrição
  de igualdade de comprimento. Não há `length predictor` nem restrição monotônica nessa
  etapa — é aqui, aliás, que o pipeline historicamente sofreu com loops de repetição em
  sentenças longas (mitigado via `no_repeat_ngram_size=3`, mesma linha 77, e via chunking
  em `inference.py`, não relacionado a CTC).
- **Quem de fato controla a duração é um `dur_predictor` dentro do `unit2av`**, não o
  `unit2unit`: em `CodeHiFiGANModel_spk.forward()`
  ([unit2av/model.py:235-240](unit2av/model.py#L235-L240)), um módulo
  `self.dur_predictor` prevê, por código discreto, quantos frames a 50Hz aquele código
  deve durar (`log_dur_pred`→`dur_out`); em seguida
  `torch.repeat_interleave(x, dur_out.view(-1), dim=2)` expande a sequência de acordo.
- **O Unit AV Renderer gera vídeo condicionado à MESMA sequência já expandida em
  duração**, não a duas sequências independentes mantidas em sincronia por uma
  restrição externa: `UnitAVRenderer.forward()`
  ([unit2av/model.py:141-224](unit2av/model.py#L141-L224)) recebe de volta
  `dedup_code` (a sequência de códigos já expandida pelo `dur_predictor`, mesmo tensor
  que gerou o áudio), calcula `padded_tgt_len = len(dedup_code) // code_frame_ratio`
  (linha 149) e usa exatamente esse número de frames para reamostrar/repetir o vídeo de
  fundo (linhas 169-188) antes de alimentar `self.face_model` com `dedup_code_seq`
  (linha 218). Ou seja: **um único tensor de (código, duração) dirige as duas
  modalidades**; a sincronia é uma consequência estrutural de derivarem do mesmo dado,
  não de uma otimização conjunta ou de uma restrição de alinhamento como CTC.

**Parágrafo para a dissertação** (adaptável):

> A isocronia entre fala e vídeo gerados não decorre de uma restrição de monotonicidade
> imposta durante o treinamento do tradutor de unidades (`unit2unit`), que na verdade
> gera sua saída por busca em feixe padrão, sem preservar o comprimento da entrada e sem
> qualquer mecanismo do tipo CTC. A sincronia é, em vez disso, uma propriedade estrutural
> do estágio final do pipeline (`unit2av`): um preditor de duração interno ao vocoder
> (`CodeHiFiGANModel_spk`) estima, para cada unidade discreta da sequência traduzida,
> quantos quadros de áudio a 50 Hz ela deve ocupar, e essa mesma sequência de unidades
> já expandida em duração — e não duas sequências independentes — é o que condiciona
> tanto a síntese de áudio (pelo decodificador convolucional) quanto a síntese de vídeo
> (pelo `FaceRenderer`), com uma razão fixa de duas unidades por quadro de vídeo (50 Hz
> de unidades para 25 fps). Áudio e vídeo são, portanto, funções determinísticas do mesmo
> tensor de (unidade, duração), o que garante o alinhamento por construção, e não por
> uma restrição aprendida de alinhamento monotônico.

---

## E4 — O Whisper é usado no caminho executado?

**Resposta categórica: NÃO — Whisper não é usado em nenhum lugar deste projeto.**

- `grep -rni "whisper" --include=*.py .`: as únicas ocorrências em todo o repositório
  estão em `fairseq/examples/mms/lid_rerank/whisper/infer_asr.py` e `infer_lid.py` —
  scripts de exemplo do fairseq para *language ID reranking* do projeto MMS, totalmente
  alheios a este pipeline, nunca importados por nada em `scripts/`, `av2unit/`,
  `unit2unit/`, `unit2av/` ou `inference.py`.
- Zero ocorrências em `scripts/prepare_data.py`, `train_full_pipeline.py`,
  `generate_synthetic_data.py` ou qualquer outro script deste projeto.
- O treino consome **apenas unidades discretas** (inteiros 0–999 + tokens de idioma),
  nunca texto. Isso é estrutural: a tarefa é `utut_pretraining`
  (`MultilingualDenoisingTask`), cujo dicionário é puramente numérico/simbólico
  ([unit2unit/model.py](unit2unit/model.py), dicionário construído em
  [scripts/train_full_pipeline.py:51-63](scripts/train_full_pipeline.py#L51-L63)) — não
  há campo de texto em nenhum ponto do `data-bin` binarizado pelo `fairseq-preprocess`.

**Achado relevante não pedido, mas importante para o capítulo**: os "alvos" de tradução
usados para o fine-tuning não vêm de um corpus paralelo humano nem de pseudo-labels
textuais — vêm da **saída do próprio pipeline pré-treinado**, rodado sobre os vídeos
brutos em português. Ver
[scripts/generate_synthetic_data.py:141-156](scripts/generate_synthetic_data.py#L141-L156):
o script chama `inference.py` (o pipeline AV2AV oficial, pré-treinado) sobre cada vídeo
bruto e faz upload do vídeo *traduzido gerado pelo próprio modelo* para a pasta
`synthetic_targets`, que depois vira o "target" de treino em
`train_full_pipeline.py`. Ou seja: o fine-tuning é uma forma de auto-destilação sobre as
próprias saídas zero-shot do modelo pré-treinado, não um aprendizado supervisionado por
tradução humana. Isso merece uma frase explícita na dissertação, independentemente da
questão do Whisper.

---

## E5 — Como o `mouth_cropped` é realmente produzido?

**NÃO CONFIRMADO — não existe, em lugar nenhum deste repositório, um script que gere
`mouth_cropped`.** Portanto nenhum dos cinco parâmetros alegados (MediaPipe Full
Range/Short Range, LMEDS, janela N=12, recorte 96×96) pode ser confirmado ou refutado
a partir deste código — eles simplesmente não têm artefato correspondente aqui.

O que encontrei:
- `mouth_cropped` aparece em 4 arquivos
  (`scripts/generate_synthetic_data.py`, `run_daily_cycle.py`, `train_drive_pipeline.py`,
  `train_full_pipeline.py`), mas em **todos** os casos é só uma *string* — o nome de
  uma pasta que o script procura no Google Drive
  (`drive_utils.find_folder(service, "mouth_cropped", root_id)`,
  [train_full_pipeline.py:186](scripts/train_full_pipeline.py#L186)) ou uma descrição de
  flag de CLI (`--use-raw-video`, "Use raw videos instead of mouth_cropped"). Nenhum
  desses arquivos **gera** o conteúdo dessa pasta.
- Não há `mediapipe` em nenhum lugar do repositório (`grep -rli mediapipe .` — vazio).
- A única biblioteca de landmark/detecção de face que existe no código deste projeto é
  `face_alignment` (usada em `inference.py`, função `extract_bbox`, para calcular
  bounding boxes na hora da *inferência*, não para gerar dados de treino) e o detector
  próprio do InsightFace dentro do submódulo `latentsync_repo` (usado só quando
  `--video-renderer latentsync`). Nenhum dos dois usa LMEDS nem janela de suavização
  N=12.
- O `README.md:44-45` afirma que o pré-processamento segue o
  [Auto-AVSR](https://github.com/mpc001/auto_avsr) — uma ferramenta **externa**, não
  vendorizada neste repositório. Se o `mouth_cropped` de fato foi gerado com
  MediaPipe/LMEDS/N=12/96×96, isso precisa ter vindo de uma execução do Auto-AVSR (ou
  outra ferramenta) fora deste código, e portanto fora do escopo do que este repositório
  consegue confirmar.
- Resolução/FPS de saída efetivos do `mouth_cropped`: **NÃO CONFIRMADO** pelo mesmo
  motivo.

---

## E6 — SpecAugment e modality dropout no fine-tuning

**Resposta: pertencem exclusivamente ao pré-treino original do mAV-HuBERT (que você não
executou), não ao fine-tuning do UTUT que você rodou.**

- `modality_dropout` e `audio_dropout` só existem em
  [av2unit/avhubert/hubert.py:240-241,355,615-616](av2unit/avhubert/hubert.py#L240-L241)
  — são parte da configuração/arquitetura do **mAV-HuBERT** (`av_hubert`), o modelo por
  trás de `checkpoints/mavhubert_large_noise.pt`. Esse checkpoint é carregado e usado
  como está (extração de unidades) — nunca é retreinado por
  `scripts/train_full_pipeline.py`, que só treina o estágio `unit2unit` (UTUT).
- O fine-tuning real que você rodou usa a tarefa `utut_pretraining`
  (`MultilingualDenoisingTask`, [unit2unit/task.py:7](unit2unit/task.py#L7)), que opera
  sobre sequências de inteiros 1-D (unidades discretas já extraídas) — não existe
  "modalidade" (áudio vs. vídeo) nesse estágio para se aplicar dropout de modalidade;
  estruturalmente não faz sentido e, de fato, não há nenhum parâmetro do tipo no
  Namespace logado (`logs_treino_full.txt:140`) nem nos params do mlflow.
  Confirmação adicional: `grep` por `modality_dropout|audio_dropout` fora de
  `av2unit/` e `fairseq/examples/` retornou vazio.
- `SpecAugment`: a única ocorrência em todo o repositório é a classe genérica
  `SpecAugmentTransform` dentro da própria biblioteca fairseq
  (`fairseq/fairseq/data/audio/feature_transforms/specaugment.py`) — código de
  biblioteca disponível para qualquer projeto fairseq, mas **nunca instanciado nem
  configurado** em `av2unit/`, `unit2unit/` ou nos scripts deste projeto (nenhum `grep`
  por `specaugment`/`spec_augment` fora de `fairseq/` retornou resultado). Não há
  evidência de que SpecAugment tenha sido usado nem no pré-treino original do
  mAV-HuBERT (que usa seu próprio esquema de mascaramento, `mask_prob_audio`/
  `mask_length_audio` etc., um mecanismo diferente e específico do HuBERT, não
  SpecAugment) nem no fine-tuning.

**Conclusão**: nenhum parâmetro efetivo para citar — nem SpecAugment "intensificado"
nem modality dropout fazem parte do fine-tuning real. Se a dissertação quiser mencionar
modality dropout, ele só pode ser atribuído ao **checkpoint pré-treinado que você usa,
não treina**.

---

## E7 — Arquitetura executada, para escrita

### Existem DUAS arquiteturas implementadas neste repositório, não uma

| | `utut_large` | `conformer_utut` |
|---|---|---|
| Registro | [unit2unit/model.py:5](unit2unit/model.py#L5) — arquitetura sobre a classe `bart` do fairseq | [unit2unit/models/conformer_utut.py:164,260](unit2unit/models/conformer_utut.py#L164) — modelo próprio |
| Encoder | Transformer (BART), 12 camadas | **Conformer** de verdade (`ConformerEncoderLayer` do fairseq, [linha 30](unit2unit/models/conformer_utut.py#L30)), 16 camadas |
| Decoder | Transformer (BART), 12 camadas | `TransformerDecoder` padrão do fairseq, 6 camadas |
| dim (embed) | 1024 | 256 |
| FFN | 4096 | 2048 |
| heads | 16 | 4 |
| Positional embedding | Senoidal, não aprendida (`encoder_learned_pos=False`) | Absoluta (`pos_enc_type='abs'`) |
| Usada por | `scripts/train_full_pipeline.py:105` (hardcoded), execução real de 2026-07-09 | Default do parser em `train_full_pipeline.py:39` e `train_drive_pipeline.py:45`; execução real de 2026-02-13 (mlflow) |
| Origem dos parâmetros | Herda quase tudo de `bart_large_architecture` ([fairseq/fairseq/models/bart/model.py:326-365](fairseq/fairseq/models/bart/model.py#L326-L365)) | Própria, com `depthwise_conv_kernel_size=31` (parâmetro específico de Conformer) |

Isso bate exatamente com o histórico do git: o commit
`0179083 feat: Implement a unit-to-unit Conformer model and pipeline...` introduziu a
arquitetura Conformer; commits posteriores (ex.: `33d3388`, e a reescrita mais recente de
`inference.py`/`train_full_pipeline.py` feita nesta própria sessão) migraram o caminho
de treino/inferência efetivo para `utut_large`, **sem remover** o código Conformer, que
continua registrado e importável (`unit2unit/models/__init__.py:1`), só não é mais o
default usado de fato.

### Detalhes de `utut_large` (a arquitetura da execução mais recente, 2026-07-09)

- **Nome e origem**: `utut_large`, arquitetura registrada sobre a classe genérica
  `bart` do fairseq ([unit2unit/model.py:5](unit2unit/model.py#L5)) — **não** é
  `mbart_large` (essa é uma arquitetura fairseq diferente, não usada aqui) nem uma
  variante nomeada oficialmente pelo fairseq; é um preset customizado deste projeto.
- Camadas: 12 encoder / 12 decoder. Dim: 1024. Heads: 16/16. FFN: 4096. Positional:
  senoidal fixa (não aprendida). (Todos confirmados em
  `logs_treino_full.txt:140`, campo `'model': Namespace(...)`.)
- **Nº de parâmetros: CONFIRMADO por leitura direta do checkpoint (atualizado
  2026-07-28)** — `checkpoints/full_model/checkpoint_best.pt`, lido com
  `sum(p.numel() for p in ck['model'].values())` na VM:
  **355.864.578 parâmetros**. A estimativa anterior (calculada, não medida) havia sido
  ~355M a partir do BART-large público (~406M com vocabulário 50265, recalculado para o
  vocabulário de 1024 deste projeto) — a estimativa ficou a ~0,07% do valor real,
  confirmando que o preset `utut_large` não altera mais nada do BART-large além dos 4
  flags já documentados acima.
- **`checkpoint_best.pt` e `checkpoint_last.pt` são o mesmo ponto de treino.** Os dois
  arquivos têm `extra_state` idêntico: `epoch=101`, `val_loss=8.38`, `best=8.38` — ou
  seja, no momento salvo, o último checkpoint também era o melhor (perda de validação
  ainda em queda, sem sinal de que um checkpoint anterior tivesse sido superado e
  descartado).
- **Nº de épocas efetivamente treinadas: 101** (não apenas até a epoch 7 que aparece em
  `logs_treino_full.txt` — esse arquivo de log é só um FRAGMENTO de uma execução
  retomada, não o histórico completo; o treino continuou muito além do que esse log
  cobre). Isso é tratado com detalhe no relatório `C-metricas.md` (Bloco C).
- **Vocabulário**: 1024 símbolos, confirmado (`logs_treino_full.txt:141`:
  `dictionary: 1024 types`). Composição confirmada por leitura direta de
  [scripts/train_full_pipeline.py:51-63](scripts/train_full_pipeline.py#L51-L63) e
  [unit2unit/inference.py:24-38](unit2unit/inference.py#L24-L38) (implementações
  idênticas, verifiquei com `diff`): **1000 unidades (0–999) + 19 tokens de idioma + 1
  `<mask>`** = 1020 símbolos escritos no dicionário, **+ 4 especiais automáticos do
  fairseq** (bos/pad/eos/unk) = 1024.
- **Partes congeladas no fine-tuning: nenhuma.** `--finetune-from-model` apenas
  inicializa os pesos a partir de `utut_sts_ft.pt` — não há `requires_grad=False`,
  `.freeze()` nem qualquer lógica de congelamento em
  `scripts/train_full_pipeline.py` ou em `unit2unit/` (busca explícita, vazio).

### Diagrama textual do fluxo (para a Figura 10)

```
[Vídeo PT bruto, 25 fps]
        │
        ├──► face_alignment (bbox) ──► crops de rosto
        │
        ▼
┌───────────────────────────────────────────┐
│ av2unit (mAV-HuBERT, PRÉ-TREINADO,          │
│ mavhubert_large_noise.pt, ~3.9GB)           │
│ entrada: áudio 16kHz + vídeo recortado      │
│ saída: sequência de unidades discretas      │
│        (vocabulário 0–999, taxa ~ nativa    │
│        do av-hubert antes da redução por    │
│        run-length)                          │
└───────────────────────────────────────────┘
        │  unidades PT (após process_units/reduce)
        ▼
┌───────────────────────────────────────────┐
│ unit2unit (UTUT, arch utut_large — BART)    │
│ checkpoint base PRÉ-TREINADO utut_sts_ft.pt,│
│ fine-tuned em checkpoints/full_model/       │
│ entrada: unidades PT + token [pt]/[en]      │
│ saída: unidades EN (comprimento LIVRE,      │
│        busca em feixe, max_len_a=1.2,       │
│        max_len_b=100 — NÃO preserva         │
│        comprimento de entrada)              │
└───────────────────────────────────────────┘
        │  unidades EN (0–1999, vocabulário do vocoder)
        ▼
┌───────────────────────────────────────────┐
│ unit2av (Unit AV Renderer, PRÉ-TREINADO,    │
│ unit_av_renderer.pt, ~425MB)                │
│  ├─ dur_predictor: prevê duração (em        │
│  │  frames a 50Hz) por unidade              │
│  ├─ repeat_interleave: expande a sequência  │
│  │  pela duração prevista → dedup_code      │
│  ├─ CodeHiFiGANModel_spk (áudio):           │
│  │  dedup_code + embedding do locutor       │
│  │  ──► forma de onda 16kHz                 │
│  └─ FaceRenderer (vídeo):                   │
│     dedup_code_seq (mesmo tensor, agrupado  │
│     de 50 em 50 = code_frame_ratio=2)       │
│     + janela de frames do vídeo original    │
│     ──► patches de boca 96×96 por frame,    │
│         em padded_tgt_len = len(dedup_code) │
│         // 2 frames (25 fps)                │
└───────────────────────────────────────────┘
        │                           │
        ▼                           ▼
   [áudio EN 16kHz]         [patches de boca 96×96,
                             mesma contagem de frames
                             que o áudio, por construção]
        │                           │
        └────────────┬──────────────┘
                      ▼
     util.save_video(): seamlessClone dos patches
     no vídeo de fundo + mux áudio/vídeo (ffmpeg)
                      ▼
            [Vídeo EN final, 25 fps]
```

---

## E8 — O capítulo de "Resultados Preliminares" descreve algo que existe?

**Resposta direta: existe uma iteração real e verificável, mas ela NÃO corresponde aos
números específicos citados (nem `AVDataset`, nem `valid_loss≈1281,17`, nem dropout 0,3,
nem SpecAugment intensificado). É uma iteração genuína, porém mais simples e
descrita incorretamente.**

Evidências, uma a uma:

1. **`class AVDataset`**: não existe em nenhum lugar. Busquei tanto na árvore de
   trabalho atual quanto em **todos os commits de todos os branches**
   (`git log --all -p -- '*.py' | grep "class AVDataset"` e
   `git rev-list --all | xargs git grep -l AVDataset`) — ambas as buscas vieram
   **vazias**. Essa classe nunca existiu neste repositório, em nenhum momento do
   histórico.

2. **`valid_loss ≈ 1281,17`**: não encontrado. Consultei `mlflow.db` diretamente
   (`SELECT ... FROM metrics WHERE value BETWEEN 1000 AND 1500` sobre **todas** as runs)
   — nenhum resultado. A única run no MLflow (`a27a9395759041689d2a58d7b8ce9c90`,
   experimento `av2av_daily`, 2026-02-13) só logou métricas `train/*`
   (`train/loss` variando entre 1,572 e 1,574) porque `disable_validation=True` estava
   ativo — **nunca houve validação nessa run**, logo nunca houve um `valid_loss` para
   ela. A execução mais recente (`logs_treino_full.txt`, 2026-07-09) teve validação
   ativa e logou `valid | epoch 007 | ... | loss 11.318` — também nada perto de 1281.

3. **Há, sim, uma iteração real e distinta, com arquitetura Conformer**: a run de
   MLflow de **2026-02-13** usou `args.arch = conformer_utut` (não `utut_large`),
   `encoder_layers=16`, `decoder_layers=6`, `embed_dim=256`, `heads=4`, `ffn=2048`,
   treinada **do zero** (`finetune_from_model=None`, não fine-tuning), rodando em
   **Windows** (`args.data = 'temp_data\train_batch_1\bin'`,
   `args.user_dir = 'C:\Users\Alexandre\...'`) — claramente uma máquina/momento
   anterior ao workflow atual na VM Linux. O código correspondente
   (`unit2unit/models/conformer_utut.py`) ainda existe no repositório, e o commit que o
   introduziu (`0179083 feat: Implement a unit-to-unit Conformer model and pipeline...`)
   está no histórico do git. Ou seja: **a arquitetura Conformer é real e localizável**,
   só não bate com os hiperparâmetros específicos citados (dropout 0,1 não 0,3;
   sem CTC; sem SpecAugment; sem AVDataset).

4. **Branches/stashes**: `git branch -a` mostra só `linux_env`/`main` (+ remotes,
   nenhum branch divergente); `git stash list` está vazio. Não há um pipeline de treino
   "escondido" em outro branch.

**Recomendação para a decisão que você precisa tomar**: não corte o capítulo — **reenquadre-o**
como a iteração Conformer de 2026-02-13 (real, rastreável, com código e run de MLflow
para citar), explicitamente descrita como uma versão preliminar mais simples (menor,
treinada do zero, sem validação), e não como a mesma configuração dos capítulos de
arquitetura final. Remova especificamente: a menção a CTC, a `AVDataset`, o valor
`1281,17` (não verificável — ou busque se você tem esse número em algum notebook/planilha
fora deste repositório) e o SpecAugment intensificado.

---

## Resumo de itens NÃO CONFIRMADO (dependem da VM/checkpoint, não deste checkout)

**Atualização 2026-07-28**: os três itens abaixo, marcados NÃO CONFIRMADO na versão
original deste relatório, foram confirmados após trazer os checkpoints da VM (ver
"Nº de parâmetros" e a nota sobre `checkpoint_best.pt`/`checkpoint_last.pt` em E7
acima). O `cfg` bit-a-bit completo ainda não foi salvo em arquivo (tentativa de gravar
`checkpoint_best_cfg.json` falhou por falta do diretório `_relatorio_dissertacao/` na
VM) — mas os campos que já vieram do `extra_state` (epoch, val_loss, best) e a
contagem de parâmetros já eliminam a maior parte da incerteza que esse item cobria.

- ~~Nº exato de parâmetros do checkpoint~~ — **confirmado**: 355.864.578 (ver E7).
- ~~Existência de `checkpoint_best.pt`~~ — **confirmado**: existe, e é idêntico ao
  `checkpoint_last.pt` (ver E7).
- `cfg` bit-a-bit completo em arquivo separado — ainda não salvo (falha de `mkdir`),
  mas os campos essenciais já vieram junto com epoch/val_loss/best.
- Todos os 5 parâmetros de geração do `mouth_cropped` (E5) — não há artefato neste
  repositório para confirmar ou refutar.
- Versão exata de PyTorch/driver CUDA da VM — ver bloco de resposta anterior desta
  conversa (não repetido aqui).
