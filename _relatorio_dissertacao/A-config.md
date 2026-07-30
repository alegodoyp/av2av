# Bloco A — Configuração experimental do fine-tuning

**Nota de execução**: gerado a partir do checkout Windows local (idêntico ao HEAD do
git após a sincronização feita nesta sessão — `ccd85f0`, ver A6), não da VM. Os 5
checkpoints já foram trazidos para este checkout (scp, sessões anteriores) e o hash
SHA256 abaixo foi calculado diretamente sobre esses arquivos locais. O `cfg` **gravado
dentro** do checkpoint (por oposição ao Namespace logado durante o treino, que é uma
fonte diferente, ainda que do mesmo evento) não foi re-extraído nesta rodada — a
tentativa anterior falhou por um `mkdir` que eu esqueci de incluir (mesmo erro descrito
no Bloco C). Dou o script corrigido no A1 para fechar esse item com 100% de certeza,
já que agora sei que essa é a única lacuna real que falta.

---

## A1 — Linha de comando efetiva do `fairseq-train`

Lista completa e literal, lida direto de
[scripts/train_full_pipeline.py:98-150](scripts/train_full_pipeline.py#L98-L150)
(conteúdo confirmado idêntico ao commit atual, `git diff` vazio para este arquivo —
ver A6). Coluna "valor resolvido" usa os valores reais logados em
`logs_treino_full.txt:140` (Namespace completo, capturado ao vivo durante a execução
que gerou `checkpoints/full_model/checkpoint_best.pt`), não os defaults do argparse:

| Argumento | Valor resolvido (execução real) | Linha no script |
|---|---|---|
| `data` (posicional) | caminho do data-bin unificado | 99 |
| `--save-dir` | `checkpoints/full_model` | 100 (default do parser, linha 30) |
| `--task` | `utut_pretraining` | 101 |
| `--arch` | `utut_large` | 105 (hardcoded — ver divergência abaixo) |
| `--langs` | `pt,en` | 108 |
| `--add-lang-token` | true | 109 |
| `--encoder-normalize-before` / `--decoder-normalize-before` | true / true | 110-111 |
| `--attention-dropout` | 0.1 | 112 |
| `--criterion` | `label_smoothed_cross_entropy` | 113 |
| `--label-smoothing` | 0.2 | 114 |
| `--optimizer` | `adam` | 115 |
| `--adam-betas` | `(0.9, 0.98)` | 116 |
| `--lr-scheduler` | `polynomial_decay` | 117 |
| `--total-num-update` | 50000 | 118 |
| `--warmup-updates` | 3000 | 119 |
| `--lr` | 1e-4 | 120 |
| `--clip-norm` | 0.1 | 121 |
| `--batch-size` | 16 (default do parser, linha 31 — não sobrescrito na chamada real) | 122 |
| `--max-tokens` | 200000 (default, linha 40) | 123 |
| `--update-freq` | 1 (default, linha 41) | 124 |
| `--max-epoch` | 100 (default, linha 42) | 125 |
| `--validate-interval` | 1 (default, linha 43) | 126 |
| `--patience` | 10 | 127 |
| `--no-epoch-checkpoints` | true | 128 |
| `--finetune-from-model` | cópia "patched" de `checkpoints/utut_sts_ft.pt` (ver `patch_utut_checkpoint`, linhas 66-84) | 129 |
| `--user-dir` | `{cwd}/unit2unit` | 130 |
| `--tokens-per-sample` | 1020 | 131 |
| `--sample-break-mode` | `eos` | 132 |
| `--max-source-positions` / `--max-target-positions` | 1024 / 1024 | 133-134 |
| `--num-workers` | 1 | 135 |
| `--skip-invalid-size-inputs-valid-test` | true | 136 |
| `--required-batch-size-multiple` | 1 (override do default fairseq=8) | 140 |
| `--shorten-method` | `truncate` | 145 |
| `--distributed-world-size` | 1 | 149 |

**Cruzamento script-atual vs. execução real**: todos os valores acima batem
exatamente com o Namespace logado (`logs_treino_full.txt:140`) — **nenhuma
divergência encontrada** entre o que o script monta hoje e o que foi de fato
executado. Isso é coerente com A6: `git diff -- scripts/train_full_pipeline.py` está
vazio neste checkout, ou seja, o arquivo não foi alterado desde a execução que gerou
`checkpoint_best.pt`.

**Ressalva sobre a fonte**: a tabela acima cruza com o **Namespace logado durante o
treino** (`logs_treino_full.txt`), não com o `cfg` lido diretamente de dentro do
arquivo `checkpoint_best.pt`. São a mesma informação por construção (fairseq serializa
o mesmo objeto de config que loga e que salva), mas para fechar esse ponto com 100% de
certeza documentável (a `--finetune-from-model` em particular, que sustenta a alegação
de transfer learning), rode isto na VM:

```bash
cd /mnt/disk/home/alexandregodoy/av2av
conda activate av2av_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p _relatorio_dissertacao

python -c "
import torch, json
ck = torch.load('checkpoints/full_model/checkpoint_best.pt', map_location='cpu')
cfg = ck['cfg']
# cfg pode ser OmegaConf DictConfig ou dict simples dependendo da versão do fairseq
try:
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
except Exception:
    cfg_dict = cfg
with open('_relatorio_dissertacao/checkpoint_best_cfg.json', 'w') as f:
    json.dump(cfg_dict, f, default=str, indent=2)
print('finetune_from_model no cfg:', cfg_dict.get('checkpoint', {}).get('finetune_from_model'))
print('arch no cfg:', cfg_dict.get('model', {}).get('_name'))
"
```

**Confirmação indireta, já disponível, da presença de `--finetune-from-model`**: o
próprio `logs_treino_full.txt:140` já mostra
`'finetune_from_model': '/dev/shm/alex/tmp/utut_sts_ft_patched.pt'` dentro do bloco
`checkpoint` do Namespace — um caminho de arquivo **temporário e derivado**
(`patch_utut_checkpoint()`, linhas 66-84, grava a cópia "stripped" em
`tempfile.gettempdir()`), não o `checkpoints/utut_sts_ft.pt` original diretamente. Isso
sustenta a alegação de transfer learning (o treino de fato partiu dos pesos do
`utut_sts_ft.pt`, só que de uma cópia com duas chaves de buffer removidas — ver
docstring de `patch_utut_checkpoint`), mas **o caminho literal no cfg não é
`checkpoints/utut_sts_ft.pt`** — é a cópia patched. Vale citar isso com precisão na
dissertação em vez de simplificar para "usou checkpoints/utut_sts_ft.pt diretamente".

---

## A2 — Arquitetura e parâmetros

- **Arquitetura registrada**: `utut_large`
  ([unit2unit/model.py:5](unit2unit/model.py#L5)), um preset customizado deste projeto
  sobre a classe `bart` genérica do fairseq (`register_model_architecture("bart",
  "utut_large")`) — **não** é `mbart_large` nem uma arquitetura oficialmente nomeada
  pelo fairseq.
- **Camadas/dimensões** (confirmadas no Namespace logado, `logs_treino_full.txt:140`,
  campo `'model': Namespace(...)`): `encoder_layers=12`, `decoder_layers=12`,
  `encoder_embed_dim=decoder_embed_dim=1024`, `encoder_ffn_embed_dim=decoder_ffn_embed_dim=4096`,
  `encoder_attention_heads=decoder_attention_heads=16`.
- **Positional embedding**: `encoder_learned_pos=False`, `decoder_learned_pos=False`
  no Namespace logado — ou seja, **senoidal, não aprendida**, por configuração
  ([unit2unit/model.py:11-12](unit2unit/model.py#L11-L12), que fixa esses dois campos
  antes de chamar `bart_large_architecture`, cujo default seria `True`). **Confirmação
  via ausência no state_dict, pedida explicitamente**: NÃO CONFIRMADO nesta rodada —
  preciso inspecionar as chaves do `state_dict` de verdade, não just confiar na flag de
  config (a regra de ouro deste bloco pede exatamente isso, e um comentário em
  `patch_utut_checkpoint` — que a própria regra de ouro manda não usar como fonte —
  afirma que o fairseq atual "no longer registers" o buffer senoidal). Comando:
  ```bash
  python -c "
  import torch
  ck = torch.load('checkpoints/full_model/checkpoint_best.pt', map_location='cpu')
  keys = [k for k in ck['model'].keys() if 'embed_positions' in k]
  print('Chaves embed_positions no state_dict:', keys if keys else '(nenhuma)')
  "
  ```
  Se vier vazio, confirma senoidal-sem-buffer (bate com o comentário, mas agora
  verificado, não assumido). Se aparecer `_float_tensor` ou `.weight`, o comentário
  está errado e isso precisa ser corrigido no texto.
- **Nº de parâmetros**: **355.864.578**, confirmado por leitura direta do checkpoint
  nesta mesma sessão (`sum(p.numel() for p in ck['model'].values())`, ver Bloco E/C).
  **Treináveis**: mesmo valor (355.864.578) — não por medição direta de
  `requires_grad` (um `state_dict` salvo é só um dicionário de tensores, sem essa
  informação; teria que instanciar o modelo ao vivo para checar), mas por busca
  exaustiva de código: `grep -rn "requires_grad\|\.freeze(" scripts/train_full_pipeline.py
  unit2unit/` não encontrou nenhuma ocorrência (Bloco E) — não existe lógica de
  congelamento em lugar nenhum do caminho de treino, logo todos os parâmetros
  permaneceram treináveis.
- **Shape dos embeddings, `[1024, 1024]`**: NÃO CONFIRMADO diretamente nesta rodada —
  é dedutível (vocab=1024 × embed_dim=1024, ambos já confirmados separadamente), mas
  como os dois números coincidem, a dedução por si só não prova que a matriz É
  `[1024, 1024]` e não, por exemplo, transposta ou com uma dimensão extra. Comando pra
  fechar com certeza:
  ```bash
  python -c "
  import torch
  ck = torch.load('checkpoints/full_model/checkpoint_best.pt', map_location='cpu')
  for k, v in ck['model'].items():
      if 'embed_tokens.weight' in k or k.endswith('output_projection.weight'):
          print(k, tuple(v.shape))
  "
  ```
- **Camadas congeladas**: **nenhuma** (mesma busca de código citada acima —
  `requires_grad`/`.freeze()` ausentes em todo o caminho de treino).

---

## A3 — Os três/cinco modelos do pipeline

**Correção de caminho**: o A3 pede `checkpoints/unit2av/encoder.pt`, mas esse caminho
**não existe**. O arquivo real é `unit2av/encoder.pt` (raiz do repo, não dentro de
`checkpoints/`) — confirmei com teste de existência direto nos dois caminhos. Usei o
caminho real na tabela abaixo.

SHA256 (primeiros 16 caracteres) calculado agora, direto sobre os arquivos locais
(idênticos aos da VM — vieram por scp nesta mesma sessão):

| arquivo | tamanho (bytes) | sha256 (16 chars) | origem | papel |
|---|---|---|---|---|
| `checkpoints/mavhubert_large_noise.pt` | 3.908.224.569 | `d11c035c09104600` | **Pré-treinado**, download oficial Facebook AV-HuBERT — URL em [scripts/download_models.py:59](scripts/download_models.py#L59): `https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt` | av2unit — extração de unidades discretas |
| `checkpoints/utut_sts_ft.pt` | 4.253.823.463 | `c35060f966fb6eb6` | **Pré-treinado**, checkpoint original dos autores do AV2AV ([README.md:51](README.md#L51), link Google Drive — `download_models.py` não baixa este) | unit2unit — ponto de partida do fine-tuning (`--finetune-from-model`) |
| `checkpoints/unit_av_renderer.pt` | 424.628.667 | `c617a20c268cc380` | **Pré-treinado**, checkpoint zero-shot original do AV2AV ([README.md:52](README.md#L52)) | unit2av — renderizador AV zero-shot |
| `unit2av/encoder.pt` | 17.090.379 | `39373b86598fa3da` | **NÃO CONFIRMADO** — não há menção a este arquivo em `download_models.py` nem no README; está commitado no git desde o commit "Init" (`git log -- unit2av/encoder.pt`), então é anterior a qualquer trabalho desta sessão, mas a origem exata (autores do AV2AV vs. gerado por alguém do projeto) não tem uma fonte que eu possa citar com confiança | Speaker encoder (GE2E), usado por `process_unit2av` para extrair o embedding do locutor |
| `checkpoints/full_model/checkpoint_best.pt` | 4.254.111.231 | `badfa647b48fe8da` | **Produzido neste trabalho** — fine-tuning de `utut_sts_ft.pt` via `scripts/train_full_pipeline.py`, execução confirmada em `logs_treino_full.txt` e no `mlflow` (ver Bloco C) | unit2unit fine-tuned — o modelo avaliado no capítulo de resultados |

`scripts/download_models.py` só baixa **dois** arquivos de fato
([linhas 59 e 65](scripts/download_models.py#L59-L65)): `mavhubert_large_noise.pt`
(citado acima) e `avhubert_base_1000.pt` (não está na lista pedida, não usado por
nenhum caminho de inferência atual — provavelmente vestígio de uma etapa de extração
de clusters não usada na versão final do pipeline). `utut_sts_ft.pt` e
`unit_av_renderer.pt` **não** têm código de download neste repositório — só o link do
README, que aponta para Google Drive, não para uma URL baixável por script.

---

## A4 — Versões do ambiente

Já confirmado nesta sessão (fontes diferentes, ver Blocos C/E), **não são medições
frescas desta rodada específica**:

| item | valor | fonte |
|---|---|---|
| Python | 3.10.20 | warning do `google.api_core` em `logs_treino_full.txt:1` |
| PyTorch (via torchaudio) | `torchaudio==2.7.1+cu118` → PyTorch quase certamente 2.7.1 | `pip list` real rodado na VM (Bloco C) — **corrige minha estimativa anterior de ~2.6.x**, que vinha só do nome do commit do fairseq, não de uma medição direta |
| GPU | NVIDIA L40S, 44,392 GB, capability 8.9 | `logs_treino_full.txt:214-216` |
| Driver NVIDIA | NÃO CONFIRMADO | nenhum log disponível mostra isso |
| fairseq | commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` ("Pt2.6 compatibility (#5611)", 2025-09-30), **não é fork** — origin é `https://github.com/facebookresearch/fairseq.git` | `git -C fairseq log -1`/`git -C fairseq remote -v`, rodado agora, working tree limpo (sem patch local) |
| av2av (este repo) | `ccd85f0dbc0b49e593f51939fb0c00cf2310adad`, 2026-07-29 18:36:42, "Added corpus BLEU" | `git log -1`, rodado agora neste checkout |

**NÃO CONFIRMADO nesta rodada** (preciso rodar na VM, comandos exatos que você já deu
no prompt — só reproduzo aqui pra deixar registrado no arquivo certo):
```bash
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
conda list --export > _relatorio_dissertacao/A-conda-env.txt
pip freeze > _relatorio_dissertacao/A-pip-freeze.txt
```
`git -C fairseq log -1`/`git log -1` já rodei localmente (linhas acima) — não precisa
repetir na VM a menos que suspeite de divergência entre o checkout local e o da VM
(não deveria haver, dado o merge feito nesta sessão, mas o checkout local é a fonte
real destes dois valores específicos, não a VM).

---

## A5 — Split treino/validação: é reproduzível?

**Resposta direta: NÃO, não é reproduzível.**

- Onde: `random.shuffle(pairs)` —
  [scripts/train_full_pipeline.py:241](scripts/train_full_pipeline.py#L241), dentro de
  `main()`. `prepare_data.py` não faz split — só recebe as listas `train_pairs`/
  `val_pairs` já definidas por `train_full_pipeline.py` e as processa.
- Seed: **nenhuma**. Busquei `random.seed(`, `np.random.seed(`, `torch.manual_seed(`
  em todo `scripts/` e `unit2unit/` — nenhuma chamada existe antes (ou depois) do
  `random.shuffle` na linha 241. O módulo `random` do Python usa entropia do SO por
  padrão quando não semeado.
- Proporção: `--split-ratio`, default **0.8**
  ([linha 36](scripts/train_full_pipeline.py#L36)), não sobrescrito na execução real
  (não aparece na lista de flags do `run_training`, que só recebe o que já foi
  decidido em `main()` antes de chamar `run_training`). Exemplos: confirmado em
  `logs_treino_full.txt:5-6`: "Found 523 source videos and 23 target videos... Matched
  23 video pairs... Split: 18 Train / 5 Validation" — bate exatamente com
  `int(23*0.8)=18`.
- `--seed` do fairseq: **não é passado explicitamente** em `cmd_args` (busca no
  arquivo confirma ausência de `--seed` na lista). O fairseq usa seu próprio default
  (`seed=1`, `fairseq/fairseq/dataclass/configs.py:143-144`), confirmado no Namespace
  logado (`'seed': 1`). **Mas esse seed controla a parte do fairseq** (inicialização de
  pesos, ordenação de batches dentro do treino) — é **completamente independente** do
  `random.shuffle` que decide QUAIS exemplos vão para treino vs. validação, que roda
  ANTES do fairseq sequer ser invocado.
- **Se você rodar o pipeline de novo hoje, obtém o mesmo split? NÃO.** O
  `random.shuffle(pairs)` não tem seed fixada, então cada execução de `main()`
  reembaralha os 23 pares de forma diferente (dependente de entropia do SO), e portanto
  os 18 pares de treino / 5 de validação seriam, com altíssima probabilidade, um
  subconjunto diferente numa nova execução — mesmo que o `--seed=1` do fairseq
  permaneça igual (esse seed não afeta o `random.shuffle`, que é Python puro, fora do
  fairseq). Isso é uma lacuna de reprodutibilidade real, não cosmética, e deve ser
  declarada explicitamente no capítulo de Ameaças à Validade: a partição
  treino/validação usada para produzir `checkpoint_best.pt` não pode ser reconstruída
  a partir do código como está — só reaproveitando os `.bin`/`.idx` já gerados (se
  ainda existirem em algum lugar persistente; o Bloco C já registrou que o data-bin
  original ficou em `/dev/shm`, provavelmente perdido).

---

## A6 — Divergências entre repositório e execução

**Correção da premissa do prompt**: neste checkout (sincronizado com o remoto nesta
mesma sessão, ver histórico de merge), `scripts/train_full_pipeline.py` **não tem
alterações não commitadas** — `git diff -- scripts/train_full_pipeline.py` retorna
vazio, e `git status --short` só mostra `_relatorio_dissertacao/C-metricas.md`
modificado (os próprios relatórios desta investigação, esperado). O arquivo está
versionado e seu histórico recente mostra exatamente os ajustes incrementais para
dataset pequeno já documentados no A1/Bloco E:

```
2cc99e6 feat: Reduce num-workers to 1 in run_training to accommodate small datasets
efbe543 feat: Override required-batch-size-multiple to 1 to prevent crashes with small datasets
08d5372 feat: Adjust update frequency to 1 for improved training granularity and prevent empty batches
```

**Ressalva importante**: isto reflete o estado deste checkout Windows local, não uma
inspeção ao vivo da VM. Dado que a sessão inteira envolveu idas e vindas de
push/pull/merge entre VM e este checkout (incluindo um merge explícito para resolver
divergência, mais cedo nesta mesma conversa), a expectativa é que estejam
sincronizados agora — mas para ter 100% de certeza antes do depósito, rode isto **na
própria VM**, não só aqui:
```bash
cd /mnt/disk/home/alexandregodoy/av2av
git status --short
git diff --stat
git log --oneline -20
```
Se aparecer qualquer coisa além de arquivos de relatório (`_relatorio_dissertacao/*`),
me mande o resultado antes de assumir que está tudo commitado.

**Diferenças entre o repositório e o que o `cfg` do checkpoint indica ter sido
executado**: nenhuma encontrada no conteúdo de `train_full_pipeline.py` (A1 já cobre
isso arg por arg). A única divergência real documentável está fora deste arquivo:
`--finetune-from-model` no cfg aponta para uma cópia temporária patched
(`/dev/shm/alex/tmp/utut_sts_ft_patched.pt`), não para `checkpoints/utut_sts_ft.pt`
diretamente — mas isso é o comportamento **esperado e documentado** do próprio código
(`patch_utut_checkpoint()`), não uma divergência acidental.

**Recomendação para commitar antes da defesa**:
1. Os arquivos deste bloco (`_relatorio_dissertacao/A-config.md` e os que ele gera) —
   fazem parte do apêndice de reprodutibilidade, vale versionar.
2. Se o comando de A4 (`conda list --export`, `pip freeze`) for rodado na VM, commitar
   os dois arquivos de saída também — são a única forma de reconstruir o ambiente
   exato depois que a VM for desligada.
3. Confirmar (comando acima) que a VM em si não tem nada pendente fora dos relatórios
   — se tiver, decidir explicitamente o que entra no repo antes do depósito, já que
   "reprodutibilidade" no apêndice só é verdade se o código commitado for de fato o
   que gerou os resultados citados.
