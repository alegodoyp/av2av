# Pacote de reprodutibilidade — pipeline AV2AV (fine-tuning PT→EN)

Documento autocontido para reprodução por terceiros. Todas as versões citadas
vêm do Bloco A (confirmadas no ambiente real, não estimadas), exceto onde
marcado `NÃO CONFIRMADO`.

## 1. Requisitos de hardware e software

- **GPU**: NVIDIA L40S ou equivalente, ≥44GB VRAM (usado na execução real);
  `CUDA_VISIBLE_DEVICES=1` foi necessário nesta máquina especificamente porque a
  GPU 0 estava ocupada por outro processo/usuário — em outra máquina, ajustar
  para o índice de GPU livre.
- **Python**: 3.10.20.
- **PyTorch**: 2.7.1 (via `torchaudio==2.7.1+cu118`, confirmado por `pip list`
  real), build CUDA 11.8.
- **fairseq**: submódulo git oficial (`https://github.com/facebookresearch/fairseq.git`,
  **não é fork**), commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` ("Pt2.6
  compatibility (#5611)").
- **Disco**: ao menos ~15GB para os 5 checkpoints principais + espaço de
  trabalho; nesta execução real, `/mnt/disk` estava compartilhado e chegou a
  100% de uso, forçando o uso de `/dev/shm` (tmpfs/RAM) como scratch — ver item 7.

## 2. Passo a passo de instalação

```bash
git clone <repo> av2av && cd av2av
git submodule update --init
pip install -e ./fairseq
pip install -r requirements.txt

# Dependências ausentes do requirements.txt (confirmado por import direto no
# código, não só citado de memória — ver Bloco D6):
pip install imageio imageio-ffmpeg face-alignment tqdm \
  google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
```

Os 5 checkpoints (ver Bloco A, tabela de origem/SHA256) precisam ser obtidos
separadamente — 3 pré-treinados dos autores do AV2AV (`mavhubert_large_noise.pt`
via script `scripts/download_models.py`; `utut_sts_ft.pt` e
`unit_av_renderer.pt` via link do Google Drive no `README.md`, sem script de
download automatizado) e o `unit2av/encoder.pt` (já commitado no git desde o
commit inicial). O checkpoint fine-tuned (`checkpoints/full_model/checkpoint_best.pt`)
é o produto deste próprio trabalho, não um pré-requisito de instalação.

## 3. Variáveis de ambiente obrigatórias

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

**Justificativa**: a partir do PyTorch 2.6, o default de `torch.load()` mudou
para `weights_only=True`, o que quebra o carregamento de checkpoints do fairseq
que armazenam um `argparse.Namespace` (não um dicionário de tensores puro) no
campo `cfg`/`args`. Sem essa variável, `torch.load()` levanta erro de
desserialização ao tentar carregar qualquer um dos checkpoints deste pipeline.

Também usadas nesta execução real, por motivo de infraestrutura local (ver
item 7), não estritamente necessárias em outra máquina:
```bash
export CUDA_VISIBLE_DEVICES=1   # GPU 0 ocupada nesta máquina especificamente
export TMPDIR=/dev/shm/alex/tmp # /mnt/disk a 100%
```

## 4. Estrutura de diretórios esperada

```
av2av/
  checkpoints/
    mavhubert_large_noise.pt
    utut_sts_ft.pt
    unit_av_renderer.pt
    full_model/checkpoint_best.pt   # produto do fine-tuning
  unit2av/
    encoder.pt                      # já vem no git
    config.json                     # já vem no git
  samples/                          # vídeos de entrada para inferência
  results/                          # saídas da inferência
  _relatorio_dissertacao/           # este próprio pacote de blocos
```

## 5. Comandos completos

**Preparação de dados → treino**: ver Bloco A (linha de comando completa e
resolvida do `fairseq-train`) e Bloco B (como o corpus pseudo-pareado é
construído, incluindo a ressalva de auto-destilação). Comando de alto nível:

```bash
python scripts/train_full_pipeline.py \
  --drive-folder videos_mestrado --src-lang pt --tgt-lang en \
  --av2unit-path checkpoints/mavhubert_large_noise.pt
```
(demais flags usam os defaults documentados no Bloco A — `--batch-size 16`,
`--max-epoch 100` etc. — a menos que se queira reproduzir exatamente a execução
real, cuja lista completa de argumentos efetivos do `fairseq-train` está na
tabela do Bloco A1.)

**Inferência (baseline)**:
```bash
python inference.py \
  --in-vid-path samples/videoN.mp4 --out-vid-path results/videoN_baseline.mp4 \
  --src-lang pt --tgt-lang en \
  --av2unit-path checkpoints/mavhubert_large_noise.pt \
  --utut-path checkpoints/utut_sts_ft.pt \
  --unit2av-path checkpoints/unit_av_renderer.pt
```

**Inferência (fine-tuned)** — só o `--utut-path` muda (ver D1, confirmado por
código que essa é a única diferença real entre as duas condições):
```bash
python inference.py \
  --in-vid-path samples/videoN.mp4 --out-vid-path results/videoN_finetuned.mp4 \
  --src-lang pt --tgt-lang en \
  --av2unit-path checkpoints/mavhubert_large_noise.pt \
  --utut-path checkpoints/full_model/checkpoint_best.pt \
  --unit2av-path checkpoints/unit_av_renderer.pt
```

Flags adicionais usadas nesta sessão para renderização de qualidade maior
(`--video-renderer latentsync`) exigem um ambiente conda **separado e
dedicado** (não o `av2av_env`) — ver Bloco C6 para o procedimento completo de
setup do LatentSync/SyncNet, incluindo por que um ambiente à parte é
necessário (conflito de versão de PyTorch/OpenCV com o ambiente principal).

## 6. Configuração do Google Drive

O pipeline de dados (`drive_utils.py`, `scripts/generate_synthetic_data.py`,
`scripts/train_full_pipeline.py`) depende de acesso à API do Google Drive via
OAuth (`client_secrets.json` + `token.json`, nenhum dos dois commitado — ver
D6). **Nota importante**: o `token.json` gerado por fluxo OAuth "Testing" no
Google Cloud Console **expira em ~7 dias** e precisa ser regenerado
manualmente a cada expiração, o que interrompe qualquer automação (ex.:
`run_daily_cycle.py`) nesse intervalo. Recomendação: migrar para uma
**service account**, compartilhando a pasta `videos_mestrado` do Drive com o
`client_email` da service account — isso elimina a expiração periódica.

## 7. Contornos de infraestrutura desta execução específica

- `/mnt/disk` (disco principal da VM) chegou a 100% de uso compartilhado por
  múltiplos usuários/projetos — todo scratch volumoso (data-bin intermediário
  do treino, temporários do LatentSync) foi redirecionado para `/dev/shm/alex/`
  (tmpfs, backed por RAM, ~756GB disponíveis nesta VM). Isso tem uma
  consequência séria para reprodutibilidade: o data-bin binarizado usado no
  treino real (`/dev/shm/alex/temp_full_dataset/unified_bin`) **não sobrevive a
  um reboot** e muito provavelmente já não existe mais — não há cópia
  persistente dele neste pacote.
- `CUDA_VISIBLE_DEVICES=1`: a GPU de índice 0 nesta máquina estava ocupada por
  outro processo/usuário; o índice correto varia por máquina e deve ser
  verificado com `nvidia-smi` antes de rodar.
- A VM usada para gerar os resultados deste trabalho está sendo desligada — este
  pacote de reprodutibilidade, junto com os checkpoints já copiados para fora
  dela (ver checklist formal), é o que resta como evidência executável depois
  disso.
