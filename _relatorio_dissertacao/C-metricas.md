# Bloco C — Métricas de treino e caminho para avaliação

**Nota de execução**: mesmo checkout Windows local das notas anteriores, não a VM.
`mlflow.db` foi re-trazido da VM nesta sessão (via scp) e está confirmado **idêntico**
ao que já estava no git — ou seja, é o estado real e atual da VM, não uma versão velha.
Os checkpoints (`checkpoints/full_model/checkpoint_best.pt` e `checkpoint_last.pt`)
também já foram trazidos e lidos (via comando rodado por você na VM, resultado colado
nesta conversa). O que ainda depende da VM está marcado `NÃO CONFIRMADO` com o comando
exato.

**Resolvido**: você confirmou que os últimos comandos de geração de resultado foram os
que eu dei nesta própria conversa — e todos usaram `--utut-path
checkpoints/utut_sts_ft.pt`. Ou seja, **os 4 vídeos em `results/` são todos baseline
(checkpoint pré-treinado); não existe nenhuma saída com o checkpoint fine-tuned
(`checkpoints/full_model/checkpoint_best.pt`) ainda.** Isso muda o C5/C6: falta gerar
os 4 equivalentes fine-tuned antes de qualquer comparação baseline-vs-fine-tuned fazer
sentido — ver comando abaixo e o C7 atualizado com esse trabalho extra no orçamento de
tempo/GPU.

---

## C1 — Extração do MLflow

```bash
sqlite3 mlflow.db ".tables"   # → não tenho sqlite3 CLI aqui, usei sqlite3 via Python; resultado idêntico
```

CSVs gerados: `C-runs.csv` (1 linha), `C-metrics-raw.csv` (348 linhas),
`C-params.csv` (478 linhas).

**Não existe o experimento `av2av_full_finetune`.** O `mlflow.db` (local e, agora
confirmado, também o da VM) só tem dois experimentos: `Default` (vazio) e
`av2av_daily` (id 1). Dentro dele, **uma única run**:

| run_uuid | name | status | start_time | end_time |
|---|---|---|---|---|
| `a27a9395759041689d2a58d7b8ce9c90` | judicious-slug-924 | FINISHED | 2026-02-13 20:59:33 | 2026-02-13 21:02:39 |

**Essa run NÃO corresponde ao `checkpoint_best.pt`.** Ela é a execução com arquitetura
`conformer_utut` já documentada no Bloco E (E7/E8) — `args.arch=conformer_utut`,
`args.finetune_from_model=None`, batch_size=2, rodada no Windows
(`args.data='temp_data\train_batch_1\bin'`). O treino que **de fato** produziu
`checkpoints/full_model/checkpoint_best.pt` (arch `utut_large`, epoch 101, confirmado
por leitura direta do checkpoint — ver C3) **nunca foi logado no MLflow**: nem antes
nem depois de eu confirmar que o `mlflow.db` da VM é byte-idêntico ao do git. A causa
mais provável, olhando `scripts/train_full_pipeline.py:154-158`
(`run_training()`), é estrutural: o script só envia métricas ao MLflow se
`--mlflow-tracking-uri`/`--mlflow-experiment-name` forem passados na linha de comando;
a execução de julho aparentemente rodou sem essas flags.

**Conclusão prática**: para a run que importa pra dissertação, a única fonte de
métricas é `logs_treino_full.txt` (parcial, só epoch 7) + o `extra_state` gravado
dentro do próprio checkpoint (só o último ponto, epoch 101). Não há dado intermediário
em lugar nenhum.

---

## C2 — Curva de aprendizado

`C-curvas.csv` gerado com o schema pedido
(`epoch,train_loss,train_nll_loss,train_ppl,valid_loss,valid_nll_loss,valid_ppl,lr,num_updates,wall_clock_s`),
mais duas colunas extras no início (`run`, `source`) para não misturar as duas
execuções na mesma tabela sem dizer de onde vem cada linha.

**Limitação central, para declarar explicitamente no texto**: só existem **dois
pontos** para a run que interessa (`utut_large`, a que gerou `checkpoint_best.pt`), não
uma curva contínua:

| epoch | train_loss | valid_loss | fonte |
|---|---|---|---|
| 7 | 11,136 | 11,318 | `logs_treino_full.txt` (linhas de progresso do fairseq) |
| 101 (final) | 9,459 | 8,380 | `extra_state` dentro de `checkpoint_best.pt` |

Os `train_ppl`/`valid_ppl` da linha do epoch 101 (578,3 e 224,7) **não estão logados
diretamente** — foram calculados a partir do `nll_loss` pela relação `ppl = 2^nll_loss`,
que confirmei bater com os pares logados no epoch 7 (`nll_loss=11,239` →
`2^11,239≈2416,9`, contra o `ppl=2416,69` realmente logado — diferença de arredondamento
apenas). Marquei isso explicitamente no CSV/aqui para não passar como valor medido.

`wall_clock_s` das duas linhas **não é comparável diretamente**: o valor da linha 7
(`19`) é relativo à sessão retomada que gerou aquele log fragmento; o valor da linha
101 (`51084,86`) é o `previous_training_time` acumulado desde o início real do treino,
gravado pelo próprio fairseq dentro do checkpoint. Não dá para saber, só com esses dois
números, quanto tempo se passou *entre* epoch 7 e epoch 101 — só o total acumulado até
o fim.

**Por que não dá pra preencher os epochs 8-100**: `logs_treino_full.txt` é um
fragmento de uma execução retomada (o log começa com "Loaded checkpoint ... epoch 7"),
não o histórico completo. Tentamos recuperar o restante nesta sessão: busca por
`*.log`/`nohup.out` (só achou logs de sistema, nada do treino), busca por conteúdo
(`fairseq_cli.train`/`begin training epoch`) em todo o disco (só achou referências no
próprio código-fonte do fairseq, nunca uma transcrição real), `screen -ls`/`tmux
list-sessions` (vazio — nenhuma sessão viva), `history` (sem rastro do comando
original). **Considero os epochs 8-100 irrecuperáveis** com os meios disponíveis
remotamente. Isso precisa ser uma frase explícita no capítulo, não uma omissão
silenciosa.

A run Conformer (fevereiro) tem 29 pontos *step*-a-*step* de verdade no MLflow
(`C-metrics-raw.csv`), mas só `train/*` (sem validação, `disable_validation=True`) e
com semântica de `step` que não consegui confirmar com certeza (os valores brutos vão
de 1726 a 1754, perto do `max_epoch=1776` configurado — parece ser número de epoch, não
de update, mas não achei uma linha de log equivalente pra cravar isso). Não coloquei
essa run no mesmo `C-curvas.csv` porque o schema pedido (com `valid_*`) não se aplica a
ela e misturar as duas tabelas confundiria mais do que ajudaria — está inteira em
`C-metrics-raw.csv`/`C-params.csv` se você quiser usá-la como comparação secundária.

---

## C3 — Sumário do treino

Tudo abaixo vem do `extra_state` lido diretamente de `checkpoints/full_model/checkpoint_best.pt`
(colado nesta conversa) — não é mais estimativa.

- **Epoch e num_updates do `checkpoint_best.pt`**: epoch **101**, `num_updates` **113**
  (contagem cumulativa de passos de otimização desde o início do treino — não é
  "por epoch", é o total; com 101 epochs e só 113 updates, dá em média ~1,1 update por
  epoch, consistente com o dataset minúsculo de 18 pares de treino já documentado no
  Bloco E). Métrica de validação nele: `val_loss=8,38`, e o campo `best=8,38` — os dois
  campos são iguais.
- **Melhor `valid_loss` e em qual epoch**: **8,38, na própria epoch 101 (a última)** —
  `best == val_loss` no ponto final indica que a perda de validação ainda estava
  melhorando (ou pelo menos não piorando) até o fim do treino configurado; não há
  registro de um checkpoint anterior ter sido "o melhor" e depois superado.
- **Por que o treino parou**: `--max-epoch` foi configurado em 100
  (`logs_treino_full.txt:140`); o checkpoint final está em **epoch 101** — exatamente
  `max_epoch + 1`, que é o comportamento padrão do fairseq quando o loop de epochs
  termina normalmente (incrementa o contador antes de checar o limite e sair). Isso é
  evidência forte de que o treino **atingiu o `--max-epoch` configurado**, não parou por
  `--patience` (que exigiria parar *antes* de completar todas as epochs, não
  exatamente uma a mais) nem por crash (não haveria `checkpoint_best.pt`/`checkpoint_last.pt`
  consistentes e com `extra_state` bem-formado se tivesse crashado no meio de um save).
  Não tenho a linha literal do terminal dizendo "reached max epoch" — a conclusão é por
  inferência do número, não uma citação direta — mas é uma inferência bem forte.
- **Tempo total de treino**: `previous_training_time = 51084,8586` segundos ≈
  **14h11min**, valor gravado pelo próprio fairseq dentro do checkpoint (não
  estimado). Tempo médio por epoch: 51084,86 / 101 ≈ **506s (~8,4 min/epoch)** — mas
  esse número é dominado por overhead fixo (dataset de 18 exemplos, ~1 update/epoch),
  não por tempo de computação real; não deve ser lido como "tempo de treino por
  exemplo".
- **GPU**: NVIDIA L40S, 44,392 GB, capability 8,9, um único dispositivo
  (`logs_treino_full.txt:214-216`, `distributed_world_size=1`). **Pico de memória: NÃO
  CONFIRMADO** — o único dado disponível é um *snapshot* de `gb_free=34,38 GB` no
  último ponto logado (ou seja, ~10GB em uso *naquele instante*, não
  necessariamente o pico da execução inteira). Para o pico real, seria necessário ter
  rodado com `nvidia-smi --query-gpu=memory.used --format=csv -l 1` durante o treino —
  não há esse registro.
- **Houve overfitting?** Com apenas 2 pontos (epoch 7 e epoch 101, 94 epochs de
  distância), **não dá para caracterizar uma trajetória** — só o comparativo dos dois
  extremos: train_loss caiu de 11,14→9,46 e valid_loss caiu de 11,32→8,38. As duas
  caíram, o que não é o padrão clássico de overfitting (que seria valid subindo
  enquanto train continua caindo). O único ponto estranho é que, na epoch 101,
  `valid_loss (8,38) < train_loss (9,46)` — validação melhor que treino, o oposto do
  usual. Dado que o conjunto de validação real é de ~5-7 exemplos (Bloco E, E8), a
  explicação mais provável é ruído estatístico de amostra pequena, não um sinal real de
  generalização superior. **Recomendo não afirmar nem descartar overfitting no texto** —
  só reportar os dois números e declarar explicitamente que a granularidade disponível
  (2 pontos) não permite a análise que a pergunta pede.

---

## C4 — Existe script de avaliação?

**Não. Não há nenhum script de avaliação (BLEU/WER/SyncNet/ASR-BLEU) neste
repositório**, fora da biblioteca genérica do fairseq.

- `scripts/` tem 10 arquivos: `diag_fairseq.py`, `diagnose_utut_ckpt.py`,
  `download_models.py`, `generate_synthetic_data.py`, `inference_folder.py`,
  `prepare_data.py`, `run_daily_cycle.py`, `train_drive_pipeline.py`,
  `train_full_pipeline.py`, `verify_vocab.py` — nenhum deles calcula BLEU, WER ou
  sincronia labial. `diag_fairseq.py`/`diagnose_utut_ckpt.py` são scripts de
  diagnóstico de checkpoint (não de avaliação de saída); `verify_vocab.py` verifica o
  dicionário, não traduções.
- `grep` por `sacrebleu|jiwer|\bwer\b|syncnet|lse[-_]?[dc]|asr[-_]?bleu` em todo o
  repositório, fora de `fairseq/examples/`: só aparece dentro da própria biblioteca
  fairseq (`fairseq/fairseq/scoring/bleu.py`, `wer.py`, `chrf.py`,
  `fairseq/fairseq/tasks/*.py`) — utilitários genéricos que fairseq oferece para
  QUALQUER tarefa, nunca chamados por nenhum script deste projeto.
- **`syncnet`: zero ocorrências em qualquer lugar do repositório** (grep por conteúdo e
  por nome de arquivo, ambos vazios). Não existe `syncnet_python` nem pesos de SyncNet
  neste código.
- **Instalado no `av2av_env`: CONFIRMADO via `pip list` real na VM (2026-07-28)**:

  | pacote | versão | relevância |
  |---|---|---|
  | `sacrebleu` | 2.6.0 | **já instalado** — confirma a hipótese (dependência transitiva do fairseq, `fairseq/setup.py:186`); não precisa instalar nada pro BLEU |
  | `torchaudio` | 2.7.1+cu118 | sugere PyTorch **2.7.x** real — corrige minha estimativa anterior (~2.6.x, inferida só do nome do commit "Pt2.6 compatibility" do fairseq) |
  | `face_alignment` | 1.5.0 | usado por `inference.py` (bbox), não relacionado à avaliação |
  | `facexlib` | 0.3.0 | dependência do GFPGAN, não relacionado à avaliação |
  | `librosa` | 0.8.1 | bate com `requirements.txt` |
  | `opencv-python` | 4.5.4.60 | bate com `requirements.txt` |

  **Ausentes** (não apareceram no `pip list` filtrado, ou seja, não instalados):
  `jiwer`, `editdistance` (nenhum dos dois — WER, seja via `jiwer` seja via o scorer
  `wer` do próprio fairseq que exige `editdistance`, precisa instalar um dos dois) e
  `whisper`/`openai-whisper`/`faster-whisper` (nenhuma variante — ASR precisa ser
  instalado do zero).

---

## C5 — Caminho mais curto para ASR-BLEU

- **Vídeos de saída já gerados**: só existe UM conjunto, em `results/`:
  `video1_pt2en.mp4` (11,68s), `video2_pt2en.mp4` (48,92s), `video3_pt2en.mp4` (21,96s),
  `video4_pt2en.mp4` (26,88s) — ~109s de áudio no total. **Confirmado: são todos
  baseline** (checkpoint pré-treinado `utut_sts_ft.pt`). **Faltam as 4 versões
  fine-tuned** — rode isso na VM pra cada vídeo (mesmos parâmetros de renderer usados
  antes, só troca `--utut-path` e `--out-vid-path`):
  ```bash
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=fairseq python inference.py \
      --in-vid-path samples/video1.mp4 \
      --out-vid-path results/video1_pt2en_finetuned.mp4 \
      --src-lang pt --tgt-lang en \
      --av2unit-path checkpoints/mavhubert_large_noise.pt \
      --utut-path checkpoints/full_model/checkpoint_best.pt \
      --unit2av-path checkpoints/unit_av_renderer.pt \
      --video-renderer latentsync \
      --latentsync-repo latentsync_repo \
      --latentsync-python /home/alexandregodoy/miniconda3/envs/latentsync/bin/python \
      --latentsync-config configs/unet/stage2_512.yaml \
      --latentsync-ckpt checkpoints/latentsync_unet.pt
  ```
  Repita trocando `video1`→`video2`/`video3`/`video4` nos dois `--*-vid-path`. Isso
  entra no orçamento de tempo/GPU do C7 — são 4 inferências completas (av2unit +
  unit2unit + unit2av + LatentSync), não uma tarefa trivial.
- **Referência em inglês**: **não existe**. `samples/` só tem os 4 vídeos originais em
  **português** (com `.bbox.pkl` cacheado) — nenhum texto ou vídeo de referência em
  inglês para nenhum dos 4. Alternativa viável, com a limitação declarada: transcrever
  o áudio PT original com Whisper, traduzir o texto com um modelo de MT leve (ex.:
  `Helsinki-NLP/opus-mt-pt-en` via `transformers`, ~300MB, baixa uma vez do Hugging
  Face), e usar isso como pseudo-referência — declarando explicitamente no capítulo que
  não é uma tradução humana e que erros do MT/ASR se propagam para o BLEU medido.
- **Pipeline concreto**:
  1. `ffmpeg` extrai o áudio de cada vídeo gerado (mesmo utilitário já usado em
     `util.py:extract_audio_from_video`).
  2. ASR sobre o áudio gerado (EN) e sobre o áudio original (PT) — Whisper (`openai-whisper`
     ou `faster-whisper`; confirmado **não instalado**, ver C4).
  3. MT do transcript PT → EN (pseudo-referência) — `transformers` + modelo
     `opus-mt-pt-en` (baixa do Hugging Face — **rede confirmada disponível na VM**, ver C6).
  4. Normalização — minúsculas + remoção de pontuação (o próprio tokenizador do
     `sacrebleu`, modo `13a`, já cobre a maior parte disso).
  5. `sacrebleu` compara a transcrição do áudio gerado contra a pseudo-referência
     — **já instalado (2.6.0)**, nada a fazer aqui.
- **Pacotes faltando, confirmado via `pip list` real (C4)**: `openai-whisper` (ou
  `faster-whisper`), `transformers` + `sentencepiece` (para o MT). `sacrebleu` já está
  presente, não precisa instalar.
- **Estimativa**: ~5-6 comandos ao todo (um a menos, já que `sacrebleu` está pronto).
  Tempo de GPU real é pequeno — só ~109s de áudio total, transcrição com Whisper (mesmo
  o modelo `small`) leva segundos por clipe. O gargalo é **download** (pesos do Whisper
  + modelo de MT) e a instalação dos pacotes, não computação — e como a rede da VM
  está confirmada funcionando, isso não deve travar. Para não pressionar `/mnt/disk` a
  100%, baixe os pesos com cache apontando para `/dev/shm/alex/` (`HF_HOME`/`XDG_CACHE_HOME`
  antes de instalar/rodar), do mesmo jeito que já foi feito para o data-bin de treino.

---

## C6 — Caminho mais curto para LSE-D e LSE-C

- **SyncNet/`syncnet_python`: confirmado que não existe em lugar nenhum**, nem no
  repositório nem no `av2av_env` (não apareceu no `pip list`, e nenhum grep achou
  vestígio). Pesos: **não estão disponíveis offline** — precisam ser baixados.
- **Restrição de rede: CONFIRMADO que NÃO há restrição.** Você rodou
  `curl -sI https://github.com --max-time 5` direto na VM e voltou `HTTP/2 200` — a VM
  tem saída para a internet. **Não precisa baixar nada pela sua máquina Windows e
  enviar por scp** — dá pra clonar o repositório e baixar os pesos direto na VM.
- **Arquivos exatos a baixar** (verifiquei direto no repositório oficial
  `joonson/syncnet_python` agora, não de memória):
  - **Pesos do SyncNet** — confirmado no próprio `download_model.sh` do repositório:
    `http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model` (salvar
    como `data/syncnet_v2.model` dentro do clone do `syncnet_python`).
  - **Detector de face S3FD** (usado por `run_pipeline.py`, etapa de rastreamento de
    rosto que roda ANTES do SyncNet): o repositório importa `from detectors import
    S3FD`, mas a pasta `detectors/` aparece vazia/como submódulo no GitHub e eu **não
    consegui confirmar** a URL exata dos pesos do S3FD a partir daqui (não achei
    `.gitmodules` nem um segundo script de download) — **NÃO CONFIRMADO**. Isso é um
    segundo download/dependência, não só o `syncnet_v2.model` — vale descobrir isso
    assim que clonar o repo com rede disponível (o próprio setup deve deixar claro o
    que falta), antes de assumir que é só um arquivo.
- **Pipeline concreto** (`run_pipeline.py` → `run_syncnet.py` → `run_visualise.py`,
  conforme o próprio README do repo): rastreamento de rosto por vídeo → cálculo do
  offset de sincronia → LSE-D/LSE-C reportados pelo `run_syncnet.py`.
- **Esforço estimado**: mais alto que o ASR-BLEU, mas a rede confirmada tira o maior
  risco logístico (não precisa mais depender de scp manual). O que resta como incerteza
  real: clonar um repositório de 2016-2020 com dependências antigas (o
  `environment.yml` do próprio repo sugere um ambiente conda dedicado, possivelmente
  incompatível com `av2av_env`), resolver a segunda dependência de pesos (S3FD, ainda
  não confirmada — só vai ficar clara ao clonar o repo e olhar o `detectors/`), e rodar
  3 scripts por vídeo. Estimativa honesta: **3-6 horas de trabalho de configuração**
  (revisada para baixo agora que a rede não é mais um obstáculo; não é tempo de GPU —
  o cálculo em si, para 4 vídeos curtos, é rápido), com risco de imprevistos ainda
  real dado que é um ambiente antigo e menos mantido que o resto do pipeline. Como a
  rede funciona, vale a pena só tentar clonar e rodar `download_model.sh` agora — a
  única forma de saber se o S3FD é 10 minutos ou 3 horas é descobrir na prática.
- **Métrica substituta, se o esforço não couber no prazo**: diferença de duração entre
  o áudio original e o gerado (proxy de isometria/pacing, não de sincronia labial
  fina). É trivial de calcular — já temos `util.get_audio_duration()` no próprio
  projeto — mas precisa ser apresentada explicitamente como proxy fraco: mede se a
  fala gerada dura perto do mesmo tempo que a original, não se os lábios batem
  quadro-a-quadro com o áudio. Não substitui LSE-D/LSE-C cientificamente, só dá algum
  número quantificável para a seção de resultados enquanto o texto principal reconhece
  a limitação.

---

## C7 — Veredito de escopo

Considerando a defesa em **02/11/2026**, uma GPU compartilhada
(`CUDA_VISIBLE_DEVICES=1`) num disco a 100%, **e que as 4 inferências fine-tuned ainda
precisam ser geradas** (confirmado no C5 — os 4 vídeos existentes são todos baseline):

| Métrica | Veredito | Justificativa |
|---|---|---|
| **ASR-BLEU** | **Viável no prazo** | Só falta instalar 2-3 pacotes (download-bound, não compute-bound) e decidir a pseudo-referência. ~109s de áudio total, GPU real necessária é mínima. |
| **WER** | **Viável, reaproveitando o ASR-BLEU** | Mesma transcrição do ASR-BLEU já dá o WER contra a mesma pseudo-referência (via `jiwer`, pacote leve, não confirmado instalado mas trivial de instalar). Vem "de graça" depois do C5. |
| **LSE-D / LSE-C** | **Vale tentar, com prazo curto e um plano B pronto** | Rede confirmada OK (era o maior risco); resta um repositório de terceiros pouco mantido, ambiente conda possivelmente incompatível, e uma segunda dependência de pesos (S3FD) ainda não confirmada. Nenhum desses é necessariamente bloqueante, mas nenhum é garantido rápido. |

**Recomendação direta**: reporte ASR-BLEU e WER como resultados quantitativos reais no
capítulo de Validação Quantitativa (ambos alcançáveis com o tempo restante, e agora
mais baratos ainda com `sacrebleu` já instalado). Para LSE-D/LSE-C, dado que a rede não
é mais obstáculo, vale **uma tentativa com prazo fixo** (sugiro no máximo meio dia de
trabalho): clonar `syncnet_python`, rodar `download_model.sh`, e ver imediatamente se
`detectors/S3FD` resolve sozinho ou exige um segundo download manual. Se em meio dia
não estiver rodando ponta a ponta, corte as perdas e vá para o plano B: declare
explicitamente no capítulo de **Ameaças à Validade** que a sincronia labial fina não
foi avaliada quantitativamente por restrição de tempo (não mais de infraestrutura, já
que a rede funciona — seja específico sobre qual foi a razão real), e ofereça a
métrica substituta (diferença de duração áudio-original-vs-gerado) como um proxy
parcial, deixando claro no texto que é isso — um proxy, não uma medição de sincronia
labial.
