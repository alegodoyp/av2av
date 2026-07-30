# Bloco D — Inferência, bugs, limitações e reprodutibilidade

**Nota de execução**: checkout Windows local, mesmo padrão dos blocos anteriores.
Boa parte deste bloco reaproveita conhecimento direto desta mesma conversa (os
comandos de inferência que eu mesmo dei e que você rodou, para as 8 saídas do
survey), não apenas leitura de código nova — vou marcar claramente o que é uma
coisa e o que é outra.

---

## D1 — Pipeline de inferência ponta a ponta

Estágios confirmados em `inference.py`, citando os pontos exatos de carregamento
de modelo (`inference.py:451-455`):

1. **Pré-processamento**: `extract_bbox()` roda `face_alignment` sobre o vídeo de
   entrada (cache em `<video>.bbox.pkl`); `extract_audio_from_video()` extrai o
   áudio a 16kHz mono PCM.
2. **av2unit** (`av2unit_model`, carregado de `args.av2unit_path`,
   `inference.py:451`): vídeo+áudio → sequência de unidades discretas PT (inteiros
   0-999, taxa nativa do av-hubert, antes da redução por run-length).
3. **unit2unit** (`unit2unit_task`/`unit2unit_generator`, carregado de
   `args.utut_path`, `inference.py:452`): unidades PT → unidades EN, por busca em
   feixe (ver Bloco E, E3 — comprimento livre, não preserva o da entrada).
4. **unit2av** (`unit2av_model`, carregado de `args.unit2av_path`,
   `inference.py:454`): unidades EN + embedding do locutor → forma de onda 16kHz +
   patches de boca 96×96 + duração determinada por um `dur_predictor` interno
   (ver Bloco E, E3 para o mecanismo completo de isocronia).
5. **Pós-processamento**: `util.save_video()` faz *seamless clone* dos patches no
   vídeo de fundo e mux de áudio+vídeo via `ffmpeg` (`libx264`/`aac`), a 25 fps —
   opcionalmente passando por GFPGAN (`face_restore.py`) e VoiceFixer
   (`audio_restore.py`) antes disso. Se `--video-renderer latentsync`, o
   `unit2av`/`save_video()` são pulados e `render_video_latentsync()` faz tudo
   (detecção de face própria, difusão, mux) como processo único.

- **Speaker encoder (`unit2av/encoder.pt`)**: carregado de forma **hardcoded**,
  independente de qualquer flag (`inference.py:455`:
  `load_speaker_encoder_model(os.path.join("unit2av", "encoder.pt"), ...)`). É um
  encoder estilo GE2E (LSTM 3 camadas, embedding 256-d — confirmado em
  `unit2av/model_speaker_encoder.py:58-82`, adaptado do
  `Real-Time-Voice-Cloning` de Corentin Jemine, citado no topo do arquivo). Recebe
  como entrada o **áudio do próprio vídeo de origem** (`speaker_ref_path`, derivado
  de `args.in_vid_path`, opcionalmente limpo por VoiceFixer). **A voz de saída é,
  portanto, clonada do falante original — não é uma voz genérica.**
- **Efeito de trocar `--utut-path` entre `utut_sts_ft.pt` e
  `checkpoints/full_model/checkpoint_best.pt`**: confirmado via as 4 linhas de
  carregamento de modelo citadas acima — `av2unit_model` (linha 451), `unit2av_model`
  (linha 454) e `speaker_encoder_model` (linha 455) usam caminhos **completamente
  independentes** de `args.utut_path`. **Só `unit2unit_task`/`unit2unit_generator`
  (linha 452) muda.** Isso sustenta diretamente a comparação controlada do survey —
  qualquer diferença percebida entre as duas condições vem exclusivamente da
  tradução de unidades, não de mudança na extração, na renderização ou na voz.
- **Tempo médio de processamento e VRAM**: **parcialmente confirmado**. Não há
  medição do tempo total do pipeline completo nem monitoramento de VRAM logado em
  nenhum artefato desta sessão. O que está diretamente medido (logs do
  `syncnet_python` rodados nesta mesma conversa, Bloco C6) é o tempo da etapa de
  **difusão do LatentSync** especificamente, por vídeo: video1 (11,7s) — 1min39s;
  video2 (48,9s) — 7min40s; video3 (22,0s) — 3min14s; video4 (26,9s) — 4min14s.
  Essa etapa é quase certamente o gargalo dominante do pipeline (av2unit/unit2unit/
  unit2av rodam em segundos, pela ordem de grandeza vista nos próprios logs de
  treino). Tempo total ponta a ponta e pico de VRAM: **NÃO CONFIRMADO** — precisaria
  rodar com `/usr/bin/time` ou `nvidia-smi --query-gpu=memory.used -l 1` durante uma
  execução real.

---

## D2 — Inventário dos artefatos de inferência

- **Diretórios**: uma única pasta, `results/`, com as duas condições
  distinguidas só pelo sufixo do nome do arquivo — baseline:
  `video{1,2,3,4}_pt2en.mp4`; fine-tuned: `video{1,2,3,4}_pt2en_finetuned.mp4`.
- **Mesmo conjunto nas duas condições?** **Sim, confirmado** — os dois conjuntos
  têm exatamente os vídeos 1, 2, 3 e 4, sem nenhum ausente de um lado ou do outro
  (verifiquei a listagem de `results/` diretamente nesta sessão, Bloco C).
- **Bug do `break` em `scripts/inference_folder.py`**: **confirmado presente no
  código atual**, e é exatamente como descrito — `inference_folder.py:100` tem um
  `break` incondicional na última linha do corpo do `try`, dentro do
  `for vid_path in videos:` (linha 52). Isso faz o script processar **só o
  primeiro vídeo** encontrado por `input_dir.glob('**/*')`, sempre, não importa
  quantos vídeos existam na pasta — contradiz o próprio propósito do script
  (processar uma pasta inteira).
- **Isso invalida os estímulos do survey? NÃO.** Confirmo com alta confiança
  porque tenho conhecimento direto desta própria conversa: os 8 vídeos do survey
  foram gerados por **chamadas individuais de `inference.py`**, uma por vídeo,
  encadeadas manualmente numa sessão `tmux` (os comandos exatos foram dados e
  executados nesta conversa, com logs colados de volta) — `inference_folder.py`
  **nunca foi invocado** neste processo. O bug é real e deveria ser corrigido antes
  de qualquer uso futuro do script em lote, mas não afeta os estímulos já
  produzidos.

---

## D3 — Lista consolidada de bugs

| arquivo:linha | descrição | status atual | impacto |
|---|---|---|---|
| `scripts/train_full_pipeline.py` — `unified_bin_dir / "bin"` | caminho citado como inexistente | **NÃO REPRODUZIDO no código/execução atuais** — o merge de binários atual (linhas 317-342) copia direto para `unified_bin_dir/<lang>/`, sem subpasta `bin` extra, e o Namespace da execução real (`logs_treino_full.txt:140`, campo `data`) aponta consistentemente para esse mesmo caminho sem erro. Se existiu, já não está presente. | — |
| `scripts/train_full_pipeline.py` — `--disable-validation` impedia `checkpoint_best.pt` | flag desabilitava validação | **NÃO PRESENTE** — não há `--disable-validation` em `cmd_args` (linhas 98-150); a execução real logou `'disable_validation': False`, e `checkpoint_best.pt` existe e foi confirmado lido (Bloco A). | — |
| `scripts/train_full_pipeline.py` — ausência de `--finetune-from-model` | treinaria do zero | **NÃO PRESENTE** — `--finetune-from-model` está em `train_full_pipeline.py:129`, confirmado no cfg real (Bloco A: aponta para a cópia patched de `utut_sts_ft.pt`). | — |
| `scripts/train_full_pipeline.py` — `--arch` default `conformer_utut` incompatível | default do parser não bate com o checkpoint | **PARCIALMENTE PRESENTE** — o default do parser (`train_full_pipeline.py:39`) ainda é `"conformer_utut"`, mas `run_training()` ignora `args.arch` e fixa `"utut_large"` (linha 105, hardcoded) — o efeito prático do bug (treinar com arch errada) está neutralizado, mas o `--help`/default continua enganoso. | *cosmético* (documentação do CLI), não *bloqueante* |
| `scripts/prepare_data.py:20` — import não registrava a task | `av_hubert_unit_pretraining` não registrada | **Import presente e defensivo** (`prepare_data.py:19-24`, `try/except` com aviso) — não reproduzi uma falha real nesta sessão. | — |
| `av2unit/task.py:43` — assinatura de `load_dataset` | incompatibilidade de assinatura | **Assinatura atual é flexível**: `def load_dataset(self, split=None, **kwargs)` (`av2unit/task.py:43`), aceita a chamada real `task.load_dataset(split="valid")` de `prepare_data.py` sem erro. | — |
| `unit2unit/inference.py` — `arg_overrides` para `data/dict.txt` inexistente | caminho de dicionário quebrado | **NÃO PRESENTE** — `arg_overrides={"user_dir": "unit2unit", "data": _ensure_extended_dict_dir()}` (`unit2unit/inference.py:45`) usa uma função que constrói um diretório de dicionário válido, não um caminho literal quebrado. | — |
| `scripts/inference_folder.py:100` — `break` prematuro | processa só o 1º vídeo da pasta | **CONFIRMADO PRESENTE, não corrigido.** Ver D2 — não afeta os estímulos já gerados (não foi usado para gerá-los), mas quebra o uso pretendido do script. | *bloqueante* para o próprio propósito do script; sem impacto nos resultados já produzidos |
| `scripts/generate_synthetic_data.py:196-197` — upload duplicado | upload do alvo sintético duplicado | **CONFIRMADO PRESENTE, não corrigido** (Bloco B). Mecanismo exato de como isso se propaga (ou não) para pares duplicados de treino já analisado no Bloco B — conclusão lá foi que esse bug isolado não basta pra explicar a duplicação observada, mas ele mesmo continua no código. | *silencioso* — não impede execução, desperdiça quota/armazenamento do Drive |

**Bugs adicionais encontrados nesta sessão, fora da lista original** (todos já
corrigidos ao longo da conversa, registrados aqui por completude do apêndice):
`unit2av/model.py` — mecanismo de ajuste de duração precisou de 3 iterações antes
de chegar num estado aceitável (ver histórico completo desta conversa: escalonamento
uniforme → tentativa de direcionar só "códigos de pausa" — revertida, causou áudio
ininteligível → piso de velocidade fixo + sem mais forçar duração); `latentsync_render.py`
— frame preto/rosto não detectado em vídeos específicos, corrigido com patch no
`image_processor.py` do próprio LatentSync; `wav2lip_render.py` — checkpoint
carregado como TorchScript archive em vez de `state_dict` puro, tratado com
`isinstance` check.

---

## D4 — Limitações técnicas conhecidas (para Ameaças à Validade)

- **Duração/pacing**: a duração do vídeo/áudio gerado é livre (sem forçar
  correspondência com a duração da fonte, decisão final desta sessão) — o vídeo
  traduzido pode ficar sensivelmente **mais curto** que o original quando a
  tradução resulta em menos unidades/tempo de fala. Não há mecanismo de
  preenchimento por pausa real implementado (uma tentativa foi feita e revertida
  por degradar a inteligibilidade — ver histórico).
- **Nº de falantes / enquadramento**: todo o pipeline (extração de bbox, patches
  96×96, `seamlessClone`) assume **um único rosto por vídeo**, enquadramento
  frontal relativamente estável. Não há tratamento para múltiplos falantes em
  quadro, mudança de plano/corte de câmera, ou ângulos não frontais.
- **Rosto fora de quadro / oclusão**: `extract_bbox()` grava `None` para frames
  sem detecção; `unit2av.get_crops()` faz preenchimento para frente/trás
  (forward/backward-fill) para seu próprio uso; o LatentSync (quando usado) tem
  seu próprio detector (InsightFace) com comportamento de fallback só depois de um
  patch aplicado nesta sessão (`_patch_image_processor_for_missing_faces`, em
  `latentsync_render.py`) — sem esse patch, uma oclusão total causava crash.
  Nenhum dos dois caminhos lida bem com oclusão **prolongada** (múltiplos
  segundos), só com falhas pontuais de frame.
- **Qualidade de renderização**: o `unit2av` nativo e o Wav2Lip operam a
  **96×96** — resolução baixa o suficiente para deixar dentes/interior da boca
  visivelmente borrados, motivo direto de ter adicionado o LatentSync (512×512)
  como alternativa nesta sessão. GFPGAN ajuda na nitidez geral do rosto, mas não
  resolve artefatos específicos de boca do renderizador base.
- **Identidade vocal/prosódia**: a clonagem de voz (D1) captura timbre via o
  encoder GE2E, mas a saída do vocoder (`CodeHiFiGANModel_spk`) é nativamente
  limitada a 16kHz/~8kHz de banda útil (`unit2av/config.json`) — soa
  "abafada"/"ruidosa" sem o pós-processamento de VoiceFixer, que ajuda mas não
  elimina esse teto de qualidade. Tentativas de melhorar a clonagem em si
  (reescalonar o embedding do locutor) ficaram marcadas como experimentais/não
  validadas.
- **Idiomas suportados vs. testados**: `inference.py` aceita `en/es/fr/it/pt` em
  qualquer par (`--src-lang`/`--tgt-lang`, `inference.py` `choices=[...]`), mas
  **só PT→EN foi de fato testado e usado** nesta sessão e nos dados de
  treino/avaliação (Blocos B, C). Qualquer outro par é tecnicamente possível mas
  sem nenhuma validação empírica registrada.
- **Falhas observadas nos logs de inferência desta sessão**: repetição de n-gramas
  no decoder do `unit2unit` para conteúdo longo/difícil (mitigado com chunking
  recursivo alinhado a pausas — ver Bloco E); erro de vocabulário raro/nomes
  próprios propagado para o BLEU medido (Bloco C — "Mário Quintana"→"Mario
  Contana" em ambos os checkpoints).

---

## D6 — Higiene do repositório antes do depósito

- **Credenciais**: `client_secrets.json`, `token.json`, `credentials.json`,
  `mycreds.txt`, `service_account.json` — **nenhum rastreado pelo git** neste
  repositório (`git ls-files | grep` vazio) e **nenhum presente localmente** neste
  checkout. `.gitignore` já lista os nomes relevantes explicitamente. Nada a fazer
  aqui para o `av2av`.
- **Arquivos grandes indevidos**: nenhum checkpoint rastreado (confirmado
  repetidamente ao longo desta sessão). O repositório rastreado soma **~5,6GB no
  total** (`git ls-files | xargs du -ch`), dominado pelos vídeos de resultado
  commitados em `results/` ao longo da sessão — não é um erro, é o efeito
  esperado de ter versionado os vídeos gerados; só vale mencionar como nota de
  tamanho de clone, não como bug.
- **O que falta para o apêndice de reprodutibilidade ser verdadeiro**: ver D5 —
  principalmente as dependências ausentes do `requirements.txt`, confirmadas por
  grep direto nos imports (não só citadas de memória): `imageio`, `face_alignment`
  (`inference.py:10-11`, `unit2av/model.py:12`), `imageio_ffmpeg`
  (`util.py:5`, `latentsync_render.py:27`), `tqdm`
  (`scripts/download_models.py:16`), `googleapiclient`/`google_auth_oauthlib`/
  `google.auth` (`drive_utils.py:6-9,115`) — nenhum desses aparece em
  `requirements.txt`.
