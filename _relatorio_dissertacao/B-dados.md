# Bloco B — Construção do corpus pseudo-pareado PT-BR

**Nota de execução**: mesmo checkout Windows local dos blocos anteriores, não a VM.
Toda a evidência de contagens/pares vem de `logs_treino_full.txt`, que — descoberta
desta rodada — contém a fase de PREPARO DE DADOS completa (tráfego do Drive, download,
extração de unidades) no início do arquivo, não só o treino do fairseq a partir da
epoch 7 como eu havia registrado nos Blocos C/E. Isso é usado extensivamente abaixo.

---

## B1 — A circularidade dos alvos sintéticos é real, confirmada por código

**Resposta categórica: SIM, é auto-destilação — e é mais circular do que uma leitura
superficial sugere, com DUAS voltas, não uma.**

- **Checkpoint carregado para gerar `synthetic_targets`**:
  `checkpoints/utut_sts_ft.pt`, citado literalmente em
  [scripts/generate_synthetic_data.py:154](scripts/generate_synthetic_data.py#L154)
  (`"--utut-path", "checkpoints/utut_sts_ft.pt"`).
- **É o mesmo checkpoint do `--finetune-from-model`?** SIM — o mesmo caminho literal
  é usado em
  [scripts/train_full_pipeline.py:96](scripts/train_full_pipeline.py#L96)
  (`patch_utut_checkpoint("checkpoints/utut_sts_ft.pt")`). Mesmo arquivo, mesmos pesos.
- **Par de idiomas e estágios usados**: PT→EN
  ([linhas 151-152](scripts/generate_synthetic_data.py#L151-L152)), e os **três**
  estágios do pipeline — `--av2unit-path`, `--utut-path` e `--unit2av-path` todos
  presentes na chamada ([linhas 153-155](scripts/generate_synthetic_data.py#L153-L155)).
  Não é só tradução de unidades: é o `inference.py` oficial completo, ponta a ponta.
- **O que vira o "alvo"**: um **vídeo renderizado completo** (áudio sintetizado +
  vídeo com os lábios gerados), não unidades nem só áudio — `--out-vid-path` gera um
  `.mp4` ([linha 150](scripts/generate_synthetic_data.py#L150)), que é isso que sobe
  pro Drive em `synthetic_targets/`.

**A segunda volta da circularidade, que não estava no seu roteiro mas é igualmente
importante**: esse vídeo sintético não entra direto no treino do fairseq — ele passa
de novo pelo `av2unit` para virar unidades discretas, em
[scripts/prepare_data.py:296](scripts/prepare_data.py#L296)
(`tgt_units = extract_units(tgt_model, tgt_task, tgt_vid, use_cuda)`). E qual modelo é
esse `tgt_model`? Por padrão, **o mesmo `av2unit` usado do lado da fonte**
([prepare_data.py:271](scripts/prepare_data.py#L271):
`tgt_model, tgt_task = src_model, src_task`, só sobrescrito se
`--tgt-av2unit-path` for passado explicitamente e diferente de `--av2unit-path`).
Confirmei que isso **não** foi sobrescrito na execução real: `prepare_data.py:273`
imprimiria `"Loading separate target model from..."` se tivesse sido, e essa linha
**não aparece em nenhum lugar** de `logs_treino_full.txt`.

Isso existe de fato um caminho no repositório que usaria um av2unit *diferente* pro
lado do alvo — `scripts/train_drive_pipeline.py`, orquestrado por
`scripts/run_daily_cycle.py`, que passa explicitamente
`--tgt-av2unit-path checkpoints/avhubert_base_1000.pt` (um checkpoint diferente,
menor). **Mas esse não é o caminho que gerou `checkpoint_best.pt`**: confirmei que a
string `"Recursively traversing Google Drive folders to find all video pairs..."`
que abre `logs_treino_full.txt` existe **só** em `train_full_pipeline.py`
(`grep` em `scripts/` não encontra essa string em `train_drive_pipeline.py`). A
execução real foi via `train_full_pipeline.py`, chamado diretamente — não via
`run_daily_cycle.py`/`train_drive_pipeline.py`. Ou seja: o caminho "menos circular"
existe no código, mas não foi o usado.

**Cadeia de circularidade completa, confirmada por código**:
```
vídeo PT bruto
  → av2unit (mavhubert_large_noise.pt)         → unidades PT
  → unit2unit (utut_sts_ft.pt)                 → unidades EN
  → unit2av (unit_av_renderer.pt)               → vídeo EN sintético ("alvo")
  → [tempo depois, no preparo de dados]
    av2unit (mavhubert_large_noise.pt de novo)  → unidades EN (isso é o rótulo real de treino)

Fine-tuning: utut_sts_ft.pt aprende a mapear
  unidades PT (do vídeo bruto)  →  unidades EN (derivadas da SUA PRÓPRIA saída anterior)
```

### Argumentos defensáveis perante a banca (já que a resposta é sim)

1. **Adaptação de domínio ao PT-BR / à fonética e prosódia locais**: mesmo sendo
   auto-destilação do lado da tradução, o **lado da entrada** (PT) continua sendo
   fala real, não sintética — o fine-tuning ainda expõe o `av2unit`+`unit2unit` a
   sotaque, prosódia e condições acústicas de falantes de PT-BR reais que não
   estavam necessariamente bem representados no pré-treino original (feito, pelo que
   os checkpoints de origem sugerem, majoritariamente sobre LRS3/mTEDx). Isso é uma
   forma legítima de adaptação de domínio na direção PT→modelo, independentemente da
   qualidade do rótulo EN.
2. **Regularização/recalibração sobre a própria distribuição de saída**: fine-tuning
   sobre as próprias saídas (self-training) é uma técnica documentada na literatura
   de tradução automática (ex.: self-training, noisy student) — pode reforçar
   padrões de saída que o modelo já produz de forma consistente/confiante, funcionando
   como uma forma de calibração, não necessariamente de aprendizado de conteúdo novo.
3. **Ausência de alternativa viável no prazo/recursos do projeto**: não há corpus
   paralelo PT-BR→EN audiovisual com sincronia labial humana disponível publicamente
   nessa escala; a auto-destilação foi a forma prática de gerar QUALQUER sinal de
   fine-tuning específico de domínio dentro do escopo de um mestrado.

### Limitações a declarar espontaneamente

1. **Não há evidência de que a tradução aprendida melhore** — e o Bloco C já mediu
   isso: ASR-BLEU/WER pioraram no checkpoint fine-tuned em relação ao baseline nos 4
   vídeos de teste. Isso é consistente com auto-destilação sem ganho real de
   qualidade: o modelo não pode aprender a traduzir melhor do que ele mesmo já
   traduzia, só reforçar (ou, como medido, levemente degradar) o que já fazia.
2. **Qualquer erro sistemático do checkpoint pré-treinado se propaga e pode se
   amplificar** — como o "alvo" é gerado pelo próprio modelo, erros de tradução
   recorrentes (ex.: nomes próprios, vocabulário raro — já documentado no Bloco C
   com "Mário Quintana"→"Mario Contana") entram como se fossem rótulo correto, sem
   nenhuma verificação humana (ver B6).

---

## B2 — Como o `mouth_cropped` é produzido

**Atualização**: não está neste repositório (`av2av`), mas está no repositório irmão
`C:\Users\lelex\source\repos\Mestrado\pre-training_pipeline`, que você indicou nesta
sessão. Lido de ponta a ponta — todos os itens pedidos agora estão **confirmados por
código**, não mais NÃO CONFIRMADO.

- **Script**: `crop_mouth_from_video.py` (repo `pre-training_pipeline`), que baixa
  cada vídeo de `videos_mestrado/<data>/` no Drive, gera o recorte, faz upload em
  `videos_mestrado/mouth_cropped/<data>/` e apaga o local — mesma convenção de pastas
  por data usada no `av2av`.
- **Detector**: **MediaPipe**, confirmado em
  `crop_mouth_from_video.py:204`: `AVSRDataLoader(modality="video",
  detector="mediapipe", convert_gray=False)`. A classe real
  (`preparation/detectors/mediapipe_detector/detector.py:10-33`) usa **dois modelos em
  cascata**: `full_range_detector` (`model_selection=1`) primeiro; só se ele não achar
  rosto em **nenhum** frame do vídeo inteiro, tenta de novo o vídeo inteiro com
  `short_range_detector` (`model_selection=0`) — confirma exatamente a alegação
  "Full Range com fallback Short Range" da dissertação, inclusive que o fallback é
  por vídeo inteiro, não por frame individual.
- **Resolução/formato/FPS**: recorte final **96×96**
  (`video_process.py:59-60`, `crop_width=96, crop_height=96`), sobre um canvas de
  alinhamento intermediário de 256×256 (`target_size=(256, 256)`,
  `video_process.py:150`). **Colorido** (RGB), não grayscale —
  `convert_gray=False` passado explicitamente
  (`crop_mouth_from_video.py:204`; o default da classe é `True`, mas não é o usado
  aqui). FPS de saída = **FPS nativo do vídeo fonte** (`crop_mouth_from_video.py:27`:
  `cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)`), não um valor fixo
  reamostrado. Formato `.mp4`, com o áudio original remesclado de volta via
  `ffmpeg` depois do recorte silencioso (`crop_mouth_from_video.py:30-59`).
- **Alinhamento geométrico**: transformada afim de similaridade via
  `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)`
  (`video_process.py:191-196`) — **confirma LMEDS**, não mínimos quadrados comuns.
  Pontos estáveis usados: olho direito, olho esquerdo, ponta do nariz e centro da
  boca (`video_process.py:177-189`), contra um rosto-médio de referência
  (`20words_mean_face.npy`).
- **Suavização temporal**: sim — média móvel das landmarks numa janela de
  **`window_margin=12`** frames (±6 ao redor de cada frame,
  `video_process.py:59-63,90-104`, `min(window_margin // 2, ...)`), recentralizada
  para não deslocar a posição do frame atual.
- **Falha de detecção por frame**: **interpolação linear entre detecções válidas**
  (`video_process.py:117-142`, `interpolate_landmarks`/`linear_interpolate`) — não
  descarta o vídeo nem repete cegamente o último frame no meio da sequência. Só nas
  bordas (início/fim do vídeo, se as primeiras/últimas detecções falharem), o código
  **propaga a landmark válida mais próxima** para preencher os frames sem detecção
  (`video_process.py:131-138`). Se **nenhum** frame do vídeo inteiro tiver detecção
  (mesmo após o fallback Full→Short Range), o vídeo inteiro é descartado
  (`assert any(l is not None for l in landmarks)`, `detector.py:32`).

---

## B3 — Regra de casamento dos pares e descartes

- **Casamento**: por **nome de arquivo** (não hash, não índice) — comparando o nome
  em `mouth_cropped/` com o nome em `synthetic_targets/`, com uma regra de remoção do
  prefixo `cropped_` quando presente
  ([scripts/train_full_pipeline.py:223-233](scripts/train_full_pipeline.py#L223-L233)).
- **Contagens, direto do log** (`logs_treino_full.txt`, linhas iniciais): **523**
  vídeos-fonte encontrados no Drive, **23** alvos sintéticos, **23** pares casados
  (ou seja, todo alvo existente achou uma fonte — o gargalo é quantos vídeos têm alvo
  sintético gerado, não o casamento em si). Split: **18 treino / 5 validação**.

### Achado central deste bloco: o corpus de 23 "pares" é, na prática, só 5 clipes únicos — com vazamento entre treino e validação

Reconstruí a lista de arquivos únicos a partir das linhas de download do log
(`[N/18] Downloading ...`/`[N/5] Downloading ...`). Tabela completa em
`_relatorio_dissertacao/B-corpus.csv`:

| clipe-base | ocorrências em treino | ocorrências em validação |
|---|---|---|
| `cropped_Monologo_Justica.mp4` | 4 | **2** |
| `cropped_Disturbios_De_Um_Monologo_Interior_-_Jefferson_Cavalcante.mp4` | 4 | 0 |
| `cropped_Wandinha_-_Monologo.mp4` | 3 | 0 |
| `cropped_Monologo_Natal_2022_-_Rodrigo_Kenji_Concurso_Five.mp4` | 3 | **3** |
| `cropped_Andre_Araujo_-_Monologo_do_filme_O_Pai_O.mp4` | 4 | 0 |

**Os 23 "pares" citáveis como exemplos de treino/validação são, na verdade, apenas 5
clipes-fonte distintos**, cada um repetido de 3 a 8 vezes ao todo. E mais grave: **os
2 clipes únicos que compõem a validação (`Monologo_Justica`, `Monologo_Natal_2022`)
aparecem, ambos, também no conjunto de treino** — ou seja, **100% do conteúdo único
de validação já apareceu (múltiplas vezes) em treino**. O `val_loss` de 8,38 reportado
no Bloco C não mede generalização a conteúdo não visto — mede, na melhor das
hipóteses, memorização de exemplares repetidos do mesmo punhado de clipes.

### O bug de upload duplicado (linhas 196-197) explica isso? Parcialmente, e a resposta honesta é mais sutil.

Verifiquei o mecanismo com precisão, lendo `drive_utils.py`:
- `upload_file()` ([drive_utils.py:113-127](drive_utils.py#L113-L127)) sempre chama
  `service.files().create(...)` — a API do Drive **não verifica duplicata por nome**,
  então rodar o upload duas vezes (como o bug das
  [linhas 196-197](scripts/generate_synthetic_data.py#L196-L197) faz) cria **dois
  arquivos de fato distintos** no Drive, com o mesmo nome, IDs diferentes.
- Só que `traverse_drive_folder()` ([drive_utils.py:81-101](drive_utils.py#L81-L101))
  alimenta um **dicionário Python** em `train_full_pipeline.py`
  (`src_files_map[norm_path] = item['id']`,
  [linha 205](scripts/train_full_pipeline.py#L205)) — se dois arquivos do Drive têm o
  **mesmo `norm_path`** (mesmo nome, mesma pasta de data), a segunda ocorrência
  simplesmente **sobrescreve** a primeira na entrada do dicionário. Ou seja, um
  upload duplicado dentro do MESMO dia/pasta **não sobrevive** como par duplicado na
  lista final de `pairs` — vira só uma entrada.
- **Logo, o bug das linhas 196-197, isoladamente, não é suficiente para explicar a
  repetição observada.** A explicação mais provável — mas que eu não consigo provar
  100% sem inspecionar a estrutura de pastas do Drive diretamente — é que o
  **mesmo clipe-fonte foi processado/carregado em MÚLTIPLAS pastas de data diferentes**
  (`<ano>/<mês>/<dia>/`), o que gera `norm_path`s diferentes (a data faz parte do
  caminho) e portanto entradas genuinamente separadas na lista de pares, mesmo sendo
  o mesmo conteúdo. Isso é uma questão de **como a coleta/geração foi operada ao
  longo de vários dias**, não um bug pontual de código — **NÃO CONFIRMADO** o
  mecanismo exato sem acesso à estrutura de pastas do Drive.

**Estimativa de quantos pares duplicados entraram no treino, pedida no B3**: dos 18
exemplos de treino, **13 são cópias de conteúdo já representado por um dos outros 5**
(18 entradas − 5 clipes únicos = 13 duplicatas). Da validação, 3 das 5 entradas são
duplicatas (5 − 2 únicos = 3).

- **Critérios de descarte logados**: nenhum critério de duração mínima/máxima ou
  contagem de frames aparece logado nesta fase — o único descarte visível no código é
  falha de extração de unidades (`if src_units is None or tgt_units is None: continue`,
  [prepare_data.py:298-300](scripts/prepare_data.py#L298-L300)), e o log confirma
  **0 descartes por esse motivo**: "Unit extraction: 18/18 pairs succeeded" e
  "Unit extraction: 5/5 pairs succeeded".

---

## B4 — Tamanho do corpus (ver também `B-corpus.csv`)

- **Exemplos**: 18 em treino, 5 em validação (23 no total) — mas só **5 clipes-fonte
  únicos** por trás desses 23 (ver B3).
- **Nº de unidades/tokens por split**: **NÃO CONFIRMADO** com precisão — o log não
  imprime contagem de tokens por arquivo nesta fase (só "Unit extraction: N/N
  succeeded"). Seria possível obter lendo os arquivos `train.pt`/`train.en`/
  `valid.pt`/`valid.en` diretamente, mas esses viviam em
  `/dev/shm/alex/temp_full_dataset/` (tmpfs) e o Bloco C já registrou que essa área é
  provavelmente perdida.
- **Duração total em horas dos 5 clipes de treino especificamente**: ainda **NÃO
  CONFIRMADO** — os vídeos brutos correspondentes a esses 5 clipes não estão
  acessíveis neste checkout.
- **Distribuição de duração — atualização parcial**: `youtube-scrapper/execution.log`
  (repo irmão) loga duração real de cada download
  (`new_scraper.py:111`: `"Downloading video '{title}' - duration {duration}s..."`).
  Extraí **32 linhas logadas, que na verdade são só 16 vídeos únicos, cada um
  aparecendo exatamente 2 vezes no log** (mais um caso do mesmo padrão de duplicação
  do B3/B5 — aqui dentro do próprio arquivo de log, não do Drive). Estatística sobre
  os 16 únicos: **mín=69s, máx=1200s (20min), média=317,6s (~5,3min), mediana=210,5s**.
  **Ressalva importante**: este log tem só 32 entradas — é uma amostra de UMA
  execução local do scraper nesta máquina Windows, não o histórico completo que
  resultou nos 523 vídeos-fonte vistos no B3/Drive, e não confirma que sejam
  exatamente os mesmos 5 clipes usados no treino (os títulos logados aqui — ex.
  "Osmar Prado recita Fernando Pessoa" — não batem com os nomes de arquivo dos 5
  clipes do B3, ex. "Monologo_Justica"). Serve como evidência **ilustrativa** da
  faixa de duração típica que esse método de coleta produz, não como censo do corpus
  de treino real.
- **Período coberto pelas partições `<ano>/<mês>/<dia>`**: **NÃO CONFIRMADO** — o log
  não imprime a(s) data(s) da pasta processada nesta execução (o script recebe
  `--date`, mas essa invocação específica de `train_full_pipeline.py` não loga o
  valor recebido). Comando pra checar na VM, se o histórico de shell ainda existir:
  ```bash
  history | grep train_full_pipeline
  ```

---

## B5 — Critérios e vieses do scraper

**Atualização**: encontrado no repositório irmão
`C:\Users\lelex\source\repos\Mestrado\youtube-scrapper` (`new_scraper.py`), lido de
ponta a ponta. Confirma e explica diretamente o achado do B3.

- **Plataforma**: **só YouTube**, via `yt_dlp` (`new_scraper.py:5`). Nenhum outro
  serviço (TikTok etc.) aparece no código.
- **Query de busca**: **uma única string fixa**,
  `YOUTUBE_SEARCH_TERM = "discurso monologo recitar"`
  (`new_scraper.py:23`), buscada como `ytsearch100:{search_term}`
  (`new_scraper.py:76`). **Isso explica diretamente o achado do B3**: o corpus inteiro
  ser só monólogos/declamação não é coincidência nem viés incidental — é
  consequência direta de haver exatamente uma query de coleta, sempre a mesma.
- **Limite por execução**: `YOUTUBE_MAX_VIDEOS = 500`
  (`new_scraper.py:24`) — praticamente idêntico aos "523 vídeos-fonte" vistos no B3,
  sugerindo poucas execuções do scraper (talvez até uma só, perto do teto).
- **Filtros de coleta reais** (`video_filter()`, `new_scraper.py:30-44`): **só
  idioma** — mantém apenas vídeos com metadado de idioma do YouTube igual a `pt`/`pt-br`
  (case-insensitive), descarta se o metadado de idioma estiver ausente. **Não há
  filtro de duração no código atual, nem de presença de rosto, nem de legendas.**
  Confirmado por você mesmo nesta sessão: havia um filtro de duração de 60s
  (`if duration and duration > 60: [rejeita]`), removido no commit `b9c2f93`
  ("Hardcode configuration variables, remove .env dependency, increase max videos to
  500, and remove video duration filter") — a mesma mudança que subiu o teto de vídeos
  para 500, coerente com sua explicação de que isso aconteceu depois de ganhar acesso
  a uma máquina mais potente.
- **Mecanismo mais provável por trás da duplicação do B3, agora com evidência
  direta**: como a busca é sempre a mesma string fixa, rodar o scraper em dias
  diferentes tende a retornar um ranking de resultados do YouTube muito parecido (ou
  idêntico) a cada execução — cada vez subindo para uma pasta de data nova no Drive.
  Isso é consistente com o mesmo punhado de vídeos aparecendo repetidamente sob
  `<ano>/<mês>/<dia>` diferentes, sem precisar do bug de upload duplicado
  (linhas 196-197 do `generate_synthetic_data.py`) para explicar tudo. Ainda **NÃO
  CONFIRMADO** o número exato de execuções/datas — precisaria da estrutura de pastas
  do Drive, que não tenho acesso direto.

**Vieses prováveis, agora fundamentados no código real do scraper** — para o
capítulo de Ameaças à Validade:
- **Tipo de conteúdo restrito por design, não por acaso**: a query fixa
  "discurso monologo recitar" torna o viés de gênero textual (monólogo/declamação)
  uma consequência direta e esperada do método de coleta, não um efeito colateral a
  investigar — vale nomear isso explicitamente como limitação de escopo da busca, não
  só como "viés observado". Os 5 clipes únicos identificados no B3 (nomes como
  "Monologo_Justica", "Monologo_Natal_2022", "Disturbios_De_Um_Monologo_Interior")
  confirmam esse gênero.
- **N de falantes efetivo é mínimo**: 5 clipes, provavelmente poucos falantes
  distintos (alguns nomes sugerem o mesmo autor/canal, ex. "Andre_Araujo") — qualquer
  generalização para sotaques/timbres/perfis demográficos não representados nesses
  poucos falantes é não testada por construção.
- **Enquadramento**: consistente com os vídeos de amostra vistos nesta sessão inteira
  (falante único, câmera frontal, planos fechados de rosto/busto) — típico de
  conteúdo vertical de redes sociais, não generalizável a múltiplos falantes em
  quadro, ângulos variados, ou fala em grupo.
- **Qualidade de áudio/vídeo variável**: não verificável sem os arquivos originais,
  mas o próprio pipeline de geração de alvo sintético (B1) e as observações
  qualitativas já feitas nesta sessão (ex.: dificuldade com nomes próprios/vocabulário
  raro) sugerem sensibilidade a condições de gravação não controladas.

---

## B6 — Sincronia dos pares pseudo-pareados

- **Mesmo número de frames entre fonte e alvo?** Não há nenhuma verificação ou
  ajuste de contagem de frames entre `src_vid` e `tgt_vid` em `prepare_data.py` —
  `extract_units()` ([prepare_data.py:181-245](scripts/prepare_data.py#L181-L245)) é
  chamado independentemente para cada um ([linhas 295-296](scripts/prepare_data.py#L295-L296)),
  produzindo cada um sua própria sequência de unidades, do comprimento que resultar
  naturalmente daquele vídeo especificamente. **Dado tudo que esta sessão já
  documentou sobre o `unit2av` gerar vídeos de duração diferente do vídeo fonte**
  (todo o trabalho de ajuste de duração/pacing feito em `inference.py` ao longo desta
  conversa), é esperado que o vídeo-alvo sintético tenha uma contagem de frames
  diferente do vídeo-fonte original — e isso **não é tratado nem detectado** em
  `prepare_data.py`.
- **Verificação de qualidade dos alvos antes do treino?** **Não encontrei nenhuma.**
  O único filtro é falha binária de extração (`None` → descarta o par, B3) — não há
  verificação de sincronia labial, confiança de tradução, ou qualquer métrica de
  qualidade sobre o conteúdo do vídeo sintético antes de ele virar rótulo de treino.
- **Consequência esperada de um alvo com sincronia labial imperfeita**: como o rótulo
  de treino são as *unidades discretas extraídas do áudio* do vídeo sintético (não do
  vídeo em si), uma imperfeição puramente *visual* de sincronia labial no alvo
  provavelmente não contamina diretamente o rótulo textual/de unidades usado no
  fine-tuning do `unit2unit` — mas qualquer artefato ou erro que exista no *áudio*
  sintetizado (incluindo os já documentados nesta sessão: pronúncia rushed, ruído do
  vocoder, erros de tradução) é herdado integralmente como se fosse o rótulo correto,
  sem nenhuma checagem, e volta a reforçar exatamente esses mesmos padrões no
  fine-tuning — coerente com o resultado do Bloco C (fine-tuning não melhorou, e em
  alguns casos piorou, a qualidade medida da tradução).

---

## Achado incidental — corrige o Bloco C: há mais curva de treino disponível do que eu havia registrado

Ao reler `logs_treino_full.txt` de ponta a ponta para este bloco, encontrei linhas
`Saved checkpoint ... (epoch N @ M updates, score S)` cobrindo **muito mais epochs
do que as 2 usadas no Bloco C** (vi contínuo de epoch 7 até pelo menos epoch 26 nesta
checagem, possivelmente mais adiante no arquivo). Se você quiser a curva de
aprendizado completa (não só os 2 pontos do C2), vale eu reprocessar esse bloco do log
por completo — é um ajuste ao Bloco C, não deste bloco, então não fiz agora para não
fugir do escopo do B, mas registro aqui para não se perder.
