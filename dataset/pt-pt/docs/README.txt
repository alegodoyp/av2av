Multilingual TEDx: a Multilingual Corpus for Speech Recognition and Translation

Release v1.0, 02/01/2021

JHU - Johns Hopkins University, Baltimore, USA
UMD - University of Maryland, College Park, USA
FBK - Fondazione Bruno Kessler, Trento, Italy


1. Overview

Multilingual TEDx is a multilingual speech recognition and translation corpus to facilitate the training of ASR and SLT models in additional languages. The corpus comprises audio recordings and transcripts from TEDx Talks in 8 languages (Spanish, French, Portuguese, Italian, Russian, Greek, Arabic, German) with translations into up to 5 languages (English, Spanish, French, Portguese, Italian). 
The audio recordings are automatically aligned at the sentence level with their manual transcriptions and translations.


2. Contents

- mTEDx_v1.0_XX-YY.tar.gz
  21 compressed archives, one for each language pair (XX-YY) and for additional ASR data (XX-XX)


2.1 Archive Contents

- data/
- docs/

2.1.1 data/

It contains three directories, one for each set on which the corpus has been partitioned:
- train
    the set utilized to train models
- valid
    the set utilized for validation during training
- test
    the set utilized to evaluate models

Each data split directory is organized in three sub-directories:
- txt/
    it contains three or four text files: 
    (1) the source language sentences,
    (2) the corresponding aligned target language sentences (if translations)
    (3) the source audio alignment info in yaml format, one line for each sentence
    (4) the source audio alignment in kaldi segments format, one line for each sentence
    note that the text files contain the transcripts/translations only, and do not have the segment ids on the same line.
    the text files are parallel line-by-line with the segments and yaml files, which contain the segment ids.
- vtt/
    it contains the original vtt files for each talk.
- wav/
    it contains the audio files in flac format, one for each TEDx talk.
    we distribute flac files for compression and easier download.
    if you need to convert between flac and wav, we recommend sox; only the file metadata will change in conversion


2.1.2 docs/

- README.txt
  (this file)
- statistics.txt
  contains statistics of the language or language pair, listed by split


3. Acknowledgments and Copyright information

- TED and TEDx talks are copyrighted by TED Conference LLC and licensed under a
  Creative Commons Attribution-NonCommercial-NoDerivs 4.0
(cfr. https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy) 

- The Multilingual TEDx corpus is released under the same Creative Commons
  Attribution-NonCommercial-NoDerivs 4.0 License.
 

If you use the Multilingual TEDx corpus in your work, please cite the following paper:

Elizabeth Salesky, Matthew Wiesner, Jacob Bremerman, Roldano Cattoni, Matteo Negri, Marco Turchi, Douglas Oard, Matt Post. 2021. Multilingual TEDx Corpus for Speech Recognition and Translation.

bibtex:
@misc{salesky2021mtedx,
      title={Multilingual TEDx Corpus for Speech Recognition and Translation},
      author={Elizabeth Salesky and Matthew Wiesner and Jacob Bremerman and Roldano Cattoni and Matteo Negri and \
Marco Turchi and Douglas W. Oard and Matt Post},
      year={2021},
}

4. Contact Information

For further information about the Multilingual TEDx corpus, please contact:
  Elizabeth Salesky <esalesky@jhu.edu>
  Matthew Wiesner <wiesner@jhu.edu>

