# Apport des émotions pour la classification des données sociales

Projet de recherche (mémoire) évaluant si l'ajout explicite de features émotionnelles améliore la détection automatique de contenus problématiques (discours haineux / commentaires toxiques) en français, en complément des embeddings contextuels de CamemBERT.

## Question de recherche

CamemBERT encode-t-il déjà implicitement le ton émotionnel d'un texte, ou l'ajout explicite de features émotionnelles (catégories d'émotions, flags booléens de type d'émotion) apporte-t-il un gain mesurable en classification binaire (haineux/non haineux, toxique/non toxique) ?

Pour y répondre, chaque corpus est traité selon deux approches en parallèle, comparées à score égal :

- **`sans_emotion/`** — pipeline « brut » : exploration du corpus, prétraitement du texte, extraction des embeddings CamemBERT (avec et sans fine-tuning), puis classification classique (régression logistique, SVM), sans aucune feature émotionnelle.
- **`avec_emotion/`** — même pipeline, enrichi de features émotionnelles extraites de deux systèmes d'annotation distincts (**Emotion/Cats** et **Emotyc**), concaténées aux embeddings CamemBERT avant classification.

## Corpus utilisés

| Corpus | Domaine | Description |
|---|---|---|
| **Haineux** | Tweets | Classification binaire haineux (1) / non haineux (0) |
| **Reddit** | Commentaires Reddit | Classification binaire toxique (1) / non toxique (0), fortement déséquilibré (~78 % non-toxique / ~22 % toxique) |

## Les deux systèmes de features émotionnelles

- **Emotion / Cats** : catégories d'émotions larges directement liées au discours haineux (colère, mépris, joie, peur, tristesse, surprise, admiration, autre...). ~8-10 dimensions, sparse (beaucoup de phrases n'ont aucune émotion détectée).
- **Emotyc** : système plus fin, à base de flags booléens décrivant le *type* d'expression émotionnelle (émotion montrée, désignée, suggérée, comportementale, complexe, de base) en plus des catégories (colère, peur, joie, tristesse, surprise, admiration, fierté, embarras...). ~15+ dimensions.

Chaque système est testé seul, puis combiné (Cats + Emotyc), afin de vérifier si les deux sources sont complémentaires ou redondantes.

## Structure du dépôt

```
avec_emotion/
└── avec emotion/
    ├── haineux/
    │   ├── CAMAMBERT_haineux_emotion_emotic.ipynb   # Pipeline principal : versions A/B/C/D, comparaison finale
    │   ├── ML_Haineux_emotic_emotion.ipynb
    │   ├── haineux_emotion_ML.ipynb
    │   ├── haineux_emotyc_ML.ipynb
    │   └── merging/
    │       ├── emotion/   construction_csv_emotion.ipynb, merging_emotion.ipynb
    │       └── emotyc/    construction_csv_emotyc.ipynb, merging_emotyc.ipynb
    └── reddit/
        ├── Camabert_sansFT_redit__emotion_emotic.ipynb  # Pipeline principal reddit
        ├── ML_Toxique_emotyc_emotion.ipynb
        ├── emotion/   merge/, reddit_emotion_ML.ipynb, sans emo avec down sample/
        └── emotyc/    merge/, reddit_emotyc_ML.ipynb, sans emo avec down sample/

sans_emotion/
└── sans emotion/
    ├── haineux_brut_sans_emotion/
    │   ├── exploration_tweets_haineux.ipynb   # EDA : classes, longueur, lexique, emojis, hashtags
    │   ├── preprocessing_haineux+ML.ipynb
    │   └── CAMAMBERT_haineux+recap.ipynb
    └── redit_brut_sans_emotion/
        ├── exploration_reddit.ipynb
        ├── preprocessing_reddit+ML.ipynb
        ├── Camambert_sans_fine_redit.ipynb        # CamemBERT sans fine-tuning + LR/SVM (baseline)
        └── Camambert_redit_fine_tuning.ipynb      # CamemBERT avec fine-tuning
```

## Méthodologie

1. **Exploration (EDA)** : chargement, distribution des classes, longueur des textes, qualité des données (nulls, doublons), analyse lexicale, emojis/URLs/mentions, nuages de mots.
2. **Prétraitement** : nettoyage du texte, parsing des fichiers d'annotation émotionnelle (`emotions_features.txt`, `emotyc.txt`) et construction des CSV correspondants (`id`, `sentence`, `has_emotion`, `tokens`, `lemmas`, `categories` pour Emotion ; flags booléens `avec_emo*` / `avec_cat_*` pour Emotyc).
3. **Fusion (`merging/`)** : alignement par index entre le corpus texte original et le corpus d'annotations émotionnelles (algorithme à deux pointeurs sur corpus ordonnés).
4. **Extraction des représentations** :
   - Embeddings **CamemBERT** (768 dimensions), avec et sans fine-tuning.
   - Vecteur émotion (one-hot / booléen) concaténé aux embeddings pour les versions « avec émotion ».
5. **Classification** : régression logistique et SVM, avec gestion du déséquilibre de classes via `class_weight` et/ou **SMOTE**.
6. **Comparaison systématique** de versions à features croissantes :
   - **A** — CamemBERT seul (baseline)
   - **B** — CamemBERT + Emotion/Cats
   - **C** — CamemBERT + Emotyc
   - **D** — CamemBERT + Cats + Emotyc combinés

## Résultats principaux

### Corpus Haineux (F1 macro, CamemBERT sans fine-tuning)

| Version | Features | F1 macro |
|---|---|---|
| A | Baseline (CamemBERT seul) | 0.9307 |
| B | + Cats | **0.9342** |
| C | + Emotyc | 0.9325 |
| D | + Cats + Emotyc | 0.9325 (= C) |

- L'ajout d'émotion améliore systématiquement la baseline, mais le gain reste **marginal (+0,18 % à +0,35 % de F1 macro)**.
- **B (Cats) > C (Emotyc)** malgré moins de dimensions : la pertinence des catégories vis-à-vis du domaine (haine) prime sur la richesse descriptive.
- **D = C** : combiner les deux sources n'apporte rien de plus — Emotyc absorbe déjà l'information de Cats (redondance).

### Corpus Reddit (SVM + class_weight, meilleur modèle par version)

| Version | Features | F1 (classe toxique) | Accuracy |
|---|---|---|---|
| A | Baseline (CamemBERT seul) | 0.7091 | 0.8704 |
| B | + Cats | 0.7156 | 0.8745 |
| C | + Emotyc | **0.7736** | **0.9028** |
| D | + Cats + Emotyc | 0.7593 | 0.8947 |

- Sur Reddit, le corpus est fortement déséquilibré et **seuls 304 textes sur 1271 contiennent une émotion détectée** (dont 301 non-toxiques contre 3 toxiques seulement) — l'émotion est donc quasi absente côté toxique, ce qui limite mécaniquement son apport potentiel malgré le gain observé en version C.

## Conclusions générales

1. **L'émotion aide, mais modestement.** Quelle que soit la représentation (Cats, Emotyc, ou les deux), CamemBERT capte déjà l'essentiel du signal émotionnel directement depuis le texte brut, ce qui plafonne naturellement l'apport de features émotionnelles explicites.
2. **La qualité prime sur la quantité.** Des catégories peu nombreuses mais alignées avec le domaine (Cats pour le discours haineux) peuvent surpasser un système plus riche mais plus générique (Emotyc).
3. **La combinaison de sources ne garantit pas un gain.** Deux systèmes d'annotation émotionnelle peuvent être largement redondants du point de vue du classifieur — ajouter plus de features n'améliore pas nécessairement la performance.
4. **La sparsité des annotations émotionnelles limite leur apport**, en particulier sur des corpus déséquilibrés comme Reddit où l'émotion est rarement détectée du côté de la classe minoritaire (toxique).

## Environnement technique

- **Langue** : Python (notebooks Jupyter, `.ipynb`)
- **Modèle de langue** : CamemBERT (`camembert-base`), utilisé comme extracteur d'embeddings (avec et sans fine-tuning)
- **Classification** : `scikit-learn` (LogisticRegression, SVC, RandomForestClassifier), `imbalanced-learn` (SMOTE)
- **Gestion du déséquilibre** : `class_weight='balanced'` et/ou sur-échantillonnage SMOTE

## Comment naviguer dans ce dépôt

- Pour comprendre la **question de recherche et les résultats finaux** : commencer par `avec_emotion/avec emotion/haineux/CAMAMBERT_haineux_emotion_emotic.ipynb` (corpus haineux) et `avec_emotion/avec emotion/reddit/Camabert_sansFT_redit__emotion_emotic.ipynb` (corpus Reddit) — ce sont les notebooks de synthèse contenant les comparaisons A/B/C/D et les conclusions.
- Pour comprendre la **construction des données** : voir les notebooks `merging/` et `construction_csv_*.ipynb` de chaque corpus.
- Pour la **baseline sans émotion** : dossier `sans_emotion/`.
- Pour l'**exploration des données brutes** : `exploration_tweets_haineux.ipynb` et `exploration_reddit.ipynb`.

## Limites et pistes futures

- Les annotations émotionnelles sont sparses (beaucoup de phrases sans émotion détectée), ce qui dilue leur apport statistique.
- Le fine-tuning complet de CamemBERT (plutôt que son usage en extracteur figé) pourrait faire évoluer l'écart entre versions avec/sans émotion.
- Une analyse par catégorie d'émotion (plutôt qu'un vecteur global) permettrait d'identifier quelles émotions spécifiques contribuent le plus au signal de haine/toxicité.
- Le déséquilibre de classes, en particulier sur Reddit, mériterait des méthodes complémentaires (undersampling combiné, focal loss) au-delà de SMOTE et class_weight.
