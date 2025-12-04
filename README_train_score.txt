Outlier pipeline ? snelle handleiding (NL)

Wat je alleen hoeft aan te passen (USER CONFIG bovenin outlier_supervised_pipeline_train_score_compat.py)
- mode: "train" (modellen trainen) of "score" (nieuwe data scoren).
- input_path: jouw bestand (.csv of .xlsx/.xls).
- output_excel: naam van het Excel-rapport.
- id_col: kolom met uniek ID, of zet op None als je die niet hebt.
- Laat de rest staan tenzij je weet wat je doet.

Data-eisen
- Train-modus: bestand moet de doelkolom (standaard: is_outlier) hebben met waarden 0/1 (1 = outlier). Alle feature-kolommen die hier staan moeten later ook in het score-bestand zitten. Als id_col is ingesteld, moet die kolom aanwezig zijn.
- Score-modus: bestand moet dezelfde feature-kolommen hebben als gebruikt in training. is_outlier is NIET nodig. Als id_col is ingesteld, moet die kolom aanwezig zijn.
- Bestandsformaten: .csv, .xlsx, .xls.
- Kolomtypes: getallen en tekst zijn prima; lege cellen zijn ok? (script vult ze aan). Datums worden als tekst behandeld; dat is prima.

Eenmalige installatie
1) Zorg voor Python 3.9+.
2) Installeer packages:
   pip install -r requirements.txt

Train (modellen fitten en opslaan)
1) Zet in USER CONFIG:
   mode = "train"
   input_path = "dummy_outliers.csv"  (of jouw gelabelde bestand)
2) Run:
   python outlier_supervised_pipeline_train_score_compat.py
Uitkomst:
- outlier_results.xlsx met: metrics (train/test), predictions, feature_importance, plus config.
- map model_artifacts/ met gekalibreerde modellen en thresholds (voor scoren).

Score (op nieuwe data met opgeslagen modellen)
1) Na een keer trainen, zet in USER CONFIG:
   mode = "score"
   input_path = "dummy_outliers_unlabeled.csv"  (of jouw weekbestand)
   output_excel = "weekly_scored.xlsx"          (naam die jij wilt)
2) Run:
   python outlier_supervised_pipeline_train_score_compat.py
Uitkomst:
- weekly_scored.xlsx met: summary (aantal geflagd per model) en scored_predictions (gerangschikt op kans).

Veelvoorkomende problemen vermijden
- Ontbrekende kolommen: train- en score-bestand moeten dezelfde feature-kolommen hebben. Als id_col staat, moet die kolom er zijn.
- Verkeerde labels: in train-modus moet is_outlier alleen 0/1 zijn.
- Verkeerd pad: bij ?file not found? het pad/bestandsnaam controleren.
- Grote wijzigingen: pas advanced settings niet aan tenzij nodig.
