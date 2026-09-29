TITOLO DEL PROGETTO
Deep Metric Learning e Data Augmentation Topologica per la Classificazione di Biomarcatori Tumorali


PANORAMICA DEL PROGETTO
Il progetto sviluppa un framework avanzato di machine learning per la classificazione multiclasse di 8 istotipi tumorali e controlli sani a partire da profili proteici di biopsia liquida. L'obiettivo principale è superare tre criticità tipiche dei dati clinici molecolari: la scarsità campionaria delle classi rare, la naturale sovrapposizione biologica dei valori tra organi differenti e il rischio di generare profili sintetici non plausibili clinicamente (chimere biologiche).


DATASET E PROTOCOLLO CLINICO
- Blind Test Reale: coorte di validazione indipendente composta da 364 pazienti mai osservati in fase di addestramento o pre-processing.
- Classi Cliniche: 9 classi complessive, corrispondenti a 8 istotipi tumorali primari e 1 coorte di controlli sani.
- Dati Molecolari: pannello di 39 biomarcatori proteici circolanti isolati dal plasma.
- Rigore Metodologico (Zero Leakage): separazione totale della coorte di test da qualsiasi fase di ottimizzazione e feature extractor congelato per garantire proiezioni deterministiche.


PIPELINE METODOLOGICO

1. Data Augmentation Topologica (KNN-Filtering a k=5):
I campioni sintetici candidati per le classi minoritarie vengono proiettati nello spazio dei dati clinici reali certificati. Il profilo sintetico viene accettato solo se i suoi 5 vicini più prossimi appartengono alla medesima classe clinica; se cade in zone di confine o miste, viene scartato per eliminare chimere biologiche e purificare i margini decisionali prima dell'addestramento.

2. Architettura a Due Stadi (Two-Stage Framework):
- Fase 1 (Strutturazione dello Spazio Latente): una rete 1D-CNN viene pre-addestrata con una testa a 9 neuroni (Softmax e CrossEntropyLoss) con Early Stopping, allo scopo di raggruppare i profili della stessa patologia e separare classi distinte nello spazio latente. Al termine, la testa di classificazione viene rimossa e i pesi della 1D-CNN vengono congelati, fissando un embedding deterministico a 64 dimensioni.
- Fase 2 (Deep Metric Learning): una Rete Siamese riceve coppie di embedding estratti dalla 1D-CNN e calcola la similarità tramite Distanza Chi-Quadro, una metrica non lineare superiore alle distanze Euclidea o Coseno nell'evidenziare variazioni relative tra profili molecolari eterogenei. Uno strato denso da 256 neuroni con attivazione sigmoide stima la probabilità finale di appartenenza alla stessa classe.

3. Setup Few-Shot Episodico:
La diagnosi dei nuovi pazienti viene eseguita mediante confronto metrico sistematico tra campioni incogniti (Query Set) e profili clinici di riferimento certificati (Support Set).


EXPLAINABLE AI E VALIDAZIONE BIOLOGICA (SHAP)
L'interpretabilità del modello è stata validata tramite analisi globale dei valori SHAP:
- Isolamento della Firma Neoplastica: biomarcatori cardine (quali OPN, Prolattina e Leptina) guidano la discriminazione netta tra soggetti sani e patologici.
- Specificità d'Organo: l'attribuzione dei pesi riflette la biologia reale (es. elevato impatto di NSE per il carcinoma polmonare), dimostrando che il modello sfrutta pattern molecolari autentici documentati in letteratura clinica e non artefatti numerici.


STACK TECNOLOGICO
Python, PyTorch, Scikit-Learn, NumPy, Pandas, SHAP, Matplotlib, Seaborn.
