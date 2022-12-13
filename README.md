# Kaggle RSNA Screening Mammografie
Dit project bevat de code voor de Kaggle RSNA Screening Mammografie-wedstrijd. Het doel van deze wedstrijd is het ontwikkelen van een machine learning-model dat borstkanker nauwkeurig kan detecteren in mammografie-afbeeldingen.


# Installatie

Om dit project op te kunnen draaien, moet je [poetry](https://python-poetry.org/) geïnstalleerd hebben.

# Gebruik
Om het model te trainen, voer je het volgende commando uit:

```
poetry run python src/models/train_model.py
```
Om een voorspelling te maken op de test set met het getrainde model, voer je het volgende commando uit:

```
poetry run python src/models/evaluate_model.py
```

# Data
De dataset voor dit project is beschikbaar op de website van Kaggle en bestaat uit mammografie-afbeeldingen in DICOM-formaat en .csv-bestanden met metadata voor de patiënten en afbeeldingen. De dataset moet worden gedownload en uitgepakt in de data/raw-map voordat je de code in dit project kan uitvoeren.


# Configuratie
Het config.yaml-bestand bevat de paden naar de dataset en de hyperparameters die worden gebruikt om het machine learning-model te trainen. Het bestand kan worden aangepast om de instellingen te wijzigen.
