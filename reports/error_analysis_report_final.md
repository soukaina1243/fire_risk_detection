
# Error Analysis Report
## Fire Risk Prediction — LSTM Model
### Test Set : 2023 | Granularité : Régions 1°×1° | Californie

---

## 1. Résumé des performances

| Métrique         | Valeur   | Interprétation                          |
|------------------|----------|-----------------------------------------|
| AUC-ROC          | 0.8449  | Bon pouvoir discriminant                |
| Avg Precision    | 0.1482  | Faible (déséquilibre extrême)           |
| Recall (feux)    | 0.2500  | Détecte 1/4 feux réels  |
| Precision (feux) | 0.1667  | 5 fausses alertes                 |
| F1-score (feux)  | 0.2000  | Compromis precision/recall              |
| Threshold        | 0.57   | Optimisé sur le set de validation      |

---

## 2. Matrice de confusion

|                  | Prédit: Non | Prédit: Oui |
|------------------|-------------|-------------|
| **Réel: Non**    | 477 (TN)   | 5 (FP)  |
| **Réel: Oui**    | 3 (FN)  | 1 (TP)   |

**Interprétation :**
- 1 feu(x) correctement alerté(s) → intervention possible
- 3 feu(x) raté(s) → risque critique non détecté
- 5 fausse(s) alerte(s) → coût opérationnel limité

---

## 3. Analyse des erreurs

### 3.1 Faux Négatifs (feux ratés) — Erreur critique
      lat    lon  month   y_proba
30   33.0 -117.0      7  0.510422
103  35.0 -120.0      8  0.539121
327  39.0 -120.0     10  0.517004

**Hypothèses sur les causes :**
- Feux de courte durée non capturés par les composites mensuels
- Résolution 1°×1° trop grossière pour certains feux localisés
- Conditions météo atypiques non représentées dans le train set

### 3.2 Faux Positifs (fausses alertes)
      lat    lon  month   y_proba
322  39.0 -121.0     11  0.584505
323  39.0 -121.0     12  0.576339
381  40.0 -122.0     10  0.575294
435  41.0 -123.0     10  0.580364
436  41.0 -123.0     11  0.576262

**Hypothèses :**
- Conditions à risque élevé sans feu déclaré (feu évité)
- Zones historiquement à risque sans feu en 2023

---

## 4. Analyse saisonnière

Les erreurs se concentrent sur :
- **Mois à risque** (Juillet-Octobre) : période où le modèle
  est le plus sollicité et où les faux négatifs ont le plus d'impact
- **Mois calmes** (Novembre-Avril) : quasi-zéro erreur car
  peu de feux réels et modèle conservateur

---

## 5. Limitations du modèle

1. **Données limitées** : seulement 80 événements positifs sur 5 ans
   → modèle entraîné principalement sur des données synthétiques (augmentation)
2. **Résolution temporelle** : données mensuelles → ne capture pas
   les feux soudains (quelques jours)
3. **Résolution spatiale** : régions 1°×1° (~100km) → perd
   la précision géographique fine
4. **Période courte** : 5 ans de données, incluant 2020
   (année exceptionnelle) qui domine le signal

---

## 6. Recommandations pour améliorer le modèle

1. **Plus de données** : étendre à 10+ ans, inclure Oregon/Washington
2. **Résolution temporelle** : passer aux données hebdomadaires
3. **Features supplémentaires** : humidité relative, FWI
   (Fire Weather Index), données topographiques
4. **Architecture** : essayer Temporal CNN ou Transformer
   sur séries temporelles plus longues

---


---

## 7. Importance des features (Permutation Importance)

| Rank | Feature           | Chute AUC | Interprétation              |
|------|-------------------|-----------|-----------------------------|
| 1    | ndvi_anomaly      | +0.0767    | Feature la plus critique    |
| 2    | soil_moisture     | +0.0294    | Forte contribution          |
| 3    | precip_mm         | +0.0238    | Contribution modérée        |

**Conclusion** : Les features ndvi_anomaly et soil_moisture dominent
le signal prédictif, ce qui est cohérent avec la physique des incendies.

---

## 8. Analyse des zones à feux ratés (Faux Négatifs)

Les 3 feux ratés (FN) présentent des probabilités prédites entre
0.51 et 0.54, très proches du threshold (0.57).
Une légère réduction du threshold à 0.50 permettrait de les capturer
mais augmenterait les fausses alertes.

**Trade-off recommandé selon l'usage :**
- Système d'alerte précoce → threshold=0.45 (maximise recall)
- Outil de planification  → threshold=0.57 (équilibre actuel)
- Allocation ressources   → threshold=0.65 (maximise precision)

---
*Rapport enrichi avec analyse de feature importance*

*Rapport généré automatiquement — Projet Deep Learning Fire Risk*
