# Etape 1 preprocessing dans preprocessing_credit_risk.ipynb

### 📋 Résumé du preprocessing effectué:

**Étapes réalisées:**
1. ✅ Chargement et exploration du dataset
2. ✅ Analyse des valeurs manquantes
3. ✅ Traitement des valeurs manquantes (médiane pour variables numériques)
4. ✅ Détection des valeurs aberrantes (méthodes IQR et Z-score)
5. ✅ Filtrage des valeurs aberrantes extrêmes (seuils métier)
6. ✅ Vérification des doublons
7. ✅ Analyse des variables catégorielles
8. ✅ Encodage des variables catégorielles (Label Encoding)
9. ✅ Sauvegarde du dataset final

**Fichier généré:**
- `credit_risk_dataset_final.csv` - Dataset complet (nettoyé et encodé, prêt pour le machine learning)

**Transformations appliquées sur df_cleaned:**
- Valeurs manquantes remplacées par la médiane
- Valeurs aberrantes extrêmes filtrées selon seuils métier
- Variables catégorielles encodées avec LabelEncoder

**Seuils appliqués pour le filtrage des outliers:**
- Age <= 90 ans
- Revenu annuel <= 2 000 000
- Ancienneté emploi <= 60 ans
- Montant prêt <= 5 000 000
- Taux d'intérêt <= 35%
- Ratio prêt/revenu <= 100%
- Historique crédit <= 80 ans

**Recommandations:**
- Le dataset `credit_risk_dataset_post_preprocessing` contient toutes les transformations et est prêt pour la modélisation
- Toutes les variables sont maintenant numériques
- Le fichier sauvegardé peut être directement utilisé pour le machine learning

--- 

 # Etape 2 : DISCRETISATION dans discretisation_finale_mixte.ipynb


## 🎓 Méthodes de discrétisation utilisées

### 1️⃣ **ChiMerge** - Fusion guidée par le Chi²
**Principe :**
- Commence avec de nombreux petits bins
- Fusionne itérativement les bins adjacents les plus similaires (Chi² minimum)
- S'arrête quand le Chi² entre bins devient significatif

**Avantages :**
- ✅ Maximise la corrélation avec la variable cible
- ✅ Optimise automatiquement le nombre de bins
- ✅ Capture les non-linéarités

**Variables utilisant ChiMerge :**
- **loan_int_rate** : Taux d'intérêt du prêt - Corrélation forte avec le défaut
- **loan_percent_income** : Ratio prêt/revenu - Indicateur de stress financier

---

### 2️⃣ **Monotone** - Garantie de cohérence logique
**Principe :**
- Crée des bins dont les taux de défaut sont strictement croissants ou décroissants
- Fusionne les bins qui violent la monotonie
- Assure une relation logique avec le risque

**Avantages :**
- ✅ Interprétabilité maximale
- ✅ Conformité réglementaire (Bâle II/III)
- ✅ Stabilité du modèle dans le temps

**Variables utilisant Monotone :**
- **person_income** : Revenu de l'emprunteur - Plus le revenu est élevé, moins le risque est élevé (relation décroissante attendue)
  - *Fusion manuelle* : Bins 3-4, 5-6, 7-8 fusionnés pour assurer la stabilité
- **loan_amnt** : Montant du prêt - Montant élevé = risque plus élevé

---

### 3️⃣ **KMeans** - Clustering par similitude
**Principe :**
- Regroupe les observations en k clusters selon leur proximité
- Minimise la variance intra-cluster
- Crée des groupes naturels dans les données

**Avantages :**
- ✅ Détection automatique de groupes homogènes
- ✅ Pas d'hypothèse sur la relation avec la cible
- ✅ Simplicité et efficacité

**Variables utilisant KMeans :**
- **person_age** : Âge de l'emprunteur - 5 clusters pour capturer les différentes phases de vie
  - *Fusion manuelle* : Bins 1-2 fusionnés (jeunes emprunteurs)
- **person_emp_length** : Ancienneté professionnelle - 3 clusters (débutants, confirmés, seniors)
- **cb_person_cred_hist_length** : Historique de crédit - 3 clusters (court, moyen, long historique)

---

## 🎯 Justification du choix par variable

| Variable | Méthode | Raison du choix |
|----------|---------|-----------------|
| **loan_int_rate** | ChiMerge | Forte corrélation non-linéaire avec le défaut. ChiMerge optimise le Chi² |
| **person_income** | Monotone + Fusion | Relation décroissante attendue (revenu ↑ → risque ↓). Fusions pour stabilité |
| **loan_amnt** | Monotone | Relation croissante logique (montant ↑ → risque ↑) |
| **loan_percent_income** | ChiMerge | Ratio complexe nécessitant optimisation du Chi² |
| **person_age** | KMeans (k=5) + Fusion | Segmentation par phases de vie. Fusion 1-2 pour jeunes |
| **person_emp_length** | KMeans (k=3) | 3 segments naturels : junior/confirmé/senior |
| **cb_person_cred_hist_length** | KMeans (k=3) | 3 profils : historique court/moyen/long |

---

## 📊 Résultats attendus

À l'issue de ce notebook, nous obtenons :

1. **Dataset discrétisé** : `credit_risk_dataset_discretized_final.csv`
2. **Boundaries optimales** : `discretization_boundaries_final.json`
3. **Rapport détaillé** : `RAPPORT_DISCRETISATION_FINALE.txt`
4. **Statistiques par bin** : Taux de défaut, Chi², significativité
5. **Visualisations** : Distribution des taux de défaut par bin

---

## ⚙️ Paramètres et optimisations

### Fusions manuelles appliquées :
- **person_income** : Bins 3-4, 5-6, 7-8 → Pour réduire le nombre de catégories et améliorer la stabilité
- **person_age** : Bins 1-2 → Pour regrouper les jeunes emprunteurs (segment homogène)

### Métriques de qualité :
- **Chi²** : Mesure de la dépendance entre variable discrétisée et défaut
- **P-value** : Significativité statistique (objectif : p < 0.05)
- **Monotonie** : Vérification de la cohérence logique
- **Taux de défaut par bin** : Distribution du risque

---

## 🚀 Utilisation

1. Exécuter toutes les cellules dans l'ordre
2. Consulter les résultats dans la section "Tableau récapitulatif"
3. Analyser les visualisations pour chaque variable
4. Vérifier le rapport détaillé généré
   
---

# Étape 3 — Analyse de Corrélation Post-Discrétisation 🔍📊 dans analyse_de_correlation.ipynb

## 1. 🎯 Objectif général

L’objectif de cette étape est d’analyser les relations entre les variables après discrétisation, afin de détecter la **multicolinéarité**, repérer les **variables redondantes**, et préparer correctement les données pour les étapes suivantes du pipeline de scoring :

- Weight of Evidence (WoE)
- Information Value (IV)
- Modélisation / Scorecard

Après la discrétisation, toutes les variables sont **catégorielles**, ce qui nécessite des méthodes adaptées.

---

## 2. 🧩 Contexte : Source du dataset

Cette étape utilise **exclusivement** le dataset issu de l’étape précédente :

📁 **credit_risk_dataset_discretized_final.csv**

Il contient :

- Les **7 variables discrétisées** (ChiMerge, Monotone, KMeans)
- La variable cible : `loan_status`
- Des bins propres, stables et cohérents

---

## 3. 🛠️ Rôle dans le pipeline de scoring

Cette étape permet de :

- ✔️ Évaluer la dépendance statistique entre variables  
- ✔️ Détecter des corrélations fortes entre variables discrétisées  
- ✔️ Identifier des variables dupliquées ou redondantes  
- ✔️ Préparer la suite du pipeline

Elle contribue à garantir un modèle :

- plus **stable**  
- plus **interprétable**  
- débarrassé de **multicolinéarité**  
- conforme aux bonnes pratiques du **crédit scoring**  

---

## 4. 📐 Méthodes de corrélation utilisées

### 4.1 💠 Test du Chi-2  
Permet de mesurer la dépendance statistique entre deux variables catégorielles.

- **p-value < 0.05** → dépendance significative  
- Un **Chi-2 élevé** → relation forte  

### 4.2 🔷 Cramer’s V  
Mesure l’intensité de corrélation (0 à 1) entre variables catégorielles.

Interprétation :

| Valeur | Signification |
|--------|---------------|
| 0.0 – 0.1 | Très faible |
| 0.1 – 0.3 | Faible |
| 0.3 – 0.5 | Moyenne |
| > 0.5 | Forte (à surveiller) |

### 4.3 🔥 Heatmap Cramer’s V  
Visualisation permettant de repérer rapidement les **clusters** de variables corrélées.

---

## 5. 📂 Résultats générés

Cette étape produit :

1. 📊 Un tableau des résultats Chi-2  
2. 📊 Un tableau complet des Cramer’s V  
3. 🔥 Une heatmap de corrélation  
4. 📝 Une liste de variables potentiellement redondantes  
5. 📄 Une section **Décisions** expliquant les choix de conservation/suppression

**Exemple :**

- `loan_percent_income` et `loan_int_rate` présentent une corrélation forte (Cramer’s V = 0.58)  
- Je conserve `loan_percent_income` car :
  - Binning plus stable  
  - Variable plus facilement interprétable pour les équipes Risques  
  - Meilleure séparation des taux de défaut  

---

## 6. 📘 Organisation du notebook

Cette étape est intégrée dans :

👉 **discretisation_finale_mixte.ipynb**

Structure :

1. Chargement du dataset discrétisé  
2. Sélection variables / cible  
3. Test du Chi-2  
4. Calcul du Cramer’s V  
5. Heatmap de corrélation  
6. Analyse & décisions  
7. Sauvegarde des résultats  


