# Analyse de sentiment
Cette tâche vise à classifier des critiques de produits selon leur polarité (positive ou négative). Le
corpus disponible sur le site du cours contient des critiques positives ou négatives (1000 de
chaque type) que vous utiliserez pour entraîner des classificateurs binaires et évaluer leur
performance.

Votre travail consiste à comparer un minimum de deux algorithmes d’apprentissage pour
accomplir cette tâche : naïf bayésien (naive bayes) et régression logistique. Vous devrez évaluer
la performance de ces algorithmes en terme de précision/rappel et d’exactitude (accuracy) et
présenter une analyse de vos résultats.

De plus, vous devez faire le prétraitement de vos textes à l’aide d’un logiciel de traitement
automatique de la langue (TALN/NLP). Les prétraitements à effectuer portent sur des techniques
que nous avons étudiées dans le cours:

## La normalisation de mots :
- En appliquant un stemming sur les mots. Quelques implantations sont
disponibles dans NLTK. Sinon plusieurs implémentations sont disponibles sur le
Web. Vous avez libre choix.
- En effectuant une lemmatisation des mots. Par ex. voir la classe
WordNetLemmatizer de NLTK.

## La sélection d’attributs (features) :
- Éliminer les mots dont la fréquence dans le corpus est faible (par ex. retirer tous
les mots dont la fréquence = 1).
- Retirer les mots outils (stop words). Voir NLTK ou Scikit-learn.
- Ne garder que les mots appartenant à des classes ouvertes (c.-à-d. les noms,
adjectifs, verbes et adverbes). Vous devez faire une analyse grammaticale (POS tagging) des textes pour identifier ces mots. 
## Valeurs d’attributs - 3 options sont possibles :
- Les comptes de mots dans chacun des textes.
- La présence des mots dans chacun des textes (valeur 0-1).
- La valeur tf-idf de chaque mot (disponible dans Scikit-learn).
## Autres attributs :
- Le nombre de mots dont la polarité est positive ou négative. Vous pouvez
utiliser SentiWordnet (NLTK) ou un autre lexique pour estimer cet attribut.

## Les configurations minimales à évaluer dans vos expérimentations sont les suivantes : 
- Classificateurs: Naive bayes, régression logistique
- Normalisation: Stemming, aucune normalisation 
- Sélection d’attributs: Tous les attributs, avec classes ouvertes seulement, sans les mots outils 
- Valeurs d’attributs: Compte de mots
- Autres attributs: Nombre de mots positifs/négatifs

Analysez clairement dans votre rapport le comportement de chacune des configurations et
identifiez l’option qui vous semble la plus intéressante (ou les options le cas échéant).
