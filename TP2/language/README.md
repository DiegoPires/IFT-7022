# Identification de la langue d’un texte
L’objectif de cette tâche est de construire un classificateur qui identifie la langue d’un texte à
partir de sous-séquences de caractères. Les attributs que vous utiliserez sont les séquences de
lettres, de signes de ponctuation et de caractères spéciaux qu’on retrouve dans un texte. Par
exemple, pour déterminer la langue de la phrase:

``` python
This_is_an_example
```

vous pourriez utiliser comme attributs les trigrammes de caractères suivants:

```
[thi, his , is_, s_i, _is , is_, s_i, _is , is_, s_a, _an , …, mpl, ple , le.]
```

Je vous laisse libre choix sur le type d’algorithmes à utiliser pour construire un classificateur
supervisé. Comme à la tâche précédente, je vous demande au minimum de comparer les
performances de 2 algorithmes d’apprentissage (par ex. naive bayes et régression logistique).
Pour entraîner votre module d’identification de langue, vous trouverez sur le site du cours des
fichiers pour 4 langues : français, anglais, espagnol et portugais. Vous trouverez également sur le
site des fichiers pour évaluer les performances de votre module. 

Les fonctions de prétraitement à appliquer sur les textes sont :
- segmenter les textes en phrase. Il n’est pas nécessaire de mettre des symboles de début et
de fin de phrases comme on le ferait pour construire des modèles N-grammes.
- convertir les caractères en minuscule.
- extraire les sous-séquences de caractères (c.-à-d. les n-grammes) pour obtenir les attributs
de vos classificateurs.

Présentez dans votre rapport les résultats que vous obtenez avec des sous-séquences de 1, 2 et 3
caractères.