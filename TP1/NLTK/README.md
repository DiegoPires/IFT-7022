# TÂCHE 2 – COMME LE DISAIT LE PROVERBE…
Pour cette tâche, vous allez construire des modèles de langue qui vous aideront à compléter des proverbes incomplets en ajoutant les bons mots aux bons endroits. Par exemple, étant donné la phrase incomplète:   

```
aide-toi, le *** t’aidera
```

et la liste de mots candidats  [médecin, ciel, professeur, whisky], votre logiciel doit  retourner la séquence la plus probable qui, souhaitons-le, sera " aide-toi, le ciel t’aidera ".

a) Modèles de langue - Construisez un logiciel Python qui entraîne des modèles N-grammes de mots (N = 1 à 3) à partir du fichier proverbes.txt. Quelques précisions :
- Un proverbe = 1 ligne du fichier. 
- Il  n’est  pas  strictement  nécessaire  de  segmenter  les  phrases.  De  plus,  j’ai  retiré  la ponctuation à la fin des proverbes. 
- Il n’est pas nécessaire d’ajouter des symboles de début et de fin de phrase (au choix). 
- Vous  pouvez  utiliser  une  librairie  externe  (par  ex.  NLTK)  pour  tokeniser  les  mots  des proverbes. 
- Aucune normalisation n’est appliquée aux mots. Tous les caractères sont en minuscule. 

b) Lissage 1 - Ajoutez à  votre  programme une  fonction pour appliquer  un  lissage  de  Laplace (add-delta smoothing) à vos modèles.

c) Lissage 2 - Ajoutez une fonction de lissage par interpolation linéaire avec poids constant ou stupid backoff (au choix). Uniquement pour les équipes de 2-3 personnes !

d) Estimation - Ajoutez des fonctions pour estimer le logarithme de la probabilité (logprob) et la perplexité d’une séquence de mots à l’aide d’un modèle.

e) Compléter les proverbes :
- En utilisant un lissage de type add-delta, évaluez la performance de vos modèles Ngrammesà l’aide du fichier de test :
- test1.txt si Équipe = 1 personne 
- test2.txt si Équipe = 2 ou 3 personnes 
- Répétez la  même  expérience  avec  un  lissage  par  interpolation  linéaire ou stupid backoff (uniquement pour les équipes de 2-3 personnes).

Analyses à inclure dans votre rapport :

- Présentez les résultats obtenus. Les modèles modélisent-ils bien le langage utilisé dans les proverbes ? Quel est l’impact de la longueur de l’historique?
- Lissage de Laplace : Faites varier la valeur de delta pour en mesurer l’impact sur les résultats. Discutez des résultats obtenus en indiquant clairement ce que vous observez et les conclusions que vous en tirez. 
- Pour les équipes de 2-3 personnes : répétez les étapes i et ii en appliquant un lissage par interpolation (ou stupid backoff). Je vous laisse libre de choisirla  manière  de  faire  varier  les  poids  de  la  fonction  de lissage (par  ex. grid search ou  aléatoire).  Indiquez  la  combinaison  de  valeurs  qui  donne  les meilleurs résultats. 