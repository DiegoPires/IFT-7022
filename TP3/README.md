# Classification de texte et analyse de sentiments dans les conversations

EmoContext est une compétition actuellement ouverte qui consiste à faire l’analyse de
conversations afin de détecter les émotions qui y sont présentes. Voir le lien suivant pour plus
d’informations : https://www.humanizing-ai.com/emocontext.html

## Proposition : 
Utilisez le corpus rendu disponible pour cette compétition afin de mener une
expérimentation. Le corpus est disponible sur le site du cours. Vous pourriez réutiliser ce que
vous avez fait pour le Travail pratique 2 (TP2) et l’adapter afin de traiter ces séquences
d’échanges. Sinon vous pourriez en profiter pour explorer de nouveaux outils et comparer les
résultats avec ceux développés au TP2.

Dans votre rapport, vous devriez aborder les points suivants :

- Expliquez la démarche que vous avez suivie.
- Indiquez les difficultés que vous avez rencontrées et les limitations de votre approche.
- Présentez les résultats que vous avez obtenus avec les jeux de données de la compétition.

## À remettre : 
Votre code et votre rapport. 

### Virtualenv
Started using virtualenv as a recommendation to make Tensorflow work for TP3

- Create as `virtualenv ./env`
- Exported with `pip freeze > requirements.txt`
- Activate with `source ./venv/bin/activate`
- Desactivate with `deactivate`
- Install requirements `$ pip install -r requirements.txt`

On VSCode should go `Shift + cmd + p` and search `python: select interpreter`. Chose the `env` environment to make it run with VSCode

### Executing the code

On VSCode just click right on the main.py and chose `Run python file on terminal`.

Or in the command prompt run `python3 <path_to_IFT-7022>/TP3/main.py` according to where the code was downloaded