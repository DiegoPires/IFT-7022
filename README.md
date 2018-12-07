# IFT-7022
TECHNIQUES ET APPLICATIONS DU TRAITEMENT DE LA LANGUE NATURELLE

## Description

L'orientation de ce cours est de familiariser le participant à des techniques du traitement automatique de la langue naturelle (TALN), un sous-domaine de l'intelligence artificielle. On y présente des techniques statistiques et symboliques, tout en mettant l'accent sur leur application pour l'analyse de textes. Des concepts et algorithmes de niveau lexical, syntaxique et sémantique sont étudiés. Des travaux permettent d'acquérir une connaissance pratique des techniques étudiées.

## Informations

Cours passé à l'autonme 2018

### Virtualenv
Started using virtualenv as a recommendation to make Tensorflow work for TP3

- Create as `virtualenv ./venv`
- Exported with `pip freeze > requirements.txt`
- Activate with `source ./venv/bin/activate`
- Desactivate with `deactivate`
- Install requirements `$ pip install -r requirements.txt --no-index --find-links file:///tmp/packages`

On VSCode should go `Shift + cmd + p` and search `python: select interpreter`. Chose the venv environment to make it run with VSCode