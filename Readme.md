## Réponses aux questions

Q1) Détection des changements de plan
-Un changement de plan implique que l'image change subitement, d'une frame à l'autre.
Il y a alors peu de chance que l'image de départ et d'arrivée aient les mêmes composantes chromatiques en même proportion.
C'est pourquoi on peut détecter les changements de plan en comparant les valeurs de l'histogramme entre 2 images:
on définit une distance entre histogrammes, et si cette distance est supérieure à un seuil, on considère que l'histogramme a "beacoup changé" entre les deux images et donc qu'il y a eu changement de plan

-

Pour une vidéo monochrome, on pourrait faire de même avec un histogramme des niveaux de gris de l'image.

Q2)
