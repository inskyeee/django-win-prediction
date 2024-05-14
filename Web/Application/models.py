from django.db import models
from django.utils import timezone
import pandas as pd
import json

with open('../Model/heroes.json', 'r') as file:
    heroes = pd.json_normalize(json.load(file)['heroes'])
    hero_names = heroes.localized_name.to_list()


HEROES = [
    (i, i) for i in hero_names
]


class Game(models.Model):
    team1_hero1 = models.CharField(max_length=20, choices=HEROES)
    team1_hero2 = models.CharField(max_length=20, choices=HEROES)
    team1_hero3 = models.CharField(max_length=20, choices=HEROES)
    team1_hero4 = models.CharField(max_length=20, choices=HEROES)
    team1_hero5 = models.CharField(max_length=20, choices=HEROES)

    team2_hero1 = models.CharField(max_length=20, choices=HEROES)
    team2_hero2 = models.CharField(max_length=20, choices=HEROES)
    team2_hero3 = models.CharField(max_length=20, choices=HEROES)
    team2_hero4 = models.CharField(max_length=20, choices=HEROES)
    team2_hero5 = models.CharField(max_length=20, choices=HEROES)

    date = models.DateField(default=timezone.now)
    prediction = models.CharField(max_length=20, default='NA')

    def __str__(self):
        return self.team1_hero1 + self.team1_hero2 + self.team1_hero3 + self.team1_hero4 + self.team1_hero5 + \
                " vs " + self.team2_hero1 + self.team2_hero2 + self.team2_hero3 + self.team2_hero4 + self.team2_hero5
    

