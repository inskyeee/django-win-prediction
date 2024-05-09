from django import forms 
from .models import Game

class GameForm(forms.ModelForm):
    class Meta:
        model = Game
        fields = ['team1_hero1', 'team1_hero2', 'team1_hero3', 'team1_hero4', 'team1_hero5',
                  'team2_hero1', 'team2_hero2', 'team2_hero3', 'team2_hero4', 'team2_hero5',
                  'date']