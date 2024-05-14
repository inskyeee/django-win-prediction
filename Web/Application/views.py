from django.shortcuts import render, redirect
from .forms import GameForm
from .models import Game
from .prediction import process_game

def index(request):
    last_prediction = None
    if Game.objects.exists():
        last_prediction = Game.objects.latest('date')
    
    if request.method == 'POST':
        form = GameForm(request.POST)
        if form.is_valid():
            game = form.save(commit=False)
            win_chance = process_game(form.cleaned_data)
            game.prediction = 'Win' if win_chance[1] >= 0.5 else 'Lose'
            game.save() 
            request.session['chance_of_win'] = win_chance[1]
            return redirect('prediction')  

    else:
        form = GameForm()  

    return render(request, 'index.html', {'form': form, 'last_prediction': last_prediction})
    
def prediction(request):
    chance_of_win = request.session.get('chance_of_win')
    win = chance_of_win >= 0.5
    return render(request, 'prediction.html', {'chance_of_win': '{chance:.0f}%'.format(chance=chance_of_win*100), 'win': win})