from django.shortcuts import render, redirect
from .forms import GameForm

def index(request):
    if request.method == 'POST':
        form = GameForm(request.POST)
        if form.is_valid():
            game = form.save()
            return redirect('prediction')  

    else:
        form = GameForm()  

    return render(request, 'index.html', {'form': form})
    
def prediction(request):
    return render(request, 'prediction.html')