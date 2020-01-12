from django.shortcuts import render
from django.http import HttpResponse
import Approri_ARM_File
import Similar_products_clustering
# Create your views here.

def home(request):
    return render(request, 'home.html')

def About(request):
    return render(request, 'About.html')

def Partners(request):
    return render(request, 'Partners.html')

def begin(request):
    return render(request, 'begin.html')

def arm(request):
    return render(request, 'arm.html')

def clust(request):
    return render(request, 'clust.html')

def dropdown(request):

    vals = int(request.POST['val'])
    pr_names, supp, total  = Approri_ARM_File.what_others_bought(vals)

    zippedList = zip(pr_names, supp, total)

    return render(request, 'result_Arm.html', {'zippedList': zippedList})

def clust_result(request):

    val1 = int(request.POST['val'])
    val2 = int(request.POST['num'])
    
    clust, price, rating = Similar_products_clustering.similar_prod(val1, val2)

    zippedList = zip(clust, price, rating)

    

    return render(request, 'result_clust.html', {'zippedList': zippedList})

