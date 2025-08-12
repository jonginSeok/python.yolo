#  from django.shortcuts import render
# Create your views here.

from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login

from django.shortcuts import render, redirect



# class CustomLoginView(LoginView):
#     template_name = "registration/login.html"  # accounts/login

# login_view = CustomLoginView.as_view()

# return render(request, 'login.html')  # 경로가 app/templates/login.html이라면 이렇게만 써도 돼요!
# 혹시 registration/login.html 처럼 되어 있다면 'login.html'로 바꿔줘야 해요.



@login_required(login_url="/login/")
def main_view(request):
    return render(request, "main.html")

# class CustomLoginView(LoginView):
#     template_name = "login.html"  # ngins7512 / 2025.08.06

# login_view = CustomLoginView.as_view()


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect("/")
        else:
            return render(request, "login.html", {"error": "아이디 또는 비밀번호가 올바르지 않습니다."})

    return render(request, "login.html")
