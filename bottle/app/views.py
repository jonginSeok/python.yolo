# from django.shortcuts import render, redirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# from django.contrib.auth import authenticate, login

from django.contrib.auth.views import LoginView


class CustomLoginView(LoginView):
    # template_name = "registration/login.html"
    template_name = "login.html"


login_view = CustomLoginView.as_view()


# def login_view(request):
#     if request.method == "POST":
#         username = request.POST.get("username")
#         password = request.POST.get("password")
#         user = authenticate(request, username=username, password=password)

#         if user:
#             login(request, user)
#             return redirect("/")
#         else:
#             return render(
#                 request,
#                 "login.html",
#                 {"error": "아이디 또는 비밀번호가 올바르지 않습니다."},
#             )

#     return render(request, "login.html")


@login_required(login_url="/login/")
def main_view(request):
    return render(request, "main.html")
