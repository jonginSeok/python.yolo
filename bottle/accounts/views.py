#  from django.shortcuts import render
# Create your views here.

from django.contrib.auth.views import LoginView


class CustomLoginView(LoginView):
    # template_name = "registration/login.html"
    template_name = "login.html"


login_view = CustomLoginView.as_view()

# return render(request, 'login.html')  # 경로가 app/templates/login.html이라면 이렇게만 써도 돼요!
# 혹시 registration/login.html 처럼 되어 있다면 'login.html'로 바꿔줘야 해요.
