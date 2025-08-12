#  from django.shortcuts import render
# Create your views here.

from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect

class CustomLoginView(LoginView):
    template_name = "registration/login.html"  # accounts/login
    

login_view = CustomLoginView.as_view()

# return render(request, 'login.html')  # 경로가 app/templates/login.html이라면 이렇게만 써도 돼요!
# 혹시 registration/login.html 처럼 되어 있다면 'login.html'로 바꿔줘야 해요.

# class CustomLoginView(LoginView):
#     template_name = "registration/login.html"
#     # template_name = "login.html"  # ngins7512 / 2025.08.06
    
#     def get(self, request, *args, **kwargs):
#         # SPA 방식으로만 접근 허용
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return super().get(request, *args, **kwargs)
#         else:
#             # 직접 접근 시 메인 페이지로 리다이렉트
#             return redirect('/')


# login_view = CustomLoginView.as_view()


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

