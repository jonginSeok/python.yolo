from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import login
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django import forms

@method_decorator(csrf_protect, name='dispatch')
class CustomLoginView(LoginView):
    template_name = "registration/login_with_intro.html"  # 인트로 애니메이션이 포함된 로그인 페이지
    # template_name = "registration/login.html"  # accounts/login #기존 로그인 페이지
    
    def get(self, request, *args, **kwargs):
        print("=== CustomLoginView.get() 호출됨 ===")
        print("세션 데이터:", dict(request.session))
        
        # 세션에서 회원가입 오류 메시지를 컨텍스트에 추가하고 삭제
        register_errors = None
        show_register_form = False
        
        if 'register_errors' in request.session:
            register_errors = request.session['register_errors']
            del request.session['register_errors']
            print("세션에서 오류 메시지 읽음:", register_errors)
        
        if 'show_register_form' in request.session:
            show_register_form = request.session['show_register_form']
            del request.session['show_register_form']
            print("세션에서 폼 표시 플래그 읽음:", show_register_form)
        
        # 부모 클래스의 get 메서드 호출
        response = super().get(request, *args, **kwargs)
        
        # context_data가 None인 경우 초기화
        if response.context_data is None:
            response.context_data = {}
        
        # 오류 메시지가 있으면 컨텍스트에 추가
        if register_errors:
            response.context_data['register_errors'] = register_errors
            print("컨텍스트에 오류 메시지 추가:", register_errors)
        
        # 회원가입 폼을 표시해야 하면 컨텍스트에 추가
        if show_register_form:
            response.context_data['show_register_form'] = True
            print("회원가입 폼 표시 플래그 설정")
        
        print("최종 컨텍스트 데이터:", response.context_data)
        return response

login_view = CustomLoginView.as_view()

# 커스텀 회원가입 폼 (이메일 필드 포함)
class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text='필수 항목입니다.')
    
    class Meta:
        model = UserCreationForm.Meta.model
        fields = UserCreationForm.Meta.fields + ('email',)
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

@csrf_protect
def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '회원가입이 완료되었습니다!')
            return redirect('/')
        else:
            # 디버깅: 폼 오류 출력
            print("=== 폼 오류 디버깅 ===")
            print("폼 오류:", form.errors)
            print("폼 데이터:", request.POST)
            
            # 폼 오류 처리
            error_messages = []
            for field, errors in form.errors.items():
                for error in errors:
                    if field == 'username':
                        if 'unique' in error:
                            error_messages.append('이미 사용 중인 아이디입니다.')
                        else:
                            error_messages.append(f'아이디: {error}')
                    elif field == 'password2':
                        if 'mismatch' in error:
                            error_messages.append('비밀번호가 일치하지 않습니다.')
                        else:
                            error_messages.append(f'비밀번호 확인: {error}')
                    elif field == 'email':
                        if 'unique' in error:
                            error_messages.append('이미 사용 중인 이메일입니다.')
                        else:
                            error_messages.append(f'이메일: {error}')
                    else:
                        error_messages.append(f'{field}: {error}')
            
            print("처리된 오류 메시지:", error_messages)
            
            # 오류 메시지를 세션에 저장하고 회원가입 폼이 표시된 상태로 리다이렉트
            request.session['register_errors'] = error_messages
            request.session['show_register_form'] = True
            print("세션에 저장된 오류:", request.session.get('register_errors'))
            return redirect('/accounts/login/')
    else:
        return redirect('/accounts/login/')

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