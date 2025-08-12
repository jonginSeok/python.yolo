# form_tags / ngins7512 / 2025.08.06
from django import template

register = template.Library()


@register.filter(name="add_class")
def add_class(field, css_class):
    # 문자열이 전달된 경우 처리
    if isinstance(field, str):
        return field
    
    # Django 폼 필드인 경우에만 as_widget 메서드 사용
    if hasattr(field, 'as_widget'):
        return field.as_widget(attrs={"class": css_class})
    
    # 그 외의 경우 원본 반환
    return field
