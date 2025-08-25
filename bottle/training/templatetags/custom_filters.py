# myapp/templatetags/custom_filters.py
from django import template

register = template.Library()


@register.filter
def mul(value, arg):
    try:
        print(f"[커스텀 태그] mul value:{value} arg:{arg}")
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ""
