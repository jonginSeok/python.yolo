from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection

def dashboard_view(request):
    return render(request, 'main.html')

def dashboard_data(request):
    if request.method == 'GET':
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        app_label,
                        model,
                        COUNT(*) as count
                    FROM django_content_type 
                    GROUP BY app_label, model
                    ORDER BY count DESC
                    LIMIT 10
                """)
                rows = cursor.fetchall()
                
            response_data = {
                'model_performance': [
                    {
                        'model': f"{row[0]}-{row[1]}" if row[0] else row[1],
                        'count': row[2],
                        'accuracy': 75 + (row[2] % 20)
                    }
                    for row in rows
                ],
                'status_breakdown': {
                    'bad-broken-large': 35,
                    'bad-broken-small': 30,
                    'bad-contamination': 35
                },
                'time_breakdown': {
                    '오전': 40,
                    '오후': 35,
                    '밤': 25
                }
            }
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'message': '데이터를 불러오는 중 오류가 발생했습니다.'
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)