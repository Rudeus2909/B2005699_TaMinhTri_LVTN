from django.shortcuts import render
import requests
from django.http import JsonResponse

# Create your views here.
from django.http import JsonResponse

def get_home(req):
    if req.method == 'POST' and req.FILES['image']:
        image_file = req.FILES['image']

        api_url = 'http://localhost:5000/api/v1/predict'

        res = requests.post(
            api_url,
            files={'upload_file': image_file}
        )

        if res.status_code == 200:
            result = res.json().get('data')  # Nhận kết quả từ API
            return JsonResponse({
                'status': 'success',
                'top_labels': result['top_labels'],
                'label_images': result['label_images']
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Dự đoán thất bại. Vui lòng thử lại.'
            }, status=500)

    return render(req, 'home.html')


