
from rest_framework.views import APIView
from rest_framework.response import Response
from .codes.api_ner import _api
# Create your views here.

class NERView(APIView):

    def get(self, request):
        text = request.data["text"]
        content = {"result": _api(text)}
        return Response(content)

    def post(self, request):
        text = request.data["text"]
        content = {"result": _api(text)}
        return Response(content)

