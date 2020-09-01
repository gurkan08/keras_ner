
from django.conf.urls import url, include
from .views import NERView

app_name = "bilstm_crf_ner"

urlpatterns = [
    url(r'^$', NERView.as_view(), name="bilstm_crf_ner"),

]
