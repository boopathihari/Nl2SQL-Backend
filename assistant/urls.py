from django.urls import path
from .views import AskQuestion

urlpatterns = [
    path("ask/", AskQuestion.as_view(), name="ask-question"),
]
