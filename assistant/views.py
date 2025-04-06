from rest_framework.views import APIView
from rest_framework.response import Response
from .langchain_logic import process_question

class AskQuestion(APIView):
    def post(self, request):
        question = request.data.get("question")
        session_id = request.data.get("session_id", "default-session")
        answer = process_question(question, session_id)
        return Response({"answer": answer})
