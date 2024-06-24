import os
from pathlib import Path

import joblib
from rest_framework.response import Response
from rest_framework.views import APIView


class ReplyFilterAPI(APIView):
    def get(self, request):
        reply_content = request.GET.get('reply-content')

        # 모델소환
        model_file_path = os.path.join(Path(__file__).resolve().parent, 'ai/reply_default_model.pkl')
        model = joblib.load(model_file_path)
        X_train = [reply_content]
        prediction = model.predict(X_train)

        if prediction[0] == 1:
            # 추가 fit
            transformed_X_train = model.named_steps['count_vectorizer'].transform(X_train)
            model.named_steps['multinomial_NB'].partial_fit(transformed_X_train, prediction)
            joblib.dump(model, model_file_path)

        return Response(prediction[0])
