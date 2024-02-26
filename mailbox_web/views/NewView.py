from django.shortcuts import render, redirect
from django.views import View
from sklearn.svm import SVC

from analysis.classification import get_model_with_words_idf, prediction
from mailbox_web.models import User, Email


class NewView(View):
    model, words_idf = get_model_with_words_idf(data_limit=100, model=SVC, kernel='linear', column_limit=500,
                                                x_column_start=2, y_column=0, test_size=0.2, score=False)
    mark = 1

    @staticmethod
    def get(request, parent_id=None):
        parent = Email.objects.get(id=parent_id) if parent_id else None
        context = {'users': User.objects.all().order_by('email'), 'user_id': request.session.get('user_id', 1),
                   'parent_id': parent_id, 'parent': parent, 'subject': parent.subject if parent else None}
        return render(request, 'new.html', context)

    @staticmethod
    def post(request, parent_id=None):
        sender_id = request.session.get('user_id', 1)
        receiver_id = request.POST.get('receiver')
        subject = request.POST.get('subject')
        content = request.POST.get('content')
        state = 10 if prediction(NewView.model, NewView.words_idf, content, log=False)[0] else 20
        email = Email.objects.create(sender_id=sender_id, receiver_id=receiver_id, subject=subject, state=state)
        email.set_content(content) # crypt content
        if parent_id:
            email.parent_id = parent_id
            email.save()
            return redirect('/sent/'+str(email.id))
        else:
            email.parent_id = email.id
            email.save()
            return redirect('sent')
