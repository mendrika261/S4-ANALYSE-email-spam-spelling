from django.shortcuts import render

from mailbox_web.models import User
from mailbox_web.views import NewView


def inbox(request, email_id=None):
    context = {
        'users': User.objects.all().order_by('email'),
        'user_id': request.session.get('user_id', 1),
        'email_id': email_id,
        'model': request.GET.get('model', None),
        'remind': NewView.mark,
    }
    return render(request, 'inbox.html', context)


def sent(request, email_id=None):
    context = {
        'users': User.objects.all().order_by('email'),
        'user_id': request.session.get('user_id', 1),
        'email_id': email_id,
        'remind': NewView.mark,
    }
    return render(request, 'sent.html', context)


def spam(request, email_id=None):
    context = {
        'users': User.objects.all().order_by('email'),
        'user_id': request.session.get('user_id', 1),
        'email_id': email_id,
        'remind': NewView.mark,
    }
    return render(request, 'spam.html', context)

