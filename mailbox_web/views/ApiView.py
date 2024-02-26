import csv
import os
import re
import shutil
from functools import lru_cache

import nltk
from bs4 import BeautifulSoup
from django.conf import settings
from django.db.models import F, Q
from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.timezone import now
from sklearn.svm import SVC
from unidecode import unidecode

from analysis.classification import get_model_with_words_idf
from mailbox_web.models import Email, User



def switch_user(request, user_id):
    request.session['user_id'] = user_id
    return JsonResponse({'user_id': user_id})


def count_json(request, receiver_id):
    receiver = User.objects.get(id=receiver_id)
    unread = Email.objects.filter(receiver=receiver, state__lte=10, read=None).count()
    spam = Email.objects.filter(receiver=receiver, state=20, read=None).count()
    # deleted = Email.objects.filter(receiver=receiver, state=30).count()
    count = {
        'unread': unread,
        'spam': spam,
        # 'deleted': deleted,
    }
    return JsonResponse(count, safe=False)


def inbox_json(request, receiver_id, page, limit):
    data = []
    start = (page - 1) * limit
    end = page * limit
    receiver = User.objects.get(id=receiver_id)
    emails = Email.objects.filter(receiver=receiver, state__lte=15).order_by('-date')[start:end]
    passed = []
    for email in emails:
        if email.parent.id in passed:
            continue
        else:
            passed.append(email.parent.id)
        data.append({
            'id': email.id,
            'sender_email': email.sender.email,
            'sender_name': email.sender.username,
            'receiver_email': email.receiver.email,
            'receiver_name': email.receiver.username,
            'subject': email.subject,
            'content': str(email.get_content())[:100] + '...',
            'date': email.date.strftime('%Y-%m-%d %H:%M:%S'),
            'read': email.read,
            'parent_id': email.parent.id,
        })
    return JsonResponse(data, safe=False)


def sent_json(request, sender_id, page, limit):
    data = []
    start = (page - 1) * limit
    end = page * limit
    sender = User.objects.get(id=sender_id)
    emails = Email.objects.filter(id=F('parent'), sender=sender).order_by('-date')[start:end]
    for email in emails:
        data.append({
            'id': email.id,
            'sender_email': email.sender.email,
            'sender_name': email.sender.username,
            'receiver_email': email.receiver.email,
            'receiver_name': email.receiver.username,
            'subject': email.subject,
            'content': str(email.get_content())[:100] + '...',
            'date': email.date.strftime('%Y-%m-%d %H:%M:%S'),
            'read': email.read,
            'parent_id': email.parent.id,
        })
    return JsonResponse(data, safe=False)


def spam_json(request, receiver_id, page, limit):
    data = []
    start = (page - 1) * limit
    end = page * limit
    receiver = User.objects.get(id=receiver_id)
    emails = Email.objects.filter(state__gte=20, state__lt=30, id=F('parent'), receiver=receiver).order_by('-date')[
             start:end]
    for email in emails:
        data.append({
            'id': email.id,
            'sender_email': email.sender.email,
            'sender_name': email.sender.username,
            'receiver_email': email.receiver.email,
            'receiver_name': email.receiver.username,
            'subject': email.subject,
            'content': str(email.get_content())[:100] + '...',
            'date': email.date.strftime('%Y-%m-%d %H:%M:%S'),
            'read': email.read,
            'parent_id': email.parent.id,
        })
    return JsonResponse(data, safe=False)


def content_email(request, email_id):
    email = Email.objects.get(id=email_id)
    emails = Email.objects.filter(parent=email.parent).order_by('date')
    data = []
    for email in emails:
        if email.receiver.id == request.session.get('user_id', 1):
            if email.read is None:
                email.read = now()
                email.save()
        data.append(email.to_json())
    return JsonResponse(data, safe=False)


def build_csv():
    emails = Email.objects.filter(Q(state=15) | Q(state=25))
    path = os.path.join(settings.BASE_DIR, 'analysis/data/')
    # copy chat.csv into new.csv
    shutil.copyfile(path + 'chat.csv', path + 'new.csv')
    with open(path + 'new.csv', 'a', newline="\n") as csvfile:
        writer = csv.writer(csvfile)
        for email in emails:
            state = 0 if email.state == 25 else 1
            writer.writerow([state, f'{email.sender.email} // {email.subject} // {email.get_content()}'])
    print('build csv done')
    return emails.count()


def rebuild_model():
    from mailbox_web.views import NewView
    nb = 100+build_csv()
    path = os.path.join(settings.BASE_DIR, 'analysis/data/')
    for file in os.listdir(path):
        if file.__contains__(str(nb)):
            os.remove(path + file)
    path = os.path.join(settings.BASE_DIR, 'analysis/model/')
    for file in os.listdir(path):
        if file.__contains__(str(nb)):
            os.remove(path + file)
    NewView.model, NewView.words_idf = get_model_with_words_idf(data_limit=nb, model=SVC, kernel='linear',
                                                                column_limit=500,
                                                                x_column_start=2, y_column=0, test_size=0.2,
                                                                score=True)
    NewView.mark = 10
    print('rebuild model done')


def train_model(request):
    rebuild_model()
    return redirect('/?model=1')


def mark_spam(request, parents_id):
    parents_id = parents_id.split(';')

    for parent_id in parents_id:
        email = Email.objects.get(parent_id=parent_id)
        email.state = 25
        email.save()
    from mailbox_web.views import NewView
    NewView.mark -= 1
    if NewView.mark == 0:
        rebuild_model()
        return JsonResponse({'build': 'ok'})
    return JsonResponse({'status': 'ok'})


def mark_normal(request, parents_id):
    parents_id = parents_id.split(';')

    for parent_id in parents_id:
        email = Email.objects.get(parent_id=parent_id)
        email.state = 15
        email.save()
    from mailbox_web.views import NewView
    NewView.mark -= 1
    if NewView.mark == 0:
        rebuild_model()
        return JsonResponse({'build': 'ok'})
    return JsonResponse({'status': 'ok'})


def delete(request, parents_id):
    parents_id = parents_id.split(';')

    for parent_id in parents_id:
        email = Email.objects.get(parent_id=parent_id)
        email.state = 30
        email.save()
    return JsonResponse({'status': 'ok'})


def remind(request):
    from mailbox_web.views import NewView
    return JsonResponse(NewView.mark, safe=False)


@lru_cache(maxsize=None)
def distance(s1, s2):
    s1 = str(s1)
    s2 = str(s2)
    m = len(s1)
    n = len(s2)

    matrix = []
    for i in range(m+1):
        temp = []
        for j in range(n+1):
            temp.append(0)
        matrix.append(temp)

    for i in range(m+1):
        matrix[i][0] = i
    for j in range(n+1):
        matrix[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                change = 0
            else:
                change = 1
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + change)

    return matrix[m][n]


def load_french_words(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        words = file.read().splitlines()
    words = map(str.lower, words)
    return sorted(set(words), key=len)


french_words = load_french_words('analysis/data/liste_francais2.txt', encoding='latin-1')


@lru_cache(maxsize=None)
def get_bad_words(text):
    text = unidecode(text)
    soup = BeautifulSoup(text, 'html.parser')  # html
    text = soup.get_text()

    text = re.sub(r'http[s]?://\S+', ' ', text)  # link
    text = re.sub(r'@\w+', ' ', text)  # mention
    text = re.sub(r'[^\w\s]', ' ', text)  # punctuations
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)  # numbers
    tokens = nltk.word_tokenize(text, language='french')
    bad_words = set()
    for token in tokens:
        if str(token).lower() not in french_words:
            bad_words.add(token)
    return bad_words


@lru_cache(maxsize=None)
def get_near_words(word):
    word = str(word).lower()
    near_words = []
    max_distance = 5
    max_near_words = 3
    word_length = len(word)

    for distance_act in range(1, max_distance + 1):
        for french_word in french_words:
            if len(french_word) < word_length - distance_act:
                continue
            if len(french_word) > word_length + distance_act:
                break
            if distance(word, french_word) <= distance_act:
                near_words.append(french_word)
                if len(near_words) >= max_near_words:
                    return '<br>'.join(near_words)
    return 'Aucun mot similaire trouvé'


def verify(request):
    text = str(request.POST.get('text', ''))

    print(text)
    pattern = r'<span style="color: red;" tooltip=".*?">(.*?)</span>'
    # pattern = r'<span style="color: red" data-bs-toggle="tooltip" title=".*?" data-bs-html=true>(.*?)</span>'
    text = re.sub(pattern, r'\1', text)

    bad_words = get_bad_words(text)
    text_marked = text

    # highlight bad words
    for bad_word in bad_words:
        if bad_word == 'p' or bad_word == 'nbsp':
            continue
        pattern = r'\b' + bad_word + r'\b'
        text_marked = re.sub(pattern, f'<span style="color: red;" tooltip="{get_near_words(bad_word)}">{bad_word}</span>', text_marked)
        # text_marked = re.sub(pattern, f'<span style="color: red" data-bs-toggle="tooltip" title="{get_near_words(bad_word)}" data-bs-html=true>{bad_word}</span>', text_marked)

    print(text_marked)
    return JsonResponse(text_marked, safe=False)
