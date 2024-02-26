"""
URL configuration for email_sys project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from mailbox_web import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('api/switch/<int:user_id>', views.switch_user, name='switch_user'),

    path('api/count/<int:receiver_id>', views.count_json, name='count_json'),
    path('api/content/<int:email_id>', views.content_email, name='content_email'),

    path('', views.inbox, name='inbox'),
    path('<int:email_id>', views.inbox, name='inbox_with_email'),
    path('api/inbox/<int:receiver_id>/<int:page>/<int:limit>', views.inbox_json, name='inbox_json'),

    path('sent/', views.sent, name='sent'),
    path('sent/<int:email_id>', views.sent, name='sent_with_email'),
    path('api/sent/<int:sender_id>/<int:page>/<int:limit>', views.sent_json, name='sent_json'),

    path('spam/', views.spam, name='spam'),
    path('spam/<int:email_id>', views.spam, name='spam_with_email'),
    path('api/spam/<int:receiver_id>/<int:page>/<int:limit>', views.spam_json, name='spam_json'),


    path('new/', views.NewView.as_view(), name='new'),
    path('new/<int:parent_id>', views.NewView.as_view(), name='new'),

    path('api/report/<str:parents_id>', views.mark_spam, name='mark_spam'),
    path('api/safe/<str:parents_id>', views.mark_normal, name='mark_safe'),
    path('api/delete/<str:parents_id>', views.delete, name='delete'),

    path('api/train', views.train_model, name='train'),
    path('api/remind', views.remind, name='remind'),

    path('api/verify', views.verify, name='verify'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
