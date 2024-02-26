from django.db import models


class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    photo = models.ImageField(upload_to='photos/%Y/%m/%d/', default='photos/default.png')

    def __str__(self):
        return f'{self.username} ({self.email})'
