# Generated by Django 4.2.4 on 2024-05-08 10:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Game',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('team1', models.CharField(max_length=100)),
                ('team2', models.CharField(max_length=100)),
                ('date', models.DateField()),
            ],
        ),
    ]