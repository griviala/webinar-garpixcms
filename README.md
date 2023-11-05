# Инструкция, как развернурть проект

0. Необходимо иметь: python, python-dev, pip, venv/pipenv, python IDE (VSCode / PyCharm)

1. Создайте файл с переменными окружения:

```
cp example.env .env
```

2. Активируйте виртуальное окружение:

* Если у вас установлен venv:

    + Для Unix, MacOS:

```

python -m venv .
sudo chmod -R 777 bin/
source bin/activate

```
*
    + Для Windows:

```

python -m venv .
Scripts/activate

```

* Если у вас установлен pipenv:

```

pipenv shell

```

3. Установите необходимые зависимости python

* venv:

```
pip install -r requirements.txt
```

* pipenv:

```
pipenv install -r requirements.txt
```

4. Примените миграции:

```
python backend/manage.py migrate
```

5. Создайте учетную запись администратора:

```
python backend/manage.py createsuperuser
```

6. Запустите сервер django, чтобы убедиться, что все работает:

```
python backend/manage.py runserver

```

Затем откройте указанный в консоли адрес в браузере. Если все прошло успешно, вы увидите приветственное окно.
