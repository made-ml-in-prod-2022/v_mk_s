# HW1 for the ML in production course

[Vladislav Melnichuk (v_mk_s)](https://github.com/made-ml-in-prod-2022/v_mk_s)

[HW1 description](https://github.com/made-ml-in-prod-2022/v_mk_s/blob/homework1/hw1_description.md)

## Work Description

`ml_project` directory contains the project with CLI application to manage feature processing, model creation and evaluation. It contains train and inference notebooks. [Heart Disease Cleveland dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) was used in the project.

I used [Cookiecutter Data Science template](https://drivendata.github.io/cookiecutter-data-science/) as a template.

For configuration management of the project, [`hydra`](https://hydra.cc/) framework is used. This experience was pretty nice for me, as I have never tickled with `yaml` files seriously (except for `docker-compose` ones).

## Project structure

## Requirements for HW1 (self-assessment done)

0) 
1)
2)
3)
4)
5)
6)
7)
8)
9)
10)
11)
12)
13)
14) Полностью выполнено (3 балла)


0) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)
1) В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

2) Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
   Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

   Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

3) Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)
4) Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

5) Проект имеет модульную структуру (2 балла)
6) Использованы логгеры (2 балла)

7) Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

8) Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки <https://faker.readthedocs.io/en/>, <https://feature-forge.readthedocs.io/en/latest/>
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

9) Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
10) Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

11) Напишите кастомный трансформер и протестируйте его (3 балла)
   <https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156>

12) В проекте зафиксированы все зависимости (1 балл)
13) Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: <https://github.com/demo-ml-cicd/ml-python-package>
14) Используется hydra  (<https://hydra.cc/docs/intro/>) (3 балла - доп баллы)

## __Usage__

### __Initial Setup__

In directory `ml_project` run the following command:

`>>> pip install -r requirements.txt`
`>>> pip install hydra-core --upgrade`

### __Run the training pipeline with hydra__

In directory `ml_project` run the following command:

`>>> python3 pipeline.py`

After this, `outputs/` dir should be automatically created by hydra to manage runs.

### __Logging__

Loggers are used in most modules and the logfile can be found in hydra's `outputs/` directory as `pipeline.log`.

### __Unit-testing__

`Pytest` framework is used for testing, modular tests can be found at `testing/` directory. I tried to advance
my knowledge of pytest by using `mark.skipif` semantics in order to manage offline runs for data fetching.
Also, I found out that `pytest` has its own `setup` and `teardown` methods. Those were used in order to
create a temporary directory to write/read data during testing. The directory is cleaned up after each run.

In order to run all unit-tests, run the following command:

`>>> python3 -m pytest`
