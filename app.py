import argparse
import json
import logging
import os

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from classifier import load_model, predict_class

# Инициализация приложения с метаданными для документации
app = FastAPI(
    title="Heart attack prediction",
    description="Предсказание рисков сердечного приступа",
    version="1.0.0",
    contact={
        "name": "Maxim Starostin",
        "email": "max-star@yandex.ru"
    }
)

# Логирование
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter(
    "%(name)s %(asctime)s %(levelname)s %(message)s"
    )
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)


# Загрузка модели при старте сервера (выполняется один раз)
try:
    model = load_model('models/model_heart.pkl')
except FileNotFoundError:
    print("Файл модели не найден, запустите train_and_save_model")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {str(e)}")

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")


# Проверка работоспособности
@app.get(
    "/health",
    summary="Поверка работы фреймворка",
    description="Возвращает старус ОК",
    tags=["health"]
    )
def health():
    return {"status": "OK"}


# Главная страница - шаблон загрузки датасета
@app.get(
    "/",
    summary="Форма загрузки датанных для предсказаний",
    description="Передает файл формата сsv с данными пациентов для предсказаний",
    tags=["index"]
    )
def main(request: Request):
    '''
    Функция рендеринкга главной страницы приложения с формой загрузки csv 
    файла данных, на базе которых нужно предсказать риск сердечного приступа
    '''
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})


# Предсказания
@app.post(
    "/predict",
    summary="Получить предсказание",
    description="Возвращает результат предсказания в формате JSON",
    tags=["predictions"]
)
def predict(file: UploadFile, request: Request):
    '''
    Функция получения предсказаний. Сохраняет полученный csv файл в папку tmp,
    проводит предсказания, выводит их на экран в формате JSON и удаляет файл
    с данными для предсказаний
    '''
    temp_dir = 'tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    data_path = os.path.join(temp_dir, file.filename)
    try:
        with open(data_path, "wb") as buffer:
            buffer.write(file.file.read())
        app_logger.info(f'processing file - {data_path}')
        predictions = predict_class(model, data_path)

        json_data = predictions.to_json(orient='records')
        return JSONResponse(
            content={
                "predictions": json.loads(json_data)
            },
            status_code=200
        )
    except Exception as e:
        app_logger.error(f"Ошибка при обработке файла: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Произошла ошибка при обработке файла"}
        )
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
