# .venv\Scripts\activate
# http://127.0.0.1:4040/inspect/http
# ngrok http 5000
# python app.py

import os
import uuid
import datetime
import asyncio
import aiohttp
import json
import httpx
import time
import datetime
import aiofiles
from sqlalchemy.orm import joinedload
import logging
import re
from urllib.parse import urlparse, unquote
from sqlalchemy import desc
from sqlalchemy.sql import select
from requests.auth import HTTPBasicAuth
from httpx import BasicAuth
from pyngrok import ngrok
from quart import Quart, jsonify, request, render_template, abort
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, PickleType, JSON
)
from sqlalchemy.orm import relationship

app = Quart(__name__)

load_dotenv('dot.env')
API_AUTH = os.getenv('API_AUTH')
IMAGE_KEY = os.getenv('IMAGE_KEY')
NGROK_AUTHTOKEN = os.getenv('NGROK_AUTHTOKEN')

DATABASE_URL = "sqlite+aiosqlite:///./newmj.db"  # Асинхронный драйвер для SQLite
Base = declarative_base()

class Imagines(Base):
    __tablename__ = 'imagines_bd'
    id = Column(Integer, primary_key=True)
    recorddate = Column(DateTime, default=datetime.datetime.utcnow)
    jobid = Column(String(50), unique=True)  # Убедитесь, что jobid уникален
    prompt = Column(String(1024))
    results = Column(String(50))
    user_created = Column(String(50))
    date_created = Column(String(50))
    url = Column(String(255))
    filename = Column(String(255))
    small_url = Column(String(255))
    upscaled_urls = Column(PickleType)
    upscaled = Column(PickleType)
    ref = Column(String(50))
    author = Column(String, default='ilia')
    upscaled_images = relationship('UpscaledImages', back_populates='imagines_bd')

class UpscaledImages(Base):
    __tablename__ = 'upscaled_images'
    id = Column(Integer, primary_key=True)
    recorddate = Column(DateTime, default=datetime.datetime.utcnow)
    imagine_id = Column(Integer, ForeignKey('imagines_bd.id'))
    upscaled_url = Column(String(255))
    filename_upscaled_url = Column(String(255))
    small_upscaled_url = Column(String(255))
    author = Column(String, default='ilia')
    imagines_bd = relationship('Imagines', back_populates='upscaled_images')

#echo=True
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_reply_url():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:4040/api/tunnels") as response:
            tunnel_url = await response.text()
            j = json.loads(tunnel_url)
            reply_url = j['tunnels'][0]['public_url']
    return reply_url

@app.before_serving
async def startup():
    global reply_url
    reply_url = await get_reply_url()
    if reply_url:
        print(f"Webhook reply URL: {reply_url}")
    else:
        print("ngrok туннель не найден.")
    async with engine.begin() as conn:
        #await conn.run_sync(Imagines.__table__.drop)
        #await conn.run_sync(UpscaledImages.__table__.drop)
        await conn.run_sync(Base.metadata.create_all)

    
@app.route('/send_prompt', methods=['POST'])
async def send_prompt():
    form_data = await request.form
    prompt = form_data.get('prompt')
    replyRef = str(uuid.uuid4())  # Генерируем уникальный ref, если нужно

    # Добавляем задачу отправки запроса к API в очередь
    await task_queue.enqueue(api_request_function, prompt, replyRef=replyRef)  
    print(f"Задача 'send_prompt' добавлена в очередь: {replyRef}")
    return jsonify({"message": "Запрос добавлен в очередь"})

async def api_request_function(prompt, replyRef):
    url = "https://cl.imagineapi.dev/items/images/"
    payload = {
        "prompt": prompt,
        "ref": replyRef
    }
    headers = {
        'Authorization': f'Bearer {API_AUTH}',
        'Content-Type': 'application/json'
    }

    # Отправка асинхронного запроса к API
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        # Можно добавить обработку ответа от API здесь
        if response.status_code == 200:
            print(f"Запрос к API успешно отправлен: {prompt}")
        else:
            print(f"Ошибка при отправке запроса к API: {response.status_code}")


@app.after_serving
async def shutdown():
    await engine.dispose()

async def download_image(url, filename, folder):
    # Создаем папку, если её нет
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code == 200:
            async with aiofiles.open(full_path, 'wb') as file:
                await file.write(response.content)
            print(f"Image {filename} saved successfully: {full_path}")
        else:
            print(f"Failed to download {filename} image from {url}")

async def process_imagekit(url, filename):
    api_key = IMAGE_KEY
    imagekituploadurl = "https://upload.imagekit.io/api/v1/files/upload"
    files = {
        'file': (None, url),
        'fileName': (None, filename),
        'transformation': (None, '{"pre":"w-500"}'),
        'useUniqueFileName': (None, 'false')
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resized_image_response = await client.post(
            imagekituploadurl, 
            auth=BasicAuth(api_key, ''), 
            files=files
        )
    response_data = json.loads(resized_image_response.text)
    small_url = response_data.get("url")
    print(f'SmallURL: {small_url}')
    return small_url

async def process_imagekit_full(url, filename):
    api_key = IMAGE_KEY
    imagekituploadurl = "https://upload.imagekit.io/api/v1/files/upload"
    files = {
        'file': (None, url),
        'fileName': (None, filename),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resized_image_response = await client.post(
            imagekituploadurl, 
            auth=BasicAuth(api_key, ''), 
            files=files
        )
    response_data = json.loads(resized_image_response.text)
    full_url = response_data.get("url")
    print(f'SmallURL: {full_url}')
    return 

class AsyncFunctionQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_task_done = asyncio.Event()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.current_task_done.set()
        logging.info(f"инициализация: {self.current_task_done}")

    async def enqueue(self, fn, *args, **kwargs):
        await self.queue.put((fn, args, kwargs))
        logging.info(f"Задача добавлена в очередь Размер очереди: {self.queue.qsize()}")
        # Автоматически запускаем обработку очереди при добавлении первой задачи
        if self.queue.qsize() == 1:
            asyncio.create_task(self.process_queue())


    async def process_queue(self):
        while not self.queue.empty():
            await self.current_task_done.wait()  # Ожидаем завершения предыдущей задачи
            self.current_task_done.clear()  # Сбрасываем событие
            logging.info("Начинается обработка задачи.")
            fn, args, kwargs = await self.queue.get()
            await fn(*args, **kwargs)

    def signal_task_completed(self):
        self.current_task_done.set()
        logging.info(f"Задача выполнена")


task_queue = AsyncFunctionQueue()


@app.route('/webhook', methods=['POST'])
async def webhook():
    data = await request.get_json()
    event = data.get('event')
    payload = data.get('payload')
    status = payload.get('status') 
    replyRef = payload.get('replyRef') 

    if event == "images.items.create":
        print(f"ID: {payload['id']}, Создание изображения")
        return '', 200

    elif event == "images.items.update":
        if status == 'in-progress':
            print(f"ID: {payload['id']}, Status: {status}, Progress: {payload.get('progress')},")
        elif status == 'completed':
            print(f"ID: {payload['id']}, выполнен")
            await handle_imagine(payload)
            task_queue.signal_task_completed()  # Сигнализируем о завершении задачи
            return '', 200
        else:
            print(f"ID: {payload['id']}, неизвестный статус: {status}")

    return '', 200

async def handle_imagine(payload):
    id = payload['id']
    prompt = payload['prompt']
    results = payload.get('results')
    user_created = payload['user_created']
    date_created = payload['date_created']
    url = payload['url']
    filename = os.path.basename(unquote(urlparse(url).path))
    small_url = await process_imagekit(url, id)
    upscaled_urls = payload.get('upscaled_urls', [])
    ref = payload['ref']

    async with async_session() as session:
        new_imagine = Imagines(
            jobid=id,  # Предполагается, что id это уникальный идентификатор задания, а не первичный ключ таблицы
            prompt=prompt,
            results=results,
            user_created=user_created,
            date_created=date_created,
            url=url,
            filename=filename,
            small_url=small_url,
            ref=ref,
        )
        session.add(new_imagine)
        await session.commit()
        imagine_id = new_imagine.id  # Получаем ID сохраненной записи для дальнейшего использования

    for upscaled_url in upscaled_urls:
        filename_upscaled_url = os.path.basename(unquote(urlparse(upscaled_url).path))
        small_upscaled_url = await process_imagekit(upscaled_url, filename_upscaled_url)
#        await download_image(upscaled_url, filename_upscaled_url, folder='upscaled1xnew')
        await process_imagekit_full(upscaled_url, filename_upscaled_url)
        # Сохранение данных в UpscaledImages
        async with async_session() as session:
            new_upscaled_image = UpscaledImages(
                imagine_id=imagine_id,
                upscaled_url=upscaled_url,
                filename_upscaled_url=filename_upscaled_url,
                small_upscaled_url=small_upscaled_url,
            )
            session.add(new_upscaled_image)
            await session.commit()


@app.route('/process_json', methods=['POST'])
async def process_json():
    data = await request.json
    total_tasks = len(data)
    print(f"Получено задач из JSON: {total_tasks}")

    for item in data:
        prompt = item if isinstance(item, str) else item.get('prompt', '')
        if prompt:
            replyRef = str(uuid.uuid4()) 
            await task_queue.enqueue(api_request_function, prompt, replyRef=replyRef)  
            print(f"Задача из файла добавлена в очередь: {replyRef}")
            await asyncio.sleep(1)  # Добавление задержки

    return jsonify({'message': f'Processed {total_tasks} items'})


@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/batch')
async def batch_uploadj():
    return await render_template('batch.html')

@app.route('/imagines')
async def show_imagines():
    async with async_session() as session:
        result = await session.execute(
            select(Imagines).options(joinedload(Imagines.upscaled_images)).order_by(Imagines.recorddate.desc())
        )
        imagines = result.unique().scalars().all()
    return await render_template('imagines.html', imagines=imagines)


if __name__ == '__main__':
    
    app.run()