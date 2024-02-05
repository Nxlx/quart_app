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
from sqlalchemy.sql import select
from requests.auth import HTTPBasicAuth
from httpx import BasicAuth
from pyngrok import ngrok
from quart import Quart, jsonify, request, render_template
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
import datetime

app = Quart(__name__)
# Загрузка переменных окружения
load_dotenv('dot.env')
USEAPI_TOKEN = os.getenv('USEAPI_TOKEN')
IMAGE_KEY = os.getenv('IMAGE_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
USEAPI_SERVER = os.getenv('USEAPI_SERVER')
USEAPI_CHANNEL = os.getenv('USEAPI_CHANNEL')
USEAPI_DISCORD = os.getenv('USEAPI_DISCORD')
NGROK_AUTHTOKEN = os.getenv('NGROK_AUTHTOKEN')
rootUrl = 'https://api.useapi.net/v2'

reply_url = ''

tasks_status = {}

semaphore_count = 0

start_time = None
tasks_completed = 0

# Настройка базы данных
DATABASE_URL = "sqlite+aiosqlite:///./test.db"  # Асинхронный драйвер для SQLite
Base = declarative_base()

# Определение модели
class Describedb(Base):
    __tablename__ = 'describedb'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    jobid = Column(String(50))
    replyRef = Column(String(50))
    describe_url = Column(String(255))
    content = Column(String)

class ImagineSingle(Base):
    __tablename__ = 'imagine_single'
    id = Column(Integer, primary_key=True)
    recorddate = Column(DateTime, default=datetime.datetime.utcnow)
    jobid = Column(String(50))
    verb = Column(String(10))
    status = Column(String(10))
    created = Column(String)
    updated = Column(String)
    prompt = Column(String(255))
    buttons = Column(PickleType())
    attachments = Column(JSON, default=[])
    discord = Column(String(50))
    channel = Column(String(50))
    server = Column(String(50))
    maxJobs = Column(Integer)
    replyUrl = Column(String(255))
    replyRef = Column(String(50))
    messageId = Column(String(50))
    content = Column(Text)
    timestamp = Column(Text)
    filename = Column(String)
    url = Column(String)
    size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    button_state = relationship('ButtonState', backref='imagine_single', uselist=False, cascade="all, delete-orphan")
    urlresized = Column(Integer)
    author = Column(Integer, default='ilia')
    seed = Column(Integer, default='')

class ButtonState(Base):
    __tablename__ = 'button_state'
    id = Column(Integer, primary_key=True)
    imagine_single_id = Column(Integer, ForeignKey('imagine_single.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    button_1 = Column(Boolean, default=False)
    button_2 = Column(Boolean, default=False)
    button_3 = Column(Boolean, default=False)
    button_4 = Column(Boolean, default=False)
    button_v1 = Column(Boolean, default=False)
    button_v2 = Column(Boolean, default=False)
    button_v3 = Column(Boolean, default=False)
    button_v4 = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    upscaled_images = relationship('UpscaledImage', backref='button_state', uselist=False)

class UpscaledImage(Base):
    __tablename__ = 'upscaled_image'
    id = Column(Integer, primary_key=True)
    button_state_id = Column(Integer, ForeignKey('button_state.id'))
    created = Column(String)
    updated = Column(String)
    jobid = Column(String(255), default='')
    url = Column(String(255), default='')
    button = Column(PickleType())
    buttons = Column(PickleType())
    button4xstate = Column(Boolean, default=False)
    attachments = Column(JSON, default=[])
    discord = Column(String(50))
    channel = Column(String(50))
    server = Column(String(50))
    maxJobs = Column(Integer)
    replyUrl = Column(String(255))
    replyRef = Column(String(50))
    messageID = Column(String(50))
    upscaled_4x_image = relationship('Upscaled4xImage', backref='upscaled_image', uselist=False)
    parentJobId = Column(String(255), default='')
    timestamp = Column(String(255), default='')
    filename = Column(String)
    size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    small_image_url = Column(Integer, default='')
    author = Column(Integer, default='ilia')
    seed = Column(Integer, default='')

class Upscaled4xImage(Base):
    __tablename__ = 'upscaled4x_image'
    id = Column(Integer, primary_key=True)
    upscaled_image_id = Column(Integer, ForeignKey('upscaled_image.id'))
    created = Column(String)
    updated = Column(String)
    jobid = Column(String(255), default='')
    url = Column(String(255), default='')
    button = Column(PickleType())
    buttons = Column(PickleType())
    attachments = Column(JSON, default=[])
    discord = Column(String(50))
    channel = Column(String(50))
    server = Column(String(50))
    maxJobs = Column(Integer)
    replyUrl = Column(String(255))
    replyRef = Column(String(50))
    messageID = Column(String(50))
    parentJobId = Column(String(255), default='')
    timestamp = Column(String(255), default='')
    filename = Column(String)
    size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    small_image_url = Column(Integer, default='')
    author = Column(Integer, default='ilia')


# Создаем асинхронный движок и сессию echo=True
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

#task_queue = asyncio.Queue()  # Очередь для хранения всех задач
#semaphore = asyncio.Semaphore(3)  # Максимум 3 задачи могут выполняться одновременно



async def get_reply_url():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:4040/api/tunnels") as response:
            tunnel_url = await response.text()
            j = json.loads(tunnel_url)
            reply_url = j['tunnels'][0]['public_url']
    return reply_url

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

class AsyncFunctionQueue:
    def __init__(self, concurrency_limit=3):
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.events = {}  # Словарь для отслеживания состояний задач

    async def enqueue(self, fn, *args, replyRef):
        await self.queue.put((fn, args, replyRef))
        # Создаём Event для каждой задачи и сохраняем его в словаре
        self.events[replyRef] = asyncio.Event()

    async def process_queue(self):
        while True:
            fn, args, replyRef = await self.queue.get()
            async with self.semaphore:
               # asyncio.create_task(self.run_task(fn, args, replyRef))
                await self.run_task(fn, args, replyRef)
                self.queue.task_done()

    async def run_task(self, fn, args, replyRef):
        print(f"RUN TASK replyRef== {replyRef}")
        await fn(*args, replyRef)
        # Ждём сигнала от вебхука
        await self.events[replyRef].wait()
        # Удаляем Event из словаря после получения сигнала
        del self.events[replyRef]


task_queue = AsyncFunctionQueue(concurrency_limit=3)

async def start_queue_processing():
    await task_queue.process_queue()

@app.before_serving
async def startup():
    global reply_url
    reply_url = await get_reply_url()
    if reply_url:
        print(f"Webhook reply URL: {reply_url}")
    else:
        print("ngrok туннель не найден.")
    async with engine.begin() as conn:
        #await conn.run_sync(UpscaledImage.__table__.drop)
        #await conn.run_sync(Upscaled4xImage.__table__.drop)
        await conn.run_sync(Base.metadata.create_all)
    asyncio.create_task(task_queue.process_queue())


@app.after_serving
async def shutdown():
    await engine.dispose()



@app.route('/webhook', methods=['POST'])
async def webhook():
    data = await request.get_json()
    replyRef = data.get('replyRef')
    status = data.get('status')
    verb = data.get('verb')

    if status == 'completed':
        if replyRef in task_queue.events:
            # Отправляем сигнал о завершении задачи
            task_queue.events[replyRef].set()
            if verb == 'describe':
                await handle_describe(data)
            elif verb == 'imagine':
                await handle_imagine(data)
            elif verb == 'button':
                await handle_button(data)
    elif status in ('moderated', 'failed', 'cancelled', 'progress', 'started'):
            jobid = data['jobid']
            content = data.get('content', '')
            matches = re.findall(r'\((.*?)\)', content)
            print(f'JobId {jobid} ReplyRef {replyRef} на сервере API в состоянии: {matches}')
    
    else:
        print(f"ReplyRef {replyRef} неизвестный статус")

    return 'Webhook received and processed', 200


async def handle_describe(data):
    jobid = data.get('jobid')
    replyRef = data.get('replyRef')
    describe_url = data.get('describeUrl')
    content = data.get('content')
    db_entry = Describedb(
        jobid=jobid, 
        replyRef=replyRef, 
        describe_url=describe_url, 
        content=content
    )
    async with async_session() as session:
        session.add(db_entry)
        await session.commit()

async def handle_imagine(data):
    jobid = data.get('jobid')
    created = data.get('created')
    updated = data.get('updated')
    prompt = data.get('prompt')
    buttons = data.get('buttons')
    attachments = data.get('attachments')
    replyRef = data.get('replyRef')
    messageId = data.get('messageId')
    content = data.get('content')
    timestamp = data.get('timestamp')
    for attachment in attachments:
        filename = attachment.get('filename')
        url = attachment.get('url')
        size = attachment.get('size')
        width = attachment.get('width')
        height = attachment.get('height') 
        small_image_url = await process_imagekit(url, filename)

    new_imagine = ImagineSingle(
        jobid=jobid,
        created=created,
        updated=updated,
        prompt=prompt,
        buttons=buttons,
        replyRef=replyRef,
        messageId=messageId,
        content=content,
        timestamp=timestamp,
        filename=filename,
        url=url,
        size=size,
        width=width,
        height=height,
        urlresized=small_image_url,
    )

    new_button_state = ButtonState(
        imagine_single=new_imagine,
        button_1=False,
        button_2=False,
        button_3=False,
        button_4=False,
        button_v1=False,
        button_v2=False,
        button_v3=False,
        button_v4=False,
    )

    async with async_session() as session:
        session.add(new_imagine)
        session.add(new_button_state)
        await session.commit()

async def handle_button(data):
    button = data.get('button')
    replyRef = data.get('replyRef')
    content = data.get('content')
    attachments = data.get('attachments', [])
    for attachment in attachments:
        filename = attachment.get('filename')
        url = attachment.get('url')
        size = attachment.get('size')
        width = attachment.get('width')
        height = attachment.get('height') 
        small_image_url = await process_imagekit(url, filename)
        if button in ['U1', 'U2', 'U3', 'U4']:
            await add_upscaled_image(data, attachment, small_image_url, 'images1x', UpscaledImage)
        elif button in ['Upscale (4x)']:
            await add_upscaled_image(data, attachment, small_image_url, 'images4x', Upscaled4xImage)
        elif button in ['V1', 'V2', 'V3', 'V4']:
            await handle_imagine(data)

async def add_upscaled_image(data, attachment, small_image_url, folder, ImageModel):
    new_upscaled = ImageModel(
        jobid=data['jobid'],
        created=data.get('created', ''),
        updated=data.get('updated', ''),
        parentJobId=data.get('parentJobId'),
        button=data.get('button'),
        buttons=data.get('buttons', []),
        attachments=data.get('attachments', []),
        discord=data.get('discord', ''),
        channel=data.get('channel', ''),
        server=data.get('server', ''),
        maxJobs=data.get('maxJobs', ''),
        replyUrl=data.get('replyUrl', ''),
        replyRef=data.get('replyRef', ''),
        messageID=data.get('messageId'),
        timestamp=data.get('timestamp', ''),
        filename=attachment.get('filename', ''),
        url=attachment.get('url', ''),
        size=attachment.get('size', ''),
        width=attachment.get('width', ''),
        height=attachment.get('height', ''),
        small_image_url=small_image_url
    )
    async with async_session() as session:
        session.add(new_upscaled)
        await session.commit()
    await download_image(attachment.get('url', ''), attachment.get('filename', ''), folder)



async def handle_task(api_request_data, verb, replyRef):
    try:
        if verb == 'describe':
            response, _ = await send_describe_request(api_request_data)
        elif verb == 'imagine':
            response, _ = await send_imagine_request(api_request_data)
        elif verb == 'button':
            response, _ = await send_pushbutton_request(api_request_data)
        else:
            raise ValueError(f"Неизвестный тип задачи: {verb}")

        if response.status_code == 429 or response.status_code == 504:
            print(f"Ошибка {response.status_code}, планируем повторную попытку...")
            await asyncio.sleep(10 if response.status_code == 429 else 3 * 60)
            await task_queue.enqueue(handle_task, api_request_data, verb, replyRef)
        elif response.status_code != 200:
            print(f'Failed to submit request: {response.text}')
    except Exception as e:
        print(f'Ошибка при выполнении задачи: {e}')

    
async def send_describe_request(data_json):
    api_url = "https://api.useapi.net/v2/jobs/describe"
    headers = {
        'Authorization': f'Bearer {USEAPI_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    describe_url = data_json.get('describeUrl', '')
    discord_token = data_json.get('discord', USEAPI_DISCORD)
    server_id = data_json.get('server', USEAPI_SERVER)
    channel_id = data_json.get('channel', USEAPI_CHANNEL)
    reply_ref = data_json.get('replyRef')

    webhook_reply_url = f"{reply_url}/webhook"

    data = {
        "describeUrl": describe_url,
        "discord": discord_token,
        "server": server_id,
        "channel": channel_id,
        "maxJobs": 3,
        "replyUrl": webhook_reply_url,
        "replyRef": reply_ref
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=data, timeout=60.0)
    
    return response, reply_ref  # Возврат и ответа API, и reply_ref

async def send_imagine_request(data_json):
    api_url = "https://api.useapi.net/v2/jobs/imagine"
    headers = {
        'Authorization': f'Bearer {USEAPI_TOKEN}',
        'Content-Type': 'application/json'
    }

    prompt = data_json.get('prompt', '')
    discord_token = data_json.get('discord', USEAPI_DISCORD)
    server_id = data_json.get('server', USEAPI_SERVER)
    channel_id = data_json.get('channel', USEAPI_CHANNEL)
    reply_ref = data_json.get('replyRef')
    webhook_reply_url = f"{reply_url}/webhook"

    data = {
        "prompt": prompt,
        "discord": discord_token,
        "server": server_id,
        "channel": channel_id,
        "maxJobs": 3,
        "replyUrl": webhook_reply_url,
        "replyRef": reply_ref
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=data, timeout=60.0)
    
    return response, reply_ref  # Возврат и ответа API, и reply_ref

async def send_pushbutton_request(data_json):
    api_url = "https://api.useapi.net/v2/jobs/button"
    headers = {
        'Authorization': f'Bearer {USEAPI_TOKEN}',
        'Content-Type': 'application/json'
    }
    jobid = data_json.get('jobid', '')
    button = data_json.get('button', '')
    prompt = data_json.get('prompt', '')
    discord_token = data_json.get('discord', USEAPI_DISCORD)
    reply_ref = data_json.get('replyRef')
    webhook_reply_url = f"{reply_url}/webhook"

    data = {
        "jobid": jobid,
        "button": button,
        "prompt": prompt,
        "discord": discord_token,
        "maxJobs": 3,
        "replyUrl": webhook_reply_url,
        "replyRef": reply_ref
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=data, timeout=60.0)
    
    return response, reply_ref  # Возврат и ответа API, и reply_ref


    

@app.route('/describe', methods=['POST'])
async def submit_describe_request():
    api_request_data = await request.json if request.is_json else await request.form.to_dict()
    api_request_data['verb'] = 'describe'  # Указываем тип задачи
    replyRef = str(uuid.uuid4())
    api_request_data['replyRef'] = replyRef

    await task_queue.enqueue(handle_task, api_request_data, 'describe', replyRef=replyRef)
    print(f"Задача 'describe' добавлена в очередь: {api_request_data}")
    return jsonify({'message': 'Describe request submitted to the queue'})

@app.route('/imagine', methods=['POST'])
async def submit_imagine_request():
    api_request_data = await request.json if request.is_json else await request.form.to_dict()
    api_request_data['verb'] = 'imagine'
    replyRef = str(uuid.uuid4())
    api_request_data['replyRef'] = replyRef

    await task_queue.enqueue(handle_task, api_request_data, 'imagine', replyRef=replyRef)
    print(f"Задача 'imagine' добавлена в очередь: {api_request_data}")

    return jsonify({'message': 'Imagine request submitted to the queue'})


@app.route('/pushbutton', methods=['POST'])
async def submit_pushbutton_request():
    api_request_data = await request.json if request.is_json else await request.form.to_dict()
    jobid = api_request_data.get('jobid', '')
    button = api_request_data.get('button', '')
    replyRef = str(uuid.uuid4())
    api_request_data['replyRef'] = replyRef

    print(f"Received pushbutton request with jobid: {jobid} and button: {button}")

    async with async_session() as session:
        print(f"Searching for ImagineSingle record with jobid: {jobid}")
        if button in ['U1', 'U2', 'U3', 'U4', 'V1', 'V2', 'V3', 'V4']:
            result = await session.execute(select(ImagineSingle).options(joinedload(ImagineSingle.button_state)).filter(ImagineSingle.jobid == jobid))
            imagine_record = result.scalars().first()
            if imagine_record:
                button_attr = f'button_{button[1:]}' if button.startswith('U') else f'button_v{button[1:]}'
                await update_button_state(session, imagine_record.button_state, button_attr)
                print(f"Updated button state for {button_attr}")
        elif button == 'Upscale (4x)':
            result = await session.execute(select(UpscaledImage).filter(UpscaledImage.jobid == jobid))
            upscaled_image_record = result.scalars().first()
            if upscaled_image_record:
                upscaled_image_record.button4xstate = True
                await session.commit()
                print(f"Кнопка Upscale 4x обновлена для jobid: {jobid}")
            else:
                print(f"Запись UpscaledImage с jobid: {jobid} не найдена")

    await task_queue.enqueue(handle_task, api_request_data, 'button', replyRef=replyRef)
    return jsonify({'message': 'Button request submitted'})

async def update_button_state(session, button_state_record, button_attr, state=True):
    if button_state_record:
        setattr(button_state_record, button_attr, state)
        await session.commit()
    else:
        print(f'Запись для кнопки {button_attr} не найдена')


@app.route('/imagestable')
async def images_table():
    async with async_session() as session:
        result = await session.execute(select(ImagineSingle).options(joinedload(ImagineSingle.button_state)))
        imagines = result.scalars().all()

    for imagine in imagines:
        button_state = imagine.button_state
        if button_state:
            imagine.button_states = {
                'U1': 'true' if button_state.button_1 else 'false',
                'U2': 'true' if button_state.button_2 else 'false',
                'U3': 'true' if button_state.button_3 else 'false',
                'U4': 'true' if button_state.button_4 else 'false',
                'V1': 'true' if button_state.button_v1 else 'false',
                'V2': 'true' if button_state.button_v2 else 'false',
                'V3': 'true' if button_state.button_v3 else 'false',
                'V4': 'true' if button_state.button_v4 else 'false',
            }
        else:
            imagine.button_states = {'U1': 'false', 'U2': 'false', 'U3': 'false', 'U4': 'false', 'V1': 'false', 'V2': 'false', 'V3': 'false', 'V4': 'false'}

    return await render_template('imagindb.html', imagines=imagines)

@app.route('/process-json', methods=['POST'])
async def process_json():
    data = await request.json
    total_tasks = len(data)
    print(f"Получено задач из JSON: {total_tasks}")

    for item in data:
        prompt = item if isinstance(item, str) else item.get('prompt', '')
        if prompt:
            api_request_data = {'prompt': prompt, 'verb': 'imagine'}
            await task_queue.put(api_request_data)
            print(f"Задача 'imagine' добавлена в очередь: {api_request_data}, Задач в очереди: {task_queue.qsize()}")
            await asyncio.sleep(3)  # Добавление задержки

    return jsonify({'message': f'Processed {total_tasks} items'})

# Остальные части вашего приложения остаются без изменений


@app.route('/')
async def index():
    return await render_template('describe_form2.html')

@app.route('/batch')
async def batch_uploadj():
    return await render_template('batch.html')

@app.route('/descriptions')
async def show_descriptions():
    async with async_session() as session:
        result = await session.execute(select(Describedb))
        entries = result.scalars().all()
    
    return await render_template('descriptions.html', entries=entries)

@app.route('/button-states')
async def show_button_states():
    async with async_session() as session:
        result = await session.execute(select(ButtonState))
        button_states = result.scalars().all()

    return await render_template('button_states.html', button_states=button_states)

@app.route('/upscaled-images_button')
async def upscaled_images_button():
    async with async_session() as session:
        result = await session.execute(select(UpscaledImage))
        images = result.scalars().all()
    return await render_template('upscaled_images_button.html', images=images)

@app.route('/UpscaledImage-states')
async def show_upbutton_states():
    async with async_session() as session:
        result = await session.execute(select(UpscaledImage))
        images = result.scalars().all()

        for image in images:
            # Установка значения состояния кнопки Upscale (4x)
            image.button_4xstate = 'true' if image.button4xstate else 'false'
            
    return await render_template('UpscaledImages.html', images=images)

@app.route('/upscaled4x-images')
async def show_upscaled4x_images():
    async with async_session() as session:
        # Выполняем запрос к базе данных с фильтрацией по автору
        result = await session.execute(
            select(Upscaled4xImage).where(Upscaled4xImage.author == 'ilia')
        )
        images = result.scalars().all()

    return await render_template('upscaled4x_images.html', images=images)


if __name__ == '__main__':
    
    app.run()
