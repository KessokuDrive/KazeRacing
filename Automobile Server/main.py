from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from PIL import Image
import pillow_heif
from io import BytesIO
import cv2
app = FastAPI()

IMAGE_LOCATION = "./Image/"

@app.get("/")
def read_root():
    return {"Attention":"009"}

@app.post("/uploadfile/")
async def create_upload_file(request: Request):
    try:
        image_data = b""  # 初始化一个空字节流，用于存储图片
        async for chunk in request.stream():
            image_data += chunk
        #request.j
        if request.headers['Content-Type'] == 'image/avif':
            image = pillow_heif.read_heif(BytesIO(image_data))
            image.save("./" + IMAGE_LOCATION + request.headers['filename'])
            return {"message": "File uploaded successfully"}
        else:
            image = Image.open(BytesIO(image_data))
            image.save("./" + IMAGE_LOCATION + request.headers['filename'])
            return {"message": "Error: File format not supported"}
    except Exception:
        return {"message": "There was an error uploading the file"}
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        heif_file = pillow_heif.from_bytes(
            mode="BGR",
            size=(800, 320),
            data=data
        )
        heif_file.save("output.avif",format="AVIF", quality=75)
        await websocket.send_text(f"Message Recieved")

"""@app.post("/uploadfile/")
async def create_file( request: Request):
    file = await request.json()
    chunk_size = 1024*1024
    img_bytes = base64.b64decode(file["image"].encode("utf-8"))
    img_arr = np.asarray(img_bytes)
    try:
        with open("output.png", mode='xb') as f:
            chunk = await img_bytes.read(chunk_size)
            if not chunk:
                return    
            f.write(chunk)
    finally:
        await request.is_disconnected()
    return {"message": "File uploaded successfully"}"""
    
import aiofiles
#Base 64 attempt
"""
@app.post('/uploadfile/')
async def upload(request: Request):
    try:
        filename = request.headers['filename']
        #img_bytes = base64.b64decode(file["image"].encode("utf-8"))
        async with aiofiles.open(filename, 'wb') as f:
            async for chunk in request.stream(): 
                await f.write(chunk)
    except Exception:
        return {"message": "There was an error uploading the file"}
     
    return {"message": f"Successfuly uploaded "}
"""

"""
def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/uploadfile/")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    print(image.data)
    return {"Hello": "World"}
"""