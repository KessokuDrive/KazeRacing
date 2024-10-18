from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import APIRouter
from app.face_module.FacePerception.FaceRecognition import faceRecognitionByByte

router = APIRouter()

@router.post("/upload/", tags=["upload"])
async def create_upload_files(request: Request):
    image_data = b""  # 初始化一个空字节流，用于存储图片

    async for chunk in request.stream():
        image_data += chunk
    
    name = faceRecognitionByByte(image_data)  #我不知道faceRecognitionByByte那个里面具体逻辑，先写着
    
    if name is not None:
        #anti-spoofing program
        #...
        #if anti-spoofing program is true:
        return {"name": name}
        #else:
        #return JSONResponse(status_code=400, content={"message": "Face is spoofed"})
    else:
        return JSONResponse(status_code=404, content={"message": "Face not detected"})
