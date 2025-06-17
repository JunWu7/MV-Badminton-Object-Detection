# -*- coding=utf-8 -*-
__author__ = "ITRI-EOSL-R300 A30335-Rachel"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import glob
from typing import List
import uvicorn
from datetime import datetime
import traceback
from ntut_yolo_inf import YOLO_inf

app = FastAPI()

# 使用當前腳本的目錄作為基礎目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(current_dir, "YOLO_uploads")
OUTPUT_DIR = os.path.join(current_dir, "YOLO_outputs")

# 確保目錄存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = 'NTUT_YOLO.pt'
MODEL_P = os.path.join(UPLOAD_DIR, MODEL)


@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    try:
        model_file_path = os.path.normpath(os.path.join(UPLOAD_DIR, MODEL))
        print(f"上傳模型到: {model_file_path}")

        # 確保上傳目錄存在
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

        # 如果模型已存在，先刪除
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

        # 保存上傳的模型
        with open(model_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 檢查文件是否成功保存
        if os.path.exists(model_file_path):
            print(f"模型成功保存到: {model_file_path}")
        else:
            print(f"模型保存失敗!")

        return {"message": "模型上傳成功", "model": MODEL}
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"上傳模型出錯: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=f"上傳模型失敗: {str(e)}")


@app.post("/upload-img/{modelp}")
async def upload_img(modelp: str, file: UploadFile = File(...)):
    try:
        # 創建唯一的時間戳目錄名稱
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_without_ext = os.path.splitext(file.filename)[0]
        baseF = f"{timestamp}_{filename_without_ext}"
        dst_folder_new = os.path.normpath(os.path.join(OUTPUT_DIR, baseF))

        print(f"處理圖片: {file.filename}, 模型: {modelp}, 儲存目錄: {dst_folder_new}")

        # 確保目錄存在
        os.makedirs(dst_folder_new, exist_ok=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 保存上傳的圖片
        img_file_path = os.path.normpath(os.path.join(UPLOAD_DIR, file.filename))
        with open(img_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"圖片已保存到: {img_file_path}")

        # 檢查模型文件是否存在
        if not os.path.exists(MODEL_P):
            return {"error": f"模型檔案不存在: {MODEL_P}"}

        # 檢查圖片文件是否存在
        if not os.path.exists(img_file_path):
            return {"error": f"圖片檔案不存在: {img_file_path}"}

        # 執行YOLO推論
        print(f"執行YOLO推論: 模型={MODEL_P}, 圖片={img_file_path}, 儲存目錄={dst_folder_new}")
        try:
            summary_file_path, time_path, gpu_info_path = YOLO_inf(
                model_path=MODEL_P,
                img_path=img_file_path,
                savedir=dst_folder_new,
                device="cuda:0"
            )
            print(f"YOLO推論完成，結果儲存在: {dst_folder_new}")
        except Exception as yolo_e:
            traceback_str = traceback.format_exc()
            print(f"YOLO推論出錯: {str(yolo_e)}\n{traceback_str}")
            return {"error": f"YOLO推論失敗: {str(yolo_e)}"}

        # 獲取處理後的圖片
        processed_images = [os.path.normpath(i) for i in glob.glob(os.path.join(dst_folder_new, "*.jpg"))]
        print(f"找到處理後的圖片: {len(processed_images)} 張")

        return {
            "message": "IMAGE 處理成功",
            "filename": file.filename,
            "processed_images": processed_images,
            "summary_file_path": os.path.normpath(summary_file_path),
            "time_path": os.path.normpath(time_path),
            "baseF": baseF
        }
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"上傳圖片出錯: {str(e)}\n{traceback_str}")
        return {"error": str(e)}


@app.get("/download-image/{file_name}/{dir}")
async def download_image(file_name: str, dir: str):
    try:
        image_path = os.path.normpath(os.path.join(OUTPUT_DIR, dir, file_name))
        print(f"請求下載文件: {image_path}")

        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            print(f"請求的文件不存在: {image_path}")
            return {"error": f"檔案不存在: {image_path}"}
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"下載圖片出錯: {str(e)}\n{traceback_str}")
        return {"error": str(e)}


if __name__ == "__main__":
    print(f"啟動伺服器...")
    print(f"上傳目錄: {UPLOAD_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"模型路徑: {MODEL_P}")
    uvicorn.run(app, host="127.0.0.1", port=8070)