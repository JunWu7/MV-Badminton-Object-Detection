####################################################################################
#############################################################
# -*- coding=utf-8 -*-
__author__ = "ITRI-EOSL-R300 A30335-Rachel"

# NTUT_2025 CV
# env: teacher_cv
# cd /home/ntut/Rachel/course/lab_0317/RTDETR
##############################################################
from ultralytics import YOLO
import os
import time
from glob import glob
import shutil
import torch
import json


def get_gpu_info():
    """獲取GPU資訊"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["current_device"] = torch.cuda.current_device()
        gpu_info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        gpu_info["available"] = False

    return gpu_info


def moveReulst(savedir):
    imglist = glob(os.path.join(savedir, '**/*.jpg'), recursive=True)
    for i in imglist:
        if ('detect' in i):
            pathname = os.path.basename(i)
            if not pathname.startswith('detection_'):
                pathname = 'detection_' + pathname
            shutil.copy2(i, os.path.join(savedir, pathname))

    # 小心地移除檔案夾，避免刪除根目錄
    for i in imglist:
        if ('detect' in i):
            dir_to_remove = os.path.dirname(i)
            # 確保不是刪除根目錄或savedir本身
            if dir_to_remove != savedir and os.path.exists(dir_to_remove) and len(dir_to_remove) > 5:
                try:
                    shutil.rmtree(dir_to_remove)
                except Exception as e:
                    print(f"無法刪除目錄 {dir_to_remove}: {e}")


def YOLO_inf(model_path="yolov8n_best.pt",
             img_path="test_image.png",
             savedir="yolo_results",
             device="cuda:0"):
    """
    執行YOLO推論

    Args:
        model_path: 模型路徑
        img_path: 圖片路徑
        savedir: 儲存結果的目錄
        device: 要使用的裝置 ('cpu', 'cuda:0', 'cuda:1', 等 或 'auto' 自動選擇)

    Returns:
        summary_file_path: 結果摘要檔案路徑
        time_path: 時間記錄檔案路徑
        gpu_info_path: GPU資訊檔案路徑
    """
    # 打印參數以便除錯
    print(f"YOLO_inf函數參數: model_path={model_path}, img_path={img_path}, savedir={savedir}, device={device}")

    # 確保所有路徑是正規化的
    model_path = os.path.normpath(model_path)
    img_path = os.path.normpath(img_path)
    savedir = os.path.normpath(savedir)

    # 確認檔案存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"圖片檔案不存在: {img_path}")

    # 確保存儲目錄存在
    os.makedirs(savedir, exist_ok=True)

    t1 = time.time()
    # 載入模型並指定設備
    try:
        model = YOLO(model_path)
        print(f"成功載入模型: {model_path}")
    except Exception as e:
        print(f"載入模型時出錯: {e}")
        raise

    # 獲取GPU資訊
    gpu_info = get_gpu_info()

    # 設定設備
    if device != "auto":
        model.to(device)

    t2 = time.time()

    source = img_path

    # 預測時指定設備
    try:
        results = model.predict(source, save=True, project=savedir, name="detection", device=device)
        print(f"模型預測完成，結果儲存在: {savedir}")
    except Exception as e:
        print(f"執行預測時出錯: {e}")
        raise

    t3 = time.time()
    print('-' * 30)
    print(results)

    # 確保儲存目錄存在
    os.makedirs(savedir, exist_ok=True)

    # 儲存GPU資訊
    gpu_info_path = os.path.join(savedir, 'gpu_info.json')
    with open(gpu_info_path, 'w') as gpu_file:
        used_device = device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        gpu_info["used_device"] = used_device
        json.dump(gpu_info, gpu_file, indent=2)

    summary_file_path = ""

    try:
        for result in results:
            print('-' * 30, 'json')
            json_result = result.to_json()
            print(json_result)
            print('-' * 30, 'summary')
            summary = result.summary()
            print(summary)
            summary_file_path = os.path.join(savedir, 'result.json')
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write(str(summary))
            print('-' * 30, 'result')
            print(result)
            try:
                # 直接在savedir中儲存裁剪結果
                result.save_crop(save_dir=savedir)
                print(f"裁剪結果已儲存至: {savedir}")
            except Exception as crop_e:
                print(f"儲存裁剪結果時出錯: {crop_e}")
    except Exception as e:
        print(f"處理結果時出錯: {e}")

    time_path = os.path.join(savedir, 'time.txt')
    tts1 = f'load model speed: {(t2 - t1) * 1000} ms'
    tts2 = f'Inference speed: {(t3 - t2) * 1000} ms'
    tts3 = f'print summary speed: {(time.time() - t3) * 1000} ms'
    with open(time_path, 'w') as f:
        f.write(f'modelpath={model_path},\nimg_path={img_path},\ndevice={device},\n{tts1},\n{tts2},\n{tts3}')

    print(tts1)
    print(tts2)
    print(tts3)

    # 移動結果
    try:
        moveReulst(savedir)
    except Exception as e:
        print(f"移動結果時出錯: {e}")

    return summary_file_path, time_path, gpu_info_path


if __name__ == "__main__":
    # 簡單測試
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_savedir = os.path.join(current_dir, "yolo_test_results")

    try:
        YOLO_inf(
            model_path=os.path.join(current_dir, "yolov8n_best.pt"),
            img_path=os.path.join(current_dir, "test_image.jpg"),
            savedir=test_savedir,
            device="auto"
        )
    except Exception as e:
        print(f"測試時出錯: {e}")