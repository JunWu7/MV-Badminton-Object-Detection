# -*- coding=utf-8 -*-
__author__ = "Group 1"

import gradio as gr
import requests
import os
import json
import time
import re
from typing import Dict, List, Tuple
import shutil
from PIL import Image
import io
import cv2
from pathlib import Path
import ffmpeg
import imageio_ffmpeg


class MultiGPU_ServiceClient:
    def __init__(self):
        self.servers = {
            "Local GPU": {"host": "127.0.0.1", "port": 8010},
            "Tesla T4": {"host": "127.0.0.1", "port": 8070},
            "AMD 7900": {"host": "140.124.181.195", "port": 8010}
        }

        # 創建工作目錄
        self.workspace_dir = "./gradio_workspace"
        self.upload_dir = os.path.join(self.workspace_dir, "uploads")
        self.download_dir = os.path.join(self.workspace_dir, "downloads")

        for dir_path in [self.workspace_dir, self.upload_dir, self.download_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_base_url(self, server_name: str) -> str:
        server_info = self.servers[server_name]
        return f"http://{server_info['host']}:{server_info['port']}"

    def validate_and_process_image(self, image_path: str) -> str:
        """驗證並處理圖片"""
        try:
            # 嘗試打開圖片檢查是否有效
            with Image.open(image_path) as img:
                # 檢查圖片格式
                if img.format not in ['JPEG', 'JPG', 'PNG', 'BMP']:
                    # 轉換為JPEG格式
                    processed_path = image_path.replace(os.path.splitext(image_path)[1], '_processed.jpg')
                    img.convert('RGB').save(processed_path, 'JPEG')
                    return processed_path

                # 檢查圖片尺寸，如果太大就縮小
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    processed_path = image_path.replace(os.path.splitext(image_path)[1], '_resized.jpg')
                    img.save(processed_path, 'JPEG')
                    return processed_path

            return image_path
        except Exception as e:
            print(f"圖片處理錯誤: {str(e)}")
            return None

    def upload_model(self, server_name: str, model_file_path: str) -> dict:
        try:
            if not os.path.exists(model_file_path):
                return {"error": f"模型檔案不存在: {model_file_path}"}

            base_url = self.get_base_url(server_name)

            with open(model_file_path, "rb") as f:
                files = {"file": (os.path.basename(model_file_path), f)}
                response = requests.post(f"{base_url}/upload-model/", files=files, timeout=30)

            return response.json()
        except Exception as e:
            return {"error": f"上傳模型失敗: {str(e)}"}

    def upload_img(self, server_name: str, img_file_path: str, model_name: str) -> dict:
        try:
            if not os.path.exists(img_file_path):
                return {"error": f"圖片檔案不存在: {img_file_path}"}

            base_url = self.get_base_url(server_name)

            with open(img_file_path, "rb") as f:
                files = {"file": (os.path.basename(img_file_path), f)}
                response = requests.post(f"{base_url}/upload-img/{model_name}", files=files, timeout=60)

            return response.json()
        except Exception as e:
            return {"error": f"上傳圖片失敗: {str(e)}"}

    def download_file(self, server_name: str, file_name: str, base_folder: str) -> str:
        try:
            base_url = self.get_base_url(server_name)
            response = requests.get(f"{base_url}/download-image/{file_name}/{base_folder}", stream=True, timeout=30)

            if response.status_code == 200:
                local_path = os.path.join(self.download_dir, base_folder, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return local_path
            else:
                return None
        except Exception as e:
            print(f"下載檔案錯誤: {str(e)}")
            return None


class YOLO_MultiGPU_UI:
    def __init__(self):
        self.client = MultiGPU_ServiceClient()

    def parse_time_info(self, time_content: str) -> str:
        """解析時間統計，格式化顯示"""
        try:
            if not time_content:
                return "無時間資訊"

            # 提取三個主要時間指標
            load_speed = re.search(r'load model speed:\s*([\d.]+)\s*ms', time_content)
            inference_speed = re.search(r'Inference speed:\s*([\d.]+)\s*ms', time_content)
            summary_speed = re.search(r'print summary speed:\s*([\d.]+)\s*ms', time_content)

            result = "⏱️ 執行時間統計\n" + "=" * 30 + "\n"

            if load_speed:
                result += f"🔄 模型載入時間: {float(load_speed.group(1)):.2f} ms\n"
            else:
                result += f"🔄 模型載入時間: N/A\n"

            if inference_speed:
                result += f"🚀 推理執行時間: {float(inference_speed.group(1)):.2f} ms\n"
            else:
                result += f"🚀 推理執行時間: N/A\n"

            if summary_speed:
                result += f"📝 結果處理時間: {float(summary_speed.group(1)):.2f} ms\n"
            else:
                result += f"📝 結果處理時間: N/A\n"

            # 計算總時間
            if load_speed and inference_speed and summary_speed:
                total_time = float(load_speed.group(1)) + float(inference_speed.group(1)) + float(
                    summary_speed.group(1))
                result += f"\n⚡ 總處理時間: {total_time:.2f} ms"

            return result

        except Exception as e:
            return f"時間解析錯誤: {str(e)}"

    def parse_object_statistics(self, result_json_content: str) -> str:
        """解析物件統計，包含數量和平均置信度"""
        try:
            if not result_json_content:
                return "無檢測結果"

            # 清理輸入數據，移除首尾空白字符
            cleaned_content = result_json_content.strip()

            # 嘗試解析JSON數據
            detection_results = None

            # 方法1: 如果已經是字典或列表，直接使用
            if isinstance(cleaned_content, (dict, list)):
                detection_results = cleaned_content
            else:
                # 方法2: 嘗試JSON解析
                try:
                    detection_results = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    # 方法3: 嘗試使用eval（僅當內容看起來安全時）
                    try:
                        # 檢查內容是否看起來像Python字典/列表格式
                        if cleaned_content.startswith('[') and cleaned_content.endswith(']'):
                            detection_results = eval(cleaned_content)
                        else:
                            return f"JSON解析錯誤: {str(e)}\n原始內容前100字符:\n{cleaned_content[:100]}"
                    except Exception as eval_error:
                        return f"數據解析失敗:\nJSON錯誤: {str(e)}\nEval錯誤: {str(eval_error)}\n原始內容:\n{cleaned_content[:200]}"

            # 驗證解析結果
            if detection_results is None:
                return "解析結果為空"

            if not isinstance(detection_results, list):
                return f"期望列表格式，但獲得: {type(detection_results)}"

            if not detection_results:
                return "未檢測到任何物件"

            # 統計每個類別的物件
            class_stats = {}

            for i, detection in enumerate(detection_results):
                try:
                    # 確保detection是字典
                    if not isinstance(detection, dict):
                        continue

                    class_name = detection.get('name', 'unknown')
                    confidence = detection.get('confidence', 0.0)

                    # 確保confidence是數字
                    if not isinstance(confidence, (int, float)):
                        try:
                            confidence = float(confidence)
                        except (ValueError, TypeError):
                            confidence = 0.0

                    if class_name not in class_stats:
                        class_stats[class_name] = {
                            'count': 0,
                            'confidences': []
                        }

                    class_stats[class_name]['count'] += 1
                    class_stats[class_name]['confidences'].append(confidence)

                except Exception as item_error:
                    print(f"處理第{i}個檢測項目時出錯: {str(item_error)}")
                    continue

            if not class_stats:
                return "沒有有效的檢測結果"

            # 格式化輸出
            result = "📊 物件檢測統計\n" + "=" * 30 + "\n"
            result += f"🎯 總檢測物件數: {len(detection_results)}\n"
            result += f"🏷️  檢測類別數: {len(class_stats)}\n\n"

            result += "📈 各類別詳細統計:\n" + "-" * 25 + "\n"

            # 按檢測數量排序
            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)

            for class_name, stats in sorted_classes:
                count = stats['count']
                confidences = stats['confidences']

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    max_confidence = max(confidences)
                    min_confidence = min(confidences)
                else:
                    avg_confidence = max_confidence = min_confidence = 0.0

                result += f"🔸 {class_name}:\n"
                result += f"   數量: {count}\n"
                result += f"   平均置信度: {avg_confidence:.3f}\n"
                result += f"   最高置信度: {max_confidence:.3f}\n"
                result += f"   最低置信度: {min_confidence:.3f}\n\n"

            return result

        except Exception as e:
            return f"物件統計解析錯誤: {str(e)}\n請檢查數據格式是否正確"

    def process_inference(self, server_name: str, image_file, model_file) -> Tuple[str, str, str, str]:
        """執行推理過程"""
        try:
            if not server_name:
                return None, "❌ 請選擇GPU服務器", "", ""

            if not image_file or not model_file:
                return None, "❌ 請上傳圖片和模型檔案", "", ""

            # 保存上傳的檔案到本地
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            log_info = f"🚀 開始處理...\n使用服務器: {server_name}\n"

            # 處理和驗證圖片
            log_info += "🖼️  驗證圖片格式...\n"
            processed_image_path = self.client.validate_and_process_image(image_file)
            if not processed_image_path:
                return None, "❌ 圖片格式無效或處理失敗", "", ""

            # 保存圖片
            img_filename = f"input_image_{timestamp}.jpg"
            img_path = os.path.join(self.client.upload_dir, img_filename)
            shutil.copy2(processed_image_path if processed_image_path != image_file else image_file, img_path)

            # 保存模型
            model_filename = f"model_{timestamp}.pt"
            model_path = os.path.join(self.client.upload_dir, model_filename)
            shutil.copy2(model_file, model_path)

            log_info += "✅ 圖片驗證成功\n"

            # 1. 上傳模型
            log_info += "📤 上傳模型中...\n"
            model_result = self.client.upload_model(server_name, model_path)
            if "error" in model_result:
                return None, f"❌ 模型上傳失敗: {model_result['error']}", "", ""

            log_info += "✅ 模型上傳成功\n"

            # 2. 上傳圖片並執行推理
            log_info += "📤 上傳圖片並執行推理...\n"
            img_result = self.client.upload_img(server_name, img_path, os.path.basename(model_path))
            if "error" in img_result:
                return None, f"❌ 圖片處理失敗: {img_result['error']}", "", ""

            log_info += "✅ 推理完成\n"

            # 3. 下載結果
            base_folder = img_result.get('baseF', '')
            if not base_folder:
                return None, "❌ 無法獲取結果資料夾信息", "", ""

            log_info += "📥 下載結果中...\n"

            # 下載處理過的圖片
            processed_image_path = None
            if 'processed_images' in img_result and img_result['processed_images']:
                for img_path in img_result['processed_images']:
                    downloaded_path = self.client.download_file(
                        server_name, os.path.basename(img_path), base_folder
                    )
                    if downloaded_path and os.path.exists(downloaded_path):
                        processed_image_path = downloaded_path
                        break

            # 下載時間資訊
            formatted_time_info = ""
            if 'time_path' in img_result:
                time_file_path = self.client.download_file(
                    server_name, os.path.basename(img_result['time_path']), base_folder
                )
                if time_file_path and os.path.exists(time_file_path):
                    with open(time_file_path, 'r', encoding='utf-8') as f:
                        time_content = f.read()
                        formatted_time_info = self.parse_time_info(time_content)

            # 下載並解析檢測結果
            formatted_object_stats = ""
            if 'summary_file_path' in img_result:
                summary_file_path = self.client.download_file(
                    server_name, os.path.basename(img_result['summary_file_path']), base_folder
                )
                if summary_file_path and os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r', encoding='utf-8') as f:
                        summary_content = f.read()

                    # 調試信息：打印原始內容
                    print(f"原始summary內容: {summary_content[:200]}...")
                    print(f"內容類型: {type(summary_content)}")
                    print(f"內容長度: {len(summary_content)}")

                    # 解析統計信息
                    formatted_object_stats = self.parse_object_statistics(summary_content)

            # 格式化最終結果
            final_log = f"""
🎯 推理完成！

📊 服務器: {server_name}
📁 結果資料夾: {base_folder}

✨ 處理狀態: 所有步驟成功完成
📈 檢測結果已生成
            """.strip()

            return processed_image_path, final_log, formatted_time_info, formatted_object_stats

        except Exception as e:
            return None, f"❌ 處理過程出錯: {str(e)}", "", ""

    def process_inference_vid(self, server_name: str, video_file, model_file) -> Tuple[str, str, str, str]:
        """執行影片推理流程（分幀+重組 + 自動轉檔）"""
        try:
            if not server_name:
                return None, "❌ 請選擇GPU服務器", "", ""

            if not video_file or not model_file:
                return None, "❌ 請上傳影片和模型檔案", "", ""

            # 檔案處理與命名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_info = f"🎬 開始處理影片...\n使用服務器: {server_name}\n"

            video_filename = f"input_video_{timestamp}.mp4"
            video_path = os.path.join(self.client.upload_dir, video_filename)
            shutil.copy2(video_file, video_path)

            model_filename = f"model_{timestamp}.pt"
            model_path = os.path.join(self.client.upload_dir, model_filename)
            shutil.copy2(model_file, model_path)

            # 分割影片為 frames
            log_info += "🔍 分割影片為影格...\n"
            base_name = Path(video_path).stem
            frames_dir = os.path.join(self.client.upload_dir, f"frames_{base_name}")
            os.makedirs(frames_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, "❌ 無法讀取影片", "", ""

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            success, frame = cap.read()
            while success:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                success, frame = cap.read()
            cap.release()

            log_info += f"✅ 成功擷取 {frame_count} 張影格\n"

            # 上傳模型
            log_info += "📤 上傳模型...\n"
            model_result = self.client.upload_model(server_name, model_path)
            if "error" in model_result:
                return None, f"❌ 模型上傳失敗: {model_result['error']}", "", ""
            log_info += "✅ 模型上傳成功\n"

            # 上傳所有 frames 並推理
            log_info += "🧠 開始影格推理...\n"
            processed_frame_paths = []
            for fname in sorted(os.listdir(frames_dir)):
                if not fname.endswith(".jpg"):
                    continue
                frame_path = os.path.join(frames_dir, fname)
                img_result = self.client.upload_img(server_name, frame_path, os.path.basename(model_path))
                if "error" in img_result:
                    return None, f"❌ 影格推理失敗: {img_result['error']}", "", ""

                base_folder = img_result.get("baseF", "")
                processed_files = img_result.get("processed_images", [])
                if processed_files:
                    downloaded_path = self.client.download_file(server_name, os.path.basename(processed_files[0]), base_folder)
                    if downloaded_path:
                        processed_frame_paths.append(downloaded_path)

            # 重組影片
            log_info += "🎞️ 重組推理後影片...\n"
            if not processed_frame_paths:
                return None, "❌ 無推理後影格可用", "", ""

            sample_img = cv2.imread(processed_frame_paths[0])
            height, width = sample_img.shape[:2]
            raw_video_path = os.path.join(self.client.upload_dir, f"{base_name}_raw_result.mp4")
            out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            for img_path in processed_frame_paths:
                frame = cv2.imread(img_path)
                out.write(frame)
            out.release()

            log_info += f"✅ 成功生成影片，共 {len(processed_frame_paths)} 幀\n"

            # 🔄 轉檔為瀏覽器相容影片格式 (H.264)
            result_video_path = os.path.join(self.client.upload_dir, f"{base_name}_result_fixed.mp4")
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()  # 自動取得內建 ffmpeg
            try:
                (
                    ffmpeg
                    .input(raw_video_path)
                    .output(
                        result_video_path,
                        vcodec='libx264',
                        acodec='aac',
                        movflags='+faststart'
                    )
                    .run(cmd=ffmpeg_bin, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                log_info += f"✅ 已轉檔為瀏覽器相容格式\n"
            except Exception as e:
                return None, f"❌ ffmpeg 轉檔過程錯誤: {str(e)}", "", ""

            # 處理時間與統計資訊
            formatted_time_info = ""
            formatted_object_stats = ""
            if 'time_path' in img_result:
                time_path = self.client.download_file(server_name, os.path.basename(img_result['time_path']), base_folder)
                if os.path.exists(time_path):
                    with open(time_path, 'r') as f:
                        formatted_time_info = self.parse_time_info(f.read())

            if 'summary_file_path' in img_result:
                summary_path = self.client.download_file(server_name, os.path.basename(img_result['summary_file_path']), base_folder)
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_content = f.read()
                        formatted_object_stats = self.parse_object_statistics(summary_content)

            final_log = f"""
    🎬 推理完成！影片重組成功 ✅

    🖥️ 使用伺服器: {server_name}
    📹 輸出影片: {result_video_path}
    📈 推理影格總數: {len(processed_frame_paths)}
    """.strip()

            return result_video_path, final_log, formatted_time_info, formatted_object_stats

        except Exception as e:
            return None, f"❌ 處理過程發生錯誤: {str(e)}", "", ""
        
    def build_interface(self):
        with gr.Blocks(
                theme=gr.themes.Soft(),
                css="""
            .server-selection {
                background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }
            .result-container {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #28a745;
            }
            .stats-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .time-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #2196f3;
            }
            """
        ) as demo:
            gr.Markdown(
                """
                # 🚀 多GPU YOLO物件檢測系統
                ### 支援 RTX 3050Ti、Tesla T4、AMD 7900 三種加速方案
                """,
                elem_classes=["server-selection"]
            )
            with gr.Tabs():
                with gr.Tab("圖片"):
                    with gr.Row():
                        # 左側控制面板
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎛️ 控制面板")

                            server_dropdown = gr.Dropdown(
                                choices=["Local GPU", "Tesla T4", "AMD 7900"],
                                label="🖥️ 選擇GPU服務器",
                                value="Local GPU"
                            )

                            image_upload = gr.File(
                                label="📷 上傳圖片",
                                file_types=[".jpg", ".jpeg", ".png", ".bmp"],
                                type="filepath"
                            )

                            model_upload = gr.File(
                                label="🤖 上傳YOLO模型",
                                file_types=[".pt"],
                                type="filepath"
                            )

                            process_btn = gr.Button(
                                "🚀 開始推理",
                                variant="primary",
                                size="lg"
                            )
                        
                        # 右側結果顯示
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 推理結果")

                            with gr.Tabs():
                                with gr.Tab("🖼️ 檢測結果"):
                                    result_image = gr.Image(
                                        label="檢測結果圖片",
                                        type="filepath"
                                    )

                                with gr.Tab("📝 詳細日誌"):
                                    log_output = gr.Textbox(
                                        label="處理日誌",
                                        lines=15,
                                        max_lines=20,
                                        elem_classes=["result-container"]
                                    )

                    # 底部統計信息
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ⏱️ 時間統計", elem_classes=["time-container"])
                            time_stats = gr.Textbox(
                                label="執行時間詳情",
                                lines=8,
                                interactive=False
                            )

                        with gr.Column():
                            gr.Markdown("### 📈 物件統計", elem_classes=["stats-container"])
                            object_stats = gr.Textbox(
                                label="檢測物件詳情",
                                lines=8,
                                interactive=False
                            )

                    
                    # 事件綁定
                    process_btn.click(
                        fn=self.process_inference,
                        inputs=[server_dropdown, image_upload, model_upload],
                        outputs=[result_image, log_output, time_stats, object_stats]
                    )

                    # 示例說明
                    gr.Markdown(
                        """
                        ---
                        ### 📋 使用說明：
                        1. **選擇GPU服務器**：根據您的需求選擇合適的加速硬體
                        2. **上傳檔案**：選擇要檢測的圖片和YOLO模型檔案
                        3. **開始推理**：點擊按鈕開始物件檢測
                        4. **查看結果**：在右側查看檢測結果圖片和詳細統計

                        ### 💡 服務器規格：
                        - **RTX 3050Ti**: 本機Windows環境，適合快速測試
                        - **Tesla T4**: 雲端Linux環境，平衡性能與成本
                        - **AMD 7900**: 高性能Linux環境，適合大型任務

                        ### 🔧 改善項目：
                        - **圖片處理**: 自動驗證並處理不同格式的圖片
                        - **時間統計**: 清晰顯示模型載入、推理、處理三個階段時間
                        - **物件統計**: 按類別顯示數量、平均置信度等詳細信息
                        """
                    )
                with gr.Tab("影片"):
                    with gr.Row():
                        # 左側控制面板
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎛️ 控制面板")

                            server_dropdown = gr.Dropdown(
                                choices=["Local GPU", "Tesla T4", "AMD 7900"],
                                label="🖥️ 選擇GPU服務器",
                                value="Local GPU"
                            )

                            video_upload = gr.File(
                                label="🎞️ 上傳影片",
                                file_types=[".mp4"],
                                type="filepath"
                            )

                            model_upload = gr.File(
                                label="🤖 上傳YOLO模型",
                                file_types=[".pt"],
                                type="filepath"
                            )

                            process_btn = gr.Button(
                                "🚀 開始推理",
                                variant="primary",
                                size="lg"
                            )
                        
                        # 右側結果顯示
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 推理結果")

                            with gr.Tabs():
                                with gr.Tab("🎬 檢測影片"):
                                    result_video = gr.Video(
                                        label="檢測結果影片",
                                        format="mp4",
                                        interactive=False
                                    )

                                with gr.Tab("📝 詳細日誌"):
                                    log_output = gr.Textbox(
                                        label="處理日誌",
                                        lines=15,
                                        max_lines=20,
                                        elem_classes=["result-container"]
                                    )

                    # 底部統計信息
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ⏱️ 時間統計", elem_classes=["time-container"])
                            time_stats = gr.Textbox(
                                label="執行時間詳情",
                                lines=8,
                                interactive=False
                            )

                        with gr.Column():
                            gr.Markdown("### 📈 物件統計", elem_classes=["stats-container"])
                            object_stats = gr.Textbox(
                                label="檢測物件詳情",
                                lines=8,
                                interactive=False
                            )

                    # 事件綁定
                    process_btn.click(
                        fn=self.process_inference_vid,
                        inputs=[server_dropdown, video_upload, model_upload],
                        outputs=[result_video, log_output, time_stats, object_stats]
                    )

                    # 使用說明
                    gr.Markdown(
                        """
                        ---
                        ### 📋 使用說明：
                        1. **選擇GPU服務器**：根據您的需求選擇合適的加速硬體
                        2. **上傳檔案**：選擇要檢測的影片和YOLO模型檔案
                        3. **開始推理**：點擊按鈕開始物件檢測
                        4. **查看結果**：在右側查看推理後影片和詳細統計

                        ### 💡 服務器規格：
                        - **RTX 3050Ti**: 本機Windows環境，適合快速測試
                        - **Tesla T4**: 雲端Linux環境，平衡性能與成本
                        - **AMD 7900**: 高性能Linux環境，適合大型任務

                        ### 🔧 改善項目：
                        - **影片處理**: 自動分幀與合併輸出
                        - **時間統計**: 清晰顯示模型載入、推理、處理三個階段時間
                        - **物件統計**: 按類別顯示數量、平均置信度等詳細信息
                        """
                    )

                    return demo


def main():
    ui = YOLO_MultiGPU_UI()
    interface = ui.build_interface()

    try:
        interface.queue().launch(
            allowed_paths=[
                "./gradio_workspace",
                os.path.abspath("./gradio_workspace")
            ],
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"啟動伺服器時發生錯誤：{str(e)}")
    finally:
        print("伺服器關閉")


if __name__ == '__main__':
    main()