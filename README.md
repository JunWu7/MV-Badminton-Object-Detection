## Server 建立

1. 本機(看各位GPU) 
   1. 先下載 code\Server\local 資料夾內的 ntut_yolo_inf_local.py 和 ntut_vm_server_yolo_local.py 並放在同一個資料夾
   2. 接著直接執行 ntut_vm_server_yolo_local.py 即可

2. Tesla T4
   1. 請助教開VM
   2. 到我們 Group1 資料夾中找到 Final_Gradio_Server 這個資料夾
   3. 直接執行裡面的 ntut_vm_server_yolo.py 即可
   >**註** : 因為不太知道怎麼遠端連到Tesla T4的IP讓他直接可以連，所以目前是都連127.0.0.1，只是不一樣port。因此執行main.py和到VM開始執行的部分請都用同一台電腦

3. AMD
   1. 請助教開VM
   2. 連線北科VPN
   3. 就可以直接執行main.py了

## 網頁使用

1. 下載 code\WebPage 資料夾內的 main.py 之後直接執行
2. 開啟瀏覽器輸入網址 : http://localhost:7860/

<!-- ## 目前出事

AMD server 的code我改不到，他的輸出圖片不會跟其他兩個有改過的一樣是整張標註過的結果圖，只會有其中一個物件的截圖。
不過其他像時間、物件數量等都是正常的。
所以看有沒有人有辦法弄弄看，不然就先這樣了。

## 新增影片功能

main_111590450.py : 主要是把影片先切成細塊再重組，所以測試的時候影片長度不要太長，testvid大約0.5秒(大概跑20秒)，是從原本2025-01-09_08-59-52_CameraReader_0三秒影片剪輯的，太長會跑很久。

## 修改了一下"時間統計"的css

原本的亮看不到字 @_@

## 已修正預覽影片跑不出來的問題 -->

重組影片格式為mpeg-4，gradio不支援，因此需要轉檔H.264
>**import imageio_ffmpeg 需要 pip install ffmpeg-python**