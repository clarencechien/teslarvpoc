# 車庫的雙眼攝影重建

兩張實拍照片可以做出可以旋轉的 3D 表面。這個實驗用 Middlebury 的 Motorcycle 車庫場景估算深度，保留 21,061 個頂點和 38,379 個三角面。滑鼠拖曳時，前輪和後方貨架會產生不同幅度的位移。轉到側面會看見遮蔽造成的孔洞，兩張正面照片只能重建局部室內。

![車庫的照片貼圖與同一個網格的相機深度，暖色較近](assets/preview.jpg)

**[直接在 GitHub Pages 開啟 3D 展示](https://clarencechien.github.io/teslarvpoc/)**

**[下載互動展示 viewer.html](https://github.com/clarencechien/teslarvpoc/raw/refs/heads/main/viewer.html)**，儲存後以支援 WebGL 的瀏覽器開啟。這個單檔內含幾何、照片貼圖與程式。若瀏覽器限制開啟本機檔，在 repo 目錄執行 `python -m http.server 8000`，前往 `http://localhost:8000/viewer.html`。

- 拖曳或用方向鍵轉動，滾輪或鍵盤 `+`、`-` 縮放。可切換照片、深度與點雲。
- 按「下載 3D 模型」可取得含貼圖、以公尺為單位的 glTF 2.0，能在 Blender 匯入並編輯網格。
- 按「測試幀率」會以固定路徑旋轉模型。暖機 2 秒、取樣 10 秒，顯示實際畫布解析度、平均 FPS 和第 95 百分位畫格間隔。這是前景頁面的 requestAnimationFrame 間隔，包含瀏覽器排程。全解析度的裝置像素比上限為 1.5。半解析度將寬和高各減半。

## 60 FPS 實測

固定 1280 × 800 視窗、裝置像素比 1，以 Linux headless Chromium 156 和 SwiftShader 軟體渲染執行內建測試。兩次以相同網格、貼圖和移動路徑量測，每個解析度各一次。

| 畫布像素 | 平均 FPS | P95 畫格間隔 |
| --- | ---: | ---: |
| 1176 × 512 | 31.9 | 50.1 ms |
| 588 × 256 | 33.7 | 50.1 ms |

60 FPS 的每幀預算是 16.7 ms。這台機器在兩種解析度都超過預算。硬體加速的筆電與手機仍需各自執行上方測試；整條街的效能還取決於視野中同時繪製的物件、三角面和貼圖。

## 從照片重建

測試使用 Python 3.12。從 repo 根目錄執行下列指令會重新下載有版本與 SHA-256 校驗的左右照片並產生 viewer.html。約 1.1 MB 的完成品已放在 repo。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/reconstruct.py
```

1. 讀取已校正到相同掃描線的左右照片，每張 741 × 500。OpenCV StereoSGBM 估算左右影像中同一點的水平位移。
2. 依已知的相機間距 0.193001 m、焦距 994.978 px 和左右主點差 31.086 px，以 `Z = focal × baseline / (disparity + doffs)` 回推公尺深度。
3. 每隔 4 像素取樣並連成網格，略過無效匹配與深度跳變，使用左相機照片貼圖。標註的參考視差僅供事後驗證。

約 91.3% 的左圖像素得到有效估計。在估計與參考視差均有效的 314,740 像素上，絕對視差誤差中位數為 0.18 px，5.8% 的像素誤差超過 2 px。金屬、輪輻和遮蔽邊緣仍可看見破面。換成手機隨拍的台灣街景時，需有同一街段的重疊視角，再用 SfM 估計相機位置，之後才能進行多視角重建。[COLMAP](https://colmap.github.io/)

## Tesla 與台灣街景工具

Musk 描述的 Reality View 會從共用視覺編碼器 UVE 取得特徵，讓獨立的約十億參數擴散模型利用上一段輸出生成新畫面。據其說明，UVE 以 18 Hz 更新，顯示端每批產生 8 幀，以 36 FPS 播放。截至 2026-11-24，在 Tesla 的公開專案中未找到這套模型的權重或推論 API。本 repo 的照片重建在離線階段完成，瀏覽器每幀只需繪製網格。[Musk 的技術說明](https://x.com/elonmusk/status/2124458557525397780) · [Tesla 官方 GitHub](https://github.com/teslamotors)

台北開發者 Jason Kuan（jason9075）的 Taipei PCG 以台北的道路層級、街廓密度與捷運周邊建築分布為原型，可以在瀏覽器調整參數並重新生成 3D 城市。作者的公開資料列出所在地為台北。目前仍缺少能將這件作品與所述 X 貼文連起來的原文。[線上展示](https://jason9075.github.io/city-pcg/) · [作品說明](https://github.com/jason9075/city-pcg) · [作者資料](https://github.com/jason9075)

## 影像來源與使用條件

兩張影像來自 Middlebury Stereo 2014 的 Motorcycle，由 scikit-image 提供縮小至四分之一的左右視圖與對應校正資料。Middlebury 允許使用與發表資料集的照片及視差圖，並請使用者引用論文。本展示由該組照片估算形狀，將左圖轉為 JPEG 貼圖。[資料及使用條件](https://vision.middlebury.edu/stereo/data/) · [原始場景](https://vision.middlebury.edu/stereo/data/scenes2014/) · [縮圖與校正參數](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle)

D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, P. Westling. *High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth*, GCPR 2014. [論文](https://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf)
