# FS_Attack
- 預設 dataset 路徑為 '/tf/datasets' (可在samples.samples.py修改)  
- 預設從 dataset 中抽取圖片放到 tmp_image 底裡

### 各檔案用途
- FS_attack.py : 主要攻擊程式
- metric.py : 計算 metric
- sample_img.py : 用來從 dataset 抽圖片放到 tmp_image 底下
- Experiment_ver7.ipynb : 最新版實驗
- experiment_script : 存一些實驗用過的 script，要移到XAI_Attack資料夾下才能用

### 用法
攻擊特定圖片 `python FS_attack.py --method "lrp" --n 100 --cuda --num 1 --dump True --origin True`  
計算 metric 範例 `python metric.py --img output/lrp_1_ori_img.npy --adv_img output/lrp_1_adv_img.npy`  




### 其他連結  
論文投影片 [[link]](https://docs.google.com/presentation/d/1eYiIPxz3XbZTnGIBF7jJyTPh-5j0uQ7gvi-614u6XvI/edit?usp=sharing)  
CISC演討會版本 [[link]](https://drive.google.com/file/d/1gtsMByaNR0fkLKrY2suWaXY2mf6gVOcj/view?usp=sharing)  
