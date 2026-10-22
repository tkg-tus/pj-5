# data/

このフォルダの中身はGitに入れない（`.gitignore` で除外）。
共通のデータがサーバ上の別の場所にある場合は、コピーせずに `--data_dir` で指定する。

```
data/
├── features/
│   ├── visual/{video_id}.npy   形（秒数, 次元） 1秒ごとの映像の特徴
│   ├── audio/{video_id}.npy    形（秒数, 次元） 1秒ごとの音の特徴
│   └── text/{video_id}.npy     形（秒数, 次元） 1秒ごとの字幕などの特徴
├── labels/{video_id}.npy       形（秒数,）      1秒ごとの正解（0〜1）
└── splits/
    ├── train.txt               1行に1つの動画ID
    └── val.txt
```
