# SwingTrace AI Server

ゴルフスイング分析用Pythonサーバー

## 機能

- 📹 **動画分析API** - スイング動画からボールを検出・追跡
- 🎯 **弾道計算** - 飛距離・最高到達点・滞空時間を計算
- 🤖 **AIコーチング** - スイングデータを分析してアドバイス
- 💎 **プレミアム機能** - 詳細なAI分析（有料ユーザー向け）

## セットアップ

### 1. Python環境の準備

```bash
# Python 3.9以上が必要
python --version
```

### 2. 仮想環境の作成

```bash
cd C:\Users\katsunori\CascadeProjects\SwingTraceServer
python -m venv venv
```

### 3. 仮想環境の有効化

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
venv\Scripts\activate.bat
```

### 4. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 5. サーバー起動

```bash
python app.py
```

サーバーが起動すると、以下のURLでアクセスできます：
- **API**: http://localhost:8000
- **ドキュメント**: http://localhost:8000/docs

## API エンドポイント

### 1. ヘルスチェック

```http
GET /
```

**レスポンス:**
```json
{
  "status": "ok",
  "message": "SwingTrace AI Server is running",
  "version": "1.0.0"
}
```

### 2. スイング分析

```http
POST /api/analyze-swing
Content-Type: multipart/form-data

video: [動画ファイル]
```

**レスポンス:**
```json
{
  "ball_detected": true,
  "trajectory": [
    {"x": 100.0, "y": 200.0, "z": 0.0, "time": 0.0},
    {"x": 150.0, "y": 180.0, "z": 0.0, "time": 0.1}
  ],
  "carry_distance": 215.5,
  "max_height": 45.2,
  "flight_time": 4.5,
  "swing_data": null,
  "confidence": 0.85
}
```

### 3. AIコーチング

```http
POST /api/ai-coaching
Content-Type: application/json

{
  "swing_speed": 95.0,
  "backswing_time": 0.8,
  "downswing_time": 0.3,
  "impact_speed": 90.0,
  "carry_distance": 210.0,
  "is_premium": true
}
```

**レスポンス:**
```json
{
  "advice": "良いスイング速度です。このペースを維持しながら、正確性を高めていきましょう。",
  "improvements": [
    "テンポを3:1に近づけましょう（理想的なリズム）"
  ],
  "strengths": [
    "スイング速度が速い",
    "十分な飛距離"
  ],
  "score": 85
}
```

## Androidアプリとの連携

### 1. ローカルネットワークで接続

サーバーのIPアドレスを確認：
```bash
ipconfig
```

Androidアプリで以下のURLを使用：
```
http://[あなたのIPアドレス]:8000
```

例: `http://192.168.1.100:8000`

### 2. ngrokで外部公開（開発用）

```bash
# ngrokをインストール
# https://ngrok.com/download

# サーバーを公開
ngrok http 8000
```

表示されたURLをAndroidアプリで使用

## 開発

### テスト実行

```bash
# curlでテスト
curl http://localhost:8000/

# 動画アップロードテスト
curl -X POST -F "video=@test_video.mp4" http://localhost:8000/api/analyze-swing
```

### ログ確認

サーバーのコンソールにリアルタイムでログが表示されます。

## 今後の拡張

- [ ] YOLO/MediaPipeによる高精度ボール検出
- [ ] OpenAI APIによる本物のAI分析
- [ ] データベース連携（ユーザー履歴保存）
- [ ] WebSocket対応（リアルタイム分析）
- [ ] クラウドデプロイ（AWS/GCP）

## トラブルシューティング

### ポート8000が使用中

別のポートを使用：
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

### OpenCVのインストールエラー

```bash
pip install opencv-python-headless
```

### Androidアプリから接続できない

1. ファイアウォールを確認
2. 同じWi-Fiネットワークに接続
3. IPアドレスが正しいか確認

## ライセンス

MIT License
