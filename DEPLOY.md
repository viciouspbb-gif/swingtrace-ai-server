# Renderへのデプロイ手順

## 📋 **前提条件**

- GitHubアカウント
- Renderアカウント（無料）

---

## 🚀 **デプロイ手順**

### **ステップ1: GitHubにコードをプッシュ**

#### **1. GitHubで新しいリポジトリを作成**

1. https://github.com/new にアクセス
2. リポジトリ名: `swingtrace-ai-server`
3. Public/Private: どちらでもOK
4. 「Create repository」をクリック

#### **2. ローカルでGit初期化**

PowerShellで実行：

```powershell
cd C:\Users\katsunori\CascadeProjects\SwingTraceServer

# Git初期化
git init

# ファイルを追加
git add .

# コミット
git commit -m "Initial commit: SwingTrace AI Server"

# GitHubリポジトリを追加
git remote add origin https://github.com/[あなたのユーザー名]/swingtrace-ai-server.git

# プッシュ
git branch -M main
git push -u origin main
```

---

### **ステップ2: Renderアカウント作成**

1. **https://render.com/ にアクセス**
2. **「Get Started」をクリック**
3. **GitHubアカウントで登録**

---

### **ステップ3: 新しいWebサービスを作成**

1. **Renderダッシュボードで「New +」をクリック**
2. **「Web Service」を選択**
3. **GitHubリポジトリを接続**
   - 「Connect a repository」
   - `swingtrace-ai-server` を選択
4. **設定を入力**
   - **Name**: `swingtrace-ai-server`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
5. **「Create Web Service」をクリック**

---

### **ステップ4: デプロイ完了を待つ**

- 初回デプロイ: 5-10分
- ログを確認してエラーがないか確認

---

### **ステップ5: URLを取得**

デプロイ完了後、以下のようなURLが表示されます：

```
https://swingtrace-ai-server.onrender.com
```

このURLをAndroidアプリで使用します。

---

## 🌐 **動作確認**

### **ブラウザでアクセス**

```
https://swingtrace-ai-server.onrender.com
```

以下のレスポンスが返ればOK：

```json
{
  "status": "ok",
  "message": "SwingTrace AI Server is running",
  "version": "1.0.0"
}
```

### **APIドキュメント**

```
https://swingtrace-ai-server.onrender.com/docs
```

---

## 📱 **Androidアプリとの連携**

Androidアプリの設定で、サーバーURLを変更：

```kotlin
// 開発環境（ローカル）
const val BASE_URL = "http://192.168.1.100:8000"

// 本番環境（Render）
const val BASE_URL = "https://swingtrace-ai-server.onrender.com"
```

---

## ⚠️ **注意事項**

### **無料プランの制限**

- **スリープ**: 15分間アクセスがないとスリープ
- **起動時間**: スリープから復帰に30秒-1分
- **月間稼働時間**: 750時間/月（約31日）

### **スリープ対策**

定期的にアクセスするCronジョブを設定（オプション）

---

## 🔄 **更新方法**

コードを更新したら：

```powershell
git add .
git commit -m "Update: 機能追加"
git push
```

Renderが自動的に再デプロイします。

---

## 🐛 **トラブルシューティング**

### **デプロイが失敗する**

1. Renderのログを確認
2. `requirements.txt` が正しいか確認
3. Pythonバージョンを確認

### **APIにアクセスできない**

1. URLが正しいか確認
2. HTTPSを使用しているか確認
3. CORSエラーの場合、`app.py`のCORS設定を確認

---

## 💰 **料金**

- **無料プラン**: $0/月
- **Starter**: $7/月（スリープなし）
- **Standard**: $25/月（より高性能）

---

## 🎉 **完了！**

これでSwingTrace AIサーバーがクラウドで稼働します！

Androidアプリから世界中どこからでもアクセスできます。
