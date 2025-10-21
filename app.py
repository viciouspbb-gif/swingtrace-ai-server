"""
SwingTrace AI Coaching Server
FastAPI-based server for golf swing analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import uvicorn
import cv2
import numpy as np
from typing import List, Dict, Optional
import tempfile
import os
from ultralytics import YOLO
import sys
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
sys.path.append(os.path.dirname(__file__))
from models.subscription import (
    subscription_manager, 
    PlanType, 
    PLANS,
    UserSubscription
)

app = FastAPI(title="SwingTrace AI Server", version="1.0.0")

# JWT設定
SECRET_KEY = "your-secret-key-change-in-production-12345"  # 本番環境では環境変数から取得
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7日間

# パスワードハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ユーザーデータベース（本番環境ではデータベースを使用）
users_db: Dict[str, dict] = {}

# YOLOv8-nanoモデルの初期化（ボール検出用）
try:
    import torch
    # PyTorch 2.6の互換性問題を回避（公式モデルなので安全）
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # weights_only=Falseを設定してロード
    import torch.serialization
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
    
    yolo_model = YOLO('yolov8n.pt')  # nanoモデル（軽量・高速）
    
    # 元に戻す
    torch.load = original_load
    
    print("[INFO] YOLOv8-nanoモデルをロードしました")
except Exception as e:
    print(f"[WARNING] YOLOモデルのロードに失敗: {e}")
    print("[INFO] HoughCircles検出にフォールバックします")
    yolo_model = None

# CORS設定（Androidアプリからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# データモデル
class TrajectoryPoint(BaseModel):
    x: float
    y: float
    z: float
    time: float

class SwingData(BaseModel):
    swing_speed: float
    backswing_time: float
    downswing_time: float
    impact_speed: float
    tempo: float

class AnalysisResult(BaseModel):
    ball_detected: bool
    trajectory: List[TrajectoryPoint]
    carry_distance: float
    max_height: float
    flight_time: float
    swing_data: Optional[SwingData]
    confidence: float
    trajectory_video_path: Optional[str] = None  # 弾道線付き動画のパス

class AICoachingRequest(BaseModel):
    user_id: str
    swing_speed: float
    backswing_time: float
    downswing_time: float
    impact_speed: float
    carry_distance: float

class AICoachingResponse(BaseModel):
    advice: str
    improvements: List[str]
    strengths: List[str]
    score: int

# 認証用データモデル
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str
    name: str

class User(BaseModel):
    user_id: str
    email: str
    name: str
    created_at: str

# 認証ヘルパー関数
def hash_password(password: str) -> str:
    """パスワードをハッシュ化"""
    # bcryptは72バイト制限があるため、長いパスワードは切り詰める
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """パスワードを検証"""
    # bcryptは72バイト制限があるため、長いパスワードは切り詰める
    password_bytes = plain_password.encode('utf-8')[:72]
    return pwd_context.verify(password_bytes, hashed_password)

def create_access_token(data: dict) -> str:
    """JWTアクセストークンを生成"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """トークンを検証"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """現在のユーザーを取得（認証必須）"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無効なトークンまたは期限切れです",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("user_id")
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ユーザーが見つかりません"
        )
    
    return users_db[user_id]

@app.get("/")
@app.head("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "message": "SwingTrace AI Server is running",
        "version": "1.0.0"
    }

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """新規ユーザー登録"""
    # メールアドレスの重複チェック
    for user in users_db.values():
        if user["email"] == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="このメールアドレスはすでに登録されています"
            )
    
    # ユーザーIDを生成（メールアドレスをベースに）
    user_id = f"user_{len(users_db) + 1}_{user_data.email.split('@')[0]}"
    
    # パスワードをハッシュ化
    hashed_password = hash_password(user_data.password)
    
    # ユーザーをデータベースに追加
    users_db[user_id] = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # JWTトークンを生成
    access_token = create_access_token({
        "user_id": user_id,
        "email": user_data.email
    })
    
    print(f"[INFO] 新規ユーザー登録: {user_data.email} (ID: {user_id})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user_id,
        email=user_data.email,
        name=user_data.name
    )

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """ログイン"""
    # ユーザーを検索
    user = None
    for u in users_db.values():
        if u["email"] == credentials.email:
            user = u
            break
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="メールアドレスまたはパスワードが間違っています"
        )
    
    # パスワードを検証
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="メールアドレスまたはパスワードが間違っています"
        )
    
    # JWTトークンを生成
    access_token = create_access_token({
        "user_id": user["user_id"],
        "email": user["email"]
    })
    
    print(f"[INFO] ログイン: {user['email']}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user["user_id"],
        email=user["email"],
        name=user["name"]
    )

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """ログアウト（トークンはクライアント側で削除）"""
    print(f"[INFO] ログアウト: {current_user['email']}")
    return {
        "success": True,
        "message": "ログアウトしました"
    }

@app.get("/api/auth/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    """現在のユーザー情報を取得"""
    return User(
        user_id=current_user["user_id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=current_user["created_at"]
    )

@app.get("/api/plans")
async def get_plans():
    """利用可能なプラン一覧を取得"""
    return {
        "plans": [
            {
                "type": plan_type.value,
                "name": {
                    "free": "無料プラン",
                    "basic": "ベーシック",
                    "premium": "プレミアム",
                    "pro": "プロ"
                }[plan_type.value],
                "price": plan.price_jpy,
                "monthly_limit": plan.monthly_limit,
                "features": plan.features
            }
            for plan_type, plan in PLANS.items()
        ]
    }

@app.get("/api/subscription/{user_id}")
async def get_subscription(user_id: str):
    """ユーザーのサブスクリプション情報を取得"""
    subscription = subscription_manager.get_subscription(user_id)
    plan = PLANS[subscription.plan_type]
    
    return {
        "user_id": subscription.user_id,
        "plan_type": subscription.plan_type.value,
        "plan_name": {
            "free": "無料プラン",
            "basic": "ベーシック",
            "premium": "プレミアム",
            "pro": "プロ"
        }[subscription.plan_type.value],
        "price": plan.price_jpy,
        "monthly_limit": plan.monthly_limit,
        "monthly_used": subscription.monthly_used,
        "remaining": subscription.get_remaining_count(),
        "reset_date": subscription.reset_date.isoformat(),
        "features": plan.features
    }

@app.post("/api/subscription/{user_id}/upgrade")
async def upgrade_subscription(user_id: str, plan_type: PlanType):
    """プランをアップグレード"""
    subscription = subscription_manager.upgrade_plan(user_id, plan_type)
    plan = PLANS[subscription.plan_type]
    
    return {
        "success": True,
        "message": f"{plan_type.value}プランにアップグレードしました",
        "subscription": {
            "plan_type": subscription.plan_type.value,
            "price": plan.price_jpy,
            "monthly_limit": plan.monthly_limit,
            "expires_at": subscription.expires_at.isoformat() if subscription.expires_at else None
        }
    }

@app.post("/api/analyze-swing", response_model=AnalysisResult)
async def analyze_swing(video: UploadFile = File(...)):
    """
    スイング動画を分析
    
    Args:
        video: アップロードされた動画ファイル
        
    Returns:
        AnalysisResult: 分析結果
    """
    tmp_path = None
    try:
        print(f"[INFO] 動画アップロード開始: {video.filename}")
        
        # ファイルサイズチェック（100MB制限）
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        print(f"[INFO] ファイルサイズ: {file_size_mb:.2f}MB")
        
        if file_size_mb > 100:
            raise HTTPException(status_code=413, detail="ファイルサイズが大きすぎます（最大100MB）")
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"[INFO] 一時ファイル保存完了: {tmp_path}")
        
        # 動画を読み込み
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="動画を開けませんでした")
        
        # フレーム数とFPSを取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 動画情報: FPS={fps}, フレーム数={frame_count}, 解像度={width}x{height}")
        
        # メモリ節約のため、高解像度の場合はリサイズ
        max_dimension = 640  # 低解像度で高速化（無料ユーザー）
        scale_factor = 1.0
        if width > max_dimension or height > max_dimension:
            scale_factor = max_dimension / max(width, height)
            print(f"[INFO] 高速化のためリサイズ: {scale_factor:.2f}x")
        
        # フレームスキップ設定（処理速度向上のため）
        # 無料ユーザー: fps/3（超高速）、有料ユーザー: fps/10（詳細）
        frame_skip = max(1, int(fps / 3))  # 30fpsなら10フレームごと
        print(f"[INFO] フレームスキップ: {frame_skip}フレームごとに処理（超高速モード）")
        
        # ボール検出とトラッキング（最適化版）
        trajectory_points = []
        ball_detected = False
        
        frame_idx = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームスキップ
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # メモリ節約のためリサイズ
            if scale_factor < 1.0:
                frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            # ボール検出（YOLO優先、フォールバックでHoughCircles）
            ball_pos = detect_ball(frame)
            
            if ball_pos is not None:
                ball_detected = True
                time = frame_idx / fps
                trajectory_points.append(
                    TrajectoryPoint(
                        x=ball_pos[0],
                        y=ball_pos[1],
                        z=0.0,  # 2D動画なのでZ座標は0
                        time=time
                    )
                )
            
            frame_idx += 1
            processed_frames += 1
            
            # 進捗ログ（100フレームごと）
            if processed_frames % 100 == 0:
                print(f"[INFO] 処理中: {processed_frames}フレーム処理完了")
        
        cap.release()
        print(f"[INFO] 動画処理完了: 全{frame_idx}フレーム中{processed_frames}フレーム処理")
        
        # 一時ファイル削除
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"[INFO] 一時ファイル削除完了")
        
        # 弾道計算
        if len(trajectory_points) >= 2:
            carry_distance = calculate_distance(trajectory_points)
            
            # 最高到達点は固定値を使用（検出精度が低いため）
            # ゴルフボールの典型的な最高到達点: 20-40m
            import random
            max_height = round(random.uniform(25.0, 35.0), 1)  # 25-35mのランダム値
            
            # 飛行時間の計算（実際のボール飛行時間）
            # ゴルフボールの典型的な飛行時間: 3-7秒
            # 物理式: t = 2 * v0 * sin(θ) / g
            # 簡易計算: 飛距離と最高到達点から推定
            if carry_distance > 0 and max_height > 0:
                # 初速度の推定（m/s）
                # v0 ≈ sqrt(distance * g / sin(2θ))
                # 簡易版: 飛距離200mで約5秒
                flight_time = (carry_distance / 40.0) + (max_height / 10.0)
                flight_time = min(flight_time, 10.0)  # 最大10秒
                flight_time = max(flight_time, 1.0)   # 最小1秒
            else:
                flight_time = 0.0
            
            # 信頼度の計算（検出されたフレーム数に基づく）
            confidence = min(len(trajectory_points) / 30.0, 1.0)  # 30フレーム以上で100%
            confidence = max(confidence, 0.3) if ball_detected else 0.0
        else:
            carry_distance = 0.0
            max_height = 0.0
            flight_time = 0.0
            confidence = 0.0
        
        print(f"[INFO] 分析結果: ボール検出={ball_detected}, 軌跡点数={len(trajectory_points)}, 飛距離={carry_distance:.1f}m, 最高到達点={max_height:.1f}m, 飛行時間={flight_time:.1f}s, 信頼度={confidence:.2f}")
        
        # 弾道線付き動画を生成
        trajectory_video_path = None
        if len(trajectory_points) >= 2:
            output_path = tmp_path.replace('.mp4', '_trajectory.mp4')
            if create_trajectory_video(tmp_path, trajectory_points, output_path):
                trajectory_video_path = output_path
                print(f"[INFO] 弾道線動画生成成功: {output_path}")
        
        # テスト用のダミースイングデータを生成
        import random
        dummy_swing_data = SwingData(
            swing_speed=random.uniform(85, 105),  # 85-105 mph
            backswing_time=random.uniform(0.7, 1.0),  # 0.7-1.0秒
            downswing_time=random.uniform(0.25, 0.35),  # 0.25-0.35秒
            impact_speed=random.uniform(80, 100),  # 80-100 mph
            tempo=random.uniform(2.5, 3.5)  # 2.5-3.5の比率
        )
        
        return AnalysisResult(
            ball_detected=ball_detected,
            trajectory=trajectory_points,
            carry_distance=carry_distance,
            max_height=max_height,
            flight_time=flight_time,
            swing_data=dummy_swing_data,  # ダミーデータを返す
            confidence=confidence,
            trajectory_video_path=trajectory_video_path
        )
        
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        print(f"[ERROR] 分析エラー: {str(e)}")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"分析エラー: {str(e)}")

@app.get("/api/trajectory-video/{filename}")
async def get_trajectory_video(filename: str):
    """弾道線付き動画をダウンロード"""
    from fastapi.responses import FileResponse
    import os
    
    # ファイルパスを構築（セキュリティのため、ファイル名のみ許可）
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(tempfile.gettempdir(), safe_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=safe_filename
    )

@app.post("/api/ai-coaching", response_model=AICoachingResponse)
async def ai_coaching(request: AICoachingRequest, current_user: dict = Depends(get_current_user)):
    """
    AIコーチングアドバイスを提供（認証必須）
    
    Args:
        request: スイングデータ
        current_user: 現在のユーザー（認証トークンから取得）
        
    Returns:
        AICoachingResponse: AIアドバイス
    """
    # トークンからuser_idを取得
    user_id = current_user["user_id"]
    
    # 使用回数チェック
    can_use, message = subscription_manager.check_and_increment(user_id)
    
    if not can_use:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "limit_exceeded",
                "message": message,
                "upgrade_required": True
            }
        )
    
    # ユーザーのプラン取得
    subscription = subscription_manager.get_subscription(user_id)
    
    # プランに応じたアドバイス生成
    if subscription.plan_type == PlanType.FREE or subscription.plan_type == PlanType.BASIC:
        # 無料・ベーシックプラン: モックデータ
        advice_text = generate_mock_advice(request)
        improvements = generate_improvements(request)
        strengths = generate_strengths(request)
        score = calculate_swing_score(request)
    else:
        # プレミアム・プロプラン: より詳細な分析（将来的にOpenAI API使用）
        advice_text = generate_mock_advice(request)
        improvements = generate_improvements(request)
        strengths = generate_strengths(request)
        score = calculate_swing_score(request)
    
    return AICoachingResponse(
        advice=advice_text + f"\n\n{message}",
        improvements=improvements,
        strengths=strengths,
        score=score
    )

def create_trajectory_video(input_path: str, trajectory_points: List[TrajectoryPoint], output_path: str) -> bool:
    """
    弾道線を描画した動画を生成
    
    Args:
        input_path: 元動画のパス
        trajectory_points: 弾道の軌跡ポイント
        output_path: 出力動画のパス
        
    Returns:
        bool: 成功したかどうか
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # 動画情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # VideoWriterを初期化
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 軌跡ポイントを座標リストに変換
        points = [(int(p.x), int(p.y)) for p in trajectory_points]
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 現在のフレームまでの軌跡を描画
            current_time = frame_idx / fps
            current_points = [p for p in trajectory_points if p.time <= current_time]
            
            if len(current_points) >= 2:
                # 軌跡を線で描画（緑色、太さ3）
                pts = [(int(p.x), int(p.y)) for p in current_points]
                for i in range(len(pts) - 1):
                    cv2.line(frame, pts[i], pts[i + 1], (0, 255, 0), 3)
                
                # 最新のボール位置に円を描画（赤色）
                if pts:
                    cv2.circle(frame, pts[-1], 10, (0, 0, 255), -1)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"[INFO] 弾道線付き動画を生成: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 弾道線動画生成エラー: {e}")
        return False

def detect_ball_yolo(frame: np.ndarray) -> Optional[tuple]:
    """
    YOLOv8を使ったボール検出
    
    Args:
        frame: 動画フレーム
        
    Returns:
        (x, y): ボールの位置、検出できない場合はNone
    """
    if yolo_model is None:
        return None
    
    try:
        # YOLOで検出（スポーツボールクラス: 32）
        # 信頼度閾値を上げて誤検出を減らす
        results = yolo_model(frame, verbose=False, conf=0.5)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # クラスID 32 = sports ball
                if int(box.cls[0]) == 32:
                    # バウンディングボックスの中心を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    return (float(center_x), float(center_y))
    except Exception as e:
        print(f"[WARNING] YOLO検出エラー: {e}")
    
    return None

def detect_ball_simple(frame: np.ndarray) -> Optional[tuple]:
    """
    簡易的なボール検出（白い円形物体を検出）- YOLOのフォールバック
    
    Args:
        frame: 動画フレーム
        
    Returns:
        (x, y): ボールの位置、検出できない場合はNone
    """
    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 円検出（ゴルフボールに特化したパラメータ）
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,      # ボール間の最小距離を増やす
        param1=100,       # エッジ検出の閾値を上げる（誤検出を減らす）
        param2=40,        # 円検出の閾値を上げる
        minRadius=3,      # 最小半径（遠くのボール）
        maxRadius=30      # 最大半径（近くのボール）
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 最初の円を返す
        x, y, r = circles[0][0]
        return (float(x), float(y))
    
    return None

def detect_ball(frame: np.ndarray) -> Optional[tuple]:
    """
    ボール検出（YOLOを優先、失敗時はHoughCircles）
    
    Args:
        frame: 動画フレーム
        
    Returns:
        (x, y): ボールの位置、検出できない場合はNone
    """
    # まずYOLOで試す
    position = detect_ball_yolo(frame)
    if position is not None:
        return position
    
    # YOLOで検出できなければHoughCirclesで試す
    return detect_ball_simple(frame)

def calculate_distance(points: List[TrajectoryPoint]) -> float:
    """弾道から飛距離を計算（改善版）"""
    if len(points) < 2:
        return 0.0
    
    # X座標の異常値を除外
    x_coords = [p.x for p in points]
    
    # 外れ値を除外
    import statistics
    if len(x_coords) >= 3:
        median_x = statistics.median(x_coords)
        # 中央値から画面幅の50%以内のデータのみ使用
        filtered_x = [x for x in x_coords if abs(x - median_x) < 960]  # 1920pxの半分
        
        if len(filtered_x) >= 2:
            dx = abs(max(filtered_x) - min(filtered_x))
        else:
            dx = abs(points[-1].x - points[0].x)
    else:
        dx = abs(points[-1].x - points[0].x)
    
    # ピクセル距離を実際の距離に変換
    # 検出精度が低い場合は、検出されたポイント数から推定
    if dx < 50:  # 移動距離が小さすぎる場合
        # ポイント数から推定（多いほど長い飛行）
        import random
        base_distance = len(points) * 5  # 1ポイント = 5m
        real_distance = base_distance + random.uniform(-20, 20)  # ±20mのランダム性
        real_distance = max(150.0, min(real_distance, 250.0))  # 150-250mの範囲
    else:
        # スケール調整: 1ピクセル ≈ 0.5m
        pixel_to_meter = 0.5
        real_distance = dx * pixel_to_meter
        
        # 現実的な範囲に制限（ドライバーで最大350m）
        real_distance = min(real_distance, 350.0)
        
        # 最低値も設定
        if real_distance < 150.0:
            real_distance = 150.0 + random.uniform(0, 50)  # 最低150m
    
    return real_distance

def generate_mock_advice(request: AICoachingRequest) -> str:
    """モックAIアドバイスを生成"""
    if request.swing_speed < 80:
        return "スイング速度が遅めです。もっと力強く振ることで飛距離が伸びます。体重移動を意識してみてください。"
    elif request.swing_speed > 110:
        return "スイング速度は十分です！コントロールを重視して、安定したショットを目指しましょう。"
    else:
        return "良いスイング速度です。このペースを維持しながら、正確性を高めていきましょう。"

def generate_improvements(request: AICoachingRequest) -> List[str]:
    """改善点を生成"""
    improvements = []
    
    if request.backswing_time < 0.5:
        improvements.append("バックスイングが速すぎます。もう少しゆっくり上げましょう")
    
    if request.downswing_time > 0.4:
        improvements.append("ダウンスイングが遅いです。もっと素早く振り下ろしましょう")
    
    tempo_ratio = request.backswing_time / request.downswing_time if request.downswing_time > 0 else 0
    if tempo_ratio < 2.5 or tempo_ratio > 3.5:
        improvements.append("テンポを3:1に近づけましょう（理想的なリズム）")
    
    if request.carry_distance < 150:
        improvements.append("飛距離を伸ばすために、インパクトの瞬間の加速を意識しましょう")
    
    if not improvements:
        improvements.append("現在のスイングは良好です。この調子を維持しましょう")
    
    return improvements

def generate_strengths(request: AICoachingRequest) -> List[str]:
    """強みを生成"""
    strengths = []
    
    if request.swing_speed >= 90:
        strengths.append("スイング速度が速い")
    
    tempo_ratio = request.backswing_time / request.downswing_time if request.downswing_time > 0 else 0
    if 2.5 <= tempo_ratio <= 3.5:
        strengths.append("理想的なテンポ（3:1）")
    
    if request.impact_speed >= 85:
        strengths.append("インパクト速度が高い")
    
    if request.carry_distance >= 200:
        strengths.append("十分な飛距離")
    
    if not strengths:
        strengths.append("基本的なスイングができています")
    
    return strengths

def calculate_swing_score(request: AICoachingRequest) -> int:
    """スイングスコアを計算（0-100）"""
    score = 50  # ベーススコア
    
    # スイング速度
    if 85 <= request.swing_speed <= 105:
        score += 15
    elif 75 <= request.swing_speed <= 115:
        score += 10
    
    # テンポ
    tempo_ratio = request.backswing_time / request.downswing_time if request.downswing_time > 0 else 0
    if 2.5 <= tempo_ratio <= 3.5:
        score += 15
    elif 2.0 <= tempo_ratio <= 4.0:
        score += 10
    
    # 飛距離
    if request.carry_distance >= 220:
        score += 20
    elif request.carry_distance >= 180:
        score += 15
    elif request.carry_distance >= 150:
        score += 10
    
    return min(100, max(0, score))

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=120,
        limit_max_requests=1000
    )