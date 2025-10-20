"""
SwingTrace AI Coaching Server
FastAPI-based server for golf swing analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from typing import List, Dict, Optional
import tempfile
import os
import sys
sys.path.append(os.path.dirname(__file__))
from models.subscription import (
    subscription_manager, 
    PlanType, 
    PLANS,
    UserSubscription
)

app = FastAPI(title="SwingTrace AI Server", version="1.0.0")

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

@app.get("/")
@app.head("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "message": "SwingTrace AI Server is running",
        "version": "1.0.0"
    }

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
        max_dimension = 1280  # HD解像度に制限
        scale_factor = 1.0
        if width > max_dimension or height > max_dimension:
            scale_factor = max_dimension / max(width, height)
            print(f"[INFO] メモリ節約のためリサイズ: {scale_factor:.2f}x")
        
        # フレームスキップ設定（処理速度向上のため）
        frame_skip = max(1, int(fps / 10))
        print(f"[INFO] フレームスキップ: {frame_skip}フレームごとに処理")
        
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
            
            # ボール検出（簡易版 - 色ベース）
            ball_pos = detect_ball_simple(frame)
            
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
            max_height = max([p.y for p in trajectory_points])
            flight_time = trajectory_points[-1].time - trajectory_points[0].time
        else:
            carry_distance = 0.0
            max_height = 0.0
            flight_time = 0.0
        
        print(f"[INFO] 分析結果: ボール検出={ball_detected}, 軌跡点数={len(trajectory_points)}")
        
        return AnalysisResult(
            ball_detected=ball_detected,
            trajectory=trajectory_points,
            carry_distance=carry_distance,
            max_height=max_height,
            flight_time=flight_time,
            swing_data=None,
            confidence=0.85 if ball_detected else 0.0
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

@app.post("/api/ai-coaching", response_model=AICoachingResponse)
async def ai_coaching(request: AICoachingRequest):
    """
    AIコーチングアドバイスを提供
    
    Args:
        request: スイングデータ
        
    Returns:
        AICoachingResponse: AIアドバイス
    """
    # 使用回数チェック
    can_use, message = subscription_manager.check_and_increment(request.user_id)
    
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
    subscription = subscription_manager.get_subscription(request.user_id)
    
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

def detect_ball_simple(frame: np.ndarray) -> Optional[tuple]:
    """
    簡易的なボール検出（白い円形物体を検出）
    
    Args:
        frame: 動画フレーム
        
    Returns:
        (x, y): ボールの位置、検出できない場合はNone
    """
    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 円検出
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 最初の円を返す
        x, y, r = circles[0][0]
        return (float(x), float(y))
    
    return None

def calculate_distance(points: List[TrajectoryPoint]) -> float:
    """弾道から飛距離を計算"""
    if len(points) < 2:
        return 0.0
    
    # 最初と最後の点の水平距離
    dx = points[-1].x - points[0].x
    dy = points[-1].y - points[0].y
    
    # ピクセル距離を実際の距離に変換（仮定: 1ピクセル = 0.1m）
    pixel_distance = np.sqrt(dx**2 + dy**2)
    real_distance = pixel_distance * 0.1
    
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
