"""
サブスクリプション管理
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

class PlanType(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    PRO = "pro"

class PlanLimits(BaseModel):
    """プランごとの制限"""
    plan_type: PlanType
    monthly_limit: int  # 月間AI分析回数（-1は無制限）
    price_jpy: int  # 月額料金（円）
    features: list[str]  # 利用可能な機能

# プラン定義
PLANS = {
    PlanType.FREE: PlanLimits(
        plan_type=PlanType.FREE,
        monthly_limit=5,
        price_jpy=0,
        features=[
            "弾道表示（無制限）",
            "AI分析（月5回）",
            "簡易アドバイス"
        ]
    ),
    PlanType.BASIC: PlanLimits(
        plan_type=PlanType.BASIC,
        monthly_limit=30,
        price_jpy=300,
        features=[
            "弾道表示（無制限）",
            "AI分析（月30回）",
            "詳細アドバイス",
            "履歴保存（30日間）"
        ]
    ),
    PlanType.PREMIUM: PlanLimits(
        plan_type=PlanType.PREMIUM,
        monthly_limit=100,
        price_jpy=980,
        features=[
            "弾道表示（無制限）",
            "AI分析（月100回）",
            "本物のAIコーチング",
            "履歴保存（無制限）",
            "動画保存"
        ]
    ),
    PlanType.PRO: PlanLimits(
        plan_type=PlanType.PRO,
        monthly_limit=-1,  # 無制限
        price_jpy=2980,
        features=[
            "すべての機能",
            "AI分析（無制限）",
            "優先サポート",
            "プロコーチ機能",
            "詳細統計"
        ]
    )
}

class UserSubscription(BaseModel):
    """ユーザーのサブスクリプション情報"""
    user_id: str
    plan_type: PlanType
    monthly_used: int  # 今月の使用回数
    reset_date: datetime  # リセット日（毎月1日）
    subscribed_at: datetime
    expires_at: Optional[datetime] = None

    def can_use_ai_analysis(self) -> bool:
        """AI分析を使用できるか"""
        plan = PLANS[self.plan_type]
        
        # 無制限プラン
        if plan.monthly_limit == -1:
            return True
        
        # 使用回数チェック
        return self.monthly_used < plan.monthly_limit
    
    def get_remaining_count(self) -> int:
        """残り使用可能回数"""
        plan = PLANS[self.plan_type]
        
        if plan.monthly_limit == -1:
            return -1  # 無制限
        
        return max(0, plan.monthly_limit - self.monthly_used)
    
    def increment_usage(self):
        """使用回数を増やす"""
        self.monthly_used += 1
    
    def should_reset(self) -> bool:
        """リセットが必要か"""
        return datetime.now() >= self.reset_date
    
    def reset_monthly_usage(self):
        """月次使用回数をリセット"""
        self.monthly_used = 0
        # 次月の1日に設定
        now = datetime.now()
        if now.month == 12:
            self.reset_date = datetime(now.year + 1, 1, 1)
        else:
            self.reset_date = datetime(now.year, now.month + 1, 1)

class SubscriptionManager:
    """サブスクリプション管理クラス"""
    
    def __init__(self):
        # 本番環境ではデータベースを使用
        # 開発環境ではメモリ上に保存
        self.subscriptions: dict[str, UserSubscription] = {}
    
    def get_subscription(self, user_id: str) -> UserSubscription:
        """ユーザーのサブスクリプション情報を取得"""
        if user_id not in self.subscriptions:
            # 新規ユーザーは無料プラン
            self.subscriptions[user_id] = UserSubscription(
                user_id=user_id,
                plan_type=PlanType.FREE,
                monthly_used=0,
                reset_date=self._get_next_reset_date(),
                subscribed_at=datetime.now()
            )
        
        subscription = self.subscriptions[user_id]
        
        # 月次リセットチェック
        if subscription.should_reset():
            subscription.reset_monthly_usage()
        
        return subscription
    
    def upgrade_plan(self, user_id: str, new_plan: PlanType) -> UserSubscription:
        """プランをアップグレード"""
        subscription = self.get_subscription(user_id)
        subscription.plan_type = new_plan
        subscription.subscribed_at = datetime.now()
        
        # 有料プランの場合、有効期限を設定（30日間）
        if new_plan != PlanType.FREE:
            subscription.expires_at = datetime.now() + timedelta(days=30)
        
        return subscription
    
    def check_and_increment(self, user_id: str) -> tuple[bool, str]:
        """
        使用可能かチェックして、使用回数を増やす
        
        Returns:
            (使用可能か, メッセージ)
        """
        subscription = self.get_subscription(user_id)
        
        if not subscription.can_use_ai_analysis():
            plan = PLANS[subscription.plan_type]
            return False, f"月間制限（{plan.monthly_limit}回）に達しました。プランをアップグレードしてください。"
        
        subscription.increment_usage()
        remaining = subscription.get_remaining_count()
        
        if remaining == -1:
            message = "無制限プランをご利用中です"
        else:
            message = f"残り{remaining}回利用可能です"
        
        return True, message
    
    def _get_next_reset_date(self) -> datetime:
        """次のリセット日を取得（翌月1日）"""
        now = datetime.now()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        else:
            return datetime(now.year, now.month + 1, 1)

# グローバルインスタンス
subscription_manager = SubscriptionManager()
