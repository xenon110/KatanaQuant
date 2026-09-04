import asyncio
from src.storage.supabase_client import SupabaseManager

async def test_supabase_connection():
    mgr = SupabaseManager()
    print(f"Supabase URL: {mgr.url}")
    print(f"Is Configured: {mgr.is_configured}")
    
    if mgr.is_configured:
        import time
        # Test 1: Order Insert
        order_res = await mgr.log_order(
            order_id=f"TEST_ORD_{int(time.time())}",
            client_order_id="CLIENT_001",
            symbol="NVDA",
            side="BUY",
            quantity=15,
            price=128.50,
            status="FILLED"
        )
        print(f"[OK] Orders table write: {order_res is not None}")

        # Test 2: Signal Log Insert
        signal_res = await mgr.log_signal(
            symbol="NVDA",
            direction="BULLISH",
            confidence=0.88,
            reasoning="Triple EMA Confluence setup confirmed."
        )
        print(f"[OK] Signal logs write: {signal_res is not None}")

        # Test 3: Risk Decision Insert
        risk_res = await mgr.log_risk_decision(
            symbol="NVDA",
            approved=True,
            allowed_quantity=15,
            violations=[],
            reasons=["All deterministic risk gate checks passed."]
        )
        print(f"[OK] Risk decision logs write: {risk_res is not None}")

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
