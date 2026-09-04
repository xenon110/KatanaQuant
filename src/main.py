"""
Main Entry Point & Command Line Interface for the Automated Multi-Agent Trading System.
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.config.settings import settings
from src.core.enums import TradingMode, AccountType, OrderSide
from src.core.models import MarketBar
from src.data.market_data import BarCache, SyntheticMarketDataProvider, AlpacaMarketDataProvider
from src.strategy.rules import EMACrossRSIStrategy
from src.backtesting.engine import BacktestEngine
from src.risk.gate import DeterministicRiskGate
from src.reconciliation.service import ReconciliationService
from src.storage.database import DatabaseManager
from src.orchestrator.pipeline import TradingOrchestrator
from src.execution.dry_run_broker import DryRunBroker
from src.execution.alpaca_broker import AlpacaBrokerClient

app = typer.Typer(help="KatanaQuant - Automated Multi-Agent Stock Trading System", invoke_without_command=True)
console = Console()


@app.callback()
def main_callback(ctx: typer.Context):
    """If no subcommand is passed, default to launching the Web UI Dashboard."""
    if ctx.invoked_subcommand is None:
        ui()


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host interface to bind"),
    port: int = typer.Option(8000, help="Port to serve dashboard UI on")
):
    """Launch the real-time visual Web Trading Dashboard."""
    import uvicorn
    actual_host = host if isinstance(host, str) else "127.0.0.1"
    actual_port = port if isinstance(port, int) else 8000

    console.print("\n")
    console.print(Panel(
        f"[bold green]🚀 KatanaQuant Trading Dashboard is Online![/bold green]\n\n"
        f"👉 [bold yellow]Click or open the link below in your browser:[/bold yellow]\n"
        f"🔗 [bold cyan underline]http://{actual_host}:{actual_port}[/bold cyan underline]\n"
        f"🔗 [bold cyan underline]http://localhost:{actual_port}[/bold cyan underline]\n\n"
        f"[dim]Multi-Agent AI Hub • Deterministic Risk Gate • Real-Time Trading Engine[/dim]",
        title="[bold white]Localhost Dashboard Server[/bold white]",
        border_style="bright_blue",
        expand=False
    ))
    console.print("\n")
    uvicorn.run("src.web.app:app", host=actual_host, port=actual_port, reload=False)




@app.command()
def backtest(
    symbol: str = typer.Option("SPY", help="Ticker symbol to backtest"),
    days: int = typer.Option(30, help="Number of historical days to backtest"),
    capital: float = typer.Option(30000.0, help="Initial account capital")
):
    """Run an event-driven backtest of the deterministic rule strategy."""
    console.print(Panel(f"[bold cyan]Running Backtest on {symbol}[/bold cyan] ({days} days, Initial Capital: ${capital:,.2f})", expand=False))

    data_provider = SyntheticMarketDataProvider(seed_price=150.0)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    bars = asyncio.run(data_provider.get_historical_bars(symbol, "5Min", start_dt, end_dt))
    strategy = EMACrossRSIStrategy()
    engine = BacktestEngine(strategy=strategy, initial_capital=capital)
    
    result = engine.run(bars)

    table = Table(title=f"Backtest Performance Summary: {symbol}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
    table.add_row("Final Equity", f"${result.final_equity:,.2f}")
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    table.add_row("Total PnL", f"[{pnl_color}]${result.total_pnl:,.2f} ({result.total_return_pct:.2f}%)[/{pnl_color}]")
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Win Rate", f"{result.win_rate * 100:.1f}% ({result.winning_trades}W / {result.losing_trades}L)")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

    console.print(table)


@app.command()
def dry_run(
    symbols: str = typer.Option("AAPL,MSFT,NVDA,SPY", help="Comma-separated symbols to monitor"),
    bars_count: int = typer.Option(50, help="Number of simulated minute bars to stream")
):
    """Run live trading pipeline in DRY-RUN mode (safe offline simulation)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    console.print(Panel(f"[bold green]Starting DRY-RUN Trading Pipeline[/bold green]\nMonitoring Universe: {', '.join(symbol_list)}", expand=False))

    asyncio.run(_run_pipeline_loop(symbol_list, mode=TradingMode.DRY_RUN, total_iterations=bars_count))


@app.command()
def live(
    confirm_real_money: bool = typer.Option(False, "--confirm-real-money", help="Explicit confirmation for real money live execution"),
    paper: bool = typer.Option(True, "--paper/--live", help="Use Alpaca paper environment or live real money")
):
    """Start live trading loop with broker connection."""
    if not paper and not confirm_real_money:
        console.print("[bold red]ERROR: To trade live real money, you must provide --confirm-real-money[/bold red]")
        raise typer.Exit(code=1)

    target_mode = TradingMode.PAPER if paper else TradingMode.LIVE
    mode_text = "[yellow]PAPER TRADING[/yellow]" if paper else "[red bold]REAL MONEY LIVE TRADING[/red bold]"
    console.print(Panel(f"Starting {mode_text}\nBroker: Alpaca\nSymbols: {', '.join(settings.watchlist_symbols)}", expand=False))

    asyncio.run(_run_pipeline_loop(settings.watchlist_symbols, mode=target_mode, total_iterations=30))


@app.command()
def kill_switch():
    """Emergency kill switch: cancel all open orders and halt trading."""
    console.print("[bold red]TRIGGERING EMERGENCY KILL SWITCH...[/bold red]")
    risk_gate = DeterministicRiskGate(settings)
    risk_gate.trip_circuit_breaker("Manual Kill Switch Invoked by User")
    console.print("[bold green]System halted. All orders blocked.[/bold green]")


async def _run_pipeline_loop(symbols: List[str], mode: TradingMode, total_iterations: int = 20):
    db = DatabaseManager(settings.database_url)
    reconciler = ReconciliationService()
    risk_gate = DeterministicRiskGate(settings)
    strategy = EMACrossRSIStrategy()
    orchestrator = TradingOrchestrator(strategy=strategy, risk_gate=risk_gate)
    bar_cache = BarCache()

    # Choose broker client
    if mode == TradingMode.DRY_RUN or not settings.alpaca_api_key or settings.alpaca_api_key == "placeholder_key":
        broker = DryRunBroker(initial_equity=settings.initial_account_equity, account_type=settings.account_type)
        data_provider = SyntheticMarketDataProvider(seed_price=150.0)
    else:
        broker = AlpacaBrokerClient(api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key, paper=settings.alpaca_paper)
        data_provider = AlpacaMarketDataProvider(api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key, paper=settings.alpaca_paper)

    console.print(f"[cyan]Initializing pipeline for {len(symbols)} symbols...[/cyan]")

    for step in range(total_iterations):
        for sym in symbols:
            # 1. Fetch / generate new bar
            now = datetime.now(timezone.utc)
            new_bar = MarketBar(
                symbol=sym,
                timestamp=now,
                open=150.0 + (step * 0.2),
                high=151.0 + (step * 0.2),
                low=149.5 + (step * 0.2),
                close=150.8 + (step * 0.2),
                volume=15000.0
            )
            bar_cache.add_bar(new_bar)
            df = bar_cache.get_dataframe(sym)

            # 2. Get live reconciled account state
            raw_acc = await broker.get_account()
            account = reconciler.reconcile_account(raw_acc)
            positions = await broker.get_positions()

            # 3. Process bar through Orchestrator and Risk Gate
            result = await orchestrator.process_bar(
                bar=new_bar,
                history_df=df,
                account=account,
                positions=positions,
                data_staleness_seconds=0.5
            )

            if result:
                trade, decision = result
                # Log audit trail to DB
                db.log_risk_decision(
                    symbol=sym,
                    approved=decision.approved,
                    status=decision.status.value,
                    orig_qty=decision.original_quantity,
                    allow_qty=decision.allowed_quantity,
                    violations=decision.rule_violations,
                    reasons=decision.rejection_reasons
                )

                status_tag = f"[bold green]APPROVED[/bold green]" if decision.approved else f"[bold red]REJECTED[/bold red]"
                console.print(
                    f"[{now.strftime('%H:%M:%S')}] Signal on {sym} | Decision: {status_tag} | "
                    f"Action: {trade.side.value} {decision.allowed_quantity} shares @ ${trade.price:.2f}"
                )

                if decision.approved and decision.allowed_quantity > 0:
                    order = await broker.submit_order(trade, decision)
                    db.save_order(
                        order_id=order.order_id,
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        qty=order.quantity,
                        status=order.status.value
                    )
                    reconciler.record_trade_fill(
                        symbol=sym,
                        side=trade.side,
                        quantity=decision.allowed_quantity,
                        price=trade.price,
                        order_id=order.order_id
                    )

        await asyncio.sleep(0.05) # Fast simulation interval

    # Summary table
    final_acc = await broker.get_account()
    console.print("\n")
    summary_table = Table(title="Execution Session Summary")
    summary_table.add_column("Account Equity", style="green")
    summary_table.add_column("Cash", style="cyan")
    summary_table.add_column("Settled Cash", style="cyan")
    summary_table.add_column("Day Trades Recorded", style="yellow")
    summary_table.add_column("Realized PnL", style="magenta")

    summary_table.add_row(
        f"${final_acc.equity:,.2f}",
        f"${final_acc.cash:,.2f}",
        f"${final_acc.settled_cash:,.2f}",
        str(final_acc.day_trade_count),
        f"${final_acc.daily_realized_pnl:,.2f}"
    )
    console.print(summary_table)


if __name__ == "__main__":
    app()
