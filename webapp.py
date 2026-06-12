from __future__ import annotations

import asyncio
import base64
import math
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from chart_generator import generate_backtest_chart
from stock_analyzer import AnalyzerError, analyze_ticker, pct, _fmt_ratio, _fmt_days

app = FastAPI(title="Long-Term Quant Analyzer")

# 결과 캐시 (1시간)
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 3600

# 백그라운드 작업 저장소
_jobs: dict[str, dict[str, Any]] = {}


def _get_cached(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


def _build_payload(result: Any) -> dict[str, Any]:
    chart_bytes = generate_backtest_chart(result)
    chart_b64 = base64.b64encode(chart_bytes).decode() if chart_bytes else None

    pf = result.profit_factor
    pf_str = "∞" if math.isinf(pf) and pf > 0 else _fmt_ratio(pf)

    trades_data: list[dict[str, Any]] = []
    if result.trades is not None and not result.trades.empty:
        for _, row in result.trades.iterrows():
            trades_data.append({
                "entry_date": str(row["Entry Date"]),
                "entry_price": row["Entry Price"],
                "exit_date": str(row["Exit Date"]),
                "exit_price": row["Exit Price"],
                "return_pct": round(row["Return"] * 100, 2),
                "holding_days": int(row["Holding Days"]),
                "exit_reason": row["Exit Reason"],
            })

    return {
        "ticker": result.ticker,
        "as_of": result.as_of,
        "start_date": result.start_date,
        "rows": result.rows,
        "action": result.action,
        "regime": result.regime,
        "confidence": result.confidence,
        "invested_now": result.invested_now,
        "entry_score": result.entry_score,
        "entry_verdict": result.entry_verdict,
        "entry_factors": result.entry_factors or [],
        "metrics": {
            "current_price": result.current_price,
            "ema50": result.ema50,
            "sma200": result.sma200,
            "atr_stop": result.trailing_stop,
            "cagr_strategy": pct(result.cagr_strategy),
            "cagr_buy_hold": pct(result.cagr_buy_hold),
            "mdd_strategy": pct(result.mdd_strategy),
            "mdd_buy_hold": pct(result.mdd_buy_hold),
            "cumulative_strategy": f"{result.cumulative_strategy:.2f}x",
            "market_exposure": pct(result.market_exposure),
            "sharpe": _fmt_ratio(result.sharpe_ratio),
            "sortino": _fmt_ratio(result.sortino_ratio),
            "win_rate": pct(result.win_rate),
            "profit_factor": pf_str,
            "num_trades": result.num_trades,
            "avg_holding": _fmt_days(result.avg_holding_days),
            "best_trade": pct(result.best_trade),
            "worst_trade": pct(result.worst_trade),
            "kelly": pct(result.kelly_fraction),
            "fractional_kelly": pct(result.suggested_fractional_kelly),
        },
        "insights": result.insights,
        "warnings": result.warnings,
        "trades": trades_data,
        "chart": chart_b64,
    }


async def _run_job(job_id: str, ticker: str, period: str, cache_key: str) -> None:
    try:
        result = await asyncio.to_thread(analyze_ticker, ticker, period)
        payload = await asyncio.to_thread(_build_payload, result)
        _set_cache(cache_key, payload)
        _jobs[job_id] = {"status": "done", "payload": payload}
    except AnalyzerError as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": f"분석 오류: {exc}"}


class AnalyzeRequest(BaseModel):
    ticker: str
    period: str = "max"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    ticker = req.ticker.strip().upper()
    if not ticker:
        return JSONResponse({"error": "티커를 입력하세요."}, status_code=400)

    cache_key = f"{ticker}:{req.period}"
    cached = _get_cached(cache_key)
    if cached:
        return JSONResponse({"status": "done", "payload": cached})

    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"status": "pending"}
    asyncio.create_task(_run_job(job_id, ticker, req.period, cache_key))
    return JSONResponse({"status": "pending", "job_id": job_id})


@app.get("/result/{job_id}")
async def get_result(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "error": "작업을 찾을 수 없습니다."}, status_code=404)
    if job["status"] == "pending":
        return JSONResponse({"status": "pending"})
    if job["status"] == "error":
        return JSONResponse({"status": "error", "error": job["error"]})
    return JSONResponse({"status": "done", "payload": job["payload"]})


HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Long-Term Quant</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #f0883e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }
  .container { max-width: 700px; margin: 0 auto; padding: 16px; }

  header { text-align: center; padding: 24px 0 20px; }
  header h1 { font-size: 1.3rem; font-weight: 700; color: var(--accent); }
  header p { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

  .search-box { display: flex; gap: 8px; margin-bottom: 20px; }
  .search-box input {
    flex: 1; background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px; padding: 12px 14px;
    font-size: 1rem; outline: none; text-transform: uppercase;
  }
  .search-box input::placeholder { text-transform: none; color: var(--muted); }
  .search-box input:focus { border-color: var(--accent); }
  .search-box button {
    background: var(--accent); color: #0d1117; border: none;
    border-radius: 8px; padding: 12px 20px; font-size: 0.95rem;
    font-weight: 600; cursor: pointer; white-space: nowrap;
  }
  .search-box button:disabled { opacity: 0.5; cursor: default; }

  .spinner { display: none; text-align: center; padding: 40px; color: var(--muted); font-size: 0.9rem; }
  .spinner.active { display: block; }

  .error-box { background: #2d1b1b; border: 1px solid var(--red); border-radius: 8px; padding: 14px; color: var(--red); font-size: 0.9rem; margin-bottom: 16px; display: none; }
  .error-box.active { display: block; }

  #result { display: none; }
  #result.active { display: block; }

  .verdict-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 22px 18px; margin-bottom: 14px; text-align: center; }
  .verdict-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }
  .verdict-main { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 14px; }
  .verdict-main.buy { color: var(--green); }
  .verdict-main.consider { color: #7ee787; }
  .verdict-main.wait { color: var(--orange); }
  .verdict-main.avoid { color: var(--red); }
  .gauge { height: 10px; background: #21262d; border-radius: 6px; overflow: hidden; margin-bottom: 8px; }
  .gauge-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
  .verdict-score { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; }
  .verdict-score-max { font-size: 0.8rem; color: var(--muted); font-weight: 400; }
  .factors { display: flex; flex-direction: column; gap: 8px; text-align: left; }
  .factor { background: var(--bg); border-radius: 8px; padding: 9px 11px; }
  .factor-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
  .factor-name { font-size: 0.82rem; font-weight: 600; }
  .factor-score { font-size: 0.82rem; font-weight: 700; }
  .factor-detail { font-size: 0.72rem; color: var(--muted); }
  .factor-bar { height: 4px; background: #21262d; border-radius: 3px; margin-top: 5px; overflow: hidden; }
  .factor-bar-fill { height: 100%; border-radius: 3px; }

  .result-header { margin-bottom: 16px; }
  .result-header .ticker-name { font-size: 1.4rem; font-weight: 700; }
  .result-header .meta { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
  .action-badge {
    display: inline-block; margin-top: 8px; padding: 6px 12px;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600;
  }
  .action-badge.bull { background: #1a3a2a; color: var(--green); border: 1px solid var(--green); }
  .action-badge.bear { background: #2d1b1b; color: var(--red); border: 1px solid var(--red); }
  .action-badge.hold { background: #1a2a3a; color: var(--accent); border: 1px solid var(--accent); }

  .section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
  .section-title { font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }

  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .metric { background: var(--bg); border-radius: 8px; padding: 10px 12px; }
  .metric .label { font-size: 0.72rem; color: var(--muted); margin-bottom: 3px; }
  .metric .value { font-size: 1.05rem; font-weight: 700; }
  .value.positive { color: var(--green); }
  .value.negative { color: var(--red); }
  .value.neutral { color: var(--accent); }

  .chart-wrap { margin-bottom: 12px; }
  .chart-wrap img { width: 100%; border-radius: 10px; display: block; }

  .trades-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
  .trades-table th { color: var(--muted); font-weight: 500; padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  .trades-table td { padding: 7px 8px; border-bottom: 1px solid #21262d; }
  .trades-table tr:last-child td { border-bottom: none; }
  .ret-pos { color: var(--green); font-weight: 600; }
  .ret-neg { color: var(--red); font-weight: 600; }

  .insights { list-style: none; }
  .insights li { font-size: 0.83rem; padding: 5px 0; border-bottom: 1px solid #21262d; color: var(--text); }
  .insights li:last-child { border-bottom: none; }
  .insights li::before { content: "✦ "; color: var(--accent); font-size: 0.7rem; }
  .warn li::before { color: var(--orange); }

  .tag { display: inline-block; font-size: 0.7rem; padding: 2px 7px; border-radius: 10px; margin-left: 4px; vertical-align: middle; }
  .tag.bull { background: #1a3a2a; color: var(--green); }
  .tag.bear { background: #2d1b1b; color: var(--red); }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>진입 타이밍 분석기</h1>
    <p>정량 지표로 "지금 사도 될지"를 0~100점으로 판단</p>
  </header>

  <div class="search-box">
    <input id="tickerInput" type="text" placeholder="티커 입력 (예: AAPL, 005930.KS)" autocomplete="off" autocorrect="off" spellcheck="false">
    <button id="analyzeBtn" onclick="runAnalysis()">분석</button>
  </div>

  <div id="errorBox" class="error-box"></div>
  <div id="spinner" class="spinner"><p>⏳ 분석 시작 중...</p><small style="font-size:0.75rem;opacity:0.7">처음에는 데이터 수집으로 1~2분 걸릴 수 있습니다</small></div>

  <div id="result">
    <div class="result-header">
      <div>
        <span class="ticker-name" id="rTicker"></span>
        <span id="rInvestedTag"></span>
      </div>
      <div class="meta" id="rMeta"></div>
    </div>

    <div class="verdict-card" id="verdictCard">
      <div class="verdict-label">지금 진입해도 될까?</div>
      <div class="verdict-main" id="vVerdict"></div>
      <div class="gauge"><div class="gauge-fill" id="vGaugeFill"></div></div>
      <div class="verdict-score"><span id="vScore"></span> <span class="verdict-score-max">/ 100점</span></div>
      <div class="factors" id="vFactors"></div>
      <div id="rAction" class="action-badge"></div>
    </div>

    <div class="chart-wrap" id="chartWrap" style="display:none">
      <img id="chartImg" src="" alt="backtest chart">
    </div>

    <div class="section">
      <div class="section-title">백테스트 성과</div>
      <div class="metrics-grid">
        <div class="metric"><div class="label">전략 CAGR</div><div class="value" id="mCagr"></div></div>
        <div class="metric"><div class="label">B&H CAGR</div><div class="value neutral" id="mCagrBh"></div></div>
        <div class="metric"><div class="label">전략 MDD</div><div class="value" id="mMdd"></div></div>
        <div class="metric"><div class="label">누적 수익</div><div class="value" id="mCumul"></div></div>
        <div class="metric"><div class="label">Sharpe</div><div class="value neutral" id="mSharpe"></div></div>
        <div class="metric"><div class="label">Sortino</div><div class="value neutral" id="mSortino"></div></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">트레이드 통계</div>
      <div class="metrics-grid">
        <div class="metric"><div class="label">총 거래 수</div><div class="value neutral" id="mTrades"></div></div>
        <div class="metric"><div class="label">승률</div><div class="value" id="mWinRate"></div></div>
        <div class="metric"><div class="label">Profit Factor</div><div class="value" id="mPF"></div></div>
        <div class="metric"><div class="label">평균 보유</div><div class="value neutral" id="mAvgHold"></div></div>
        <div class="metric"><div class="label">최고 거래</div><div class="value positive" id="mBest"></div></div>
        <div class="metric"><div class="label">최저 거래</div><div class="value negative" id="mWorst"></div></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">현재 수준</div>
      <div class="metrics-grid">
        <div class="metric"><div class="label">현재가</div><div class="value neutral" id="mPrice"></div></div>
        <div class="metric"><div class="label">ATR 손절선</div><div class="value" id="mStop"></div></div>
        <div class="metric"><div class="label">EMA50</div><div class="value neutral" id="mEma"></div></div>
        <div class="metric"><div class="label">SMA200</div><div class="value neutral" id="mSma"></div></div>
        <div class="metric"><div class="label">시장 노출</div><div class="value neutral" id="mExposure"></div></div>
        <div class="metric"><div class="label">Kelly f*</div><div class="value neutral" id="mKelly"></div></div>
      </div>
    </div>

    <div class="section" id="tradesSection" style="display:none">
      <div class="section-title">거래 내역</div>
      <div style="overflow-x:auto">
        <table class="trades-table">
          <thead><tr><th>진입</th><th>청산</th><th>수익률</th><th>보유</th><th>이유</th></tr></thead>
          <tbody id="tradesBody"></tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <div class="section-title">핵심 해석</div>
      <ul class="insights" id="insightsList"></ul>
    </div>

    <div class="section">
      <div class="section-title">주의</div>
      <ul class="insights warn" id="warnList"></ul>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

function colorClass(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return '';
  return n >= 0 ? 'positive' : 'negative';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function setSpinner(msg) {
  const p = $('spinner').querySelector('p');
  if (p) p.textContent = msg;
}

async function postAnalyze(ticker) {
  // 서버 콜드스타트 대응: 실패 시 최대 10회(~60초) 재시도
  for (let attempt = 0; attempt < 10; attempt++) {
    if (attempt > 0) {
      setSpinner(`⏳ 서버 시작 중... ${attempt * 6}초 경과 (최대 60초)`);
      await sleep(6000);
    }
    try {
      const resp = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, period: 'max' }),
      });
      if (!resp.ok) continue; // 503 등 재시도
      const data = await resp.json();
      return data;
    } catch (_) { /* 네트워크 오류 → 재시도 */ }
  }
  return null;
}

async function runAnalysis() {
  const ticker = $('tickerInput').value.trim().toUpperCase();
  if (!ticker) { $('tickerInput').focus(); return; }

  $('analyzeBtn').disabled = true;
  $('spinner').classList.add('active');
  $('errorBox').classList.remove('active');
  $('result').classList.remove('active');
  setSpinner('⏳ 분석 시작 중...');

  try {
    // 1단계: 분석 작업 시작 (즉시 응답, 서버 콜드스타트 시 재시도 포함)
    const init = await postAnalyze(ticker);
    if (!init) { showError('서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.'); return; }
    if (init.error) { showError(init.error); return; }

    // 캐시 히트: 바로 렌더
    if (init.status === 'done') { render(init.payload); $('result').classList.add('active'); return; }

    // 2단계: 결과 폴링 (3초 간격, 최대 5분)
    const jobId = init.job_id;
    const deadline = Date.now() + 5 * 60 * 1000;
    let elapsed = 0;
    while (Date.now() < deadline) {
      await sleep(3000);
      elapsed += 3;
      setSpinner(`⏳ 분석 중... ${elapsed}초`);

      try {
        const poll = await fetch(`/result/${jobId}`);
        const data = await poll.json();
        if (data.status === 'done') {
          render(data.payload);
          $('result').classList.add('active');
          return;
        }
        if (data.status === 'error') { showError(data.error); return; }
      } catch (_) { /* 일시적 네트워크 오류 → 계속 폴링 */ }
    }
    showError('시간이 너무 오래 걸립니다. 잠시 후 다시 시도해주세요.');
  } finally {
    $('spinner').classList.remove('active');
    $('analyzeBtn').disabled = false;
  }
}

function showError(msg) {
  $('errorBox').textContent = msg;
  $('errorBox').classList.add('active');
}

function scoreColor(s) {
  if (s >= 75) return '#3fb950';
  if (s >= 60) return '#7ee787';
  if (s >= 40) return '#f0883e';
  return '#f85149';
}

function render(d) {
  const m = d.metrics;
  $('rTicker').textContent = d.ticker;
  $('rInvestedTag').innerHTML = d.invested_now
    ? '<span class="tag bull">보유 중</span>'
    : '<span class="tag bear">현금</span>';
  $('rMeta').textContent = `${d.start_date} ~ ${d.as_of} | ${d.rows.toLocaleString()}거래일 | 신뢰도: ${d.confidence}`;

  // 진입 판정 카드
  const score = d.entry_score ?? 0;
  const verdict = d.entry_verdict || '-';
  const vMain = $('vVerdict');
  vMain.textContent = verdict;
  const vClass = verdict.includes('적극') ? 'buy' : verdict.includes('고려') ? 'consider' : verdict.includes('관망') ? 'wait' : 'avoid';
  vMain.className = 'verdict-main ' + vClass;
  $('vScore').textContent = Math.round(score);
  const col = scoreColor(score);
  $('vGaugeFill').style.width = score + '%';
  $('vGaugeFill').style.background = col;

  const fWrap = $('vFactors');
  fWrap.innerHTML = (d.entry_factors || []).map(f => {
    const c = scoreColor(f.score);
    return `<div class="factor">
      <div class="factor-top">
        <span class="factor-name">${f.name} <span style="color:var(--muted);font-weight:400">(비중 ${f.weight}%)</span></span>
        <span class="factor-score" style="color:${c}">${f.score}점</span>
      </div>
      <div class="factor-detail">${f.detail}</div>
      <div class="factor-bar"><div class="factor-bar-fill" style="width:${f.score}%;background:${c}"></div></div>
    </div>`;
  }).join('');

  const act = $('rAction');
  act.textContent = '전략 신호: ' + d.action;
  act.className = 'action-badge ' + (d.invested_now ? 'bull' : d.action.includes('매수') ? 'bull' : d.action.includes('매도') || d.action.includes('청산') ? 'bear' : 'hold');

  if (d.chart) {
    $('chartImg').src = 'data:image/png;base64,' + d.chart;
    $('chartWrap').style.display = 'block';
  }

  function setMetric(id, val, forceClass) {
    const el = $(id);
    el.textContent = val;
    if (forceClass) el.className = 'value ' + forceClass;
    else el.className = 'value ' + colorClass(val);
  }

  setMetric('mCagr', m.cagr_strategy);
  setMetric('mCagrBh', m.cagr_buy_hold, 'neutral');
  setMetric('mMdd', m.mdd_strategy, 'negative');
  setMetric('mCumul', m.cumulative_strategy, 'neutral');
  setMetric('mSharpe', m.sharpe, 'neutral');
  setMetric('mSortino', m.sortino, 'neutral');
  $('mTrades').textContent = m.num_trades + '회';
  setMetric('mWinRate', m.win_rate);
  setMetric('mPF', m.profit_factor, parseFloat(m.profit_factor) >= 1 ? 'positive' : 'negative');
  $('mAvgHold').textContent = m.avg_holding;
  $('mBest').textContent = m.best_trade;
  $('mWorst').textContent = m.worst_trade;

  const fmt = n => n >= 100 ? n.toLocaleString('en', {maximumFractionDigits:2}) : n.toFixed(2);
  $('mPrice').textContent = fmt(m.current_price);
  $('mStop').textContent = fmt(m.atr_stop);
  $('mEma').textContent = fmt(m.ema50);
  $('mSma').textContent = fmt(m.sma200);
  $('mExposure').textContent = m.market_exposure;
  $('mKelly').textContent = m.fractional_kelly;

  const body = $('tradesBody');
  body.innerHTML = '';
  if (d.trades && d.trades.length > 0) {
    d.trades.slice().reverse().forEach(t => {
      const cls = t.return_pct >= 0 ? 'ret-pos' : 'ret-neg';
      const sign = t.return_pct >= 0 ? '+' : '';
      body.innerHTML += `<tr>
        <td>${t.entry_date}</td>
        <td>${t.exit_date}</td>
        <td class="${cls}">${sign}${t.return_pct}%</td>
        <td>${t.holding_days}일</td>
        <td>${t.exit_reason}</td>
      </tr>`;
    });
    $('tradesSection').style.display = 'block';
  } else {
    $('tradesSection').style.display = 'none';
  }

  const iList = $('insightsList');
  iList.innerHTML = d.insights.map(i => `<li>${i}</li>`).join('');

  const wList = $('warnList');
  wList.innerHTML = d.warnings.map(w => `<li>${w}</li>`).join('');
}

$('tickerInput').addEventListener('keydown', e => { if (e.key === 'Enter') runAnalysis(); });
</script>
</body>
</html>"""


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    port = int(os.getenv("PORT", 8000))
    local_url = f"http://127.0.0.1:{port}"

    # 로컬 실행일 때만 브라우저 자동 열기 (PORT 환경변수가 없으면 로컬로 간주)
    if not os.getenv("PORT"):
        print("\n" + "=" * 50)
        print("  Long-Term Quant 앱이 실행됩니다")
        print(f"  브라우저 주소: {local_url}")
        print("  같은 와이파이의 폰에서 접속하려면:")
        print(f"    http://<이 컴퓨터의 IP>:{port}")
        print("  종료하려면 이 창에서 Ctrl+C 를 누르세요")
        print("=" * 50 + "\n")
        threading.Timer(1.5, lambda: webbrowser.open(local_url)).start()

    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
